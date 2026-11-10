import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_deterministic_ortho_mix(hidden_dim: int, seed: int = 42) -> torch.Tensor:
    """
    Builds a fixed deterministic orthogonal channel-mixing matrix of shape (hidden_dim, hidden_dim)
    using QR decomposition with positive diagonal sign convention (matches HiPPOOracleCUDA).
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    rand_mat = torch.randn((hidden_dim, hidden_dim), generator=gen, dtype=torch.float32)
    q, r = torch.linalg.qr(rand_mat)
    diag_sign = torch.sign(torch.diagonal(r))
    diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
    ortho_mix = q * diag_sign.unsqueeze(0)
    return ortho_mix


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim, frequency_embedding_size=256):
        super().__init__()
        self.freq_dim = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.freq_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        # Scale t in [0, 1] by 1000.0 so sinusoidal frequencies span full periods
        t_scaled = t.float() * 1000.0
        emb = t_scaled[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


class AdaLN(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.linear = nn.Linear(cond_dim, 2 * hidden_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        if cond_dim == hidden_dim:
            with torch.no_grad():
                self.linear.weight[:hidden_dim, :].copy_(0.1 * torch.eye(hidden_dim))

    def forward(self, x: torch.Tensor, h_potential: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D), h_potential: (B, L, cond_dim) or (B, cond_dim)
        x_norm = self.norm(x)
        shift, scale = self.linear(h_potential).chunk(2, dim=-1)
        if shift.dim() == 2:
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        return x_norm * (1.0 + scale) + shift


class OrthogonalAttractor(nn.Module):
    """
    Implements Span-Aware Knowledge & Symbol Extraction via 3-Gram Cosine Resonance
    plus Exact Single-Char Value Grafting (`ExactSpanPointer`):
      - Queries `x` and keys `k` incorporate immediate left/right neighbor context via a
        depthwise 1D convolution (`kernel_size=3, groups=hidden_dim`) gated by `span_mix`,
        disambiguating repeated characters inside multi-char entities ("布鲁因斯玛") or
        code identifiers (`user_account_id`).
      - Values `k_val` remain the pure, un-blurred single-character `TinyCharEncoderCUDA`
        embeddings so copied characters retain sharp 1024-D dictionary coordinates.
      - Masks out <PAD> prompt tokens so target tokens are not overwritten by padding keys.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm_x = nn.LayerNorm(hidden_dim)
        self.norm_k = nn.LayerNorm(hidden_dim)
        self.channel_weight = nn.Parameter(torch.ones(hidden_dim))
        self.logit_scale = nn.Parameter(torch.tensor(15.0))
        self.snap_threshold = 0.08

        # Upgrade A: 3-gram local span context mixer for query/key disambiguation
        self.span_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )
        nn.init.normal_(self.span_conv.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.span_conv.bias)
        self.span_mix = nn.Parameter(torch.tensor(0.10))

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        k_val: torch.Tensor = None,
        k_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        x: (B, L_x, D) - Unet's current generative trajectory
        k: (B, L_k, D) - Knowledge attractor keys
        k_val: (B, L_k, D) - Pure un-blurred single-char attractor values (defaults to k)
        k_mask: (B, L_k) - Boolean mask indicating non-PAD tokens in k (True=valid, False=PAD)
        """
        if k_val is None:
            k_val = k

        mix = torch.tanh(self.span_mix)
        x_span = x + mix * self.span_conv(x.transpose(1, 2)).transpose(1, 2)

        if k_mask is not None:
            k_clean = k * k_mask.unsqueeze(-1).to(k.dtype)
        else:
            k_clean = k
        k_span = k_clean + mix * self.span_conv(k_clean.transpose(1, 2)).transpose(1, 2)

        q_unit = F.normalize(self.norm_x(x_span) * self.channel_weight, dim=-1, eps=1e-6)
        k_unit = F.normalize(self.norm_k(k_span) * self.channel_weight, dim=-1, eps=1e-6)

        # Cosine Similarity Matrix: (B, L_x, L_k) in [-1, 1]
        sim = torch.bmm(q_unit, k_unit.transpose(1, 2))

        # Diagonal Span Relay + Adjacent Same-Key Exclusion (Pauli anti-collapse against AABB stuttering):
        # 1. `diag_relay`: rewards contiguous span progression (l-1, m-1) -> (l, m) -> (l+1, m+1);
        # 2. `same_key_competitor`: suppresses two adjacent target positions (l, l+1) from locking
        #    onto the exact same single source key index m ("随机" -> "随随", "方法" -> "法法").
        if sim.shape[1] > 1 and sim.shape[2] > 1:
            s_pos = F.relu(sim.masked_fill(~k_mask.unsqueeze(1), 0.0) if k_mask is not None else sim)
            diag_prev = torch.zeros_like(s_pos)
            diag_next = torch.zeros_like(s_pos)
            diag_prev[:, 1:, 1:] = s_pos[:, :-1, :-1]
            diag_next[:, :-1, :-1] = s_pos[:, 1:, 1:]
            diag_relay = 0.5 * (diag_prev + diag_next)

            same_prev = torch.zeros_like(s_pos)
            same_next = torch.zeros_like(s_pos)
            same_prev[:, 1:, :] = s_pos[:, :-1, :]
            same_next[:, :-1, :] = s_pos[:, 1:, :]
            same_key_competitor = torch.maximum(same_prev, same_next)

            sim = sim + 0.25 * diag_relay - 0.30 * same_key_competitor

        # Mask out <PAD> keys so padding receives strictly 0 attention
        if k_mask is not None:
            sim = sim.masked_fill(~k_mask.unsqueeze(1), -1e4)

        # Snap-to-Grid Threshold Gate: suppresses uncorrelated / negated distractors
        max_sim = sim.max(dim=-1, keepdim=True).values
        gate = torch.sigmoid((max_sim - self.snap_threshold) * 20.0)

        scale = self.logit_scale.clamp(min=1.0, max=35.0)
        attn = F.softmax(sim * scale, dim=-1) * gate

        # Exact single-character value copy from un-blurred k_val
        force = torch.bmm(attn, k_val)
        return force


class Gated1DResBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()

        self.time_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.normal_(self.time_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.time_proj.bias)

        self.adaLN1 = AdaLN(hidden_dim, cond_dim)
        self.attractor = OrthogonalAttractor(hidden_dim)
        # Initialized to 0.25 so Step 0 actively engages k_attractors and opens smoothly via gradient descent
        self.attractor_gate = nn.Parameter(torch.full((hidden_dim,), 0.25))

        # Upgrade B: Full-Clause Structural Self-Attention (32x32 global syntax & bracket weaver)
        self.struct_norm = nn.LayerNorm(hidden_dim)
        self.struct_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.struct_gate = nn.Parameter(torch.full((hidden_dim,), 0.15))

        padding = ((kernel_size - 1) * dilation) // 2
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation)
        self.act1 = nn.SiLU()

        self.adaLN2 = AdaLN(hidden_dim, cond_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.conv2.bias)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        h_potential: torch.Tensor,
        k_attractors: torch.Tensor = None,
        k_val: torch.Tensor = None,
        k_mask: torch.Tensor = None,
    ):
        residual = x

        # 1. Inject Time
        t_spatial = self.time_proj(t_emb).unsqueeze(1)
        x = x + t_spatial

        # 2. Inject Momentum (Macroscopic Wind via AdaLN)
        x = self.adaLN1(x, h_potential)

        # 3. Inject Knowledge Attractor smoothly into the residual stream
        if k_attractors is not None:
            force = self.attractor(x, k_attractors, k_val=k_val, k_mask=k_mask)
            x = x + torch.tanh(self.attractor_gate) * force

        # 4. Full-Clause Structural Coordination (Global 32x32 Self-Attention for brackets & syntax)
        x_struct = self.struct_norm(x)
        attn_out, _ = self.struct_attn(x_struct, x_struct, x_struct, need_weights=False)
        x = x + torch.tanh(self.struct_gate) * attn_out

        # 5. Sculpt Local Geometry (1D Dilated Conv)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = x.transpose(1, 2)
        x = self.act1(x)

        x = self.adaLN2(x, h_potential)
        x = x.transpose(1, 2)
        x = self.conv2(x)
        x = x.transpose(1, 2)

        return residual + x


class CavitySpectralSlotRefiner(nn.Module):
    """
    Mean-Field Cavity Debiasing + High-Band Spectral Mode Completion + 32-D Spatial Slot Deblurring (#050):
      1. Restores centered clause fluctuation amplitude (`target_fluct_std`) when multi-clause cavity
         superposition shrinks orthogonal fluctuations around the common clause mean `z_mu`, while leaving
         clean unit-shell clauses untouched (`is_cavity == 0`).
      2. Synthesizes / rescales attenuated high Legendre bands (k >= 6) from low bands (k = 0..5) via
         `spectral_mode_mlp`.
      3. Applies contextual 1D slot deblurring (`slot_deblur_conv`) strictly inside the 32-D `subspace_proj`
         coordinates to prevent 992-D nullspace overfitting (#049).
    """

    def __init__(self, hidden_dim: int = 1024, num_subbands: int = 32, seq_len: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_subbands = num_subbands
        self.d_sub = hidden_dim // num_subbands
        self.seq_len = seq_len

        self.register_buffer("z_mu", torch.zeros(1, hidden_dim, dtype=torch.float32))
        self.register_buffer("target_fluct_std", torch.tensor([[0.90]], dtype=torch.float32))
        init_bands = torch.arange(1, num_subbands + 1, dtype=torch.float32).pow(-0.25)
        init_bands = init_bands / init_bands.pow(2).mean().sqrt().clamp(min=1e-5)
        self.register_buffer("band_rms", init_bands)
        self.register_buffer("slot_mu", torch.zeros(1, seq_len, self.d_sub, dtype=torch.float32))
        self.register_buffer("calib_steps", torch.tensor(0, dtype=torch.long))

        self.spectral_mode_mlp = nn.Sequential(
            nn.Linear(num_subbands, num_subbands * 2),
            nn.SiLU(),
            nn.Linear(num_subbands * 2, num_subbands),
        )
        nn.init.normal_(self.spectral_mode_mlp[0].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.spectral_mode_mlp[0].bias)
        nn.init.normal_(self.spectral_mode_mlp[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.spectral_mode_mlp[-1].bias)

        self.slot_deblur_conv = nn.Sequential(
            nn.Conv1d(self.d_sub * 2, self.d_sub * 4, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(self.d_sub * 4, self.d_sub * 4, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(self.d_sub * 4, self.d_sub, kernel_size=3, padding=1),
        )
        nn.init.normal_(self.slot_deblur_conv[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.slot_deblur_conv[-1].bias)

    @torch.no_grad()
    def update_stats(
        self,
        z_clean: torch.Tensor,
        basis: torch.Tensor,
        max_calib_steps: int = 100,
        momentum: float = 0.05,
    ) -> bool:
        step = int(self.calib_steps.item())
        if step >= max_calib_steps or z_clean.shape[0] < 2:
            return False

        device = z_clean.device
        amp_dev = device.type if device.type in ("cuda", "cpu") else "cpu"
        with torch.amp.autocast(device_type=amp_dev, enabled=False):
            zf = z_clean.detach().float()
            B = zf.shape[0]
            batch_z_mu = zf.mean(dim=0, keepdim=True)
            proj = (zf * batch_z_mu).sum(dim=-1, keepdim=True) / (batch_z_mu.pow(2).sum(dim=-1, keepdim=True) + 1e-8)
            fluct = zf - proj * batch_z_mu
            batch_fluct_std = fluct.std(dim=-1).mean().clamp(min=0.20, max=0.995).view(1, 1)

            z_sub = zf.view(B, self.num_subbands, self.d_sub)
            batch_band_rms = z_sub.pow(2).mean(dim=(0, 2)).sqrt().clamp(min=1e-5)

            if basis.shape[1] == self.seq_len:
                hw_32 = torch.einsum("bkd,kl->bld", z_sub, basis.float()) * math.sqrt(float(self.num_subbands))
                batch_slot_mu = hw_32.mean(dim=0, keepdim=True).float()
            else:
                batch_slot_mu = self.slot_mu.clone()

            batch_z_mu = batch_z_mu.to(device=self.z_mu.device, dtype=self.z_mu.dtype)
            batch_fluct_std = batch_fluct_std.to(device=self.target_fluct_std.device, dtype=self.target_fluct_std.dtype)
            batch_band_rms = batch_band_rms.to(device=self.band_rms.device, dtype=self.band_rms.dtype)
            batch_slot_mu = batch_slot_mu.to(device=self.slot_mu.device, dtype=self.slot_mu.dtype)

            if step == 0:
                self.z_mu.copy_(batch_z_mu)
                self.target_fluct_std.copy_(batch_fluct_std)
                self.band_rms.copy_(batch_band_rms)
                self.slot_mu.copy_(batch_slot_mu)
            else:
                self.z_mu.lerp_(batch_z_mu, momentum)
                self.target_fluct_std.lerp_(batch_fluct_std, momentum)
                self.band_rms.lerp_(batch_band_rms, momentum)
                self.slot_mu.lerp_(batch_slot_mu, momentum)

            self.calib_steps.add_(1)
            return True

    def forward(
        self,
        z_in: torch.Tensor,
        basis: torch.Tensor,
        subspace_proj: torch.Tensor,
    ):
        """
        Args:
            z_in: (B, D) input 2D clause vector
            basis: (K, L) orthonormal Legendre basis
            subspace_proj: (D, d_sub) principal character subspace projection
        Returns:
            z_ref: (B, D) refined 2D clause vector
            h_wave_ref: (B, L, D) unpacked 1024-D wave in span(subspace_proj)
            h_wave_32: (B, L, d_sub) deblurred 32-D slot trajectory
            corrupt_strength: (B, 1) estimated cavity/spectral corruption level in [0, 1]
        """
        B, D = z_in.shape
        K = self.num_subbands
        d_sub = self.d_sub
        L = basis.shape[1]

        z_sub = z_in.view(B, K, d_sub)
        band_energy = z_sub.pow(2).mean(dim=-1).sqrt()  # (B, K)
        band_rms = self.band_rms.to(device=z_in.device, dtype=z_in.dtype).unsqueeze(0).clamp(min=1e-5)

        hi_rms_in = band_energy[:, 6:].pow(2).mean(dim=-1, keepdim=True).sqrt()
        hi_rms_ref = band_rms[:, 6:].pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-5)
        hi_ratio = hi_rms_in / hi_rms_ref
        hi_alive = torch.where(
            hi_ratio >= 0.65,
            torch.ones_like(hi_ratio),
            (hi_ratio / 0.65).clamp(min=0.0, max=1.0),
        )
        per_band_gate = ((band_energy / band_rms) / 0.45).clamp(min=0.0, max=1.0)
        active_band = torch.where(hi_ratio >= 0.65, (band_energy > 1e-3).to(z_in.dtype), per_band_gate)

        # 1. Mean-Field Cavity Debiasing
        t_std = self.target_fluct_std.to(device=z_in.device, dtype=z_in.dtype).clamp(min=0.20, max=0.995)
        if self.z_mu.norm().item() > 1e-5:
            z_mu = self.z_mu.to(device=z_in.device, dtype=z_in.dtype)
        elif B >= 2:
            bm = z_in.mean(dim=0, keepdim=True)
            bm_unit = F.normalize(bm, dim=-1, eps=1e-6)
            canon_mu_norm = math.sqrt(float(D)) * (1.0 - t_std.pow(2)).clamp(min=0.02).sqrt()
            z_mu = bm_unit * canon_mu_norm
        else:
            z_mu = self.z_mu.to(device=z_in.device, dtype=z_in.dtype)

        mu_norm_sq = z_mu.pow(2).sum(dim=-1, keepdim=True)
        if mu_norm_sq.item() > 1e-5:
            proj_c = (z_in * z_mu).sum(dim=-1, keepdim=True) / (mu_norm_sq + 1e-6)
            z_fluct = z_in - proj_c * z_mu
            fluct_std = z_fluct.std(dim=-1, keepdim=True).clamp(min=1e-5)
            is_cavity = ((fluct_std < 0.96 * t_std) | (proj_c > 1.03)).to(dtype=z_in.dtype)
            gain = (t_std / fluct_std).clamp(min=1.0, max=4.0)
            z_deb_raw = z_mu + z_fluct * gain
            z_deb_raw = z_deb_raw / z_deb_raw.std(dim=-1, keepdim=True).clamp(min=1e-5)
            z_deb = is_cavity * z_deb_raw + (1.0 - is_cavity) * z_in
            cavity_severity = is_cavity * (1.0 - (fluct_std / t_std).clamp(max=1.0))
        else:
            z_deb = z_in
            cavity_severity = torch.zeros((B, 1), device=z_in.device, dtype=z_in.dtype)

        # 2. High-Band Spectral Mode Completion (synthesizes attenuated k >= 6 from k = 0..5)
        expected_e = (
            (band_rms.pow(2) * active_band).sum(dim=-1, keepdim=True)
            / band_rms.pow(2).sum().clamp(min=1e-5)
        ).clamp(min=0.05, max=1.0)
        z_trunc_scaled = z_sub * expected_e.sqrt().unsqueeze(-1)
        mode_delta = self.spectral_mode_mlp(z_trunc_scaled.transpose(1, 2)).transpose(1, 2)
        mode_gate = (1.0 - active_band).unsqueeze(-1)
        if self.training:
            mode_gate = mode_gate.clamp(min=0.05)
        z_synth_modes = z_trunc_scaled + mode_gate * mode_delta
        z_spec = z_synth_modes.reshape(B, D)
        z_spec = z_spec / z_spec.std(dim=-1, keepdim=True).clamp(min=1e-5)

        spec_mix = 1.0 - hi_alive
        if self.training:
            spec_mix = spec_mix.clamp(min=0.05)
        z_ref = (1.0 - spec_mix) * z_deb + spec_mix * z_spec
        z_ref = z_ref / z_ref.std(dim=-1, keepdim=True).clamp(min=1e-5)

        corrupt_strength = torch.maximum(1.0 - hi_alive, cavity_severity).clamp(min=0.0, max=1.0)

        # 3. 32-D Spatial Slot Unpacking & Deblurring
        scale_k = math.sqrt(float(K))
        x_low_eff = torch.einsum("bkd,kl->bld", z_ref.view(B, K, d_sub), basis) * scale_k
        x_low_raw = torch.einsum("bkd,kl->bld", z_sub, basis) * scale_k
        if self.slot_mu.shape[1] == L:
            slot_mu = self.slot_mu.to(device=z_in.device, dtype=z_in.dtype)
        else:
            slot_mu = F.interpolate(
                self.slot_mu.to(device=z_in.device, dtype=z_in.dtype).transpose(1, 2),
                size=L,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        deblur_in = torch.cat([x_low_eff, x_low_raw - slot_mu], dim=-1).transpose(1, 2)
        deblur_delta = self.slot_deblur_conv(deblur_in).transpose(1, 2)
        deblur_gate = corrupt_strength.unsqueeze(-1)
        if self.training:
            deblur_gate = deblur_gate.clamp(min=0.05)
        h_wave_32 = x_low_eff + deblur_gate * deblur_delta
        h_wave_ref = torch.matmul(h_wave_32, subspace_proj.T)
        return z_ref, h_wave_ref, h_wave_32, corrupt_strength


class SubspaceSlotGeneratorHead(nn.Module):
    """
    Orthogonally-Isolated 32-D Character Subspace Endpoint Generator (#049 / #050):
    Operates strictly on 32-D principal subspace coordinates `[h_wave_32, x_normed_32, x_t_32]`
    plus scalar timestep embedding `t_emb` so the 32-D character prediction cannot overfit to
    992-D orthogonal nullspace noise.
    """

    def __init__(self, hidden_dim: int = 1024, d_sub: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_sub = d_sub
        self.t_proj = nn.Linear(hidden_dim, d_sub)
        nn.init.normal_(self.t_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.t_proj.bias)

        self.conv_net = nn.Sequential(
            nn.Conv1d(d_sub * 4, d_sub * 4, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(d_sub * 4, d_sub * 4, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(d_sub * 4, d_sub, kernel_size=3, padding=1),
        )
        nn.init.normal_(self.conv_net[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.conv_net[-1].bias)

        self.res_linear = nn.Linear(d_sub * 3, d_sub)
        nn.init.normal_(self.res_linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.res_linear.bias)

    def forward(
        self,
        h_wave_32: torch.Tensor,
        x_normed_32: torch.Tensor,
        x_t_32: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        t_32 = self.t_proj(t_emb).unsqueeze(1).expand_as(h_wave_32)
        feat_4 = torch.cat([h_wave_32, x_normed_32, x_t_32, t_32], dim=-1)
        conv_delta = self.conv_net(feat_4.transpose(1, 2)).transpose(1, 2)
        lin_delta = self.res_linear(torch.cat([h_wave_32, x_normed_32, x_t_32], dim=-1))
        return conv_delta + lin_delta


class FlowMatching1DUnet(nn.Module):
    """
    Stateless Universal Left-Brain Crystallizer v2 (`LeftBrainCrystallizer` / `FlowMatching1DUnet` #025 / #050 / #059):
    - Unpacks a 2D Punctuation-Clause Vector `h_potential: (B, 1024)` into a full-spectrum
      32-mode Legendre trajectory `h_wave_ref: (B, L, 1024)` over the 32-D principal manifold subspace
      via `CavitySpectralSlotRefiner` (mean-field cavity debiasing + high-band mode completion + 32-D slot deblur).
    - Supports direct binding of Phase 1 v2's native orthogonal `encoder.char_subspace_proj` (`bind_encoder_subspace`)
      so `subspace_proj` is locked to the isotropic 32-D character anchor subspace (`subspace_locked == 1`)
      without scalar-mean PC1 bias, while preserving backward compatibility with already-calibrated checkpoints.
    - Removes the legacy `x_1_hat = h_wave + ...` hard skip cheat and restores `x_t` with full `1.0` weight
      as the primary ODE trajectory: `x = x_t + sigmoid(wave_in_gate) * h_wave_ref + 0.1 * pos_x`.
    - Enforces the 992-D Nullspace Isolation Rule (#049):
      * 32-D character subspace endpoint `x_1_hat_32` is generated strictly from 32-D features
        `[h_wave_32, x_normed @ P, x_t @ P]` + `t_emb` via `slot_subspace_head` + gated `alpha_skip_32`;
      * 992-D complement endpoint `x_1_hat_orth` projects out `P`: `raw_full - (raw_full @ P) @ P.T`;
      * Combines `x_1_hat = x_1_hat_32 @ P.T + x_1_hat_orth` + optional frozen `Translator` codebook snap.
    """
    def __init__(
        self,
        hidden_dim: int = 1024,
        cond_dim: int = 1024,
        num_layers: int = 6,
        num_subbands: int = 32,
    ):
        super().__init__()
        if hidden_dim % num_subbands != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_subbands ({num_subbands})")

        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.num_subbands = num_subbands
        self.d_sub = hidden_dim // num_subbands

        self.t_embedder = TimestepEmbedder(hidden_dim)

        # 2D Sentence Vector Legendre Unpacker (matches HiPPOOracleCUDA seed=42 QR basis)
        ortho_mix = build_deterministic_ortho_mix(hidden_dim, seed=42)
        self.register_buffer("ortho_mix", ortho_mix)
        self.register_buffer("subspace_proj", ortho_mix[:, : self.d_sub].contiguous())
        self.register_buffer("subspace_locked", torch.tensor(0, dtype=torch.long))

        # Cavity-Tolerant Spectral & 32-D Slot Refiner (#050)
        self.cavity_refiner = CavitySpectralSlotRefiner(
            hidden_dim=hidden_dim,
            num_subbands=num_subbands,
            seq_len=32,
        )

        self.sent_wave_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.normal_(self.sent_wave_proj[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.sent_wave_proj[-1].bias)

        # Primary stream wave conditioning gate (x_t has full 1.0 weight)
        self.wave_in_gate = nn.Parameter(torch.full((1, 1, hidden_dim), 0.85))

        self.blocks = nn.ModuleList([
            Gated1DResBlock(
                hidden_dim=hidden_dim,
                cond_dim=cond_dim,
                kernel_size=5,
                dilation=2**(i % 4)
            ) for i in range(num_layers)
        ])

        self.final_norm = AdaLN(hidden_dim, cond_dim)
        # 992-D Orthogonal Complement projection head (32-D subspace component is projected out)
        self.final_proj = nn.Linear(hidden_dim, hidden_dim)
        with torch.no_grad():
            self.final_proj.weight.mul_(0.01)
            nn.init.zeros_(self.final_proj.bias)

        # Dedicated 32-D Principal Subspace Generator Head + Gated 32-D Wave Skip (#049 / #050)
        # sigmoid(0.75) ~= 0.679 matches canonical valid-span Parseval ratio at Step 0
        self.alpha_skip_32 = nn.Parameter(torch.full((1, 1, self.d_sub), 0.75))
        self.slot_subspace_head = SubspaceSlotGeneratorHead(hidden_dim=hidden_dim, d_sub=self.d_sub)

        # Optional In-Flow Frozen Translator Codebook Langevin Snap parameters & non-persistent buffer
        self.snap_temp = nn.Parameter(torch.tensor(0.18, dtype=torch.float32))
        self.snap_step = nn.Parameter(torch.tensor(-0.80, dtype=torch.float32))
        self.register_buffer("_char_codebook_norm", None, persistent=False)
        self.__dict__["_bound_translator"] = None

        # In-Model Adaptive Manifold Auto-Calibration Buffers (persisted in unet.pt)
        self.register_buffer("manifold_mean", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("manifold_std", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("calib_steps", torch.tensor(0, dtype=torch.long))

        self._pos_cache = {}
        self._basis_cache = {}

    def upgrade_legacy_state_dict(self, state_dict: dict, prefix: str = "") -> dict:
        """
        Upgrades a legacy Phase-4 v1/v2 `unet.pt` state_dict in-place so `load_state_dict(..., strict=True)`
        succeeds seamlessly:
          1. Pre-fills any missing v2/#059 parameters/buffers (`cavity_refiner.*`, `slot_subspace_head.*`,
             `wave_in_gate`, `alpha_skip_32`, `snap_temp`, `snap_step`, `subspace_locked`) from current model defaults.
          2. If an older checkpoint lacked `subspace_locked` and already has `calib_steps > 0`, marks
             `subspace_locked = 1` so its loaded `subspace_proj` is preserved across resumed runs.
          3. Wakes up dormant legacy `struct_gate` (< 0.13 -> 0.15) and `attractor_gate` (< 0.13 -> 0.25)
             across all 6 blocks while preserving already-active gates (such as block 0's 0.981 attractor_gate).
        """
        is_legacy = f"{prefix}alpha_skip_32" not in state_dict
        had_subspace_locked = f"{prefix}subspace_locked" in state_dict
        own_state = self.state_dict()
        for k, v in own_state.items():
            full_k = f"{prefix}{k}"
            if full_k not in state_dict:
                state_dict[full_k] = v.detach().clone()

        if not had_subspace_locked:
            cs_key = f"{prefix}calib_steps"
            sl_key = f"{prefix}subspace_locked"
            cs_val = int(state_dict[cs_key].item()) if cs_key in state_dict else 0
            lock_val = 1 if cs_val > 0 else 0
            state_dict[sl_key] = torch.tensor(
                lock_val,
                dtype=torch.long,
                device=state_dict[sl_key].device,
            )

        if is_legacy:
            for i in range(len(self.blocks)):
                sg_key = f"{prefix}blocks.{i}.struct_gate"
                if sg_key in state_dict:
                    sg = state_dict[sg_key]
                    if sg.float().abs().mean().item() < 0.13:
                        sign = torch.where(sg >= 0, torch.ones_like(sg), -torch.ones_like(sg))
                        state_dict[sg_key] = sign * sg.abs().clamp(min=0.15)
                ag_key = f"{prefix}blocks.{i}.attractor_gate"
                if ag_key in state_dict:
                    ag = state_dict[ag_key]
                    if ag.float().abs().mean().item() < 0.13:
                        sign = torch.where(ag >= 0, torch.ones_like(ag), -torch.ones_like(ag))
                        state_dict[ag_key] = sign * ag.abs().clamp(min=0.25)
        return state_dict

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self.upgrade_legacy_state_dict(state_dict, prefix=prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.no_grad()
    def bind_encoder_subspace(self, encoder: nn.Module, force: bool = False) -> bool:
        """
        Locks `self.subspace_proj` (`(hidden_dim, d_sub)`) onto Phase 1 v2's native orthogonal
        `encoder.char_subspace_proj.weight` (`(d_sub, hidden_dim)`), orthonormalized via QR in float32
        with positive diagonal sign convention.
        If `self.calib_steps > 0` and `not force`, preserves the already-calibrated `self.subspace_proj`
        and returns False.
        """
        if int(self.calib_steps.item()) > 0 and not force:
            return False

        if (
            encoder is not None
            and hasattr(encoder, "char_subspace_proj")
            and hasattr(encoder.char_subspace_proj, "weight")
            and tuple(encoder.char_subspace_proj.weight.shape) == (self.d_sub, self.hidden_dim)
        ):
            w_t = encoder.char_subspace_proj.weight.detach().float().T.contiguous()  # (D, d_sub)
            q, r = torch.linalg.qr(w_t.cpu())
            diag_r = torch.diagonal(r)
            diag_sign = torch.where(torch.sign(diag_r) == 0, torch.ones_like(diag_r), torch.sign(diag_r))
            q = q * diag_sign.unsqueeze(0)
            self.subspace_proj.copy_(
                q.to(device=self.subspace_proj.device, dtype=self.subspace_proj.dtype)
            )
            self.subspace_locked.fill_(1)
            return True

        return False

    def bind_translator_codebook(self, translator: nn.Module = None) -> None:
        """
        Binds a frozen `ContinuousToTextTranslatorCUDA` (or raw `(V, D)` codebook tensor) for in-flow
        character-manifold Langevin snapping without registering `translator` as a persistent submodule
        in `unet.state_dict()`.
        """
        if translator is None:
            self.__dict__["_bound_translator"] = None
            self._char_codebook_norm = None
            return

        if isinstance(translator, torch.Tensor):
            self.__dict__["_bound_translator"] = None
            self._char_codebook_norm = F.normalize(translator.detach().float(), dim=-1).to(
                device=self.subspace_proj.device
            )
        else:
            self.__dict__["_bound_translator"] = translator
            if hasattr(translator, "fc") and hasattr(translator.fc, "weight"):
                w_norm = F.normalize(translator.fc.weight.detach().float(), dim=-1)
                self._char_codebook_norm = w_norm.to(device=self.subspace_proj.device)

    def apply_char_manifold_snap(
        self,
        x_1_raw: torch.Tensor,
        corrupt_strength: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        When `_char_codebook_norm` is bound, performs a gated soft-argmax Langevin step toward
        the frozen Translator's character prototypes `W_fc_norm`.
        """
        if self._char_codebook_norm is None:
            return x_1_raw

        cb = self._char_codebook_norm.to(device=x_1_raw.device, dtype=x_1_raw.dtype)
        temp = self.snap_temp.to(device=x_1_raw.device, dtype=x_1_raw.dtype).clamp(min=0.05, max=0.50)
        bound_trans = self.__dict__.get("_bound_translator", None)
        if bound_trans is not None:
            bound_trans = bound_trans.to(device=x_1_raw.device)
            logits = bound_trans(self.denormalize(x_1_raw).float()).to(dtype=x_1_raw.dtype)
            probs = F.softmax(logits / temp, dim=-1)
        else:
            x_u = F.normalize(x_1_raw, dim=-1, eps=1e-6)
            probs = F.softmax(torch.matmul(x_u, cb.T) / temp, dim=-1)

        char_dir = torch.matmul(probs, cb)
        if corrupt_strength is not None:
            cs_gate = 0.20 + 0.80 * corrupt_strength.unsqueeze(-1).to(dtype=x_1_raw.dtype)
        else:
            cs_gate = 0.35
        step_sz = F.softplus(self.snap_step.to(device=x_1_raw.device, dtype=x_1_raw.dtype))
        return x_1_raw + cs_gate * step_sz * char_dir

    @torch.no_grad()
    def calibrate_cavity_stats(
        self,
        z_clean: torch.Tensor,
        seq_len: int = 32,
        max_calib_steps: int = 100,
        momentum: float = 0.05,
    ) -> bool:
        basis = self._get_legendre_basis(seq_len, z_clean.device, torch.float32)
        return self.cavity_refiner.update_stats(
            z_clean,
            basis=basis,
            max_calib_steps=max_calib_steps,
            momentum=momentum,
        )

    def _get_legendre_basis(self, L: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Computes the first K = self.num_subbands (32) discrete-orthonormal Legendre polynomials
        evaluated on the fixed sequence window [0, L-1] with uniform midpoint coordinates
        tau_l = (l + 0.5) / L in (0, 1), orthonormalized via two-pass Gram-Schmidt in float64 so that
        basis @ basis.T == I_K and (when K == L == 32) basis.T @ basis == I_32 with 0.0000 neighbor bleed.
        Bit-for-bit identical to HiPPOOracleCUDA._get_legendre_basis.
        """
        key = (L, device, dtype)
        if key in self._basis_cache:
            return self._basis_cache[key]

        K = self.num_subbands
        tau = (torch.arange(L, device="cpu", dtype=torch.float64) + 0.5) / float(max(L, 1))
        u = 2.0 * tau - 1.0

        polys = []
        for k in range(K):
            if k == 0:
                p_k = torch.ones_like(u)
            elif k == 1:
                p_k = u
            else:
                p_k = ((2.0 * k - 1.0) * u * polys[-1] - (k - 1.0) * polys[-2]) / float(k)
            polys.append(p_k)

        raw_basis = torch.stack(polys, dim=0)  # (K, L)

        ortho_rows = []
        for k in range(K):
            v = raw_basis[k]
            if k < L:
                for _ in range(2):
                    for prev in ortho_rows:
                        v = v - torch.dot(v, prev) * prev
            norm = v.norm(p=2).clamp(min=1e-12)
            ortho_rows.append(v / norm)

        basis = torch.stack(ortho_rows, dim=0).to(device=device, dtype=dtype)
        self._basis_cache[key] = basis
        return basis

    def unpack_sentence_vector(self, z_sent: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Unpacks a 2D Clause Vector `z_sent` of shape `(B, D)` into a full-spectrum Legendre
        temporal wave `h_wave` of shape `(B, seq_len, D)` over the 32-D principal subspace.
        Scaled by sqrt(K) so total Parseval energy matches the K orthonormal modes on the valid span.
        """
        B, D = z_sent.shape
        K = self.num_subbands
        d_sub = self.d_sub

        z_sub = z_sent.reshape(B, K, d_sub)
        basis = self._get_legendre_basis(seq_len, z_sent.device, z_sent.dtype)  # (K, seq_len)

        x_low_rec = torch.einsum("bkd,kl->bld", z_sub, basis) * math.sqrt(K)  # (B, seq_len, d_sub)
        proj = self.subspace_proj.to(device=z_sent.device, dtype=z_sent.dtype)  # (D, d_sub)
        return torch.matmul(x_low_rec, proj.T)

    def _get_pos_emb(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Generates 1D Sinusoidal Positional Embeddings of shape (1, seq_len, hidden_dim).
        Breaks translation symmetry so every token position l in [0, L-1] knows its exact order.
        """
        key = (seq_len, device, dtype)
        if key in self._pos_cache:
            return self._pos_cache[key]

        half_dim = self.hidden_dim // 2
        pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=device) * (-math.log(10000.0) / max(half_dim - 1, 1))
        ).unsqueeze(0)  # (1, D/2)
        angles = pos * div_term
        pos_emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).unsqueeze(0).to(dtype=dtype)  # (1, L, D)
        self._pos_cache[key] = pos_emb
        return pos_emb

    @torch.no_grad()
    def calibrate_manifold(
        self,
        x_sample: torch.Tensor,
        valid_mask: torch.Tensor = None,
        max_calib_steps: int = 50,
        momentum: float = 0.05
    ) -> bool:
        """
        Online EMA Auto-Calibration for target manifold scale + Step-0 Principal Subspace Calibration:
        - Strictly computes mean and std over valid (non-PAD) tokens when valid_mask is provided,
          preventing high-padding short sequences from shrinking manifold_std.
        - Step 0: Initializes manifold_mean, manifold_std, and ONLY if `self.subspace_locked == 0`,
          calibrates the 32-D principal subspace `self.subspace_proj` via per-feature centered
          covariance eigendecomposition (avoiding scalar-mean PC1 bias).
        - Steps 1 .. max_calib_steps-1: Smooths manifold_mean and manifold_std via EMA.
        - Step >= max_calib_steps: Locks coordinate scale permanently for stable flow training.
        """
        step = int(self.calib_steps.item())
        if step >= max_calib_steps:
            return False

        x_float = x_sample.detach().float()
        if valid_mask is not None and valid_mask.any():
            valid_tokens = x_float[valid_mask.bool()]
            batch_mean = valid_tokens.mean()
            batch_std = valid_tokens.std().clamp(min=1e-5)
        else:
            valid_tokens = None
            batch_mean = x_float.mean()
            batch_std = x_float.std().clamp(min=1e-5)

        batch_mean_cast = batch_mean.to(device=self.manifold_mean.device, dtype=self.manifold_mean.dtype)
        batch_std_cast = batch_std.to(device=self.manifold_std.device, dtype=self.manifold_std.dtype)

        if step == 0:
            self.manifold_mean.copy_(batch_mean_cast)
            self.manifold_std.copy_(batch_std_cast)

            if int(self.subspace_locked.item()) == 0:
                v_tokens = valid_tokens if (valid_mask is not None and valid_mask.any()) else x_float.reshape(-1, self.hidden_dim)
                if v_tokens.shape[0] >= 2:
                    tok_mean_vec = v_tokens.mean(dim=0, keepdim=True)
                    v_centered = (v_tokens - tok_mean_vec) / batch_std
                    cov = torch.matmul(v_centered.T, v_centered) / float(max(v_centered.shape[0], 1))
                    # Add tiny diagonal jitter for numerical stability before eigh (on CPU for universal device support)
                    cov_cpu = cov.cpu().float() + 1e-6 * torch.eye(self.hidden_dim, dtype=torch.float32)
                    _, eigvecs_cpu = torch.linalg.eigh(cov_cpu)
                    top_vecs = eigvecs_cpu[:, -self.d_sub:].flip(-1).contiguous().to(device=cov.device, dtype=torch.float32)
                    # Deterministic column sign convention
                    max_abs_idx = top_vecs.abs().argmax(dim=0, keepdim=True)
                    col_signs = torch.sign(torch.gather(top_vecs, 0, max_abs_idx))
                    col_signs = torch.where(col_signs == 0, torch.ones_like(col_signs), col_signs)
                    top_vecs = top_vecs * col_signs
                    self.subspace_proj.copy_(
                        top_vecs.to(device=self.subspace_proj.device, dtype=self.subspace_proj.dtype)
                    )
        else:
            self.manifold_mean.lerp_(batch_mean_cast, float(momentum))
            self.manifold_std.lerp_(batch_std_cast, float(momentum))

        self.calib_steps.add_(1)
        return True

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Maps raw encoder trajectory (Mean=mu, Std=sigma) -> Normalized Flow Manifold (Mean=0, Std=1)."""
        mean = self.manifold_mean.to(device=x.device, dtype=x.dtype)
        std = self.manifold_std.to(device=x.device, dtype=x.dtype).clamp(min=1e-5)
        return (x - mean) / std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Restores generated normalized trajectory (Mean=0, Std=1) -> Original Encoder Manifold (Mean=mu, Std=sigma)."""
        mean = self.manifold_mean.to(device=x.device, dtype=x.dtype)
        std = self.manifold_std.to(device=x.device, dtype=x.dtype).clamp(min=1e-5)
        return x * std + mean

    def predict_x1(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        h_potential: torch.Tensor,
        k_attractors: torch.Tensor = None,
        k_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Predicts the denoised normalized endpoint `x_1_hat` (`(B, L, D)`) without the hard `h_wave` skip cheat:
          1. Refines 2D `z_sent` via `self.cavity_refiner` into `z_ref`, `h_wave_ref`, and `h_wave_32`.
          2. Runs `x = x_t + sigmoid(wave_in_gate) * h_wave_ref + 0.1 * pos_x` through all 6 `Gated1DResBlock`s.
          3. Synthesizes 32-D character coordinates `x_1_hat_32` strictly from `[h_wave_32, x_normed @ P, x_t @ P]` + `t_emb`
             and 992-D orthogonal complement `x_1_hat_orth` by projecting `P` out of `final_proj(x_normed)`.
          4. Applies optional frozen-Translator character manifold Langevin snap when bound.
        """
        B, L, D = x_t.shape
        t_emb = self.t_embedder(t)
        pos_x = self._get_pos_emb(L, x_t.device, x_t.dtype)
        P = self.subspace_proj.to(device=x_t.device, dtype=x_t.dtype)

        if h_potential.dim() == 2:
            h_pot = h_potential.to(device=x_t.device, dtype=x_t.dtype)
            basis = self._get_legendre_basis(L, x_t.device, x_t.dtype)
            z_ref, h_wave_ref, h_wave_32, corrupt_strength = self.cavity_refiner(h_pot, basis, P)
            h_global = z_ref.unsqueeze(1).expand(B, L, D)
            h_cond = h_wave_ref + 0.5 * h_global + self.sent_wave_proj(
                torch.cat([h_wave_ref, h_global + pos_x], dim=-1)
            )
        else:
            h_wave_ref = h_potential.to(device=x_t.device, dtype=x_t.dtype)
            h_wave_32 = torch.matmul(h_wave_ref, P)
            corrupt_strength = torch.zeros((B, 1), device=x_t.device, dtype=x_t.dtype)
            h_cond = h_wave_ref

        # Primary ODE trajectory x_t with full 1.0 weight + gated refined wave anchor
        x = x_t + torch.sigmoid(self.wave_in_gate.to(dtype=x_t.dtype)) * h_wave_ref + 0.1 * pos_x

        k_pos = None
        if k_attractors is not None:
            pos_k = self._get_pos_emb(k_attractors.shape[1], k_attractors.device, k_attractors.dtype)
            k_pos = k_attractors + 0.02 * pos_k

        for block in self.blocks:
            x = block(x, t_emb, h_cond, k_attractors=k_pos, k_val=k_attractors, k_mask=k_mask)

        x_normed = self.final_norm(x, h_cond)

        # Orthogonally-Isolated 32-D Subspace Generator + 992-D Complement Projection (#049 / #050)
        x_normed_32 = torch.matmul(x_normed, P)
        x_t_32 = torch.matmul(x_t, P)
        delta_32 = self.slot_subspace_head(h_wave_32, x_normed_32, x_t_32, t_emb)
        skip_32 = torch.sigmoid(self.alpha_skip_32.to(dtype=x_t.dtype))
        x_1_hat_32 = skip_32 * h_wave_32 + delta_32

        raw_full = self.final_proj(x_normed)
        x_1_hat_orth = raw_full - torch.matmul(torch.matmul(raw_full, P), P.T)
        x_1_hat = torch.matmul(x_1_hat_32, P.T) + x_1_hat_orth

        x_1_hat = self.apply_char_manifold_snap(x_1_hat, corrupt_strength=corrupt_strength)
        return x_1_hat

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        h_potential: torch.Tensor,
        k_attractors: torch.Tensor = None,
        k_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Sign-Aligned Velocity Field Predictor for OT-CFM (`LeftBrainCrystallizer` v2):
        Predicts velocity via no-skip subspace-isolated endpoint `x_1_hat = self.predict_x1(...)`.
        """
        B = x_t.shape[0]
        x_1_hat = self.predict_x1(
            x_t=x_t,
            t=t,
            h_potential=h_potential,
            k_attractors=k_attractors,
            k_mask=k_mask,
        )
        t_expand = t.view(B, 1, 1).to(dtype=x_t.dtype)
        denom = (1.0 - t_expand).clamp(min=0.05)
        v_pred = (x_1_hat - x_t) / denom

        return v_pred


# Canonical alias in the Modular Bicameral Architecture (#025)
LeftBrainCrystallizer = FlowMatching1DUnet
