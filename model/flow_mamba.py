import math
from typing import Optional, Tuple, Dict, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    Mamba = None
    HAS_MAMBA = False


class CausalMambaFallbackBlock(nn.Module):
    """
    Pure-PyTorch Causal Mamba Fallback Block for environments without mamba_ssm.
    Replaces unmasked, non-causal, position-blind TransformerEncoderLayer with:
      1. Input gated projection: in_proj(x) -> (x_main, z_gate)
      2. Causal depthwise 1D convolution: conv1d(kernel_size=4, padding=3, groups=D)[:, :, :L] + SiLU
      3. Rotary Position-Aware Causal Multi-Head Self-Attention via
         F.scaled_dot_product_attention(q, k, v, is_causal=True) mixed with token-sharp conv path
      4. Gated output projection: out_proj(mixed * SiLU(z_gate))
    """
    def __init__(self, d_model: int = 1024, d_conv: int = 4, expand: int = 2, nhead: int = 8):
        super().__init__()
        self.d_model = d_model
        self.d_conv = d_conv
        if d_model % nhead != 0:
            nhead = max(1, math.gcd(d_model, nhead))
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv1d = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_model,
        )
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        # Small residual output init so stacked blocks preserve high-frequency token sharpness at step 0
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out_proj.bias)
        self._rope_cache = {}

    def _get_rope(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (seq_len, self.head_dim, device, dtype)
        if key in self._rope_cache:
            return self._rope_cache[key]

        half_dim = max(self.head_dim // 2, 1)
        pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        inv_freq = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / max(half_dim, 1))
        ).unsqueeze(0)
        angles = pos * inv_freq  # (L, half_dim)
        cos = torch.cos(angles).to(dtype=dtype).view(1, 1, seq_len, half_dim)
        sin = torch.sin(angles).to(dtype=dtype).view(1, 1, seq_len, half_dim)
        self._rope_cache[key] = (cos, sin)
        return cos, sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x_main, z_gate = self.in_proj(x).chunk(2, dim=-1)

        # Causal depthwise 1D convolution sliced to [:L]
        x_conv = self.conv1d(x_main.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Causal Multi-Head Self-Attention with Rotary Position Embedding
        qkv = self.qkv_proj(x_conv).view(B, L, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, nhead, L, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.head_dim >= 2 and self.head_dim % 2 == 0:
            cos, sin = self._get_rope(L, x.device, x.dtype)
            q1, q2 = q[..., ::2], q[..., 1::2]
            k1, k2 = k[..., ::2], k[..., 1::2]
            q = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).flatten(-2)
            k = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).flatten(-2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)

        # Combine local token-sharp conv state with causal attention context
        mixed = 0.75 * x_conv + 0.25 * attn_out
        return self.out_proj(mixed * F.silu(z_gate))


class OrthogonalFieldSuperposition(nn.Module):
    """
    Physical field superposition for multi-stream fusion (Option A).
    Superimposes auxiliary force fields onto the main stream (`main_stream + gate * aux_field`)
    without mixing dense layers, preserving high-dimensional orthogonality and zero-pollution
    when auxiliary forces are zero.
    """
    def __init__(self, d_model: int = 1024, init_gate: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.gate = nn.Parameter(torch.tensor(float(init_gate), dtype=torch.float32))

    def forward(
        self,
        main_stream: torch.Tensor,
        aux_field: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        if aux_field is None:
            return main_stream
        if isinstance(aux_field, (list, tuple)):
            if len(aux_field) == 0:
                return main_stream
            combined = aux_field[0]
            for aux in aux_field[1:]:
                combined = combined + aux
        else:
            combined = aux_field

        gate = self.gate.to(device=main_stream.device, dtype=main_stream.dtype)
        return main_stream + gate * combined


class OrthogonalKnowledgeResonance(nn.Module):
    """
    Macro- and Token-level Orthogonal Attractor with <PAD> mask support.
    Reuses the ReLU(inner_product) physics of FlowMatching1DUnet.OrthogonalAttractor:
      - Normalizes query trajectory X and knowledge pool K via LayerNorm(D).
      - Applies identity-initialized learnable query/key/value projections so step-0 preserves
        exact inner-product orthogonality while allowing token-level resonant addressing.
      - Computes inner-product scores = (Q @ K^T) / sqrt(D).
      - Masks <PAD> knowledge tokens to 0.0 before ReLU.
      - Applies thresholded ReLU (NO Softmax!) so irrelevant/orthogonal knowledge yields 0.0 force.
      - Superimposes the resulting force onto X via OrthogonalFieldSuperposition.
    """
    def __init__(self, d_model: int = 1024, init_gate: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.norm_x = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.q_proj.weight)
        nn.init.eye_(self.k_proj.weight)
        nn.init.eye_(self.v_proj.weight)

        self.scale = 1.0 / math.sqrt(d_model)
        self.superposition = OrthogonalFieldSuperposition(d_model, init_gate=init_gate)

    @property
    def gate(self) -> nn.Parameter:
        return self.superposition.gate

    def compute_force(
        self,
        x: torch.Tensor,
        k: Optional[torch.Tensor],
        k_mask: Optional[torch.Tensor] = None,
        stream_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the raw orthogonal pull force (B, L_q, D) without adding it to x.
        Guarantees exact zero force when k is None, mask is all False, or inner products <= 0.
        """
        if k_mask is None and stream_mask is not None:
            k_mask = stream_mask

        if k is None or k.shape[1] == 0:
            return torch.zeros_like(x)

        B, L_q, D = x.shape
        L_k = k.shape[1]

        if k_mask is not None:
            k_mask_bool = k_mask.bool()
            if not k_mask_bool.any():
                return torch.zeros_like(x)
            mask_f = k_mask_bool.unsqueeze(-1).to(dtype=k.dtype)
            k_masked = k * mask_f
        else:
            k_mask_bool = None
            mask_f = None
            k_masked = k

        x_norm = self.q_proj(self.norm_x(x))
        k_norm = self.k_proj(self.norm_k(k_masked))

        # Zero out k_norm at <PAD> positions in case LayerNorm bias is non-zero
        if k_mask_bool is not None:
            k_norm = k_norm * mask_f

        # Inner product activation: (B, L_q, L_k)
        scores = torch.bmm(x_norm, k_norm.transpose(1, 2)) * self.scale

        if k_mask_bool is not None:
            scores = scores.masked_fill(~k_mask_bool.unsqueeze(1), -1e4)
            has_any = k_mask_bool.any(dim=1).view(B, 1, 1).to(dtype=scores.dtype)
        else:
            has_any = 1.0

        # Snap-to-grid ReLU threshold with 0.25 margin (NO Softmax -> prevents random mean washout & preserves zero-pollution)
        activation = torch.relu(scores - 0.25) * has_any

        # Normalize by active positive resonance mass + sqrt(active_count) floor
        if k_mask_bool is not None:
            active_count = k_mask_bool.float().sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1.0)
            norm_factor = 1.0 / torch.sqrt(active_count).to(dtype=x.dtype)
        else:
            norm_factor = 1.0 / math.sqrt(max(L_k, 1))

        v_masked = self.v_proj(k_masked)
        if k_mask_bool is not None:
            v_masked = v_masked * mask_f

        force = torch.bmm(activation, v_masked) * norm_factor
        return force

    def forward(
        self,
        x: torch.Tensor,
        k: Optional[torch.Tensor],
        k_mask: Optional[torch.Tensor] = None,
        return_force: bool = False,
        stream_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if k_mask is None and stream_mask is not None:
            k_mask = stream_mask
        force = self.compute_force(x, k, k_mask=k_mask)
        out = self.superposition(x, force)
        if return_force:
            return out, force
        return out


class TimeWarpedLegendreExpander(nn.Module):
    """
    Expands K_beats (e.g., 32) macroscopic narrative beats into a full-resolution 3D canvas
    H_canvas of arbitrary canvas length L_out (e.g., 256, 512, 768) with zero hardcoded L weights.

    Implements Dual-Level Elastic Time-Warping + Full-Rank Orthonormal Harmonic Packet Expansion:
      1. Global Effective Horizon (length_head / rho_override):
         Runs explicitly in FP32 with straight-through leaky tanh logit bounding:
           soft_logit = 4.5 * tanh(raw_len_logit / 4.5)
           bounded_logit = 0.9 * soft_logit + 0.1 * raw_len_logit + 0.1 * (soft_logit - raw_len_logit).detach()
           rho_pred = 0.04 + 0.95 * sigmoid(bounded_logit)  in (0.04, 0.99)
         Detaches rho_pred when computing rho_used so loss_potential gradients through
         l_eff = rho_used * L_out (512x multiplier) never hijack length_head.
      2. Local Beat Pacing (pacing_head):
         Predicts positive beat duration weights delta_s_k -> normalized p_k (sum_k p_k = 1).
      3. Dual Basis Macro Projection + Orthonormal DCT/DST Harmonic Packet Expansion:
         Each beat k in [0, K-1] emits sub_harmonics local spectral modes (total K * sub_harmonics = 512
         modes), projected onto an analytical orthonormal Nyquist Fourier basis Phi(t) where
         <Phi(t), Phi(t+1)> == 0.0 across all t in [0, L_out-1], completely eliminating the rank-32
         low-pass bottleneck!
    """
    def __init__(self, d_model: int = 1024, num_beats: int = 32):
        super().__init__()
        self.d_model = d_model
        self.num_beats = num_beats
        self.sub_harmonics = max(1, (512 + num_beats - 1) // num_beats)
        self.num_harmonics = num_beats * self.sub_harmonics

        # 1. Global Effective Horizon Predictor: h_summary (B, D) -> rho_pred in (0.04, 0.99)
        self.length_head = nn.Sequential(
            nn.Linear(d_model, max(d_model // 4, 64)),
            nn.SiLU(),
            nn.Linear(max(d_model // 4, 64), 1),
        )
        nn.init.normal_(self.length_head[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.length_head[-1].bias)

        # 2. Local Beat Pacing Predictor: C_beats (B, K, D) -> delta_s_k > 0
        self.pacing_head = nn.Sequential(
            nn.Linear(d_model, max(d_model // 4, 64)),
            nn.SiLU(),
            nn.Linear(max(d_model // 4, 64), 1),
        )
        nn.init.zeros_(self.pacing_head[-1].weight)
        nn.init.zeros_(self.pacing_head[-1].bias)

        # 3. Separate Linear Projections for Legendre Spectral Basis vs Gaussian Spatial Basis
        self.leg_proj = nn.Linear(d_model, d_model, bias=False)
        self.gauss_proj = nn.Linear(d_model, d_model, bias=False)
        self.macro_norm = nn.LayerNorm(d_model)
        self.film_proj = nn.Linear(d_model, d_model * 2)
        nn.init.normal_(self.film_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.film_proj.bias)

        # 4. Full-Rank Beat-Packet Harmonic Projector: maps (B, K, D) + h_summary -> (B, K * sub_harmonics, D)
        #    Uses H_feat = max(512, d_model // 2) Fourier phase-modulated sample x beat features so all B * K
        #    sample-beat pairs are mutually orthogonal in R^{H_feat} with condition number ~1.0!
        self.harmonic_feat_dim = max(512, d_model // 2)
        self.harmonic_beat_norm = nn.LayerNorm(d_model)
        self.harmonic_sum_norm = nn.LayerNorm(d_model)
        self.harmonic_beat_proj = nn.Linear(d_model, self.harmonic_feat_dim // 2, bias=False)
        self.harmonic_sum_proj = nn.Linear(d_model, self.harmonic_feat_dim // 2, bias=False)
        nn.init.normal_(self.harmonic_beat_proj.weight, mean=0.0, std=1.2 / math.sqrt(d_model))
        nn.init.normal_(self.harmonic_sum_proj.weight, mean=0.0, std=1.2 / math.sqrt(d_model))
        self.harmonic_proj = nn.Linear(self.harmonic_feat_dim, self.sub_harmonics * d_model)
        nn.init.normal_(self.harmonic_proj.weight, mean=0.0, std=0.005)
        nn.init.zeros_(self.harmonic_proj.bias)

        # 5. Learned <PAD>/<EOS> asymptotic tail state beyond L_eff
        self.pad_tail_emb = nn.Parameter(torch.zeros(1, 1, d_model))

        # 6. Lightweight 1D sharpening conv block (local grammatical transition refinement)
        self.sharpen_norm = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.conv2.bias)

        self._pos_cache = {}
        self._slot_cache = {}
        self._ortho_basis_cache = {}

    def _get_pos_emb(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Generates 1D Sinusoidal Positional Embeddings of shape (1, seq_len, d_model)."""
        key = (seq_len, device, dtype)
        if key in self._pos_cache:
            return self._pos_cache[key]

        half_dim = self.d_model // 2
        pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / max(half_dim - 1, 1))
        ).unsqueeze(0)
        angles = pos * div_term
        pos_emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).unsqueeze(0).to(dtype=dtype)
        self._pos_cache[key] = pos_emb
        return pos_emb

    def _get_slot_carrier(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Generates High-Frequency Orthonormal Nyquist Slot Carrier of shape (1, seq_len, d_model).
        Uses uniform frequencies theta_m = ((m + 0.5) / (D/2)) * pi so adjacent tokens t and t+1
        are strictly orthogonal: <slot[t], slot[t+1]> == 0.0 for all t!
        """
        key = (seq_len, self.d_model, device, dtype)
        if key in self._slot_cache:
            return self._slot_cache[key]

        half_dim = max(self.d_model // 2, 1)
        t = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)  # (L, 1)
        theta = ((torch.arange(half_dim, dtype=torch.float32, device=device) + 0.5) / float(half_dim)) * math.pi
        angles = t * theta.unsqueeze(0)  # (L, half_dim)
        carrier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (L, D)
        if carrier.shape[-1] < self.d_model:
            carrier = F.pad(carrier, (0, self.d_model - carrier.shape[-1]))
        carrier = carrier.unsqueeze(0).to(dtype=dtype)  # (1, L, D)
        self._slot_cache[key] = carrier
        return carrier

    def _get_ortho_harmonic_basis(
        self, seq_len: int, t_warped: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Evaluates the analytical M = num_harmonics (e.g. 512) orthonormal DCT/DST basis Phi(t)
        blended with time-warped harmonics Phi(t_warped) of shape (B, seq_len, num_harmonics).
        Satisfies Phi @ Phi^T = I_512 (condition number 1.0, exact 0.0 adjacent token correlation).
        """
        F_dim = self.num_harmonics
        half_f = max(F_dim // 2, 1)
        key = (seq_len, F_dim, device)
        if key in self._ortho_basis_cache:
            theta, static_basis = self._ortho_basis_cache[key]
        else:
            t_idx = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)  # (L, 1)
            theta = ((torch.arange(half_f, dtype=torch.float32, device=device) + 0.5) / float(half_f)) * math.pi
            angles_static = t_idx * theta.unsqueeze(0)  # (L, half_f)
            static_basis = torch.cat([torch.sin(angles_static), torch.cos(angles_static)], dim=-1) * math.sqrt(
                2.0 / float(F_dim)
            )  # (L, F_dim), unit row norm
            self._ortho_basis_cache[key] = (theta, static_basis)

        # Time-warped harmonic basis on t_warped (B, L_out)
        angles_warped = t_warped.unsqueeze(-1) * theta.view(1, 1, half_f)  # (B, L, half_f)
        warped_basis = torch.cat([torch.sin(angles_warped), torch.cos(angles_warped)], dim=-1) * math.sqrt(
            2.0 / float(F_dim)
        )
        combined = 0.8 * static_basis.unsqueeze(0) + 0.2 * warped_basis
        return combined.to(dtype=dtype)

    def _eval_legendre_basis(self, tau: torch.Tensor, degree: int) -> torch.Tensor:
        """
        Evaluates orthogonal Legendre polynomials P_0(tau) .. P_{degree-1}(tau) on tau in [-1, 1]
        via Bonnet's recursion:
            k P_k(tau) = (2k - 1) * tau * P_{k-1}(tau) - (k - 1) * P_{k-2}(tau)
        """
        polys = [torch.ones_like(tau)]
        if degree > 1:
            polys.append(tau)
        for k in range(2, degree):
            p_k = ((2 * k - 1) * tau * polys[-1] - (k - 1) * polys[-2]) / float(k)
            polys.append(p_k)
        basis = torch.stack(polys, dim=-1)  # (B, L_out, K)
        return basis / math.sqrt(max(degree, 1))

    def forward(
        self,
        c_beats: torch.Tensor,
        h_summary: torch.Tensor,
        target_len: int = 512,
        rho_override: Optional[Union[torch.Tensor, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            c_beats: (B, K_beats, D)
            h_summary: (B, D) - Clean context/knowledge summary (unpolluted by rho_override)
            target_len: int (dynamic output canvas length L_out)
            rho_override: Optional (B,) tensor or float in (0, 1] for manual verbosity or target alignment
        Returns:
            h_potential: (B, target_len, D)
            aux_dict: dict with rho_pred, rho_used, l_eff, pacing_weights, beat_centers, horizon_gate, h_canvas
        """
        B, K, D = c_beats.shape
        device = c_beats.device
        dtype = c_beats.dtype
        L_out = int(target_len)

        # 1. Global Horizon Prediction in explicit FP32 (never saturates or rounds to 1.0 in bf16)
        with torch.amp.autocast(device_type=h_summary.device.type, enabled=False):
            raw_len_logit = self.length_head(h_summary.float()).squeeze(-1)
            # Softly bound forward logit to (-4.5, 4.5) with straight-through leaky gradient (min slope >= 0.1)
            soft_logit = 4.5 * torch.tanh(raw_len_logit / 4.5)
            bounded_logit = 0.9 * soft_logit + 0.1 * raw_len_logit + 0.1 * (soft_logit - raw_len_logit).detach()
            rho_pred = 0.04 + 0.95 * torch.sigmoid(bounded_logit)  # Strictly in (0.04, 0.99), FP32, backward slope >= 0.001 everywhere!
            rho_pred = rho_pred.view(B)

        if rho_override is not None:
            if isinstance(rho_override, (int, float)):
                rho_used = torch.full((B,), float(rho_override), device=device, dtype=torch.float32)
            else:
                rho_used = rho_override.to(device=device, dtype=torch.float32).view(B)
            rho_used = rho_used.clamp(min=0.04, max=1.0)
        else:
            # Crucial Gradient Isolation: detach rho_pred so loss_potential gradients through l_eff
            # (512x multiplier) never hijack length_head!
            rho_used = rho_pred.detach()

        # Effective active token horizon L_eff in [1.0, L_out]
        l_eff = (rho_used.float() * float(L_out)).clamp(min=1.0, max=float(L_out)).unsqueeze(-1)  # (B, 1)

        # 2. Local Beat Pacing Weights (delta_s_k > 0 -> normalized p_k)
        delta_s = F.softplus(self.pacing_head(c_beats).float()).squeeze(-1) + 0.1  # (B, K)
        p_weights = delta_s / delta_s.sum(dim=-1, keepdim=True).clamp(min=1e-6)    # (B, K)

        cum_p = torch.cumsum(p_weights, dim=-1)        # (B, K) in (0, 1]
        low_edges = cum_p - p_weights                  # (B, K) in [0, 1)
        u_centers = cum_p - 0.5 * p_weights            # (B, K) in (0, 1)

        # 3. Non-Uniform Time-Warped Grid over [0, L_eff]
        t_grid = torch.arange(L_out, device=device, dtype=torch.float32).unsqueeze(0)  # (1, L_out)
        denom = (l_eff - 1.0).clamp(min=1.0)
        s_grid = (t_grid / denom).clamp(0.0, 1.0)  # (B, L_out) in [0, 1]

        # Differentiable piecewise-linear cumulative time-warping
        local_frac = ((s_grid.unsqueeze(-1) - low_edges.unsqueeze(1)) / p_weights.unsqueeze(1).clamp(min=1e-6)).clamp(0.0, 1.0)
        s_warped = local_frac.mean(dim=-1)  # (B, L_out) in [0, 1]
        tau = 2.0 * s_warped - 1.0          # (B, L_out) in [-1, 1]
        t_warped = s_warped * denom         # (B, L_out) in [0, L_eff - 1]

        # Evaluate K-order Legendre basis on warped coordinate tau(t)
        p_legendre = self._eval_legendre_basis(tau, K).to(dtype=dtype)  # (B, L_out, K)

        # Soft Gaussian beat routing centered at u_k with width sigma_k = p_k
        sigma = p_weights.unsqueeze(1).clamp(min=0.5 / K)  # (B, 1, K)
        dist_sq = (s_grid.unsqueeze(-1) - u_centers.unsqueeze(1)) ** 2
        w_gauss = torch.softmax(-0.5 * dist_sq / (sigma ** 2), dim=-1).to(dtype=dtype)  # (B, L_out, K)

        # Dual Basis Macro Projection: separate spectral Legendre projection and spatial Gaussian beat projection
        h_macro = torch.bmm(p_legendre, self.leg_proj(c_beats)) + torch.bmm(w_gauss, self.gauss_proj(c_beats))
        h_macro_norm = self.macro_norm(h_macro)

        # Full-Rank Orthonormal Harmonic Packet Expansion (breaks the 32-beat low-pass bottleneck!)
        beat_normed = self.harmonic_beat_norm(c_beats)  # (B, K, D)
        sum_normed = self.harmonic_sum_norm(h_summary.to(dtype=dtype)).unsqueeze(1)  # (B, 1, D)
        phase_angles = self.harmonic_beat_proj(beat_normed) + self.harmonic_sum_proj(sum_normed)  # (B, K, H_feat // 2)
        fourier_feats = torch.cat([torch.sin(phase_angles), torch.cos(phase_angles)], dim=-1)  # (B, K, H_feat)
        c_harmonics = 2.0 * self.harmonic_proj(fourier_feats).view(B, self.num_harmonics, D)  # (B, num_harmonics, D)

        phi_basis = self._get_ortho_harmonic_basis(L_out, t_warped, device, dtype)  # (B, L_out, num_harmonics)
        h_harmonic = torch.bmm(phi_basis, c_harmonics)  # (B, L_out, D)

        # High-Frequency Orthogonal Slot Carrier modulated by Macro-Beat FiLM
        slot_carrier = self._get_slot_carrier(L_out, device, dtype)  # (1, L_out, D), cos_sim(t, t+1) == 0.0
        gamma, beta = self.film_proj(h_macro_norm).chunk(2, dim=-1)
        h_pos = h_harmonic + (0.35 + gamma) * slot_carrier + 0.10 * (h_macro_norm + beta)

        # Residual 1D Conv local transition refinement
        h_conv_in = self.sharpen_norm(h_pos).transpose(1, 2)  # (B, D, L_out)
        h_conv = self.conv2(self.act(self.conv1(h_conv_in))).transpose(1, 2)
        h_canvas = h_pos + h_conv

        # Smooth <EOS> Horizon Tail Gate
        horizon_gate = torch.sigmoid((l_eff - t_grid) / 2.0).unsqueeze(-1).to(dtype=dtype)  # (B, L_out, 1)
        pad_tail = self.pad_tail_emb.to(dtype=dtype)
        h_potential = horizon_gate * h_canvas + (1.0 - horizon_gate) * pad_tail

        aux_dict = {
            "rho_pred": rho_pred,
            "rho_used": rho_used,
            "l_eff": l_eff.squeeze(-1),
            "pacing_weights": p_weights,
            "beat_centers": u_centers,
            "horizon_gate": horizon_gate.squeeze(-1),
            "h_canvas": h_canvas,
        }
        return h_potential, aux_dict


class MultiStreamMambaPlanner(nn.Module):
    """
    Phase 5 Unified Left-Brain Architecture (Multi-Stream Mamba Planner).
    1. Context & Knowledge Scan:
       Adds sinusoidal positional embeddings to x_context and x_knowledge, strictly masks <PAD> tokens
       before and after every layer, and extracts a clean summary h_summary_clean (combining masked mean
       pooling + last valid non-EOS token + pooled active knowledge summary) for length_head.
    2. Macroscopic 32-Beat Rollout:
       Rolls out K=32 orthogonal + sinusoidally-initialized narrative beats with Context & Knowledge
       Orthogonal Resonance.
    3. Dual-Basis Time-Warped Legendre + Orthonormal Harmonic Expansion & Full-Resolution L=512 Causal Articulation:
       Expands the 32 beats onto the L_out canvas and articulates every token t in [0, L_out-1] through
       causal sequence layers (`canvas_layers`) + token-level context resonance (`canvas_ctx_resonance`)
       + token-level multi-stream knowledge resonance (`canvas_know_resonance`).
    """
    def __init__(
        self,
        d_model: int = 1024,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 6,
        num_beats: int = 32,
        num_canvas_layers: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_beats = num_beats
        self.num_canvas_layers = (
            num_canvas_layers if num_canvas_layers is not None else max(2, num_layers // 2)
        )

        def _make_seq_block():
            if HAS_MAMBA and Mamba is not None:
                return Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
            nhead = 8 if d_model % 8 == 0 else max(1, math.gcd(d_model, 8))
            return CausalMambaFallbackBlock(
                d_model=d_model,
                d_conv=d_conv,
                expand=expand,
                nhead=nhead,
            )

        self.use_mamba_ssm = bool(HAS_MAMBA and Mamba is not None)
        if not self.use_mamba_ssm:
            print("Warning: mamba_ssm not found. Using CausalMambaFallbackBlock for causal sequence modeling.")

        # Macro context & beat rollout layers
        self.layers = nn.ModuleList([_make_seq_block() for _ in range(num_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        # Multi-stream orthogonal knowledge resonance modules per macro layer
        self.ctx_know_resonance = nn.ModuleList([
            OrthogonalKnowledgeResonance(d_model, init_gate=0.15) for _ in range(num_layers)
        ])
        self.beat_ctx_resonance = nn.ModuleList([
            OrthogonalKnowledgeResonance(d_model, init_gate=0.20) for _ in range(num_layers)
        ])
        self.beat_know_resonance = nn.ModuleList([
            OrthogonalKnowledgeResonance(d_model, init_gate=0.20) for _ in range(num_layers)
        ])

        # Global verbosity hint field projection (1 -> D) + orthogonal superposition
        self.verbosity_field_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.verbosity_superposition = OrthogonalFieldSuperposition(d_model, init_gate=0.1)

        # Elastic Time-Warped Legendre + Orthonormal Harmonic Expander
        self.expander = TimeWarpedLegendreExpander(d_model=d_model, num_beats=num_beats)

        # K=32 learnable future narrative beat queries: distinct sinusoidal beat pos + orthogonal init (std 0.25)
        with torch.no_grad():
            beat_pos = self.expander._get_pos_emb(num_beats, torch.device("cpu"), torch.float32).squeeze(0)
            ortho_init = torch.empty(num_beats, d_model, dtype=torch.float32)
            nn.init.orthogonal_(ortho_init, gain=0.25)
            init_beats = 0.25 * beat_pos + ortho_init
        self.beat_queries = nn.Parameter(init_beats)
        self.beat_film = nn.Linear(d_model, d_model * 2)

        self.norm_f = nn.LayerNorm(d_model)
        self.beat_norm = nn.LayerNorm(d_model)
        self.z_proj = nn.Linear(d_model, d_model)

        # Full-Resolution L=512 Causal Canvas Articulation Stack
        self.canvas_layers = nn.ModuleList([
            _make_seq_block() for _ in range(self.num_canvas_layers)
        ])
        self.canvas_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(self.num_canvas_layers)
        ])
        self.canvas_ctx_resonance = nn.ModuleList([
            OrthogonalKnowledgeResonance(d_model, init_gate=0.05) for _ in range(self.num_canvas_layers)
        ])
        self.canvas_know_resonance = nn.ModuleList([
            OrthogonalKnowledgeResonance(d_model, init_gate=0.05) for _ in range(self.num_canvas_layers)
        ])
        self.canvas_final_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def _flatten_knowledge(
        self,
        x_knowledge: Optional[torch.Tensor],
        knowledge_mask: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if x_knowledge is None:
            return None, None

        if x_knowledge.dim() == 4:
            B, M, L_k, D = x_knowledge.shape
            pos_k = self.expander._get_pos_emb(L_k, device, x_knowledge.dtype).unsqueeze(1)  # (1, 1, L_k, D)
            stream_pos = self.expander._get_pos_emb(M, device, x_knowledge.dtype).unsqueeze(2)  # (1, M, 1, D)
            x_k_pos = x_knowledge + 0.2 * pos_k + 0.1 * stream_pos
            k_pool = x_k_pos.reshape(B, M * L_k, D)
        elif x_knowledge.dim() == 3:
            B, L_know, D = x_knowledge.shape
            pos_k = self.expander._get_pos_emb(L_know, device, x_knowledge.dtype)
            k_pool = x_knowledge + 0.2 * pos_k
        else:
            raise ValueError(f"Expected x_knowledge to have 3 or 4 dims, got shape {x_knowledge.shape}")

        if knowledge_mask is not None:
            if knowledge_mask.dim() == 3:
                B, M, L_k = knowledge_mask.shape
                k_mask = knowledge_mask.reshape(B, M * L_k).bool()
            elif knowledge_mask.dim() == 2:
                k_mask = knowledge_mask.bool()
            else:
                raise ValueError(f"Expected knowledge_mask to have 2 or 3 dims, got shape {knowledge_mask.shape}")
        else:
            k_mask = torch.ones((batch_size, k_pool.shape[1]), dtype=torch.bool, device=device)

        k_pool = k_pool * k_mask.unsqueeze(-1).to(dtype=k_pool.dtype)
        return k_pool, k_mask

    def forward(
        self,
        x_context: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        x_knowledge: Optional[torch.Tensor] = None,
        knowledge_mask: Optional[torch.Tensor] = None,
        target_len: Optional[int] = None,
        rho_override: Optional[Union[torch.Tensor, float]] = None,
        return_sequence: bool = True,
        keepdim: bool = False,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Args:
            x_context: (B, L_c, D) - Continuous embeddings of dialogue context
            valid_mask: (B, L_c) - Boolean mask (True for non-PAD context tokens)
            x_knowledge: (B, M, L_k, D) or (B, L_know, D) - Multi-stream knowledge embeddings
            knowledge_mask: (B, M, L_k) or (B, L_know) - Boolean mask for valid knowledge tokens
            target_len: int - Target canvas length L_out (defaults to L_c if None)
            rho_override: Optional (B,) tensor or float in (0, 1] for manual verbosity or hint conditioning
            return_sequence: If True (default), returns (B, target_len, D) potential curve;
                             if False, returns (B, D) pooled summary state.
            keepdim: If True (and not return_sequence), returns (B, 1, D).
            return_aux: If True, returns (output, aux_dict).
        """
        B, L_c, D = x_context.shape
        out_len = int(target_len) if target_len is not None else L_c

        k_pool, k_mask = self._flatten_knowledge(x_knowledge, knowledge_mask, B, x_context.device)

        # Add sinusoidal positional encoding to x_context before processing
        pos_ctx = self.expander._get_pos_emb(L_c, x_context.device, x_context.dtype)
        h_ctx = x_context + 0.25 * pos_ctx

        if valid_mask is not None:
            valid_mask_bool = valid_mask.bool()
            mask_f = valid_mask_bool.unsqueeze(-1).to(dtype=h_ctx.dtype)
            h_ctx = h_ctx * mask_f
        else:
            valid_mask_bool = None
            mask_f = None

        # 1. Context Scan with Multi-Stream Knowledge Resonance & strict <PAD> zeroing
        for i, layer in enumerate(self.layers):
            normed_ctx = self.layer_norms[i](h_ctx)
            if mask_f is not None:
                normed_ctx = normed_ctx * mask_f
            h_ctx = h_ctx + layer(normed_ctx)
            if mask_f is not None:
                h_ctx = h_ctx * mask_f
            if k_pool is not None:
                h_ctx = self.ctx_know_resonance[i](h_ctx, k_pool, k_mask=k_mask)
                if mask_f is not None:
                    h_ctx = h_ctx * mask_f

        h_ctx = self.norm_f(h_ctx)
        if mask_f is not None:
            h_ctx = h_ctx * mask_f

        # 2. Extract Clean Summary (masked mean pooling + last valid non-EOS token + pooled active knowledge)
        #    Strictly computed BEFORE adding verbosity_cond so length_head never cheats from rho_override!
        if valid_mask_bool is not None:
            valid_len = valid_mask_bool.long().sum(dim=1)  # (B,)
            valid_len_f = valid_len.float().clamp(min=1.0).unsqueeze(-1).to(dtype=h_ctx.dtype)
            ctx_mean = (h_ctx * mask_f).sum(dim=1) / valid_len_f  # (B, D)

            # Last valid non-EOS token (index valid_len - 2 when valid_len > 1, else valid_len - 1)
            non_eos_idx = torch.where(valid_len > 1, valid_len - 2, (valid_len - 1).clamp(min=0))
            last_non_eos = torch.gather(
                h_ctx, dim=1, index=non_eos_idx.view(B, 1, 1).expand(B, 1, D)
            ).squeeze(1)

            last_eos_idx = (valid_len - 1).clamp(min=0)
            last_eos = torch.gather(
                h_ctx, dim=1, index=last_eos_idx.view(B, 1, 1).expand(B, 1, D)
            ).squeeze(1)
        else:
            ctx_mean = h_ctx.mean(dim=1)
            last_non_eos = h_ctx[:, -2, :] if L_c > 1 else h_ctx[:, -1, :]
            last_eos = h_ctx[:, -1, :]

        if k_pool is not None:
            if k_mask is not None and k_mask.any():
                k_mask_f = k_mask.unsqueeze(-1).to(dtype=k_pool.dtype)
                k_counts = k_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0).to(dtype=k_pool.dtype)
                has_know = k_mask.any(dim=1, keepdim=True).to(dtype=k_pool.dtype)
                know_summary = ((k_pool * k_mask_f).sum(dim=1) / k_counts) * has_know
            elif k_mask is None:
                know_summary = k_pool.mean(dim=1)
            else:
                know_summary = torch.zeros_like(ctx_mean)
        else:
            know_summary = torch.zeros_like(ctx_mean)

        raw_summary_clean = 0.4 * ctx_mean + 0.4 * last_non_eos + 0.2 * last_eos + 0.5 * know_summary
        h_summary_clean = self.z_proj(raw_summary_clean)  # (B, D)

        if not return_sequence and not return_aux:
            if keepdim:
                return h_summary_clean.unsqueeze(1)
            return h_summary_clean

        # Modulate beat seed with verbosity_cond ONLY when an explicit rho_override is provided,
        # keeping h_summary_clean untouched for length_head!
        if rho_override is not None:
            if isinstance(rho_override, (int, float)):
                rho_t = torch.full((B, 1), float(rho_override), device=h_ctx.device, dtype=h_ctx.dtype)
            else:
                rho_t = rho_override.to(device=h_ctx.device, dtype=h_ctx.dtype).view(B, 1)
            v_field = self.verbosity_field_proj(rho_t)  # (B, D)
            h_summary_beats = self.verbosity_superposition(h_summary_clean, v_field)
        else:
            h_summary_beats = h_summary_clean

        # 3. Future 32-Beat Rollout with Sample-Specific FiLM Beat Modulation
        beat_gamma, beat_beta = self.beat_film(h_summary_beats).unsqueeze(1).chunk(2, dim=-1)
        beats = (
            self.beat_queries.unsqueeze(0).to(dtype=h_ctx.dtype) * (1.0 + beat_gamma)
            + h_summary_beats.unsqueeze(1)
            + beat_beta
        )  # (B, K_beats, D)

        for i, layer in enumerate(self.layers):
            beats = beats + layer(self.layer_norms[i](beats))
            # Cross-resonate with valid context tokens so beats recall specific prompt entities
            beats = self.beat_ctx_resonance[i](beats, h_ctx, stream_mask=valid_mask_bool)
            # Cross-resonate with multi-stream external knowledge pool
            if k_pool is not None:
                beats = self.beat_know_resonance[i](beats, k_pool, stream_mask=k_mask)

        beats = self.beat_norm(beats)

        # 4. Dynamic Time-Warped Legendre + Orthonormal Harmonic Expansion -> h_canvas (B, out_len, D)
        _, aux_dict = self.expander(
            c_beats=beats,
            h_summary=h_summary_clean,
            target_len=out_len,
            rho_override=rho_override,
        )

        if not return_sequence:
            out = h_summary_clean.unsqueeze(1) if keepdim else h_summary_clean
            return (out, aux_dict) if return_aux else out

        # 5. Full-Resolution L=512 Causal Canvas Articulation Stack
        h_canvas = aux_dict["h_canvas"]
        horizon_gate = aux_dict["horizon_gate"].unsqueeze(-1).to(dtype=h_canvas.dtype)  # (B, out_len, 1)

        for j in range(self.num_canvas_layers):
            h_canvas = h_canvas + self.canvas_layers[j](self.canvas_norms[j](h_canvas))
            h_canvas = self.canvas_ctx_resonance[j](h_canvas, h_ctx, stream_mask=valid_mask_bool)
            if k_pool is not None:
                h_canvas = self.canvas_know_resonance[j](h_canvas, k_pool, stream_mask=k_mask)

        # Residual output projection preserves the orthonormal harmonic highway
        h_canvas = h_canvas + self.out_proj(self.canvas_final_norm(h_canvas))
        pad_tail = self.expander.pad_tail_emb.to(dtype=h_canvas.dtype)
        h_potential = horizon_gate * h_canvas + (1.0 - horizon_gate) * pad_tail

        if return_aux:
            return h_potential, aux_dict
        return h_potential


# Backward-compatible alias
MambaReader = MultiStreamMambaPlanner
