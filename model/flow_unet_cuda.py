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
    Implements Knowledge Extraction via Dimensional Cosine Resonance.
    Preserves the shared 1024-D embedding coordinate axes via diagonal `channel_weight`
    so `h_wave` unpacked from `z_sent` directly resonates with matching token embeddings in `k_attractors`.
    Masks out <PAD> prompt tokens so target tokens are not overwritten by padding keys.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm_x = nn.LayerNorm(hidden_dim)
        self.norm_k = nn.LayerNorm(hidden_dim)
        self.channel_weight = nn.Parameter(torch.ones(hidden_dim))
        self.logit_scale = nn.Parameter(torch.tensor(15.0))
        self.snap_threshold = 0.08

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
        k_val: (B, L_k, D) - Knowledge attractor values (defaults to k)
        k_mask: (B, L_k) - Boolean mask indicating non-PAD tokens in k (True=valid, False=PAD)
        """
        if k_val is None:
            k_val = k

        q_unit = F.normalize(self.norm_x(x) * self.channel_weight, dim=-1, eps=1e-6)
        k_unit = F.normalize(self.norm_k(k) * self.channel_weight, dim=-1, eps=1e-6)

        # Cosine Similarity Matrix: (B, L_x, L_k) in [-1, 1]
        sim = torch.bmm(q_unit, k_unit.transpose(1, 2))

        # Mask out <PAD> keys so padding receives strictly 0 attention
        if k_mask is not None:
            sim = sim.masked_fill(~k_mask.unsqueeze(1), -1e4)

        # Snap-to-Grid Threshold Gate: suppresses uncorrelated / negated distractors
        max_sim = sim.max(dim=-1, keepdim=True).values
        gate = torch.sigmoid((max_sim - self.snap_threshold) * 20.0)

        scale = self.logit_scale.clamp(min=1.0, max=35.0)
        attn = F.softmax(sim * scale, dim=-1) * gate

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
        # Initialized to 0.0 so Step 0 starts prior to attractor lock-in, then opens smoothly via gradient descent
        self.attractor_gate = nn.Parameter(torch.zeros(hidden_dim))

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

        # 4. Sculpt Geometry (1D Conv)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = x.transpose(1, 2)
        x = self.act1(x)

        x = self.adaLN2(x, h_potential)
        x = x.transpose(1, 2)
        x = self.conv2(x)
        x = x.transpose(1, 2)

        return residual + x


class FlowMatching1DUnet(nn.Module):
    """
    Standard Stable Vector Field Predictor for OT-CFM (Right Brain / Motor Cortex).
    - Driven by a 2D Sentence Vector `h_potential: (B, 1024)` (unpacked via 16-mode shifted Legendre
      basis + position-aware residual adapter) as `h_cond`.
    - Primary generative path driven directly by `x_t`: `x = x_t + 0.1 * pos_x`.
    - Knowledge attractors with `<PAD>` key masking injected smoothly into the residual stream.
    - Velocity output: `v_pred = self.final_proj(x_normed)` without 1/(1-t) division or x_t subtraction.
    - Includes built-in Adaptive Manifold Auto-Calibration (manifold_mean & manifold_std).
    """
    def __init__(
        self,
        hidden_dim: int = 1024,
        cond_dim: int = 1024,
        num_layers: int = 6,
        num_subbands: int = 16,
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

        self.sent_wave_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.normal_(self.sent_wave_proj[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.sent_wave_proj[-1].bias)

        self.blocks = nn.ModuleList([
            Gated1DResBlock(
                hidden_dim=hidden_dim,
                cond_dim=cond_dim,
                kernel_size=5,
                dilation=2**(i % 4)
            ) for i in range(num_layers)
        ])

        self.final_norm = AdaLN(hidden_dim, cond_dim)
        # Full-rank 1024x1024 projection head for velocity output
        self.final_proj = nn.Linear(hidden_dim, hidden_dim)
        with torch.no_grad():
            self.final_proj.weight.mul_(0.01)
            nn.init.zeros_(self.final_proj.bias)

        # In-Model Adaptive Manifold Auto-Calibration Buffers (persisted in unet.pt)
        self.register_buffer("manifold_mean", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("manifold_std", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("calib_steps", torch.tensor(0, dtype=torch.long))

        self._pos_cache = {}
        self._basis_cache = {}

    def _get_legendre_basis(self, L: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Computes the first K = self.num_subbands (16) orthonormal shifted Legendre polynomials P_k(tau)
        evaluated at tau_l = (l + 0.5) / L in (0, 1) for l = 0..L-1, with sum_l basis[k, l]^2 == 1.0.
        """
        key = (L, device, dtype)
        if key in self._basis_cache:
            return self._basis_cache[key]

        K = self.num_subbands
        tau = (torch.arange(L, device=device, dtype=torch.float32) + 0.5) / float(max(L, 1))
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
                for prev in ortho_rows:
                    v = v - torch.dot(v, prev) * prev
            norm = v.norm(p=2).clamp(min=1e-6)
            ortho_rows.append(v / norm)

        basis = torch.stack(ortho_rows, dim=0).to(dtype=dtype)
        self._basis_cache[key] = basis
        return basis

    def unpack_sentence_vector(self, z_sent: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Unpacks a 2D Sentence Vector `z_sent` of shape `(B, D)` into a rank-16 smooth Legendre
        temporal wave `h_wave` of shape `(B, seq_len, D)`.
        """
        B, D = z_sent.shape
        K = self.num_subbands
        d_sub = self.d_sub

        z_sub = z_sent.reshape(B, K, d_sub)
        basis = self._get_legendre_basis(seq_len, z_sent.device, z_sent.dtype)  # (K, seq_len)

        x_mixed_rec = torch.einsum("bkd,kl->blkd", z_sub, basis) * math.sqrt(K)
        x_mixed_rec = x_mixed_rec.reshape(B, seq_len, D)

        ortho_T = self.ortho_mix.to(device=z_sent.device, dtype=z_sent.dtype).T
        return torch.matmul(x_mixed_rec, ortho_T)

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
        Online EMA Auto-Calibration for target manifold scale:
        - Strictly computes mean and std over valid (non-PAD) tokens when valid_mask is provided,
          preventing high-padding short sequences from shrinking manifold_std.
        - Step 0: Directly initializes manifold_mean and manifold_std from the first batch.
        - Steps 1 .. max_calib_steps-1: Smooths estimate via EMA.
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
            batch_mean = x_float.mean()
            batch_std = x_float.std().clamp(min=1e-5)

        if step == 0:
            self.manifold_mean.copy_(batch_mean)
            self.manifold_std.copy_(batch_std)
        else:
            self.manifold_mean.lerp_(batch_mean, momentum)
            self.manifold_std.lerp_(batch_std, momentum)

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

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        h_potential: torch.Tensor,
        k_attractors: torch.Tensor = None,
        k_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Standard Stable Velocity Field Predictor for OT-CFM:
        - Primary generative path: driven by x_t! `x = x_t + 0.1 * pos_x`
        - Condition path: `h_cond = h_wave + 0.5 * h_global + sent_wave_proj(...)`
        - Velocity output: simply `v_pred = self.final_proj(x_normed)`!
        - Completely free of /(1-t) division and x_t_gate * x_t subtraction.
        """
        B, L, D = x_t.shape
        t_emb = self.t_embedder(t)

        pos_x = self._get_pos_emb(L, x_t.device, x_t.dtype)

        if h_potential.dim() == 2:
            h_pot = h_potential.to(device=x_t.device, dtype=x_t.dtype)
            h_wave = self.unpack_sentence_vector(h_pot, L)
            h_global = h_pot.unsqueeze(1).expand(B, L, D)
            h_cond = h_wave + 0.5 * h_global + self.sent_wave_proj(
                torch.cat([h_wave, h_global + pos_x], dim=-1)
            )
        else:
            h_cond = h_potential

        # Primary generative trajectory driven by x_t!
        x = x_t + 0.1 * pos_x

        k_pos = None
        if k_attractors is not None:
            pos_k = self._get_pos_emb(k_attractors.shape[1], k_attractors.device, k_attractors.dtype)
            k_pos = k_attractors + 0.02 * pos_k

        for block in self.blocks:
            x = block(x, t_emb, h_cond, k_attractors=k_pos, k_val=k_attractors, k_mask=k_mask)

        x_normed = self.final_norm(x, h_cond)
        v_pred = self.final_proj(x_normed)

        return v_pred
