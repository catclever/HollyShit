import math
from typing import Optional, Tuple, Dict, Union, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    Mamba = None
    HAS_MAMBA = False


class TimestepEmbedder(nn.Module):
    """
    Continuous Flow Timestep Embedder: maps scalar timestep t in [0, 1] -> R^{d_model}
    using sinusoidal frequency embeddings + 2-layer SiLU MLP (matching FlowMatching1DUnet).
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / max(half, 1)
        )
        args = t[:, None].float() * freqs[None] * 1000.0
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_dtype = self.mlp[0].weight.dtype
        return self.mlp(t_freq.to(dtype=t_dtype))


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
    Parameter-Free Metric-Preserving Orthogonal Attractor with <PAD> mask support.
    Directly measures inner-product similarity in the shared Phase 1 geometric manifold
    without redundant Q/K/V linear projections (saving 75.5M parameters across 24 modules and
    ~3.5 GB of intermediate activation memory at B=64, L=512):
      - Normalizes query trajectory X and knowledge pool K via LayerNorm(D).
      - Computes inner-product scores = (X_norm @ K_norm^T) / sqrt(D).
      - Masks <PAD> knowledge tokens to 0.0 before ReLU.
      - Applies thresholded ReLU (NO Softmax!) so irrelevant/orthogonal knowledge yields 0.0 force.
      - Superimposes the resulting force onto X via 1-parameter OrthogonalFieldSuperposition.
    """
    def __init__(self, d_model: int = 1024, init_gate: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.norm_x = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)
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
        if k.dim() == 4:
            _, M, L_k_sub, _ = k.shape
            L_k = M * L_k_sub
            k = k.reshape(B, L_k, D)
            if k_mask is not None:
                k_mask = k_mask.reshape(B, L_k)
        else:
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

        x_norm = self.norm_x(x)
        k_norm = self.norm_k(k_masked)

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

        # Normalize by sqrt(active_count) floor
        if k_mask_bool is not None:
            active_count = k_mask_bool.float().sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1.0)
            norm_factor = 1.0 / torch.sqrt(active_count).to(dtype=x.dtype)
        else:
            norm_factor = 1.0 / math.sqrt(max(L_k, 1))

        force = torch.bmm(activation, k_masked) * norm_factor
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
    H_canvas of arbitrary canvas length L_out (e.g., 256, 512, 768) with zero hardcoded L weights
    and without the 8.9M-parameter harmonic_proj bottleneck.

    Implements Dual-Level Elastic Time-Warping + Orthonormal Nyquist Slot Articulation:
      1. Global Effective Horizon (length_head / rho_override):
         Runs explicitly in FP32 with straight-through leaky tanh logit bounding:
           soft_logit = 4.5 * tanh(raw_len_logit / 4.5)
           bounded_logit = 0.9 * soft_logit + 0.1 * raw_len_logit + 0.1 * (soft_logit - raw_len_logit).detach()
           rho_pred = 0.04 + 0.95 * sigmoid(bounded_logit)  in (0.04, 0.99)
         Detaches rho_pred when computing rho_used so loss_potential gradients through
         l_eff = rho_used * L_out (512x multiplier) never hijack length_head.
      2. Local Beat Pacing (pacing_head):
         Predicts positive beat duration weights delta_s_k -> normalized p_k (sum_k p_k = 1).
      3. Dual Basis Macro Projection + Orthonormal Nyquist Slot Carrier Articulation:
         Combines K-order Legendre polynomials + Gaussian beat kernels with an analytical
         orthonormal Nyquist carrier (`<slot[t], slot[t+1]> == 0.0` for all t) modulated by
         macro beat & summary projections (`self.slot_proj`).
    """
    def __init__(self, d_model: int = 1024, num_beats: int = 32):
        super().__init__()
        self.d_model = d_model
        self.num_beats = num_beats

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

        # 4. Lightweight Orthonormal Nyquist Slot Articulation (replaces 8.9M harmonic_proj)
        self.slot_proj = nn.Linear(d_model, d_model)
        nn.init.normal_(self.slot_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.slot_proj.bias)

        # 5. Learned <PAD>/<EOS> asymptotic tail state beyond L_eff (calibrated at runtime from <PAD> embedding)
        self.pad_tail_emb = nn.Parameter(torch.zeros(1, 1, d_model))

        # 6. Lightweight bottleneck 1D sharpening conv block (local grammatical transition refinement)
        conv_mid = max(d_model // 2, 64)
        self.sharpen_norm = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, conv_mid, kernel_size=5, padding=2)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(conv_mid, d_model, kernel_size=5, padding=2)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.conv2.bias)

        self._pos_cache = {}
        self._slot_cache = {}

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
            h_macro_potential: (B, target_len, D)
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
            rho_pred = 0.04 + 0.95 * torch.sigmoid(bounded_logit)
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

        # Evaluate K-order Legendre basis on warped coordinate tau(t)
        p_legendre = self._eval_legendre_basis(tau, K).to(dtype=dtype)  # (B, L_out, K)

        # Soft Gaussian beat routing centered at u_k with width sigma_k = p_k
        sigma = p_weights.unsqueeze(1).clamp(min=0.5 / K)  # (B, 1, K)
        dist_sq = (s_grid.unsqueeze(-1) - u_centers.unsqueeze(1)) ** 2
        w_gauss = torch.softmax(-0.5 * dist_sq / (sigma ** 2), dim=-1).to(dtype=dtype)  # (B, L_out, K)

        # Dual Basis Macro Projection: separate spectral Legendre projection and spatial Gaussian beat projection
        h_macro = torch.bmm(p_legendre, self.leg_proj(c_beats)) + torch.bmm(w_gauss, self.gauss_proj(c_beats))
        h_macro_norm = self.macro_norm(h_macro)

        # High-Frequency Orthonormal Nyquist Slot Carrier modulated by Macro-Beat & Summary FiLM
        slot_carrier = self._get_slot_carrier(L_out, device, dtype)  # (1, L_out, D), cos_sim(t, t+1) == 0.0
        gamma, beta = self.film_proj(h_macro_norm).chunk(2, dim=-1)
        sum_normed = self.macro_norm(h_summary.to(dtype=dtype)).unsqueeze(1)  # (B, 1, D)
        h_slot = self.slot_proj((0.5 * h_macro_norm + 0.5 * sum_normed + 0.5) * (2.5 * slot_carrier))
        h_pos = h_macro + (0.35 + gamma) * slot_carrier + h_slot + 0.10 * beta

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
            "h_macro_potential": h_potential,
        }
        return h_potential, aux_dict


class MultiStreamMambaPlanner(nn.Module):
    """
    Phase 5 Unified Left-Brain Architecture (Two-Stage Macro-Once + Micro-Flow (X_t, t) Closed-Loop Planner).
    1. Stage 1 (`encode_macro` — Executed ONCE per dialogue turn):
       Scans context & multi-stream knowledge, extracts clean summary `h_summary_clean` for `length_head`,
       rolls out K=32 narrative beats, and expands them onto the L_out canvas `h_canvas_macro`.
    2. Stage 2 (`forward_canvas` — Executed at each Flow Timestep t in [0, 1]):
       Injects strictly 1-step right-shifted causal prefix state `x_t_prev` (position l ONLY observes
       indices 0 .. l-1, NEVER index l!) weighted by the Bayesian Wiener filter `w_wiener(t) = t / (t^2 + (1-t)^2)`
       + macro-centered causal innovation + continuous timestep `t_emb` into the 3-layer causal `canvas_layers`
       stack, outputting the refined trajectory `h_potential(X_t, t)` and instantaneous OT-CFM velocity
       vector field `v_mamba(X_t, t)`.
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

        # Parameter-free metric-preserving orthogonal knowledge resonance modules per macro layer
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

        # Elastic Time-Warped Legendre + Orthonormal Nyquist Expander
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

        # Stage 2 Micro-Flow (X_t, t) Conditioning & Causal Canvas Articulation Stack
        self.t_embedder = TimestepEmbedder(hidden_size=d_model, frequency_embedding_size=256)
        self.canvas_prev_proj = nn.Linear(d_model, d_model)
        nn.init.normal_(self.canvas_prev_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.canvas_prev_proj.bias)

        self.canvas_t_proj = nn.Linear(d_model, d_model)
        nn.init.normal_(self.canvas_t_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.canvas_t_proj.bias)
        self.canvas_t_gates = nn.Parameter(torch.full((self.num_canvas_layers,), 0.2, dtype=torch.float32))

        # Causal prefix branch discriminator + orthonormal slot carrier projection
        self.canvas_branch_gate = nn.Linear(d_model, 1, bias=False)
        nn.init.normal_(self.canvas_branch_gate.weight, mean=0.0, std=1.0 / math.sqrt(d_model))
        self.canvas_slot_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.canvas_slot_proj.weight)
        nn.init.zeros_(self.canvas_slot_proj.bias)

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

        # Instantaneous OT-CFM Velocity Head v_mamba(X_t, t)
        self.vel_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.vel_proj.weight)
        nn.init.zeros_(self.vel_proj.bias)

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

    def encode_macro(
        self,
        x_context: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        x_knowledge: Optional[torch.Tensor] = None,
        knowledge_mask: Optional[torch.Tensor] = None,
        target_len: Optional[int] = None,
        rho_override: Optional[Union[torch.Tensor, float]] = None,
    ) -> Dict[str, Any]:
        """
        Stage 1 (Executed ONCE per dialogue turn):
        Scans context and knowledge streams, extracts clean summary `h_summary_clean`,
        rolls out K=32 narrative beats, and expands them onto the L_out macroscopic canvas.
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
            beats = self.beat_ctx_resonance[i](beats, h_ctx, stream_mask=valid_mask_bool)
            if k_pool is not None:
                beats = self.beat_know_resonance[i](beats, k_pool, stream_mask=k_mask)

        beats = self.beat_norm(beats)

        # 4. Dynamic Time-Warped Legendre + Orthonormal Slot Expansion -> h_macro_potential (B, out_len, D)
        h_macro_potential, aux_dict = self.expander(
            c_beats=beats,
            h_summary=h_summary_clean,
            target_len=out_len,
            rho_override=rho_override,
        )
        aux_dict["beats"] = beats
        aux_dict["h_macro_potential"] = h_macro_potential

        h_canvas_macro = aux_dict["h_canvas"]
        horizon_gate = aux_dict["horizon_gate"].unsqueeze(-1).to(dtype=h_canvas_macro.dtype)
        pad_tail = self.expander.pad_tail_emb.to(dtype=h_canvas_macro.dtype)
        slot_carrier = self.expander._get_slot_carrier(out_len, x_context.device, h_canvas_macro.dtype)

        return {
            "h_ctx": h_ctx,
            "valid_mask_bool": valid_mask_bool,
            "k_pool": k_pool,
            "k_mask": k_mask,
            "h_summary_clean": h_summary_clean,
            "beats": beats,
            "h_canvas_macro": h_canvas_macro,
            "h_macro_potential": h_macro_potential,
            "horizon_gate": horizon_gate,
            "pad_tail": pad_tail,
            "slot_carrier": slot_carrier,
            "out_len": out_len,
            "aux_dict": aux_dict,
        }

    def forward_canvas(
        self,
        x_t: torch.Tensor,
        t: Union[torch.Tensor, float],
        macro_state: Dict[str, Any],
        return_velocity: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Stage 2 (Executed at each Flow Matching timestep t in [0, 1]):
        Takes the current noisy trajectory `x_t` (B, L, D), shifts it right by 1 position (`x_t_prev`)
        so position l ONLY observes causal prefix `x_t[:, :l, :]` (strictly 0.0 intra-step leakage!),
        applies the Bayesian Wiener filter `w_wiener(t) = t / (t^2 + (1-t)^2)` (which is 0.0 at t=0.0,
        preventing white-noise amplification at Euler step 0), and runs the 3-layer causal `canvas_layers`
        stack to articulate `h_potential(X_t, t)` and `v_mamba(X_t, t)`.
        """
        h_canvas_macro = macro_state["h_canvas_macro"]
        horizon_gate = macro_state["horizon_gate"]
        pad_tail = macro_state["pad_tail"]
        slot_carrier = macro_state["slot_carrier"]
        h_ctx = macro_state["h_ctx"]
        valid_mask_bool = macro_state["valid_mask_bool"]
        k_pool = macro_state["k_pool"]
        k_mask = macro_state["k_mask"]

        B, L, D = h_canvas_macro.shape
        device = h_canvas_macro.device
        dtype = h_canvas_macro.dtype

        if isinstance(t, (int, float)):
            t_tensor = torch.full((B,), float(t), device=device, dtype=torch.float32)
        elif t.dim() == 0:
            t_tensor = t.view(1).expand(B).to(device=device, dtype=torch.float32)
        else:
            t_tensor = t.reshape(B).to(device=device, dtype=torch.float32)

        t_emb = self.t_embedder(t_tensor).to(dtype=dtype)  # (B, D)
        t_expand = t_tensor.view(B, 1, 1).to(dtype=dtype)

        # Bayesian Wiener filter weight: w_wiener(0.0) == 0.0 (zero white-noise amplification at Step 0!),
        # w_wiener(0.5) == 1.0, and w_wiener(1.0) == 1.0.
        w_wiener = (t_tensor / (t_tensor.pow(2) + (1.0 - t_tensor).pow(2) + 1e-6)).view(B, 1, 1).to(dtype=dtype)

        # Strictly 1-step right-shifted causal prefix of x_t:
        # At index l in [0, L-1], x_t_prev[:, :l+1, :] ONLY contains x_t[:, :l, :] (NEVER index l!)
        x_t_cast = x_t.to(dtype=dtype)
        x_t_gated = x_t_cast * horizon_gate
        x_t_prev = F.pad(x_t_gated[:, :-1, :], (0, 0, 1, 0))  # (B, L, D)

        # Macro-centered causal innovation weighted by w_wiener(t)
        h_macro_gated = h_canvas_macro.detach() * horizon_gate
        h_macro_prev = F.pad(h_macro_gated[:, :-1, :], (0, 0, 1, 0))  # (B, L, D)
        innov_prev = w_wiener * (x_t_prev - t_expand * h_macro_prev)  # (B, L, D)

        if L > 1:
            has_prefix = F.pad(torch.ones((1, L - 1, 1), device=device, dtype=dtype), (0, 0, 1, 0))
            innov_first = innov_prev[:, 1:2, :] * has_prefix  # (B, L, D): 0 at l=0, innov[:, 0, :] at l>=1
        else:
            has_prefix = torch.zeros((1, L, 1), device=device, dtype=dtype)
            innov_first = torch.zeros_like(innov_prev)

        counts = torch.arange(L, device=device, dtype=torch.float32).clamp(min=1.0).view(1, L, 1).to(dtype=dtype)
        innov_cum = torch.cumsum(innov_prev, dim=1) / counts  # (B, L, D): 0 at l=0, prefix mean over 0..l-1 at l>=1
        prefix_innov = 0.4 * innov_first + 0.4 * innov_cum + 0.2 * innov_prev

        # Causal branch gate z_branch in (-1, 1): strictly 0 at t=0.0 and at l=0, causal over 0..l-1 for l>=1
        z_branch = torch.tanh(8.0 * self.canvas_branch_gate(prefix_innov)) * has_prefix  # (B, L, 1)

        # Fuse macro canvas with Wiener-filtered causal prefix state and flow timestep embedding
        t_proj_shared = self.canvas_t_proj(t_emb).unsqueeze(1)  # (B, 1, D)
        h_canvas = (
            h_canvas_macro
            + self.canvas_prev_proj(w_wiener * x_t_prev + 0.5 * prefix_innov)
            + t_proj_shared
        )

        for j in range(self.num_canvas_layers):
            t_gate_j = self.canvas_t_gates[j].to(dtype=dtype)
            normed_c = self.canvas_norms[j](h_canvas + t_gate_j * t_proj_shared)
            h_canvas = h_canvas + self.canvas_layers[j](normed_c)
            h_canvas = self.canvas_ctx_resonance[j](h_canvas, h_ctx, stream_mask=valid_mask_bool)
            if k_pool is not None:
                h_canvas = self.canvas_know_resonance[j](h_canvas, k_pool, stream_mask=k_mask)

        h_normed = self.canvas_final_norm(h_canvas)
        h_branch_slot = self.canvas_slot_proj(z_branch * (4.0 * slot_carrier))
        h_refined = h_canvas + self.out_proj(h_normed) + h_branch_slot
        h_potential = horizon_gate * h_refined + (1.0 - horizon_gate) * pad_tail

        if return_velocity:
            inv_one_minus_t = 1.0 / (1.0 - t_tensor).clamp(min=0.05).view(B, 1, 1).to(dtype=dtype)
            v_base = (h_potential - x_t_cast) * inv_one_minus_t
            v_mamba = v_base + self.vel_proj(F.silu(h_normed - x_t_cast))
            return h_potential, v_mamba

        return h_potential

    def forward(
        self,
        x_context: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        x_knowledge: Optional[torch.Tensor] = None,
        knowledge_mask: Optional[torch.Tensor] = None,
        target_len: Optional[int] = None,
        rho_override: Optional[Union[torch.Tensor, float]] = None,
        x_t: Optional[torch.Tensor] = None,
        t: Optional[Union[torch.Tensor, float]] = None,
        return_sequence: bool = True,
        keepdim: bool = False,
        return_aux: bool = False,
        return_velocity: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        """
        Unified Two-Stage Forward Pass:
          1. Runs `encode_macro(...)` once on context + multi-stream knowledge.
          2. Runs `forward_canvas(x_t, t, macro_state)` on the flow-matching canvas.
             If `x_t` is omitted (legacy caller), defaults to `x_t = h_macro_potential.detach(), t = 1.0`.
        """
        B = x_context.shape[0]
        macro_state = self.encode_macro(
            x_context=x_context,
            valid_mask=valid_mask,
            x_knowledge=x_knowledge,
            knowledge_mask=knowledge_mask,
            target_len=target_len,
            rho_override=rho_override,
        )
        h_summary_clean = macro_state["h_summary_clean"]
        aux_dict = macro_state["aux_dict"]

        if not return_sequence:
            out = h_summary_clean.unsqueeze(1) if keepdim else h_summary_clean
            return (out, aux_dict) if return_aux else out

        if x_t is None:
            x_t = macro_state["h_macro_potential"].detach()
        if t is None:
            t = torch.ones((B,), device=x_context.device, dtype=torch.float32)

        h_potential, v_mamba = self.forward_canvas(
            x_t=x_t,
            t=t,
            macro_state=macro_state,
            return_velocity=True,
        )
        aux_dict["v_mamba"] = v_mamba

        if return_velocity and return_aux:
            return h_potential, v_mamba, aux_dict
        if return_velocity:
            return h_potential, v_mamba
        if return_aux:
            return h_potential, aux_dict
        return h_potential


# Backward-compatible alias
MambaReader = MultiStreamMambaPlanner
