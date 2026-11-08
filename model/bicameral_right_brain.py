import math
from typing import Optional, Tuple, Sequence, Dict, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.flow_unet_cuda import TimestepEmbedder


class ClauseMambaBlock(nn.Module):
    """
    Pure-PyTorch GPU-friendly Selective State-Space Model (Mamba) block operating on
    punctuation-clause sequences `(B, T, D)` where each discrete time step `t` is
    1 punctuation clause (`6-28` chars packed via HiPPOOracleCUDA into R^D).

    Supports both:
      - `forward(s_seq, state=None, return_state=False)` over sequence `(B, T, D)`
      - `step(s_t, state=None) -> (h_t, new_state)` for single-clause recurrent stepping `(B, D)`.
    Both execution modes share identical selective SSM recurrences:
      Delta_t = softplus(dt_proj(delta_t))
      A_bar_t = exp(Delta_t * A)
      B_bar_t = Delta_t * B_t
      state_t = A_bar_t * state_{t-1} + B_bar_t * u_t
      y_t = sum_n(state_t * C_t) + D_skip * u_t
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        d_state: int = 16,
        expand: int = 2,
        dt_rank: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_state = d_state
        self.expand = expand
        self.d_inner = hidden_dim * expand
        self.dt_rank = dt_rank

        self.norm = nn.LayerNorm(hidden_dim)
        self.in_proj = nn.Linear(hidden_dim, 2 * self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        # Initialize dt_proj bias so softplus(bias) lies in [0.01, 0.1]
        with torch.no_grad():
            dt_init = torch.exp(
                torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.01)) + math.log(0.01)
            )
            inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
            self.dt_proj.bias.copy_(inv_dt)
            nn.init.normal_(self.dt_proj.weight, mean=0.0, std=0.02)

        # HiPPO-inspired diagonal state decay A_log: A = -exp(A_log) in (-d_state, -1)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, d_state).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D_skip = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, hidden_dim)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros((batch_size, self.d_inner, self.d_state), device=device, dtype=dtype)

    def step(
        self,
        s_t: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-clause recurrent Mamba update:
          s_t: (B, D)
          state: (B, d_inner, d_state) or None
        Returns:
          h_t: (B, D)
          new_state: (B, d_inner, d_state)
        """
        B, _ = s_t.shape
        if state is None:
            state = self.init_state(B, s_t.device, dtype=torch.float32)
        else:
            state = state.float()

        x_norm = self.norm(s_t)
        u_t, z_t = self.in_proj(x_norm).chunk(2, dim=-1)  # (B, d_inner) each
        u_act = F.silu(u_t)

        ssm_params = self.x_proj(u_act)
        dt_raw, B_t, C_t = torch.split(ssm_params, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw)).float()  # (B, d_inner)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))  # (B, d_inner, d_state)
        dB_u = (dt * u_act.float()).unsqueeze(-1) * B_t.float().unsqueeze(1)  # (B, d_inner, d_state)

        new_state = dA * state + dB_u
        y_t = (new_state * C_t.float().unsqueeze(1)).sum(dim=-1) + self.D_skip.float().unsqueeze(0) * u_act.float()
        y_gated = y_t.to(dtype=s_t.dtype) * F.silu(z_t)

        h_t = s_t + self.out_proj(y_gated)
        return h_t, new_state

    def forward(
        self,
        s_seq: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ):
        """
        Sequence-level Mamba pass over clause trajectory `s_seq: (B, T, D)`.
        Vectorizes input projections across all `T` clauses and unrolls the selective state
        recurrence so `forward` and `T` sequential calls to `step` match exactly.
        """
        B, T, _ = s_seq.shape
        if state is None:
            curr_state = self.init_state(B, s_seq.device, dtype=torch.float32)
        else:
            curr_state = state.float()

        x_norm = self.norm(s_seq)
        u_seq, z_seq = self.in_proj(x_norm).chunk(2, dim=-1)  # (B, T, d_inner)
        u_act = F.silu(u_seq)

        ssm_params = self.x_proj(u_act)
        dt_raw, B_seq, C_seq = torch.split(ssm_params, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_seq = F.softplus(self.dt_proj(dt_raw)).float()  # (B, T, d_inner)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        D_skip_f = self.D_skip.float().unsqueeze(0)  # (1, d_inner)

        ys: List[torch.Tensor] = []
        for t in range(T):
            dt_t = dt_seq[:, t, :]
            u_t = u_act[:, t, :].float()
            B_t = B_seq[:, t, :].float()
            C_t = C_seq[:, t, :].float()

            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            dB_u = (dt_t * u_t).unsqueeze(-1) * B_t.unsqueeze(1)
            curr_state = dA * curr_state + dB_u
            y_t = (curr_state * C_t.unsqueeze(1)).sum(dim=-1) + D_skip_f * u_t
            ys.append(y_t.to(dtype=s_seq.dtype))

        y_seq = torch.stack(ys, dim=1)  # (B, T, d_inner)
        y_gated = y_seq * F.silu(z_seq)
        h_seq = s_seq + self.out_proj(y_gated)

        if return_state:
            return h_seq, curr_state
        return h_seq


class ClauseMambaStream(nn.Module):
    """
    Multi-layer Selective SSM Stream for Punctuation Clause Trajectories.
    Optionally bridges external `input_dim` (`d_clause`) to internal Mamba dimension `hidden_dim` (`d_mamba`)
    when `input_dim != hidden_dim`.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        d_state: int = 16,
        num_layers: int = 2,
        expand: int = 2,
        input_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = hidden_dim if input_dim is None else int(input_dim)
        self.num_layers = num_layers

        if self.input_dim != self.hidden_dim:
            self.in_bridge: Optional[nn.Module] = nn.Linear(self.input_dim, self.hidden_dim)
            self.out_bridge: Optional[nn.Module] = nn.Linear(self.hidden_dim, self.input_dim)
        else:
            self.in_bridge = None
            self.out_bridge = None

        self.layers = nn.ModuleList([
            ClauseMambaBlock(hidden_dim=hidden_dim, d_state=d_state, expand=expand)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(self.input_dim)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> List[torch.Tensor]:
        return [layer.init_state(batch_size, device, dtype=dtype) for layer in self.layers]

    def step(
        self,
        s_t: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if states is None:
            states = [None] * len(self.layers)
        new_states: List[torch.Tensor] = []
        h = self.in_bridge(s_t) if self.in_bridge is not None else s_t
        for layer, st in zip(self.layers, states):
            h, next_st = layer.step(h, st)
            new_states.append(next_st)
        if self.out_bridge is not None:
            h = s_t + self.out_bridge(h)
        return self.final_norm(h), new_states

    def forward(
        self,
        s_seq: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
        return_state: bool = False,
    ):
        if states is None:
            states = [None] * len(self.layers)
        new_states: List[torch.Tensor] = []
        h = self.in_bridge(s_seq) if self.in_bridge is not None else s_seq
        for layer, st in zip(self.layers, states):
            h, next_st = layer(h, state=st, return_state=True)
            new_states.append(next_st)
        if self.out_bridge is not None:
            h = s_seq + self.out_bridge(h)
        h = self.final_norm(h)
        if return_state:
            return h, new_states
        return h


class AdaLN2DResMLPBlock(nn.Module):
    """
    2D AdaLN-Modulated Residual MLP block for `ThoughtFlowMatcher` in clause-vector space `(B, D)`.
    """

    def __init__(self, hidden_dim: int = 1024, mlp_ratio: int = 2):
        super().__init__()
        inner_dim = hidden_dim * mlp_ratio
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ada_mod = nn.Linear(hidden_dim, 3 * hidden_dim)
        nn.init.zeros_(self.ada_mod.weight)
        nn.init.zeros_(self.ada_mod.bias)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, hidden_dim),
        )
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.ada_mod(cond).chunk(3, dim=-1)
        z_mod = self.norm(z) * (1.0 + scale) + shift
        return z + (1.0 + torch.tanh(gate)) * self.mlp(z_mod)


# =====================================================================================
# Clause-vector frequency-band layout helpers (Task #029, Right-Brain v2)
# -------------------------------------------------------------------------------------
# `HiPPOOracleCUDA` packs a clause as `z_sub: (K=32 Legendre orders, d_sub=32 PCA channels)`
# and flattens it row-major, so a clause vector `z: (..., 1024)` is BAND-MAJOR:
#   band b (Legendre order b) == dims [b * d_sub : (b + 1) * d_sub].
# `ThoughtFlowMatcher.normalize` is a scalar mean/std affine map, so the layout survives
# normalization. Low orders (b < sem_bands, default 6) are *hypothesised* to carry content /
# semantics and high orders word-order / syntax; this split is NOT verified -> keep
# `sem_bands` configurable and monitor the per-band sigma curve.
# =====================================================================================
DEFAULT_NUM_BANDS = 32
DEFAULT_SEM_BANDS = 6


def to_bands(z: torch.Tensor, num_bands: int = DEFAULT_NUM_BANDS) -> torch.Tensor:
    """Reshapes a band-major clause vector `z: (..., D)` into `(..., num_bands, D // num_bands)`."""
    D = z.shape[-1]
    if D % num_bands != 0:
        raise ValueError(f"Clause dim {D} is not divisible by num_bands={num_bands}.")
    return z.reshape(*z.shape[:-1], num_bands, D // num_bands)


def from_bands(z_bands: torch.Tensor) -> torch.Tensor:
    """Inverse of `to_bands`: `(..., num_bands, d_sub) -> (..., num_bands * d_sub)`."""
    return z_bands.reshape(*z_bands.shape[:-2], z_bands.shape[-2] * z_bands.shape[-1])


def band_gaussian_nll(
    z1: torch.Tensor,
    mu: torch.Tensor,
    sigma_band: torch.Tensor,
    num_bands: int = DEFAULT_NUM_BANDS,
) -> torch.Tensor:
    """
    Band-wise heteroscedastic Gaussian NLL (float32, constant terms dropped), per row:
      nll = mean_b [ ||z1_b - mu_b||^2 / (2 * d_sub * sigma_b^2) + log sigma_b ]
    Optimum: sigma_b^2 = E[ ||z1_b - mu_b||^2 / d_sub | condition ].
    Shapes: z1, mu: (..., D); sigma_band: (..., num_bands). Returns (...,).
    """
    diff_b = to_bands(z1.float() - mu.float(), num_bands)  # (..., nb, d_sub)
    d_sub = diff_b.shape[-1]
    sq_b = diff_b.pow(2).sum(dim=-1)  # (..., nb)
    s = sigma_band.float()
    nll_b = sq_b / (2.0 * d_sub * s.pow(2)) + torch.log(s)
    return nll_b.mean(dim=-1)


def _build_discrete_legendre_basis(K: int, L: int) -> torch.Tensor:
    """
    Computes the first K discrete-orthonormal Legendre polynomials evaluated on [0, L-1]
    with uniform midpoint coordinates tau_l = (l + 0.5) / L in (0, 1), orthonormalized via
    two-pass Gram-Schmidt in float64 (matching HiPPOOracleCUDA._get_legendre_basis).
    Returns `basis` of shape (K, L).
    """
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
    raw_basis = torch.stack(polys, dim=0)
    ortho_rows = []
    for k in range(K):
        v = raw_basis[k]
        if k < L:
            for _ in range(2):
                for prev in ortho_rows:
                    v = v - torch.dot(v, prev) * prev
        norm = v.norm(p=2).clamp(min=1e-12)
        ortho_rows.append(v / norm)
    return torch.stack(ortho_rows, dim=0).to(dtype=torch.float32)


# [Design Change Documentation - Task #029: ThoughtFlowMatcher v2]
# 1. What changed vs #027:
#    - REMOVED the hard residual `v_pred = (h_t - z_tau) + final_proj(...)`. That residual made every
#      ODE trajectory collapse onto the single point `h_t` (a conditional *mean*), so the Right Brain
#      could only express one interpretation per step. `v_pred = final_proj(final_norm(...))` now.
#    - The ODE start point is no longer N(0, I) but the input-dependent adaptive prior cone
#      `z0 = mu + sigma_band (x) eps` produced by `PriorHead` (32 Legendre bands, 1 sigma per band).
#    - Training uses K-branch soft competition (`compute_branch_loss`): K blind Euler roll-outs
#      (with grad, no ground-truth leakage); the closest branch (winner) gets CFM velocity MSE +
#      endpoint MSE + 0.25 (1 - cos); the losers are only constrained to remain *legal* clause
#      vectors: low-band Mahalanobis (scaled by detached sigma) + low-band cosine, and per-band
#      L2 *energy* matching on high bands (no direction constraint -> syntax freedom).
# 2. Why: the design intent (design/tentative/2026-11-02_phase5_right_brain_v2_adaptive_prior.md)
#    is "Mamba state = a region of plausible clauses; Flow Matching samples inside that region".
#    A single-target CFM + residual regresses to the blurry mean.
# 3. Caveats:
#    - Calibration buffers (`z_mean`, `z_std`, `calib_steps`) and `normalize` / `denormalize` are
#      unchanged; mu / sigma / all losses live in the NORMALIZED space.
#    - `mu` / `sigma_band` are always `.detach()`-ed inside the flow losses: the prior head is
#      trained only by the band NLL (`band_gaussian_nll`).
#    - The flow cost is ~ K * (train_ode_steps + 1) forward passes of this MLP per valid row.
#    - Checkpoints of #027 are NOT compatible (residual semantics + new modules) -> use a new
#      checkpoint prefix (e.g. `bicameral_right_brain_v2`).
class SlotContextCrossAttn(nn.Module):
    """
    1D Spatial-Slot Cross-Attention over candidate context clauses' character slots (Right-Brain v3, #042).
    Queries: `h_slot: (N, L=32, d_slot)`.
    Keys/Values: `ctx_kv: (N, M_ctx, d_slot)` (`M_ctx = K_ctx * 32`), plus direct 32-D slot value bypass
    from `ctx_raw_slots: (N, M_ctx, d_sub)` and a 1D diagonal span relay (`(l-1, c-1) -> (l, c)`).
    """

    def __init__(self, d_slot: int = 128, d_sub: int = 32, num_heads: int = 4, n_slots: int = 32):
        super().__init__()
        self.d_slot = int(d_slot)
        self.d_sub = int(d_sub)
        self.num_heads = int(num_heads)
        self.n_slots = int(n_slots)
        self.head_dim = self.d_slot // self.num_heads

        self.q_norm = nn.LayerNorm(self.d_slot)
        self.k_norm = nn.LayerNorm(self.d_slot)
        self.q_proj = nn.Linear(self.d_slot, self.d_slot)
        self.k_proj = nn.Linear(self.d_slot, self.d_slot)
        self.v_proj = nn.Linear(self.d_slot, self.d_slot)
        self.out_proj = nn.Linear(self.d_slot, self.d_slot)
        self.raw_to_slot = nn.Linear(self.d_sub, self.d_slot)

        self.sink_logit = nn.Parameter(torch.full((self.num_heads,), 1.0))
        self.sink_v = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.qk_scale = nn.Parameter(torch.full((self.num_heads,), 8.0))
        self.raw_vel_scale = nn.Parameter(torch.tensor(0.50))

        self.diag_conv = nn.Conv1d(
            self.num_heads, self.num_heads, kernel_size=3, padding=1, bias=False
        )
        with torch.no_grad():
            self.diag_conv.weight.zero_()
            for h in range(self.num_heads):
                # Tap 0 in Conv1d with padding=1 reads c-1 at shifted target slot l-1 -> (+1, +1) diagonal relay
                self.diag_conv.weight[h, h, 0] = 0.25
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.out_proj.bias)
            nn.init.normal_(self.raw_to_slot.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.raw_to_slot.bias)

    def forward(
        self,
        h_slot: torch.Tensor,
        ctx_kv: torch.Tensor,
        ctx_raw_slots: torch.Tensor,
        ctx_slot_weights: Optional[torch.Tensor] = None,
        x_tau: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N, L, _ = h_slot.shape
        M_ctx = ctx_kv.shape[1]
        H, dh = self.num_heads, self.head_dim

        q = F.normalize(
            self.q_proj(self.q_norm(h_slot)).view(N, L, H, dh).transpose(1, 2).float(),
            dim=-1,
            eps=1e-6,
        )  # (N, H, L, dh)
        k = F.normalize(
            self.k_proj(self.k_norm(ctx_kv)).view(N, M_ctx, H, dh).transpose(1, 2).float(),
            dim=-1,
            eps=1e-6,
        )  # (N, H, M_ctx, dh)
        v = self.v_proj(ctx_kv).view(N, M_ctx, H, dh).transpose(1, 2).float()  # (N, H, M_ctx, dh)

        scale = F.softplus(self.qk_scale.float()).clamp(min=1.0, max=16.0).view(1, H, 1, 1)
        logits = scale * torch.matmul(q, k.transpose(-2, -1))  # (N, H, L, M_ctx)

        if L >= 2 and M_ctx >= 2:
            prev_l = F.pad(F.relu(logits[:, :, :-1, :]), (0, 0, 1, 0))  # (N, H, L, M_ctx)
            relay = self.diag_conv(
                prev_l.permute(0, 2, 1, 3).reshape(N * L, H, M_ctx)
            ).reshape(N, L, H, M_ctx).permute(0, 2, 1, 3)
            if self.n_slots > 0 and M_ctx > self.n_slots:
                c_idx = torch.arange(M_ctx, device=logits.device)
                boundary = (c_idx % self.n_slots == 0).view(1, 1, 1, M_ctx)
                relay = relay.masked_fill(boundary, 0.0)
            logits = logits + relay

        if ctx_slot_weights is not None:
            w_f = ctx_slot_weights.float()  # (N, M_ctx)
            valid_c = w_f > 1e-4
            w_log = torch.log(w_f.clamp(min=1e-4)).view(N, 1, 1, M_ctx)
            logits = logits + w_log
            logits = logits.masked_fill(~valid_c.view(N, 1, 1, M_ctx), -1e4)

        sink = self.sink_logit.float().view(1, H, 1, 1).expand(N, H, L, 1)
        logits_all = torch.cat([logits, sink], dim=-1)  # (N, H, L, M_ctx + 1)
        attn_all = F.softmax(logits_all, dim=-1)
        attn = attn_all[..., :M_ctx]       # (N, H, L, M_ctx)
        attn_sink = attn_all[..., M_ctx:]  # (N, H, L, 1)
        if ctx_slot_weights is not None:
            attn = attn * (ctx_slot_weights.float() > 1e-4).view(N, 1, 1, M_ctx).float()

        out_heads = torch.matmul(attn, v) + attn_sink * self.sink_v.float().view(1, H, 1, dh)
        out_h = self.out_proj(
            out_heads.transpose(1, 2).contiguous().view(N, L, self.d_slot).to(dtype=h_slot.dtype)
        )

        attn_mean = attn.mean(dim=1)  # (N, L, M_ctx)
        borrowed_raw = torch.bmm(attn_mean, ctx_raw_slots.float())  # (N, L, d_sub)
        out_h = out_h + self.raw_to_slot(borrowed_raw.to(dtype=h_slot.dtype))
        if x_tau is not None:
            attn_mass = attn_mean.sum(dim=-1, keepdim=True)  # (N, L, 1)
            delta_v_raw = torch.tanh(self.raw_vel_scale.float()) * (borrowed_raw - attn_mass * x_tau.float())
        else:
            delta_v_raw = torch.tanh(self.raw_vel_scale.float()) * borrowed_raw
        return out_h, delta_v_raw


class SlotFlowBlock(nn.Module):
    """
    1D Spatial-Slot Flow Block (Right-Brain v3, #042):
      1) AdaLN-modulated 1D Local Conv (kernel_size=3, padding=1) across slots (l-1, l, l+1)
      2) AdaLN-modulated 1D Bidirectional Slot Self-Attention (num_heads=4) across all 32 slots
      3) SlotContextCrossAttn attending over context character slots `ctx_kv` when provided
      4) AdaLN-modulated Channel MLP (d_slot -> 2 * d_slot -> d_slot)
    """

    def __init__(self, d_slot: int = 128, d_sub: int = 32, num_heads: int = 4, mlp_ratio: int = 2):
        super().__init__()
        self.d_slot = int(d_slot)
        self.d_sub = int(d_sub)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_slot // self.num_heads

        self.norm_conv = nn.LayerNorm(self.d_slot, elementwise_affine=False, eps=1e-6)
        self.local_conv = nn.Conv1d(self.d_slot, self.d_slot, kernel_size=3, padding=1)
        self.conv_proj = nn.Linear(self.d_slot, self.d_slot)

        self.norm_attn = nn.LayerNorm(self.d_slot, elementwise_affine=False, eps=1e-6)
        self.self_qkv = nn.Linear(self.d_slot, self.d_slot * 3)
        self.self_proj = nn.Linear(self.d_slot, self.d_slot)

        self.cross_attn = SlotContextCrossAttn(d_slot=self.d_slot, d_sub=self.d_sub, num_heads=self.num_heads)

        self.norm_mlp = nn.LayerNorm(self.d_slot, elementwise_affine=False, eps=1e-6)
        mlp_hidden = self.d_slot * int(mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(self.d_slot, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, self.d_slot),
        )

        self.ada_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.d_slot, 9 * self.d_slot),
        )
        with torch.no_grad():
            nn.init.zeros_(self.ada_mod[-1].weight)
            nn.init.zeros_(self.ada_mod[-1].bias)
            # Open residual gates slightly at init so slot self-attn / local conv / MLP transmit immediately
            for g_idx in (2, 5, 8):
                self.ada_mod[-1].bias[g_idx * self.d_slot : (g_idx + 1) * self.d_slot].fill_(0.15)
            nn.init.normal_(self.conv_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.conv_proj.bias)
            nn.init.normal_(self.self_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.self_proj.bias)
            nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        h_slot: torch.Tensor,
        c_ada: torch.Tensor,
        ctx_kv: Optional[torch.Tensor] = None,
        ctx_raw_slots: Optional[torch.Tensor] = None,
        ctx_slot_weights: Optional[torch.Tensor] = None,
        x_tau: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N, L, D_s = h_slot.shape
        H, dh = self.num_heads, self.head_dim
        (
            shift_c, scale_c, gate_c,
            shift_a, scale_a, gate_a,
            shift_m, scale_m, gate_m,
        ) = self.ada_mod(c_ada).unsqueeze(1).chunk(9, dim=-1)

        # 1. AdaLN-modulated 1D Local Conv across adjacent character slots (l-1, l, l+1)
        h_c = self.norm_conv(h_slot) * (1.0 + scale_c) + shift_c
        h_c = F.silu(self.local_conv(h_c.transpose(1, 2)).transpose(1, 2))
        h_slot = h_slot + gate_c * self.conv_proj(h_c)

        # 2. AdaLN-modulated 1D Bidirectional Slot Self-Attention across all 32 slots
        h_a = self.norm_attn(h_slot) * (1.0 + scale_a) + shift_a
        qkv = self.self_qkv(h_a).view(N, L, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (N, H, L, dh)
        attn_out = F.scaled_dot_product_attention(q, k, v)  # (N, H, L, dh), no all-masked rows
        attn_out = attn_out.transpose(1, 2).contiguous().view(N, L, D_s)
        h_slot = h_slot + gate_a * self.self_proj(attn_out)

        # 3. Context Slot Cross-Attention (sub-clause span borrowing & diagonal relay)
        if ctx_kv is not None and ctx_raw_slots is not None:
            delta_h, delta_v_raw = self.cross_attn(
                h_slot, ctx_kv, ctx_raw_slots, ctx_slot_weights=ctx_slot_weights, x_tau=x_tau
            )
            h_slot = h_slot + delta_h
        else:
            delta_v_raw = torch.zeros((N, L, self.d_sub), device=h_slot.device, dtype=torch.float32)

        # 4. AdaLN-modulated Channel MLP
        h_m = self.norm_mlp(h_slot) * (1.0 + scale_m) + shift_m
        h_slot = h_slot + gate_m * self.mlp(h_m)

        return h_slot, delta_v_raw


class ThoughtFlowMatcher(nn.Module):
    """
    1D Spatial-Slot Flow Matcher with Context Slot Cross-Attention & Decoupled Slot Validity Head
    (Right-Brain v3, #042).
    Operates in the isomorphic 32-slot spatial domain `X_slot: (B, L=32, d_sub=32)` via the exact
    orthogonal Legendre transform while exposing the same normalized 1024-D clause interface.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        d_clause: Optional[int] = None,
        *,
        num_bands: int = DEFAULT_NUM_BANDS,
        sem_bands: int = DEFAULT_SEM_BANDS,
        num_branches: int = 4,
        train_ode_steps: int = 3,
        win_weight: float = 0.65,
        other_temp: float = 1.0,
        prior_noise_scale: float = 0.2,
    ):
        super().__init__()
        if d_clause is not None:
            hidden_dim = int(d_clause)
        self.hidden_dim = hidden_dim
        self.d_clause = hidden_dim
        self.num_bands = int(num_bands)
        self.n_slots = self.num_bands
        self.d_sub = hidden_dim // self.num_bands
        self.d_slot = 128
        self.sem_bands = int(sem_bands)
        self.num_branches = int(num_branches)
        self.train_ode_steps = int(train_ode_steps)
        self.win_weight = float(win_weight)
        self.other_temp = float(other_temp)
        self.prior_noise_scale = float(prior_noise_scale)
        self._upgraded_from_v1 = False
        self._cached_sample_ctx: Optional[Dict[str, Any]] = None

        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, self.d_slot),
            nn.SiLU(),
            nn.Linear(self.d_slot, self.d_slot),
        )
        self.h_to_slots = nn.Linear(hidden_dim, self.n_slots * self.d_sub)
        self.slot_pos_emb = nn.Parameter(torch.randn(1, self.n_slots, self.d_slot) * 0.02)
        self.slot_in_proj = nn.Linear(self.d_sub * 5, self.d_slot)

        self.ctx_slot_proj = nn.Linear(self.d_sub, self.d_slot)
        self.ctx_clause_emb = nn.Parameter(torch.randn(1, 4, 1, self.d_slot) * 0.02)

        self.slot_blocks = nn.ModuleList([
            SlotFlowBlock(d_slot=self.d_slot, d_sub=self.d_sub, num_heads=4, mlp_ratio=2)
            for _ in range(num_layers)
        ])
        self.slot_final_norm = nn.LayerNorm(self.d_slot)
        self.slot_out_proj = nn.Linear(self.d_slot, self.d_sub)
        self.ctx_direct_proj = nn.Linear(self.d_slot, self.d_sub)
        nn.init.normal_(self.slot_out_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.slot_out_proj.bias)
        nn.init.normal_(self.ctx_direct_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.ctx_direct_proj.bias)

        # Decoupled Clause Length / Slot Validity Head (#042)
        self.slot_valid_in = nn.Linear(self.d_sub * 2, self.d_slot)
        self.slot_valid_head = nn.Sequential(
            nn.LayerNorm(self.d_slot),
            nn.Linear(self.d_slot, self.d_slot // 2),
            nn.SiLU(),
            nn.Linear(self.d_slot // 2, 1),
        )

        # In-Model Per-Dimension Manifold Auto-Calibration Buffers
        self.register_buffer("z_mean", torch.zeros(hidden_dim, dtype=torch.float32))
        self.register_buffer("z_std", torch.ones(hidden_dim, dtype=torch.float32))
        self.register_buffer("calib_steps", torch.tensor(0, dtype=torch.long))
        self.register_buffer("flow_version", torch.tensor(3, dtype=torch.long))
        self.register_buffer(
            "legendre_basis",
            _build_discrete_legendre_basis(self.num_bands, self.num_bands),
            persistent=False,
        )
        self._register_load_state_dict_pre_hook(self._upgrade_scalar_calib_buffers)

    def _upgrade_scalar_calib_buffers(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        for name in ("z_mean", "z_std"):
            k = prefix + name
            if k in state_dict and state_dict[k].shape != (self.hidden_dim,):
                state_dict[k] = state_dict[k].reshape(-1).expand(self.hidden_dim).clone()
        self._upgraded_from_v1 = False

    @torch.no_grad()
    def calibrate_z(
        self,
        z_sample: torch.Tensor,
        max_calib_steps: int = 50,
        momentum: float = 0.1,
    ) -> bool:
        """
        Online EMA Per-Dimension Auto-Calibration for 2D clause vectors `z_sample` over the first
        `max_calib_steps` steps. Standardizes each of the `D` dimensions independently to N(0, 1).
        """
        step = int(self.calib_steps.item())
        if step >= max_calib_steps:
            return False

        z_float = z_sample.detach().float().reshape(-1, self.hidden_dim)
        valid = z_float.abs().sum(dim=-1) > 1e-6
        z_valid = z_float[valid]
        if z_valid.numel() == 0:
            return False

        batch_mean = z_valid.mean(dim=0)
        if z_valid.shape[0] >= 2:
            batch_std = z_valid.std(dim=0, unbiased=False).clamp(min=1e-3)
        else:
            s = z_valid.std().clamp(min=1e-3) if z_valid.numel() > 1 else torch.tensor(1.0, device=z_valid.device)
            batch_std = s.expand(self.hidden_dim).clone()

        if step == 0:
            self.z_mean.copy_(batch_mean)
            self.z_std.copy_(batch_std)
        else:
            self.z_mean.lerp_(batch_mean, momentum)
            self.z_std.lerp_(batch_std, momentum)

        self.calib_steps.add_(1)
        return True

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        """Normalizes clause vector `z (..., D)` from `N(z_mean, diag(z_std^2))` to whitened `N(0, I_D)`."""
        mean = self.z_mean.to(device=z.device, dtype=z.dtype)
        std = self.z_std.to(device=z.device, dtype=z.dtype).clamp(min=1e-3)
        return (z - mean) / std

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        """Denormalizes whitened vector `z (..., D)` back to physical `z_clause` coordinate space."""
        mean = self.z_mean.to(device=z.device, dtype=z.dtype)
        std = self.z_std.to(device=z.device, dtype=z.dtype).clamp(min=1e-3)
        return z * std + mean

    def reconstruct_spatial_slots(self, z_norm: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the 32 spatial character-slot vectors `X_slot: (..., L, d_sub)` from a whitened
        clause vector `z_norm: (..., D)` via the exact orthogonal inverse Legendre transform `B^T`.
        """
        nb = self.num_bands
        z_phys = self.denormalize(z_norm.float())  # (..., D)
        z_bands = to_bands(z_phys, nb)  # (..., nb, d_sub)
        std_b = to_bands(self.z_std.to(device=z_norm.device, dtype=torch.float32).clamp(min=1e-3), nb)  # (nb, d_sub)
        chan_scale = std_b.pow(2).mean(dim=0).sqrt().clamp(min=1e-3)  # (d_sub,)
        z_chan_wht = z_bands / chan_scale  # (..., nb, d_sub)
        basis = self.legendre_basis.to(device=z_norm.device, dtype=torch.float32)  # (nb, L)
        return torch.einsum("...kd,kl->...ld", z_chan_wht, basis)

    def from_spatial_slots(self, x_slot: torch.Tensor) -> torch.Tensor:
        """
        Exact inverse of `reconstruct_spatial_slots`: maps spatial character-slot vectors
        `x_slot: (..., L=32, d_sub=32)` back to whitened Legendre clause coordinates `z_norm: (..., D=1024)`.
        """
        nb = self.num_bands
        basis = self.legendre_basis.to(device=x_slot.device, dtype=torch.float32)  # (nb, L)
        z_chan_wht = torch.einsum("...ld,kl->...kd", x_slot.float(), basis)
        std_b = to_bands(self.z_std.to(device=x_slot.device, dtype=torch.float32).clamp(min=1e-3), nb)
        chan_scale = std_b.pow(2).mean(dim=0).sqrt().clamp(min=1e-3)
        z_phys = from_bands(z_chan_wht * chan_scale)
        return self.normalize(z_phys).to(dtype=x_slot.dtype)

    def velocity_to_spatial_slots(self, v_norm: torch.Tensor) -> torch.Tensor:
        """
        Homogeneous linear tangent map corresponding to `reconstruct_spatial_slots` (no `z_mean` offset):
        maps whitened Legendre velocity `v_norm: (..., D)` to spatial-slot velocity `v_slot: (..., L, d_sub)`.
        """
        nb = self.num_bands
        std = self.z_std.to(device=v_norm.device, dtype=torch.float32).clamp(min=1e-3)
        v_phys = v_norm.float() * std
        v_bands = to_bands(v_phys, nb)
        std_b = to_bands(std, nb)
        chan_scale = std_b.pow(2).mean(dim=0).sqrt().clamp(min=1e-3)
        v_chan_wht = v_bands / chan_scale
        basis = self.legendre_basis.to(device=v_norm.device, dtype=torch.float32)
        return torch.einsum("...kd,kl->...ld", v_chan_wht, basis).to(dtype=v_norm.dtype)

    def velocity_from_spatial_slots(self, v_slot: torch.Tensor) -> torch.Tensor:
        """
        Exact inverse of `velocity_to_spatial_slots` (homogeneous linear tangent map of `from_spatial_slots`):
        maps spatial-slot velocity `v_slot: (..., L, d_sub)` back to whitened Legendre velocity `v_norm: (..., D)`.
        """
        nb = self.num_bands
        basis = self.legendre_basis.to(device=v_slot.device, dtype=torch.float32)
        v_chan_wht = torch.einsum("...ld,kl->...kd", v_slot.float(), basis)
        std = self.z_std.to(device=v_slot.device, dtype=torch.float32).clamp(min=1e-3)
        std_b = to_bands(std, nb)
        chan_scale = std_b.pow(2).mean(dim=0).sqrt().clamp(min=1e-3)
        v_phys = from_bands(v_chan_wht * chan_scale)
        return (v_phys / std).to(dtype=v_slot.dtype)

    def predict_slot_validity(self, cond: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Predicts per-slot active-character validity logits `(..., 32)` from `cond` (`h_t`) and `mu`.
        """
        orig_lead = cond.shape[:-1]
        D = self.d_clause
        cond_2d = cond.reshape(-1, D)
        mu_2d = mu.reshape(-1, D)
        x_h = (
            0.5 * self.reconstruct_spatial_slots(cond_2d.float())
            + 0.5 * self.h_to_slots(cond_2d).float().view(-1, self.n_slots, self.d_sub)
        )
        x_mu = self.reconstruct_spatial_slots(mu_2d.float())
        v_in = torch.cat([x_h, x_mu], dim=-1).to(dtype=self.slot_valid_in.weight.dtype)
        h_v = self.slot_valid_in(v_in) + self.slot_pos_emb.to(dtype=v_in.dtype)
        logits = self.slot_valid_head(h_v).squeeze(-1).float()
        return logits.reshape(*orig_lead, self.n_slots)

    def compute_polar_band_loss(
        self,
        z_pred_norm: torch.Tensor,
        z_target_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decouples per-band RMS radial norm matching from per-band and global angular alignment
        so imperfect high-band angles never shrink high-frequency band amplitudes by cos(theta_k).
        """
        nb = self.num_bands
        zp = z_pred_norm.float()
        zt = z_target_norm.float()
        z_pred_b = to_bands(zp, nb)  # (..., nb, d_sub)
        z_gt_b = to_bands(zt, nb)    # (..., nb, d_sub)

        r_pred = (z_pred_b.pow(2).mean(dim=-1) + 1e-6).sqrt()  # (..., nb)
        r_gt = (z_gt_b.pow(2).mean(dim=-1) + 1e-6).sqrt()      # (..., nb)
        band_norm_loss = F.mse_loss(r_pred, r_gt)

        band_cos = F.cosine_similarity(z_pred_b, z_gt_b, dim=-1, eps=1e-6)
        band_cos_loss = 1.0 - band_cos.mean()

        u_pred_b = z_pred_b / r_pred.unsqueeze(-1)
        u_gt_b = z_gt_b / r_gt.unsqueeze(-1)
        zp_iso = from_bands(u_pred_b * r_gt.detach().unsqueeze(-1))
        global_cos = F.cosine_similarity(zp_iso, zt, dim=-1, eps=1e-6)
        global_cos_loss = 1.0 - global_cos.mean()

        if self.d_sub > 1:
            u_pred_0 = u_pred_b[..., 0]  # (..., nb)
            u_gt_0 = u_gt_b[..., 0]      # (..., nb)
            c0_dir_mse = F.mse_loss(u_pred_0, u_gt_0)
            c0_cos_loss = 1.0 - F.cosine_similarity(u_pred_0, u_gt_0, dim=-1, eps=1e-6).mean()
            chan0_loss = 0.5 * c0_dir_mse + 0.5 * c0_cos_loss
        else:
            chan0_loss = band_norm_loss * 0.0

        total_polar = band_norm_loss + band_cos_loss + 0.5 * global_cos_loss + 0.5 * chan0_loss
        return total_polar, {
            "band_norm_loss": band_norm_loss.detach(),
            "band_cos_loss": band_cos_loss.detach(),
            "global_cos_loss": global_cos_loss.detach(),
            "chan0_loss": chan0_loss.detach(),
        }

    def compute_spatial_slot_loss(
        self,
        z_pred_norm: torch.Tensor,
        z_target_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes inverse-Legendre spatial slot cosine alignment (`slot_cos_loss`), per-slot RMS
        envelope matching (`slot_norm_loss`), radially-decoupled Channel-0 character-vs-<PAD> step
        envelope alignment (`env_slot_loss`), and polar adjacent-slot first-difference transition
        alignment (`slot_diff_loss` with step-size norm matching + anti-stagnation hinge).
        """
        nb = self.num_bands
        zp_b = to_bands(z_pred_norm.float(), nb)
        zt_b = to_bands(z_target_norm.float(), nb)
        u_pred_b = F.normalize(zp_b, dim=-1, eps=1e-6) * math.sqrt(self.d_sub)
        if u_pred_b.requires_grad:
            u_unit_d = F.normalize(zp_b.detach(), dim=-1, eps=1e-6)
            u_pred_b.register_hook(lambda g: g - (g * u_unit_d).sum(dim=-1, keepdim=True) * u_unit_d)
        rt = (zt_b.pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt().detach()
        z_pred_iso = from_bands(u_pred_b * rt)

        x_pred = self.reconstruct_spatial_slots(z_pred_norm)      # (..., L, d_sub)
        x_pred_iso = self.reconstruct_spatial_slots(z_pred_iso)   # (..., L, d_sub) radially decoupled (`d/dr_k == 0`)
        x_gt = self.reconstruct_spatial_slots(z_target_norm)      # (..., L, d_sub)

        slot_rms_pred = (x_pred.pow(2).mean(dim=-1) + 1e-6).sqrt()  # (..., L)
        slot_rms_gt = (x_gt.pow(2).mean(dim=-1) + 1e-6).sqrt()      # (..., L)
        slot_norm_loss = F.mse_loss(slot_rms_pred, slot_rms_gt)

        slot_norm_gt = x_gt.norm(dim=-1)  # (..., L)
        active_mask = (slot_norm_gt > 0.15).float()
        if active_mask.sum().item() < 1.0:
            active_mask = torch.ones_like(slot_norm_gt)

        if self.d_sub > 1:
            lex_cos = F.cosine_similarity(x_pred_iso[..., 1:], x_gt[..., 1:], dim=-1, eps=1e-6)
            full_cos = F.cosine_similarity(x_pred_iso, x_gt, dim=-1, eps=1e-6)
            slot_cos = 0.75 * lex_cos + 0.25 * full_cos
            env_pred = x_pred_iso[..., 0]  # (..., L)
            env_gt = x_gt[..., 0]          # (..., L)
            env_slot_loss = 1.0 - F.cosine_similarity(env_pred, env_gt, dim=-1, eps=1e-6).mean()
        else:
            slot_cos = F.cosine_similarity(x_pred_iso, x_gt, dim=-1, eps=1e-6)
            env_slot_loss = slot_norm_loss * 0.0
        slot_cos_loss = ((1.0 - slot_cos) * active_mask).sum() / active_mask.sum().clamp(min=1.0)

        dx_pred_iso = x_pred_iso[..., 1:, :] - x_pred_iso[..., :-1, :]  # (..., L-1, d_sub)
        dx_gt = x_gt[..., 1:, :] - x_gt[..., :-1, :]                # (..., L-1, d_sub)
        dx_norm_gt = dx_gt.norm(dim=-1)
        trans_mask = (dx_norm_gt > 0.15).float()
        if trans_mask.sum().item() < 1.0:
            trans_mask = torch.ones_like(dx_norm_gt)
        trans_cos = F.cosine_similarity(dx_pred_iso, dx_gt, dim=-1, eps=1e-6)
        slot_diff_cos_loss = ((1.0 - trans_cos) * trans_mask).sum() / trans_mask.sum().clamp(min=1.0)

        dx_rms_pred = (dx_pred_iso.pow(2).mean(dim=-1) + 1e-6).sqrt()  # (..., L-1) radially decoupled (d/dr_k == 0)
        dx_rms_gt = (dx_gt.pow(2).mean(dim=-1) + 1e-6).sqrt()          # (..., L-1)
        slot_diff_norm_loss = F.mse_loss(dx_rms_pred, dx_rms_gt)
        anti_stag = (F.relu(0.85 * dx_rms_gt - dx_rms_pred).pow(2) * trans_mask).sum() / trans_mask.sum().clamp(min=1.0)
        slot_diff_loss = slot_diff_cos_loss + 0.5 * slot_diff_norm_loss + 0.5 * anti_stag

        total_slot_loss = slot_cos_loss + 0.25 * slot_norm_loss + slot_diff_loss + 0.5 * env_slot_loss
        return total_slot_loss, {
            "slot_cos_loss": slot_cos_loss.detach(),
            "slot_norm_loss": slot_norm_loss.detach(),
            "slot_diff_loss": slot_diff_loss.detach(),
            "slot_diff_norm_loss": slot_diff_norm_loss.detach(),
            "anti_stag_loss": anti_stag.detach(),
            "env_slot_loss": env_slot_loss.detach(),
        }

    def forward(
        self,
        z_tau: torch.Tensor,
        tau: torch.Tensor,
        h_t: torch.Tensor,
        *,
        z_lookback: Optional[torch.Tensor] = None,
        gate_lb: Optional[torch.Tensor] = None,
        ctx_clauses: Optional[torch.Tensor] = None,
        ctx_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predicts velocity `v_pred: (B, D)` (or `(B, T, D)`) at flow time `tau` conditioned on `h_t`
        by evolving the 32 character slots `(N, L=32, d_sub=32)` through `SlotFlowBlock` layers and
        projecting the resulting slot velocity back to normalized Legendre space via `velocity_from_spatial_slots`.
        """
        orig_shape = z_tau.shape
        if z_tau.dim() == 3:
            B, T, D = orig_shape
            N = B * T
            z_tau = z_tau.reshape(N, D)
            h_t = h_t.reshape(N, D)
            if tau.numel() == B:
                tau = tau.unsqueeze(1).expand(B, T).reshape(N)
            elif tau.numel() == N:
                tau = tau.reshape(N)
            if z_lookback is not None:
                z_lookback = z_lookback.reshape(N, D)
            if gate_lb is not None:
                gate_lb = gate_lb.reshape(N, -1)
            if ctx_clauses is not None:
                ctx_clauses = ctx_clauses.reshape(N, -1, D)
            if ctx_weights is not None:
                ctx_weights = ctx_weights.reshape(N, -1)
        else:
            N, D = z_tau.shape

        w_dtype = self.slot_in_proj.weight.dtype
        x_tau = self.reconstruct_spatial_slots(z_tau.float())  # (N, 32, 32)
        dx = x_tau[:, 1:, :] - x_tau[:, :-1, :]
        dx_prev = F.pad(dx, (0, 0, 1, 0))
        dx_next = F.pad(dx, (0, 0, 0, 1))
        x_h_cond = (
            0.5 * self.reconstruct_spatial_slots(h_t.float())
            + 0.5 * self.h_to_slots(h_t.to(dtype=w_dtype)).float().view(N, self.n_slots, self.d_sub)
        )
        if z_lookback is not None:
            x_lb = self.reconstruct_spatial_slots(z_lookback.reshape(N, D).float())
            g_lb = (
                gate_lb.reshape(N, 1, 1).float()
                if gate_lb is not None
                else torch.ones((N, 1, 1), device=z_tau.device, dtype=torch.float32)
            )
            x_lb_gated = g_lb * x_lb
        else:
            x_lb_gated = torch.zeros_like(x_tau)

        slot_in = torch.cat([x_tau, dx_prev, dx_next, x_h_cond, x_lb_gated], dim=-1).to(dtype=w_dtype)
        h_slot = self.slot_in_proj(slot_in) + self.slot_pos_emb.to(dtype=w_dtype)

        t_emb = self.t_embedder(tau).to(dtype=w_dtype)
        c_ada = self.cond_proj(torch.cat([h_t.to(dtype=w_dtype), t_emb], dim=-1))

        if ctx_clauses is None and z_lookback is not None:
            ctx_clauses = z_lookback.reshape(N, 1, D)
            ctx_weights = (
                gate_lb.reshape(N, 1).float()
                if gate_lb is not None
                else torch.ones((N, 1), device=z_tau.device, dtype=torch.float32)
            )

        if ctx_clauses is not None:
            K_ctx = min(int(ctx_clauses.shape[1]), 4)
            ctx_c = ctx_clauses[:, :K_ctx, :].reshape(N, K_ctx, D)
            ctx_raw = self.reconstruct_spatial_slots(ctx_c.float())  # (N, K_ctx, 32, 32)
            ctx_emb = self.ctx_slot_proj(ctx_raw.to(dtype=w_dtype))
            ctx_emb = (
                ctx_emb
                + self.slot_pos_emb.unsqueeze(1).to(dtype=w_dtype)
                + self.ctx_clause_emb[:, :K_ctx, :, :].to(dtype=w_dtype)
            )
            M_ctx = K_ctx * self.n_slots
            ctx_kv = ctx_emb.reshape(N, M_ctx, self.d_slot)
            ctx_raw_slots = ctx_raw.reshape(N, M_ctx, self.d_sub)
            slot_active = (ctx_raw.detach().norm(dim=-1) > 0.12).float()  # (N, K_ctx, 32)
            clause_valid = (ctx_c.detach().abs().sum(dim=-1, keepdim=True) > 1e-6).float()  # (N, K_ctx, 1)
            if ctx_weights is not None:
                cw = ctx_weights[:, :K_ctx].reshape(N, K_ctx, 1).float().detach()
                cw_eff = torch.where((cw > 1e-5) & (clause_valid > 0.5), 0.35 + 0.65 * cw, torch.zeros_like(cw))
            else:
                cw_eff = clause_valid
            ctx_slot_weights = (cw_eff * (0.20 + 0.80 * slot_active)).reshape(N, M_ctx)
        else:
            ctx_kv = None
            ctx_raw_slots = None
            ctx_slot_weights = None

        v_ctx_accum = torch.zeros_like(x_tau)
        for block in self.slot_blocks:
            h_slot, delta_v_raw = block(
                h_slot,
                c_ada,
                ctx_kv=ctx_kv,
                ctx_raw_slots=ctx_raw_slots,
                ctx_slot_weights=ctx_slot_weights,
                x_tau=x_tau,
            )
            v_ctx_accum = v_ctx_accum + delta_v_raw

        h_final = self.slot_final_norm(h_slot)
        v_slot = (
            self.slot_out_proj(h_final).float()
            + self.ctx_direct_proj(h_final).float()
            + v_ctx_accum
        )
        v_pred = self.velocity_from_spatial_slots(v_slot).to(dtype=h_t.dtype)

        if len(orig_shape) == 3:
            v_pred = v_pred.reshape(orig_shape)
        return v_pred

    def expand_sigma(self, sigma_band: torch.Tensor) -> torch.Tensor:
        """Broadcasts per-band sigma `(..., num_bands)` to per-dim `(..., D)` (band-major layout)."""
        return sigma_band.repeat_interleave(self.d_sub, dim=-1)

    def _euler_integrate(
        self,
        z0: torch.Tensor,
        cond: torch.Tensor,
        steps: int,
        *,
        z_lookback: Optional[torch.Tensor] = None,
        gate_lb: Optional[torch.Tensor] = None,
        ctx_clauses: Optional[torch.Tensor] = None,
        ctx_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Euler-integrates the ODE from tau=0 to tau=1 in float32 state precision (keeps grad)."""
        x = z0.float()
        n_steps = max(int(steps), 1)
        dt = 1.0 / float(n_steps)
        for i in range(n_steps):
            tau = torch.full((x.shape[0],), i * dt, device=x.device, dtype=torch.float32)
            v = self.forward(
                x,
                tau,
                cond,
                z_lookback=z_lookback,
                gate_lb=gate_lb,
                ctx_clauses=ctx_clauses,
                ctx_weights=ctx_weights,
            )
            x = x + v.float() * dt
        return x

    def _euler_integrate_with_trajectory(
        self,
        z0: torch.Tensor,
        cond: torch.Tensor,
        steps: int,
        *,
        z_lookback: Optional[torch.Tensor] = None,
        gate_lb: Optional[torch.Tensor] = None,
        ctx_clauses: Optional[torch.Tensor] = None,
        ctx_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Euler-integrates the ODE from tau=0 to tau=1 in float32 state precision (keeps grad),
        returning `(x1, x_traj, v_traj)` where `x_traj` and `v_traj` have shape `(steps, B, D)`.
        """
        x = z0.float()
        n_steps = max(int(steps), 1)
        dt = 1.0 / float(n_steps)
        xs, vs = [], []
        for i in range(n_steps):
            tau = torch.full((x.shape[0],), i * dt, device=x.device, dtype=torch.float32)
            v = self.forward(
                x,
                tau,
                cond,
                z_lookback=z_lookback,
                gate_lb=gate_lb,
                ctx_clauses=ctx_clauses,
                ctx_weights=ctx_weights,
            ).float()
            xs.append(x)
            vs.append(v)
            x = x + v * dt
        return x, torch.stack(xs, dim=0), torch.stack(vs, dim=0)

    def compute_branch_loss(
        self,
        z1_norm: torch.Tensor,
        cond: torch.Tensor,
        mu: torch.Tensor,
        sigma_band: torch.Tensor,
        num_branches: Optional[int] = None,
        train_ode_steps: Optional[int] = None,
        *,
        z_lookback: Optional[torch.Tensor] = None,
        gate_lb: Optional[torch.Tensor] = None,
        ctx_clauses: Optional[torch.Tensor] = None,
        ctx_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        K-branch soft-competition flow loss on ALREADY-FILTERED, NORMALIZED rows.
          z1_norm: (N, D) normalized targets, cond: (N, D) fused Mamba state,
          mu: (N, D) prior mean (normalized space), sigma_band: (N, num_bands).
        Returns `(L_flow, stats)` where `stats` holds detached tensors for metrics.
        """
        K = int(num_branches) if num_branches is not None else self.num_branches
        K = max(K, 1)
        S = int(train_ode_steps) if train_ode_steps is not None else self.train_ode_steps
        n_steps = max(S, 1)
        N, D = z1_norm.shape
        nb, ds, L = self.num_bands, self.d_sub, self.sem_bands
        device = z1_norm.device

        z1f = z1_norm.float()
        mu_d = mu.detach().float()
        sig_d = sigma_band.detach().float()  # (N, nb)
        sig_full = self.expand_sigma(sig_d)  # (N, D)

        # ---- 1. K blind roll-outs from the adaptive prior cone (with grad, no target leakage)
        eps = torch.randn((K, N, D), device=device, dtype=torch.float32)
        z0 = mu_d.unsqueeze(0) + (self.prior_noise_scale * sig_full).unsqueeze(0) * eps  # (K, N, D)
        cond_k = cond.unsqueeze(0).expand(K, N, D).reshape(K * N, D)
        z_lb_k = (
            z_lookback.reshape(N, D).unsqueeze(0).expand(K, N, D).reshape(K * N, D)
            if z_lookback is not None else None
        )
        gate_lb_k = (
            gate_lb.reshape(N, 1).unsqueeze(0).expand(K, N, 1).reshape(K * N, 1)
            if gate_lb is not None else None
        )
        if ctx_clauses is not None:
            K_ctx = ctx_clauses.shape[1]
            ctx_clauses_k = ctx_clauses.reshape(N, K_ctx, D).unsqueeze(0).expand(K, N, K_ctx, D).reshape(K * N, K_ctx, D)
        else:
            ctx_clauses_k = None
        if ctx_weights is not None:
            K_w = ctx_weights.shape[1]
            ctx_weights_k = ctx_weights.reshape(N, K_w).unsqueeze(0).expand(K, N, K_w).reshape(K * N, K_w)
        else:
            ctx_weights_k = None

        x1_flat, x_traj_flat, v_traj_flat = self._euler_integrate_with_trajectory(
            z0.reshape(K * N, D),
            cond_k,
            n_steps,
            z_lookback=z_lb_k,
            gate_lb=gate_lb_k,
            ctx_clauses=ctx_clauses_k,
            ctx_weights=ctx_weights_k,
        )
        x1 = x1_flat.reshape(K, N, D)
        x_traj = x_traj_flat.reshape(n_steps, K, N, D)
        v_traj = v_traj_flat.reshape(n_steps, K, N, D)

        # ---- 2. Winner selection (no grad, spectral + polar band norm/direction + spatial slot transition alignment)
        with torch.no_grad():
            x1_d = x1.detach()
            spec_err = (x1_d - z1f.unsqueeze(0)).pow(2).mean(dim=-1)  # (K, N)
            x1_b_d = to_bands(x1_d, nb)                               # (K, N, nb, ds)
            z1_b_d = to_bands(z1f, nb).unsqueeze(0)                   # (1, N, nb, ds)
            band_cos_k = F.cosine_similarity(x1_b_d, z1_b_d, dim=-1, eps=1e-6).mean(dim=-1)  # (K, N)
            r_x1 = (x1_b_d.pow(2).mean(dim=-1) + 1e-6).sqrt()         # (K, N, nb)
            r_z1 = (z1_b_d.pow(2).mean(dim=-1) + 1e-6).sqrt()         # (1, N, nb)
            band_norm_err = (r_x1 - r_z1).pow(2).mean(dim=-1)         # (K, N)
            x1_slots = self.reconstruct_spatial_slots(x1_d)           # (K, N, L_slots, ds)
            z1_slots = self.reconstruct_spatial_slots(z1f).unsqueeze(0)  # (1, N, L_slots, ds)
            dx1_slots = x1_slots[..., 1:, :] - x1_slots[..., :-1, :]
            dz1_slots = z1_slots[..., 1:, :] - z1_slots[..., :-1, :]
            trans_err = (dx1_slots - dz1_slots).pow(2).mean(dim=(-1, -2))  # (K, N)
            env_cos_k = F.cosine_similarity(x1_slots[..., 0], z1_slots[..., 0], dim=-1, eps=1e-6)  # (K, N)
            err = (
                0.5 * spec_err
                + (1.0 - band_cos_k)
                + 0.5 * band_norm_err
                + 0.5 * trans_err
                + 0.5 * (1.0 - env_cos_k)
            )
            k_star = err.argmin(dim=0)  # (N,)
        rows = torch.arange(N, device=device)
        x1_win = x1[k_star, rows]  # (N, D)
        x_traj_win = x_traj[:, k_star, rows, :]  # (n_steps, N, D)
        v_traj_win = v_traj[:, k_star, rows, :]  # (n_steps, N, D)

        # ---- 3. Winner loss: Leak-free polar band velocity/endpoint loss + spatial slot alignment + slot validity
        step_idx = torch.arange(n_steps, device=device, dtype=torch.float32).view(n_steps, 1, 1)
        rem_time = 1.0 - step_idx / float(n_steps)  # (n_steps, 1, 1) in (0, 1]
        x1_implied = x_traj_win.detach() + rem_time * v_traj_win  # (n_steps, N, D)
        z1_expand = z1f.unsqueeze(0).expand_as(x1_implied)

        polar_vel_loss, _ = self.compute_polar_band_loss(x1_implied, z1_expand)
        cart_vel_mse = F.mse_loss(rem_time * v_traj_win, z1f.unsqueeze(0) - x_traj_win.detach())
        vel_loss = 0.80 * polar_vel_loss + 0.20 * cart_vel_mse

        polar_end_loss, polar_stats = self.compute_polar_band_loss(x1_win, z1f)
        cart_end_mse = F.mse_loss(x1_win, z1f)
        end_mse = 0.80 * polar_end_loss + 0.20 * cart_end_mse

        win_cos = F.cosine_similarity(x1_win, z1f, dim=-1, eps=1e-6)  # (N,)
        cos_loss = 1.0 - win_cos.mean()

        traj_slot_loss, _ = self.compute_spatial_slot_loss(x1_implied, z1_expand)
        end_slot_loss, slot_stats = self.compute_spatial_slot_loss(x1_win, z1f)
        slot_loss = 0.5 * traj_slot_loss + 0.5 * end_slot_loss

        x_gt = self.reconstruct_spatial_slots(z1f)  # (N, 32, d_sub)
        valid_targets = (x_gt.norm(dim=-1) > 0.15).float().detach().clone()
        valid_targets[:, :2] = 1.0
        valid_logits = self.predict_slot_validity(cond, mu_d)
        slot_valid_loss = F.binary_cross_entropy_with_logits(valid_logits, valid_targets)
        x1_win_slots = self.reconstruct_spatial_slots(x1_win)
        pad_mask = 1.0 - valid_targets
        pad_tail_loss = (x1_win_slots.pow(2).mean(dim=-1) * pad_mask).sum() / pad_mask.sum().clamp(min=1.0)

        win_loss = (
            0.5 * vel_loss
            + 0.5 * end_mse
            + 0.75 * slot_loss
            + 0.25 * slot_valid_loss
            + 0.25 * pad_tail_loss
        )

        # ---- 4. Non-winner branch regularization: prior-cone anchor + pairwise angular repulsion
        x1_b = to_bands(x1, nb)  # (K, N, nb, ds)
        mu_b = to_bands(mu_d, nb)  # (N, nb, ds)
        prior_dev = (
            (x1_b - mu_b.unsqueeze(0)).pow(2).sum(dim=-1)
            / (ds * sig_d.pow(2).clamp(min=1e-4)).unsqueeze(0)
        ).mean(dim=-1)  # (K, N)

        win_onehot = F.one_hot(k_star, num_classes=K).transpose(0, 1).bool()  # (K, N)
        if K > 1:
            disp = F.normalize((x1 - mu_d.unsqueeze(0)).transpose(0, 1), dim=-1, eps=1e-6)  # (N, K, D)
            sim = torch.bmm(disp, disp.transpose(1, 2))  # (N, K, K)
            eye = torch.eye(K, device=device, dtype=torch.bool).unsqueeze(0)
            repulse = F.relu(sim.masked_select(~eye) - 0.25).mean()
            with torch.no_grad():
                logits = -prior_dev.detach() / max(self.other_temp, 1e-6)
                logits = logits.masked_fill(win_onehot, float("-inf"))
                w_other = torch.softmax(logits, dim=0)  # (K, N), winner weight == 0
            other_loss = 0.1 * (w_other * prior_dev).sum(dim=0).mean() + repulse
        else:
            other_loss = prior_dev.sum() * 0.0

        ww = self.win_weight
        loss = ww * win_loss + (1.0 - ww) * other_loss

        with torch.no_grad():
            if K > 1:
                x_low = x1.detach()[..., : L * ds].transpose(0, 1)  # (N, K, L*ds)
                pd = torch.cdist(x_low, x_low)  # (N, K, K)
                diversity = pd.sum(dim=(-1, -2)) / float(K * (K - 1))
                diversity = diversity.mean()
            else:
                diversity = torch.zeros((), device=device)
            hist = torch.bincount(k_star, minlength=K)

        stats = {
            "vel_loss": vel_loss.detach(),
            "end_mse": end_mse.detach(),
            "cos_loss": cos_loss.detach(),
            "slot_loss": slot_loss.detach(),
            "slot_cos_loss": slot_stats["slot_cos_loss"],
            "slot_diff_loss": slot_stats["slot_diff_loss"],
            "slot_valid_loss": slot_valid_loss.detach(),
            "pad_tail_loss": pad_tail_loss.detach(),
            "thought_cos": win_cos.mean().detach(),
            "win_loss": win_loss.detach(),
            "other_loss": other_loss.detach(),
            "branch_diversity": diversity,
            "winner_hist": hist,
        }
        return loss, stats

    def compute_loss(
        self,
        z_1: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        sigma_band: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
        *,
        h_t: Optional[torch.Tensor] = None,
        z_target: Optional[torch.Tensor] = None,
        z_lookback: Optional[torch.Tensor] = None,
        gate_lb: Optional[torch.Tensor] = None,
        ctx_clauses: Optional[torch.Tensor] = None,
        ctx_weights: Optional[torch.Tensor] = None,
    ):
        """
        Stand-alone flow loss (filters `valid_mask`, auto-calibrates, normalizes, then runs
        `compute_branch_loss`). If `mu` / `sigma_band` are None the prior falls back to N(0, I).
        `MultiStreamRightBrain.compute_loss` is the primary entry point (it also adds prior NLL + stop BCE).
        """
        if z_1 is None and z_target is not None:
            z_1 = z_target
        if cond is None and h_t is not None:
            cond = h_t
        if z_1 is None or cond is None:
            raise ValueError("ThoughtFlowMatcher.compute_loss requires `z_1` and `cond` (or `h_t` and `z_target`).")

        D = z_1.shape[-1]
        nb = self.num_bands
        z_1 = z_1.reshape(-1, D)
        cond = cond.reshape(-1, D)
        mu = torch.zeros_like(z_1, dtype=torch.float32) if mu is None else mu.reshape(-1, D)
        sigma_band = (
            torch.ones((z_1.shape[0], nb), device=z_1.device, dtype=torch.float32)
            if sigma_band is None else sigma_band.reshape(-1, nb)
        )
        if z_lookback is not None:
            z_lookback = z_lookback.reshape(-1, D)
        if gate_lb is not None:
            gate_lb = gate_lb.reshape(-1, 1)
        if ctx_clauses is not None:
            ctx_clauses = ctx_clauses.reshape(-1, ctx_clauses.shape[-2], D)
        if ctx_weights is not None:
            ctx_weights = ctx_weights.reshape(-1, ctx_weights.shape[-1])

        empty_valid = False
        if valid_mask is not None:
            mask_bool = valid_mask.reshape(-1).bool()
            if mask_bool.any():
                z_1, cond, mu, sigma_band = z_1[mask_bool], cond[mask_bool], mu[mask_bool], sigma_band[mask_bool]
                if z_lookback is not None:
                    z_lookback = z_lookback[mask_bool]
                if gate_lb is not None:
                    gate_lb = gate_lb[mask_bool]
                if ctx_clauses is not None:
                    ctx_clauses = ctx_clauses[mask_bool]
                if ctx_weights is not None:
                    ctx_weights = ctx_weights[mask_bool]
            else:
                empty_valid = True
                z_1, cond, mu, sigma_band = z_1[:1], cond[:1], mu[:1], sigma_band[:1]
                if z_lookback is not None:
                    z_lookback = z_lookback[:1]
                if gate_lb is not None:
                    gate_lb = gate_lb[:1]
                if ctx_clauses is not None:
                    ctx_clauses = ctx_clauses[:1]
                if ctx_weights is not None:
                    ctx_weights = ctx_weights[:1]

        if self.training and not empty_valid and int(self.calib_steps.item()) < 50:
            self.calibrate_z(z_1)
        z_1_norm = self.normalize(z_1.float())

        loss, stats = self.compute_branch_loss(
            z_1_norm,
            cond,
            mu,
            sigma_band,
            z_lookback=z_lookback,
            gate_lb=gate_lb,
            ctx_clauses=ctx_clauses,
            ctx_weights=ctx_weights,
        )
        if empty_valid:
            loss = loss * 0.0
        if return_metrics:
            metrics = {k: (v.tolist() if v.dim() > 0 else float(v.item())) for k, v in stats.items()}
            return loss, metrics
        return loss

    @torch.no_grad()
    def sample(
        self,
        cond: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        sigma_band: Optional[torch.Tensor] = None,
        steps: int = 10,
        temperature: float = 1.0,
        renorm: bool = True,
        *,
        h_t: Optional[torch.Tensor] = None,
        z_lookback: Optional[torch.Tensor] = None,
        gate_lb: Optional[torch.Tensor] = None,
        ctx_clauses: Optional[torch.Tensor] = None,
        ctx_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Samples ONE eps per row: `z0 = mu + temperature * sigma (x) eps` (normalized space), Euler-integrates
        `steps` steps conditioned on `cond` and context slots, applies decoupled slot validity `<PAD>` tail
        masking in spatial slot space, and returns `self.denormalize(z)` in `z_clause` coordinates.
        """
        if cond is None and h_t is not None:
            cond = h_t
        if cond is None:
            raise ValueError("ThoughtFlowMatcher.sample requires `cond` (or `h_t`).")

        orig_shape = cond.shape
        D = self.d_clause
        cond_2d = cond.reshape(-1, D)
        B = cond_2d.shape[0]
        mu_2d = (
            torch.zeros((B, D), device=cond.device, dtype=torch.float32)
            if mu is None else mu.reshape(-1, D).float()
        )
        sig_2d = (
            torch.ones((B, self.num_bands), device=cond.device, dtype=torch.float32)
            if sigma_band is None else sigma_band.reshape(-1, self.num_bands).float()
        )

        cached = getattr(self, "_cached_sample_ctx", None)
        if z_lookback is None and ctx_clauses is None and cached is not None:
            c_zlb = cached.get("z_lookback")
            c_glb = cached.get("gate_lb")
            c_clauses = cached.get("ctx_clauses")
            c_weights = cached.get("ctx_weights")
            if c_zlb is not None:
                if c_zlb.reshape(-1, D).shape[0] == B:
                    z_lookback = c_zlb.reshape(B, D)
                    if c_glb is not None:
                        gate_lb = c_glb.reshape(B, 1)
                elif c_zlb.dim() == 3 and c_zlb.shape[0] == B and len(orig_shape) == 2:
                    z_lookback = c_zlb[:, -1, :]
                    if c_glb is not None:
                        gate_lb = c_glb[:, -1, :].reshape(B, 1)
            if c_clauses is not None:
                k_top = c_clauses.shape[-2]
                if c_clauses.reshape(-1, k_top, D).shape[0] == B:
                    ctx_clauses = c_clauses.reshape(B, k_top, D)
                    if c_weights is not None:
                        ctx_weights = c_weights.reshape(B, k_top)
                elif c_clauses.dim() == 4 and c_clauses.shape[0] == B and len(orig_shape) == 2:
                    ctx_clauses = c_clauses[:, -1, :, :]
                    if c_weights is not None:
                        ctx_weights = c_weights[:, -1, :]

        if z_lookback is not None:
            z_lookback = z_lookback.reshape(B, D).to(device=cond.device)
        if gate_lb is not None:
            gate_lb = gate_lb.reshape(B, 1).to(device=cond.device)
        if ctx_clauses is not None:
            ctx_clauses = ctx_clauses.reshape(B, -1, D).to(device=cond.device)
        if ctx_weights is not None:
            ctx_weights = ctx_weights.reshape(B, -1).to(device=cond.device)

        eps = torch.randn((B, D), device=cond.device, dtype=torch.float32)
        eff_temp = float(temperature) * self.prior_noise_scale
        z0 = mu_2d + eff_temp * self.expand_sigma(sig_2d) * eps
        z = self._euler_integrate(
            z0,
            cond_2d,
            steps,
            z_lookback=z_lookback,
            gate_lb=gate_lb,
            ctx_clauses=ctx_clauses,
            ctx_weights=ctx_weights,
        )

        valid_probs = torch.sigmoid(self.predict_slot_validity(cond_2d, mu_2d))  # (B, 32)
        n_valid = (valid_probs >= 0.5).sum(dim=-1).clamp(min=2, max=self.n_slots)  # (B,)
        slot_mask = (
            torch.arange(self.n_slots, device=z.device).unsqueeze(0) < n_valid.unsqueeze(1)
        ).unsqueeze(-1).float()  # (B, 32, 1)

        if renorm:
            z_rms = (z.pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
            z = z * (z_rms.clamp(min=0.35, max=2.5) / z_rms)

        x_slot = self.reconstruct_spatial_slots(z) * slot_mask
        z = self.from_spatial_slots(x_slot)
        z_out = self.denormalize(z)
        z_out = z_out.to(dtype=cond.dtype)
        if len(orig_shape) == 3:
            z_out = z_out.reshape(orig_shape)
        return z_out


class PriorHead(nn.Module):
    """
    32-Band Adaptive Prior Head (#029).
    Input: concat[h0_t, F_aux_t, max_i g_ti, sum_i g_ti] -> mu (D, normalized z space) and
    sigma_band (num_bands,) = sigma_min + (sigma_max - sigma_min) * sigmoid(.).
    Trained ONLY by `band_gaussian_nll` (callers must detach mu / sigma for any other loss).
    Gradients of the NLL DO flow back into the Mamba streams through h0 / F_aux / gates.
    """

    def __init__(
        self,
        d_clause: int = 1024,
        num_bands: int = DEFAULT_NUM_BANDS,
        hidden_dim: int = 1024,
        sigma_min: float = 0.05,
        sigma_max: float = 1.0,
        sigma_init: float = 0.5,
        zero_init_aux: bool = True,
    ):
        super().__init__()
        self.d_clause = int(d_clause)
        self.num_bands = int(num_bands)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        in_dim = 2 * self.d_clause + 2
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # [Design Change - Task #033: zero-init companion inputs of the prior head]
        # 1. What: the input columns of `trunk[0]` that consume [F_aux (D), gate_max (1), gate_sum (1)]
        #    (= columns d_clause .. 2*d_clause+1) are zero-initialised.
        # 2. Why: Stage 1 (design/tentative/2026-11-02_phase5_staged_training_plan.md §2) trains a single
        #    main stream; the companion stream (Stage 3) is attached later as an EXACT identity branch, so
        #    a freshly attached aux stream must not perturb mu / sigma of the trained Stage-1 prior.
        # 3. Caveats: shapes are unchanged (old #029 checkpoints still load strictly). The columns DO
        #    receive gradients as soon as non-zero F_aux / gates are fed, so they only stay zero while
        #    N == 0 (Stage 1). With aux=None the inputs in those columns are zeros anyway.
        if zero_init_aux:
            with torch.no_grad():
                self.trunk[0].weight[:, self.d_clause:].zero_()
        self.mu_out = nn.Linear(hidden_dim, self.d_clause)
        self.sigma_out = nn.Linear(hidden_dim, self.num_bands)
        nn.init.normal_(self.mu_out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.mu_out.bias)
        nn.init.normal_(self.sigma_out.weight, mean=0.0, std=0.02)
        frac = (float(sigma_init) - self.sigma_min) / max(self.sigma_max - self.sigma_min, 1e-6)
        frac = min(max(frac, 1e-4), 1.0 - 1e-4)
        nn.init.constant_(self.sigma_out.bias, math.log(frac / (1.0 - frac)))

    def forward(
        self,
        h0: torch.Tensor,
        F_aux: torch.Tensor,
        gate_max: torch.Tensor,
        gate_sum: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h0, F_aux: (..., D); gate_max, gate_sum: (...,).
        Returns mu: (..., D) float32, sigma_band: (..., num_bands) float32.
        """
        x = torch.cat(
            [h0, F_aux.to(dtype=h0.dtype), gate_max.unsqueeze(-1).to(dtype=h0.dtype), gate_sum.unsqueeze(-1).to(dtype=h0.dtype)],
            dim=-1,
        )
        feat = self.trunk(x)
        mu = self.mu_out(feat).float()
        raw = self.sigma_out(feat).float()
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(raw)
        return mu, sigma


# =====================================================================================
# [Feature Addition & Bug Fix - Task #038: CausalContextLookback & Successor Relay]
# 1. Problem fixed:
#    In long-context QA (`qa#383582` with 15 clauses, `qa#405886` with 18 clauses), the question
#    appears at the very end of the passage. A pure Mamba RNN (`main_stream`) reads the 15-18 clauses
#    before seeing the question, compressing all prior clauses into a single 1024-D vector `h0_t`,
#    and has no mechanism to look back at earlier clauses `0 .. t` once the question arrives at step `t`
#    or to relay contiguous follow-up clauses during multi-clause answer generation.
# 2. Method used:
#    - `CausalContextLookback`: multi-head causal attention over `(h0_0..t, z_in_0..t)` with deterministic
#      1D sinusoidal clause-position embeddings on keys, a learnable per-head null/sink key (`sink_logit`),
#      and an identity-initialized value/output residual path on `z_in` so attended context clauses pass
#      their exact whitened Legendre spectrum forward.
#    - Clause-Level Successor Relay (100% causal, zero target leakage): when `T >= 3`, if the current
#      input clause `z_in_t` matches an earlier prefix clause `z_in_tau` (`tau <= t - 2`), boosts the
#      attention logit of its immediate context successor `j = tau + 1 <= t - 1` by
#      `softplus(succ_scale) * relu(sim(t, j - 1))`.
#    - `step(h0_t, z_t, history_h, history_z)`: delegates to `self.forward(history_h, history_z)[:, -1, :]`
#      for exact equivalence between parallel `forward_context` and sequential `step` rollouts.
# 3. Caveats for subsequent development:
#    - `MultiStreamRightBrain._upgrade_right_brain_state_dict` pre-hook populates missing
#      `context_lookback.*` keys from `self.state_dict()` when loading older pre-#038 checkpoints.
# =====================================================================================
class CausalContextLookback(nn.Module):
    """
    Causal Clause-Level Context Lookback, Top-2 Context Clause Extraction & Successor Relay (#038 / #042).
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_heads: int = 8,
        num_bands: int = DEFAULT_NUM_BANDS,
        sem_bands: int = DEFAULT_SEM_BANDS,
        max_clauses: int = 128,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim={self.hidden_dim} must be divisible by num_heads={self.num_heads}.")
        self.head_dim = self.hidden_dim // self.num_heads
        self.num_bands = int(num_bands)
        self.sem_bands = int(sem_bands)
        self.d_sub = self.hidden_dim // self.num_bands
        self.max_clauses = int(max_clauses)

        self.q_norm = nn.LayerNorm(self.hidden_dim)
        self.kv_norm = nn.LayerNorm(self.hidden_dim)
        self.q_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.sink_logit = nn.Parameter(torch.full((self.num_heads,), 1.5))
        self.qk_scale = nn.Parameter(torch.full((self.num_heads,), 10.0))
        self.succ_scale = nn.Parameter(torch.full((self.num_heads,), 8.0))
        self.alpha_lookback = nn.Parameter(torch.tensor(0.50))
        self.beta_prior = nn.Parameter(torch.tensor(2.00))

        self.register_buffer(
            "legendre_basis",
            _build_discrete_legendre_basis(self.num_bands, self.num_bands),
            persistent=False,
        )
        self._last_ctx_clauses: Optional[torch.Tensor] = None
        self._last_ctx_weights: Optional[torch.Tensor] = None
        self._last_j0_header_mask: Optional[torch.Tensor] = None

        with torch.no_grad():
            nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.q_proj.bias)
            nn.init.normal_(self.k_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.normal_(self.v_proj.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.v_proj.bias)
            # Zero-init residual refinement so direct spectral copy `z_copy` is clean from Step 0
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.002)
            nn.init.zeros_(self.out_proj.bias)

    def _get_clause_pos_emb(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Deterministic 1D sinusoidal clause-index embeddings of shape `(1, T, hidden_dim)`."""
        pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)  # (T, 1)
        half = max(self.hidden_dim // 2, 1)
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / float(half)
        ).unsqueeze(0)  # (1, half)
        angles = pos * freqs  # (T, half)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (T, 2 * half)
        if emb.shape[-1] < self.hidden_dim:
            emb = F.pad(emb, (0, self.hidden_dim - emb.shape[-1]))
        elif emb.shape[-1] > self.hidden_dim:
            emb = emb[:, :self.hidden_dim]
        return (0.25 * emb).unsqueeze(0).to(dtype=dtype)

    def forward(
        self,
        h0: torch.Tensor,
        z_in: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        return_delta: bool = False,
        j0_header_mask: Optional[torch.Tensor] = None,
    ):
        """
        Causal context lookback over `(B, T, D)`.
        Strictly causal: output at step `t` attends ONLY over strictly prior steps `j <= t - 1`
        (plus the learnable sink key), so `j == t` never self-copies the current input clause.
        """
        B, T, D = h0.shape
        if T == 0:
            self._last_peak_logit = torch.zeros((B, 0, 1), device=h0.device, dtype=torch.float32)
            self._last_ctx_clauses = torch.zeros((B, 0, 1, D), device=h0.device, dtype=torch.float32)
            self._last_ctx_weights = torch.zeros((B, 0, 1), device=h0.device, dtype=torch.float32)
            if return_delta:
                return h0, torch.zeros_like(h0), torch.zeros((B, 0, 1), device=h0.device, dtype=h0.dtype)
            return h0
        H, dh = self.num_heads, self.head_dim
        if valid_mask is None:
            valid_mask = z_in.abs().sum(dim=-1) > 1e-6
        else:
            valid_mask = valid_mask.bool()

        if j0_header_mask is None and T >= 2:
            z0_b = to_bands(z_in[:, 0].float(), self.num_bands)  # (B, nb, d_sub)
            basis = self.legendre_basis.to(device=z_in.device, dtype=torch.float32)
            x0_slots = torch.einsum("bkd,kl->bld", z0_b, basis)
            j0_active_slots = (x0_slots.norm(dim=-1) > 0.20).sum(dim=-1)
            j0_header_mask = (j0_active_slots <= 7) & valid_mask[:, 0] & valid_mask[:, 1]
        self._last_j0_header_mask = j0_header_mask

        z_in_cast = z_in.to(dtype=h0.dtype)
        h_q = self.q_norm(h0)
        # Local Mamba state innovation for Key/Value so j=0's short-header state does not persist into j=1
        h_prev = F.pad(h0[:, :-1, :], (0, 0, 1, 0))
        h_kv = self.kv_norm(h0 - 0.85 * h_prev)

        q_in = torch.cat([h_q, z_in_cast], dim=-1)
        k_in = torch.cat([h_kv, z_in_cast], dim=-1)
        v_in = torch.cat([h_kv, z_in_cast], dim=-1)

        Q = F.normalize(self.q_proj(q_in).view(B, T, H, dh).transpose(1, 2).float(), dim=-1, eps=1e-6)  # (B, H, T, dh)
        K = F.normalize(self.k_proj(k_in).view(B, T, H, dh).transpose(1, 2).float(), dim=-1, eps=1e-6)  # (B, H, T, dh)
        V = self.v_proj(v_in).view(B, T, H, dh).transpose(1, 2).float()  # (B, H, T, dh)

        scale_qk = F.softplus(self.qk_scale.float()).clamp(min=1.0, max=12.0).view(1, H, 1, 1)
        logits = scale_qk * torch.matmul(Q, K.transpose(-2, -1))  # (B, H, T, T)
        strict_prev = torch.tril(torch.ones((T, T), device=h0.device, dtype=torch.bool), diagonal=-1)  # j <= t - 1

        # Clause-Level Successor Relay & Self-Repeat Suppression (100% Causal, Zero Target Leakage)
        if T >= 2:
            sem_dim = min(self.sem_bands * self.d_sub, D)
            z_sem = F.normalize(z_in[..., :sem_dim].float(), dim=-1, eps=1e-6)
            z_full = F.normalize(z_in.float(), dim=-1, eps=1e-6)
            sim_sem = torch.bmm(z_sem, z_sem.transpose(1, 2))
            sim_full = torch.bmm(z_full, z_full.transpose(1, 2))
            sim = 0.6 * sim_sem + 0.4 * sim_full
            vm_f = valid_mask.float()
            sim = sim * vm_f.unsqueeze(1) * vm_f.unsqueeze(2)
            sim_full = sim_full * vm_f.unsqueeze(1) * vm_f.unsqueeze(2)

            if T >= 3:
                succ_boost = torch.zeros_like(sim)
                succ_boost[:, :, 1:] = F.relu(sim[:, :, :-1] - 0.35) / 0.65
                succ_boost = succ_boost.masked_fill(~strict_prev.unsqueeze(0), 0.0)
                if j0_header_mask is not None:
                    j0_m = j0_header_mask.bool().view(B, 1, 1)
                    succ_boost[:, 2:, :2] = succ_boost[:, 2:, :2].masked_fill(j0_m, 0.0)
                # Protect succ_boost from negative base <Q_t, K_{tau+1}> dot products
                logits = torch.where(succ_boost.unsqueeze(1) > 0.15, logits.clamp(min=0.0), logits)
                logits = logits + F.softplus(self.succ_scale.float()).view(1, H, 1, 1) * succ_boost.unsqueeze(1)

            # Suppress copying an earlier clause `j <= t - 1` that is already identical to current input `z_in[t]`
            repeat_pen = F.relu(sim_full - 0.65).masked_fill(~strict_prev.unsqueeze(0), 0.0)
            logits = logits - 25.0 * repeat_pen.unsqueeze(1)

        key_allow = strict_prev.unsqueeze(0) & valid_mask.unsqueeze(1)  # (B, T, T), strictly j <= t - 1
        if T >= 3 and j0_header_mask is not None:
            key_allow = key_allow.clone()
            key_allow[:, 2:, 0] = key_allow[:, 2:, 0] & (~j0_header_mask.bool().view(B, 1))
        logits = logits.masked_fill(~key_allow.unsqueeze(1), -1e4)

        n_prev = key_allow.float().sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, T, 1)
        sink = (self.sink_logit.float().view(1, H, 1, 1) + 0.5 * n_prev.log().unsqueeze(1)).expand(B, H, T, 1)  # (B, H, T, 1)
        logits_with_sink = torch.cat([logits, sink], dim=-1)  # (B, H, T, T + 1)
        attn = F.softmax(logits_with_sink, dim=-1)[..., :T]   # (B, H, T, T)
        attn = attn.masked_fill(~key_allow.unsqueeze(1), 0.0)
        attn = attn * valid_mask.unsqueeze(1).unsqueeze(-1).float()

        clause_logits_with_sink = 0.5 * logits_with_sink.mean(dim=1) + 0.5 * logits_with_sink.amax(dim=1)  # (B, T, T + 1)
        attn_clause = F.softmax(clause_logits_with_sink, dim=-1)[..., :T]  # (B, T, T)
        attn_clause = attn_clause.masked_fill(~key_allow, 0.0) * valid_mask.unsqueeze(-1).float()
        attn_mass = attn_clause.sum(dim=-1, keepdim=True)  # (B, T, 1)
        probs_norm = attn_clause / attn_mass.clamp(min=1e-6)
        peak_idx = attn_clause.argmax(dim=-1, keepdim=True)  # (B, T, 1)
        one_hot = torch.zeros_like(attn_clause).scatter_(-1, peak_idx, 1.0) * key_allow.float()
        vm_col = valid_mask.unsqueeze(-1).float()
        has_prev = key_allow.any(dim=-1, keepdim=True).float()  # (B, T, 1), 0 at t=0

        z_hard = torch.bmm(one_hot * vm_col * has_prev, z_in.float())     # (B, T, D)
        z_soft = torch.bmm(probs_norm * vm_col * has_prev, z_in.float())  # (B, T, D)
        hb = to_bands(z_hard, self.num_bands)  # (B, T, nb, d_sub)
        sb = to_bands(z_soft, self.num_bands)  # (B, T, nb, d_sub)
        if self.d_sub > 1:
            r_h0 = (hb[..., :1].pow(2).mean(dim=-2, keepdim=True) + 1e-6).sqrt().detach()
            r_s0 = (sb[..., :1].pow(2).mean(dim=-2, keepdim=True) + 1e-6).sqrt()
            s0_unit = sb[..., :1] * (r_h0 / r_s0)
            r_h_lex = (hb[..., 1:].pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt().detach()
            r_s_lex = (sb[..., 1:].pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
            s_lex_unit = sb[..., 1:] * (r_h_lex / r_s_lex)
            z_soft_unit = from_bands(torch.cat([s0_unit, s_lex_unit], dim=-1)) * vm_col * has_prev
        else:
            r_h = (hb.pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt().detach()
            r_s = (sb.pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
            z_soft_unit = from_bands(sb * (r_h / r_s)) * vm_col * has_prev
        z_copy_unit = z_hard.detach() + z_soft_unit - z_soft_unit.detach()  # (B, T, D)

        max_clause_logit = clause_logits_with_sink[..., :T].amax(dim=-1, keepdim=True)  # (B, T, 1)
        sink_clause_logit = clause_logits_with_sink[..., T:]                           # (B, T, 1)
        peak_logit = (max_clause_logit - sink_clause_logit) * vm_col * has_prev        # (B, T, 1)
        self._last_peak_logit = peak_logit
        self._last_clause_rel_logits = (clause_logits_with_sink[..., :T] - sink_clause_logit) * vm_col * has_prev
        attn_peak = torch.sigmoid(peak_logit) * vm_col * has_prev

        # Top-2 candidate context clauses for SlotContextCrossAttn (#042)
        k_top = min(2, T)
        clause_logits = clause_logits_with_sink[..., :T].masked_fill(~key_allow, -1e4)
        topk_logits, topk_idx = clause_logits.topk(k=k_top, dim=-1)  # (B, T, k_top)
        topk_weights = torch.sigmoid(topk_logits - sink_clause_logit) * vm_col * has_prev  # (B, T, k_top)
        idx_exp = topk_idx.unsqueeze(-1).expand(B, T, k_top, D)
        z_in_exp = z_in.float().unsqueeze(1).expand(B, T, T, D)
        ctx_clauses = torch.gather(z_in_exp, dim=2, index=idx_exp) * vm_col.unsqueeze(-1) * has_prev.unsqueeze(-1)
        if k_top >= 2:
            one_hot_k1 = torch.zeros_like(attn_clause).scatter_(-1, topk_idx[..., 1:2], 1.0) * key_allow.float()
            attn_rem = attn_clause * (1.0 - one_hot)
            probs_rem = attn_rem / attn_rem.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            z_k1_hard = torch.bmm(one_hot_k1 * vm_col * has_prev, z_in.float())
            z_k1_soft = torch.bmm(probs_rem * vm_col * has_prev, z_in.float())
            hb1 = to_bands(z_k1_hard, self.num_bands)
            sb1 = to_bands(z_k1_soft, self.num_bands)
            if self.d_sub > 1:
                r_h0_1 = (hb1[..., :1].pow(2).mean(dim=-2, keepdim=True) + 1e-6).sqrt().detach()
                r_s0_1 = (sb1[..., :1].pow(2).mean(dim=-2, keepdim=True) + 1e-6).sqrt()
                s0_u1 = sb1[..., :1] * (r_h0_1 / r_s0_1)
                r_hl_1 = (hb1[..., 1:].pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt().detach()
                r_sl_1 = (sb1[..., 1:].pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
                sl_u1 = sb1[..., 1:] * (r_hl_1 / r_sl_1)
                z_k1_soft_unit = from_bands(torch.cat([s0_u1, sl_u1], dim=-1)) * vm_col * has_prev
            else:
                r_h1 = (hb1.pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt().detach()
                r_s1 = (sb1.pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
                z_k1_soft_unit = from_bands(sb1 * (r_h1 / r_s1)) * vm_col * has_prev
            z_k1_unit = z_k1_hard.detach() + z_k1_soft_unit - z_k1_soft_unit.detach()
            ctx_clauses = torch.cat([z_copy_unit.unsqueeze(2), z_k1_unit.unsqueeze(2), ctx_clauses[:, :, 2:, :]], dim=2)
        else:
            ctx_clauses = z_copy_unit.unsqueeze(2)
        self._last_ctx_clauses = ctx_clauses
        self._last_ctx_weights = topk_weights

        out_heads = torch.matmul(attn, V.to(dtype=attn.dtype))  # (B, H, T, dh)
        out_feat = self.out_proj(out_heads.transpose(1, 2).contiguous().view(B, T, D).to(dtype=h0.dtype)).float()
        out_h = (attn_peak * z_copy_unit + out_feat) * vm_col
        out_lb = z_copy_unit * vm_col * has_prev

        h_out = (h0.float() + torch.tanh(self.alpha_lookback.float()) * out_h).to(dtype=h0.dtype)
        if return_delta:
            gate_lb = (torch.sigmoid(self.beta_prior.float()) * attn_peak).to(dtype=h0.dtype)
            return h_out, out_lb.to(dtype=h0.dtype), gate_lb
        return h_out

    def step(
        self,
        h0_t: torch.Tensor,
        z_t: torch.Tensor,
        history_h: torch.Tensor,
        history_z: torch.Tensor,
        history_valid: Optional[torch.Tensor] = None,
        return_delta: bool = False,
        j0_header_mask: Optional[torch.Tensor] = None,
    ):
        """
        Single-step causal lookback over `history_h: (B, T_hist, D)` (ending with `h0_t`)
        and `history_z: (B, T_hist, D)` (ending with `z_t`).
        """
        if return_delta:
            h_all, z_lb_all, gate_lb_all = self.forward(
                history_h, history_z, valid_mask=history_valid, return_delta=True, j0_header_mask=j0_header_mask
            )
            return h_all[:, -1, :], z_lb_all[:, -1, :], gate_lb_all[:, -1, :]
        return self.forward(
            history_h, history_z, valid_mask=history_valid, j0_header_mask=j0_header_mask
        )[:, -1, :]


# [Design Change Documentation - Task #029: MultiStreamRightBrain v2 (1 + N dynamic streams)]
# 1. What changed vs #027:
#    - The 3 fixed streams (discourse / knowledge / goal) + scalar gates alpha_k / alpha_g are replaced by
#      `main_stream` (ClauseMambaStream, 4 layers) + ONE shared `aux_stream` (2 layers) that encodes
#      N >= 0 isomorphic companion streams `aux: (B, N, S, D)` (no stream-type embedding; identity is
#      not preset). Each companion contributes its LAST VALID state h_aux_i.
#    - Per-step dynamic gates g[t, i] = sigmoid(<q(h0_t), k(h_aux_i)> / sqrt(d_gate) + b) and an
#      orthogonal force F_aux_t = sum_i g * OrthProj(v(h_aux_i), h0_t); h_t = h0_t + tanh(alpha_aux) F_aux_t.
#      Implementation note: OrthProj is linear in its first argument, so we compute
#      OrthProj(sum_i g_ti v_i, h0_t) — mathematically identical, O(B T D) memory instead of O(B T N D).
#    - New heads: `prior_head` (adaptive 32-band prior, NLL-trained) and `stop_head` (end-of-answer BCE).
#    - Flow loss = K-branch soft competition (see ThoughtFlowMatcher v2).
# 2. Why: information may come from any stream; when the condition is decisive the low-band sigma
#    must shrink (only syntax variation left), when ambiguous it must widen. See the design doc §2.
# 3. Caveats:
#    - #027 checkpoints are INCOMPATIBLE (different module names) -> new ckpt prefix
#      `bicameral_right_brain_v2`; resume checks must skip old checkpoints.
#    - Legacy kwargs `z_knowledge_seq` / `z_goal_seq` are auto-packed (goal -> 1 stream of G clauses,
#      each knowledge clause -> its own length-1 stream). Validity of legacy rows is inferred from
#      non-zero rows (the trainer zeroes padded clauses). Companion streams must be RIGHT-padded.
#    - `alpha_k` / `alpha_g` are read-only aliases of `alpha_aux` kept only for old log lines.
#    - `sigma_max` (default 1.0) caps per-band std in normalized space; bands whose residual std
#      exceeds it will saturate — watch `sigma_band` metrics and raise `sigma_max` if needed.
class MultiStreamRightBrain(nn.Module):
    """
    Right-Brain v2 (#029): 1 main + N companion Mamba streams, 32-band adaptive prior cone,
    K-branch soft-competition flow matcher and stop head.
    """

    GROUP_NAMES = {0: "full", 1: "goal_only", 2: "none"}

    def __init__(
        self,
        hidden_dim: int = 1024,
        d_state: int = 16,
        num_mamba_layers: int = 2,
        num_flow_layers: int = 4,
        *,
        d_clause: Optional[int] = None,
        d_mamba: Optional[int] = None,
        n_layers_main: Optional[int] = None,
        n_layers_aux: Optional[int] = None,
        n_layers_discourse: Optional[int] = None,
        n_layers_knowledge: Optional[int] = None,
        n_layers_goal: Optional[int] = None,
        active_streams: Optional[Sequence[str]] = None,
        num_bands: int = DEFAULT_NUM_BANDS,
        sem_bands: int = DEFAULT_SEM_BANDS,
        d_gate: int = 128,
        prior_hidden: int = 1024,
        sigma_min: float = 0.05,
        sigma_max: float = 1.0,
        sigma_init: float = 0.5,
        num_branches: int = 4,
        train_ode_steps: int = 3,
        win_weight: float = 0.65,
        other_temp: float = 1.0,
        prior_weight: float = 1.0,
        stop_weight: float = 0.2,
        alpha_aux_init: float = 0.0,
        zero_init_aux: bool = True,
    ):
        super().__init__()
        if d_clause is not None:
            hidden_dim = int(d_clause)
        self.hidden_dim = hidden_dim
        self.d_clause = hidden_dim
        self.d_mamba = int(d_mamba) if d_mamba is not None else hidden_dim
        self.num_bands = int(num_bands)
        self.sem_bands = int(sem_bands)
        self.d_sub = hidden_dim // self.num_bands
        self.d_gate = int(d_gate)
        self.prior_weight = float(prior_weight)
        self.stop_weight = float(stop_weight)

        # Layer counts: explicit v2 args > legacy #027 args > v2 defaults (main=4, aux=2).
        # `num_mamba_layers` is accepted for signature compatibility only.
        if n_layers_main is not None:
            l_main = int(n_layers_main)
        elif n_layers_discourse is not None:
            l_main = int(n_layers_discourse)
        else:
            l_main = 4
        if n_layers_aux is not None:
            l_aux = int(n_layers_aux)
        elif n_layers_goal is not None:
            l_aux = int(n_layers_goal)
        elif n_layers_knowledge is not None:
            l_aux = int(n_layers_knowledge)
        else:
            l_aux = 2

        if active_streams is None:
            self.active_streams: Tuple[str, ...] = ("discourse", "aux")
        elif isinstance(active_streams, str):
            self.active_streams = tuple(s.strip() for s in active_streams.split(",") if s.strip())
        else:
            self.active_streams = tuple(active_streams)

        self.main_stream = ClauseMambaStream(
            hidden_dim=self.d_mamba, d_state=d_state, num_layers=l_main, input_dim=self.d_clause,
        )
        self.aux_stream = ClauseMambaStream(
            hidden_dim=self.d_mamba, d_state=d_state, num_layers=l_aux, input_dim=self.d_clause,
        )

        # Dynamic per-step companion gates + orthogonal force value projection
        self.gate_q = nn.Linear(self.d_clause, self.d_gate)
        self.gate_k = nn.Linear(self.d_clause, self.d_gate)
        self.gate_bias = nn.Parameter(torch.zeros(1))
        self.aux_v = nn.Linear(self.d_clause, self.d_clause)
        # [Design Change - Task #033] `alpha_aux_init` default 0.1 -> 0.0 (tanh(alpha) == 0): together with
        # the zero-initialised companion columns of `PriorHead` (`zero_init_aux`), a fresh model given
        # `aux` produces bit-identical (mu, sigma, h) to `aux=None`. Stage 3 attaches companions as an
        # exact identity branch (staged plan §2/§4). Caveat: while tanh(alpha_aux) == 0 the gate / aux_v /
        # aux_stream weights get ZERO gradient; alpha_aux itself does get a gradient (dh/dalpha = F_aux),
        # so it leaves 0 after the first aux-stage step and the branch then opens smoothly.
        a0 = min(max(float(alpha_aux_init), -0.999), 0.999)
        self.alpha_aux = nn.Parameter(torch.full((1,), math.atanh(a0)))

        self.prior_head = PriorHead(
            d_clause=self.d_clause, num_bands=self.num_bands, hidden_dim=prior_hidden,
            sigma_min=sigma_min, sigma_max=sigma_max, sigma_init=sigma_init,
            zero_init_aux=zero_init_aux,
        )
        self.stop_head = nn.Linear(self.d_clause, 1)

        self.flow_matcher = ThoughtFlowMatcher(
            hidden_dim=self.d_clause, num_layers=num_flow_layers,
            num_bands=self.num_bands, sem_bands=self.sem_bands, num_branches=num_branches,
            train_ode_steps=train_ode_steps, win_weight=win_weight, other_temp=other_temp,
        )
        self.context_lookback = CausalContextLookback(
            hidden_dim=self.d_clause,
            num_heads=8,
            num_bands=self.num_bands,
            sem_bands=self.sem_bands,
        )
        self._upgraded_context_lookback = False
        self._register_load_state_dict_pre_hook(self._upgrade_right_brain_state_dict)

    def _upgrade_right_brain_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """
        #038 backward compatibility: populates missing `context_lookback.*` parameters/buffers from
        `self.state_dict()` when loading an older Stage-1 / v2 checkpoint so `strict=True` succeeds.
        """
        own_sd = None
        filled = 0
        for name, _ in self.context_lookback.named_parameters():
            key = f"{prefix}context_lookback.{name}"
            if key not in state_dict:
                if own_sd is None:
                    own_sd = self.state_dict()
                state_dict[key] = own_sd[f"context_lookback.{name}"].detach().clone()
                filled += 1
        for name, _ in self.context_lookback.named_buffers():
            key = f"{prefix}context_lookback.{name}"
            if key not in state_dict:
                if own_sd is None:
                    own_sd = self.state_dict()
                own_key = f"context_lookback.{name}"
                # [Bug Fix - Task #045] Skip non-persistent buffers (e.g. v3 `legendre_basis`, registered with
                # `persistent=False`): they never appear in any state_dict, so looking them up raised
                # `KeyError` and made `find_compatible_ckpt` treat every v3 checkpoint as incompatible
                # (silent restart from step 0 under `--continue`).
                if own_key not in own_sd:
                    continue
                state_dict[key] = own_sd[own_key].detach().clone()
                filled += 1
        self._upgraded_context_lookback = filled > 0
        if filled > 0:
            print(
                f"ℹ️ MultiStreamRightBrain: upgraded checkpoint with fresh CausalContextLookback ({filled} tensors)."
            )

    # ------------------------------------------------------------------ legacy aliases
    @property
    def alpha_k(self) -> torch.Tensor:
        """Read-only #027 log alias of `alpha_aux`."""
        return self.alpha_aux

    @property
    def alpha_g(self) -> torch.Tensor:
        """Read-only #027 log alias of `alpha_aux`."""
        return self.alpha_aux

    # ------------------------------------------------------------------ static helpers
    @staticmethod
    def _project_orthogonal(
        stream_vec: torch.Tensor,
        main_vec: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Projects `stream_vec` onto the hyperplane strictly orthogonal to `main_vec` along the last dimension:
          v_orth = stream_vec - (<stream_vec, main_vec> / (||main_vec||^2 + eps)) * main_vec
        Prevents secondary streams from overwriting the main discourse trajectory's direction.
        """
        dot = (stream_vec.float() * main_vec.float()).sum(dim=-1, keepdim=True)
        norm_sq = (main_vec.float() * main_vec.float()).sum(dim=-1, keepdim=True).clamp(min=eps)
        proj = (dot / norm_sq) * main_vec.float()
        return (stream_vec.float() - proj).to(dtype=main_vec.dtype)

    @staticmethod
    def pack_aux_streams(
        z_goal: Optional[torch.Tensor] = None,
        goal_valid: Optional[torch.Tensor] = None,
        z_cand: Optional[torch.Tensor] = None,
        cand_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Packs companion streams into `aux: (B, N, S, D)` + `aux_valid: (B, N, S)`:
          - stream 0 = goal clause sequence `z_goal: (B, G, D)` (or (B, D) -> G = 1),
          - streams 1..C = each candidate/knowledge clause `z_cand: (B, C, D)` as its own length-1 stream.
        Missing validity masks are inferred as "row is non-zero". Returns (None, None) if nothing to pack.
        """
        parts: List[Tuple[torch.Tensor, torch.Tensor]] = []  # (B, n, S_i, D), (B, n, S_i)
        if z_goal is not None:
            if z_goal.dim() == 2:
                z_goal = z_goal.unsqueeze(1)
                if goal_valid is not None and goal_valid.dim() == 1:
                    goal_valid = goal_valid.unsqueeze(1)
            gv = goal_valid.bool() if goal_valid is not None else (z_goal.abs().sum(dim=-1) > 0)
            if z_goal.shape[1] > 0:
                parts.append((z_goal.unsqueeze(1), gv.unsqueeze(1)))
        if z_cand is not None:
            if z_cand.dim() == 2:
                z_cand = z_cand.unsqueeze(1)
                if cand_valid is not None and cand_valid.dim() == 1:
                    cand_valid = cand_valid.unsqueeze(1)
            cv = cand_valid.bool() if cand_valid is not None else (z_cand.abs().sum(dim=-1) > 0)
            if z_cand.shape[1] > 0:
                parts.append((z_cand.unsqueeze(2), cv.unsqueeze(2)))
        if not parts:
            return None, None
        S = max(p[0].shape[2] for p in parts)
        auxs, valids = [], []
        for a, v in parts:
            pad = S - a.shape[2]
            if pad > 0:
                a = F.pad(a, (0, 0, 0, pad))
                v = F.pad(v, (0, pad), value=False)
            auxs.append(a)
            valids.append(v)
        dtype = auxs[0].dtype
        return torch.cat([a.to(dtype=dtype) for a in auxs], dim=1), torch.cat(valids, dim=1)

    def _normalize_clause_input(self, z: torch.Tensor) -> torch.Tensor:
        """Whitens non-zero clause rows via `flow_matcher.normalize`, preserving all-zero padding rows at 0."""
        nonzero = z.abs().sum(dim=-1, keepdim=True) > 1e-6
        z_norm = self.flow_matcher.normalize(z.float()).to(dtype=z.dtype)
        return torch.where(nonzero, z_norm, z)

    @torch.no_grad()
    def _jitter_high_bands_impl(
        self: Optional["MultiStreamRightBrain"],
        z: torch.Tensor,
        p: float = 0.3,
        sem_bands: int = DEFAULT_SEM_BANDS,
        num_bands: int = DEFAULT_NUM_BANDS,
    ) -> torch.Tensor:
        """
        Exposure-bias augmentation (§2.6): for each clause row independently with probability `p`,
        rotates every HIGH band (b >= sem_bands) by an independent random orthogonal matrix
        (QR of a Gaussian, sign-fixed -> Haar). When called on a calibrated `MultiStreamRightBrain`
        instance, rotation is performed in the whitened `N(0, I_D)` space and denormalized back so
        channel-0 variance does not leak into channels 1..31; low bands and all-zero (padded) rows
        are returned bit-identical. Works on any `(..., D)` tensor.
        """
        if p <= 0.0 or z.numel() == 0 or sem_bands >= num_bands:
            return z
        orig_shape = z.shape
        D = orig_shape[-1]
        flat = z.reshape(-1, D)
        R = flat.shape[0]
        nonzero = flat.abs().sum(dim=-1) > 0
        sel = (torch.rand((R,), device=z.device) < float(p)) & nonzero
        idx = torch.nonzero(sel, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return z
        d_sub = D // num_bands
        n_hi = num_bands - int(sem_bands)
        n_sel = idx.numel()
        rows = flat[idx].float()
        if self is not None:
            rows = self.flow_matcher.normalize(rows)
        rows_b = to_bands(rows, num_bands)  # (n_sel, nb, d_sub)
        hi = rows_b[:, sem_bands:, :]  # (n_sel, n_hi, d_sub)
        qr_dev = torch.device("cpu") if z.device.type == "mps" else z.device
        G = torch.randn((n_sel, n_hi, d_sub, d_sub), device=qr_dev, dtype=torch.float32)
        Q, Rm = torch.linalg.qr(G)
        sgn = torch.sign(torch.diagonal(Rm, dim1=-2, dim2=-1))
        sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
        Q = (Q * sgn.unsqueeze(-2)).to(device=z.device)
        hi_rot = torch.einsum("nbij,nbj->nbi", Q, hi)
        rows_rot = rows.clone()
        rows_rot[:, sem_bands * d_sub:] = from_bands(hi_rot)
        if self is not None:
            rows_rot = self.flow_matcher.denormalize(rows_rot)
        new_rows = flat[idx].clone()
        new_rows[:, sem_bands * d_sub:] = rows_rot[:, sem_bands * d_sub:].to(dtype=z.dtype)
        out = flat.clone()
        out[idx] = new_rows
        return out.reshape(orig_shape)

    class _DualMethod:
        def __init__(self, fn):
            self.fn = fn
        def __get__(self, instance, owner=None):
            return lambda *args, **kwargs: self.fn(instance, *args, **kwargs)

    jitter_high_bands = _DualMethod(_jitter_high_bands_impl)

    # ------------------------------------------------------------------ stream encoding
    def _use_aux(self, active_streams: Optional[Sequence[str]]) -> bool:
        streams = self.active_streams if active_streams is None else active_streams
        if isinstance(streams, str):
            streams = tuple(s.strip() for s in streams.split(",") if s.strip())
        return any(s != "discourse" for s in streams)

    def _resolve_aux(
        self,
        aux: Optional[torch.Tensor],
        aux_valid: Optional[torch.Tensor],
        active_streams: Optional[Sequence[str]] = None,
        z_knowledge_seq: Optional[torch.Tensor] = None,
        z_goal_seq: Optional[torch.Tensor] = None,
        knowledge_valid: Optional[torch.Tensor] = None,
        goal_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Merges explicit `aux` with legacy knowledge/goal kwargs; honours `active_streams=('discourse',)`."""
        if not self._use_aux(active_streams):
            return None, None
        legacy_aux, legacy_valid = self.pack_aux_streams(z_goal_seq, goal_valid, z_knowledge_seq, knowledge_valid)
        if aux is not None and aux.dim() == 3:  # (B, N, D) -> N length-1 streams
            aux = aux.unsqueeze(2)
            if aux_valid is not None and aux_valid.dim() == 2:
                aux_valid = aux_valid.unsqueeze(2)
        if aux is not None and aux_valid is None:
            aux_valid = aux.abs().sum(dim=-1) > 0
        if legacy_aux is None:
            return aux, aux_valid
        if aux is None:
            return legacy_aux, legacy_valid
        S = max(aux.shape[2], legacy_aux.shape[2])
        aux = F.pad(aux, (0, 0, 0, S - aux.shape[2]))
        aux_valid = F.pad(aux_valid.bool(), (0, S - aux_valid.shape[2]), value=False)
        legacy_aux = F.pad(legacy_aux, (0, 0, 0, S - legacy_aux.shape[2])).to(dtype=aux.dtype)
        legacy_valid = F.pad(legacy_valid, (0, S - legacy_valid.shape[2]), value=False)
        return torch.cat([aux, legacy_aux], dim=1), torch.cat([aux_valid, legacy_valid], dim=1)

    def encode_aux(
        self,
        aux: Optional[torch.Tensor],
        aux_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encodes companion streams `aux: (B, N, S, D)` with the shared `aux_stream` batched as (B*N, S, D)
        and gathers the state at each stream's LAST VALID position.
        Returns `h_aux: (B, N, D)` (zeros for invalid streams) and `stream_valid: (B, N)` bool,
        or `(None, None)` if `aux` is None or N == 0.
        """
        if aux is None or aux.shape[1] == 0 or aux.shape[2] == 0:
            return None, None
        B, N, S, D = aux.shape
        if aux_valid is None:
            aux_valid = aux.abs().sum(dim=-1) > 0
        valid = aux_valid.reshape(B * N, S).bool()
        pos = torch.arange(S, device=aux.device).unsqueeze(0).expand(B * N, S)
        last_idx = torch.where(valid, pos, torch.full_like(pos, -1)).max(dim=-1).values  # (B*N,)
        stream_valid = last_idx >= 0
        if not bool(stream_valid.any()):
            return (
                torch.zeros((B, N, D), device=aux.device, dtype=aux.dtype),
                stream_valid.reshape(B, N),
            )
        # Trim trailing all-padding positions (no stream needs them) before the sequential scan.
        S_eff = int(last_idx.max().item()) + 1
        flat = self._normalize_clause_input(aux.reshape(B * N, S, D)[:, :S_eff])
        h_seq = self.aux_stream(flat)  # (B*N, S_eff, D)
        gidx = last_idx.clamp(min=0).view(B * N, 1, 1).expand(B * N, 1, h_seq.shape[-1])
        h_last = torch.gather(h_seq, dim=1, index=gidx).squeeze(1)  # (B*N, D)
        h_last = h_last * stream_valid.unsqueeze(-1).to(dtype=h_last.dtype)
        return h_last.reshape(B, N, -1), stream_valid.reshape(B, N)

    def _fuse(
        self,
        h0: torch.Tensor,
        h_aux: Optional[torch.Tensor],
        stream_valid: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        h0: (B, T, D); h_aux: (B, N, D) or None. Returns dict with
        h (B,T,D), h0, F_aux (B,T,D), gates (B,T,N) float32, gate_max (B,T), gate_sum (B,T).
        """
        B, T, D = h0.shape
        if h_aux is None or h_aux.shape[1] == 0:
            zeros_bt = torch.zeros((B, T), device=h0.device, dtype=torch.float32)
            F_aux = torch.zeros_like(h0)
            return {
                "h": h0, "h0": h0, "F_aux": F_aux,
                "gates": torch.zeros((B, T, 0), device=h0.device, dtype=torch.float32),
                "gate_max": zeros_bt, "gate_sum": zeros_bt,
            }
        N = h_aux.shape[1]
        if stream_valid is None:
            stream_valid = torch.ones((B, N), device=h0.device, dtype=torch.bool)
        q = self.gate_q(h0).float()  # (B, T, dg)
        k = self.gate_k(h_aux).float()  # (B, N, dg)
        logits = torch.einsum("btd,bnd->btn", q, k) / math.sqrt(self.d_gate) + self.gate_bias.float()
        gates = torch.sigmoid(logits) * stream_valid.unsqueeze(1).float()  # (B, T, N)
        v = self.aux_v(h_aux).float()  # (B, N, D)
        agg = torch.einsum("btn,bnd->btd", gates, v)  # sum_i g_ti v_i
        F_aux = self._project_orthogonal(agg, h0.float())  # == sum_i g_ti OrthProj(v_i, h0_t)
        h = (h0.float() + torch.tanh(self.alpha_aux.float()) * F_aux).to(dtype=h0.dtype)
        return {
            "h": h, "h0": h0, "F_aux": F_aux.to(dtype=h0.dtype), "gates": gates,
            "gate_max": gates.amax(dim=-1), "gate_sum": gates.sum(dim=-1),
        }

    def forward_context(
        self,
        z_main_seq: Optional[torch.Tensor] = None,
        aux: Optional[torch.Tensor] = None,
        aux_valid: Optional[torch.Tensor] = None,
        *,
        z_lookback_seq: Optional[torch.Tensor] = None,
        active_streams: Optional[Sequence[str]] = None,
        h_aux: Optional[torch.Tensor] = None,
        stream_valid: Optional[torch.Tensor] = None,
        main_states: Optional[List[torch.Tensor]] = None,
        history_h: Optional[torch.Tensor] = None,
        history_z: Optional[torch.Tensor] = None,
        return_state: bool = False,
        discourse_seq: Optional[torch.Tensor] = None,
        z_knowledge_seq: Optional[torch.Tensor] = None,
        z_goal_seq: Optional[torch.Tensor] = None,
        knowledge_seq: Optional[torch.Tensor] = None,
        goal_seq: Optional[torch.Tensor] = None,
        knowledge_valid: Optional[torch.Tensor] = None,
        goal_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Runs `main_stream` over `z_main_seq: (B, T, D)`, applies `context_lookback`, and fuses companion streams.
        Pre-encoded `h_aux` / `stream_valid` may be passed to skip `encode_aux`.
        Returns dict: h (B,T,D), h0 (B,T,D), F_aux (B,T,D), gates (B,T,N), gate_max (B,T),
        gate_sum (B,T), h_aux, stream_valid (+ `main_states`, `history_h`, `history_z` if return_state).
        """
        if z_main_seq is None:
            z_main_seq = discourse_seq
        if z_main_seq is None:
            raise ValueError("forward_context requires `z_main_seq`.")
        if z_main_seq.dim() == 2:
            z_main_seq = z_main_seq.unsqueeze(1)
        if z_lookback_seq is not None and z_lookback_seq.dim() == 2:
            z_lookback_seq = z_lookback_seq.unsqueeze(1)
        if z_knowledge_seq is None:
            z_knowledge_seq = knowledge_seq
        if z_goal_seq is None:
            z_goal_seq = goal_seq

        if h_aux is None:
            aux_r, aux_valid_r = self._resolve_aux(
                aux, aux_valid, active_streams, z_knowledge_seq, z_goal_seq, knowledge_valid, goal_valid
            )
            h_aux, stream_valid = self.encode_aux(aux_r, aux_valid_r)
        elif not self._use_aux(active_streams):
            h_aux, stream_valid = None, None

        z_main_in = self._normalize_clause_input(z_main_seq)
        z_lb_in = self._normalize_clause_input(z_lookback_seq) if z_lookback_seq is not None else z_main_in
        h0_raw, new_main = self.main_stream(z_main_in, states=main_states, return_state=True)
        if history_h is not None and history_z is not None:
            full_h = torch.cat([history_h.to(dtype=h0_raw.dtype), h0_raw], dim=1)
            full_z = torch.cat([history_z.to(dtype=z_lb_in.dtype), z_lb_in], dim=1)
            if full_z.shape[1] >= 2:
                j0_slots = (self.flow_matcher.reconstruct_spatial_slots(full_z[:, 0]).norm(dim=-1) > 0.20).sum(dim=-1)
                j0_valid = full_z[:, 0].abs().sum(dim=-1) > 1e-6
                j1_valid = full_z[:, 1].abs().sum(dim=-1) > 1e-6
                j0_header_mask = (j0_slots <= 7) & j0_valid & j1_valid
            else:
                j0_header_mask = None
            h_lb, z_lb, g_lb = self.context_lookback(
                full_h, full_z, return_delta=True, j0_header_mask=j0_header_mask
            )
            T_cur = h0_raw.shape[1]
            h0 = h_lb[:, -T_cur:, :]
            z_lookback = z_lb[:, -T_cur:, :]
            gate_lb = g_lb[:, -T_cur:, :]
            peak_logit_all = getattr(self.context_lookback, "_last_peak_logit", None)
            peak_logit = peak_logit_all[:, -T_cur:, :] if peak_logit_all is not None else None
            rel_logits_all = getattr(self.context_lookback, "_last_clause_rel_logits", None)
            clause_rel_logits = rel_logits_all[:, -T_cur:, -T_cur:] if rel_logits_all is not None else None
            ctx_clauses_all = getattr(self.context_lookback, "_last_ctx_clauses", None)
            ctx_clauses = ctx_clauses_all[:, -T_cur:, :, :] if ctx_clauses_all is not None else None
            ctx_weights_all = getattr(self.context_lookback, "_last_ctx_weights", None)
            ctx_weights = ctx_weights_all[:, -T_cur:, :] if ctx_weights_all is not None else None
        else:
            full_h = h0_raw
            full_z = z_lb_in
            if full_z.shape[1] >= 2:
                j0_slots = (self.flow_matcher.reconstruct_spatial_slots(full_z[:, 0]).norm(dim=-1) > 0.20).sum(dim=-1)
                j0_valid = full_z[:, 0].abs().sum(dim=-1) > 1e-6
                j1_valid = full_z[:, 1].abs().sum(dim=-1) > 1e-6
                j0_header_mask = (j0_slots <= 7) & j0_valid & j1_valid
            else:
                j0_header_mask = None
            h0, z_lookback, gate_lb = self.context_lookback(
                h0_raw, z_lb_in, return_delta=True, j0_header_mask=j0_header_mask
            )
            peak_logit = getattr(self.context_lookback, "_last_peak_logit", None)
            clause_rel_logits = getattr(self.context_lookback, "_last_clause_rel_logits", None)
            ctx_clauses = getattr(self.context_lookback, "_last_ctx_clauses", None)
            ctx_weights = getattr(self.context_lookback, "_last_ctx_weights", None)
        out = self._fuse(h0, h_aux, stream_valid)
        out["z_lookback"] = z_lookback
        out["gate_lb"] = gate_lb
        out["peak_logit"] = peak_logit
        out["clause_rel_logits"] = clause_rel_logits
        out["ctx_clauses"] = ctx_clauses
        out["ctx_weights"] = ctx_weights
        out["h_aux"] = h_aux
        out["stream_valid"] = stream_valid
        if return_state:
            out["main_states"] = new_main
            out["history_h"] = full_h.detach()
            out["history_z"] = full_z.detach()
        return out

    def predict_prior(self, ctx: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies `prior_head` + decoupled Channel-0 / Lexical polar-band pointer-generator `z_lookback` interpolation -> (mu (…,D), sigma_band (…,nb)), float32."""
        mu_head, sigma = self.prior_head(ctx["h0"], ctx["F_aux"], ctx["gate_max"], ctx["gate_sum"])
        ctx["mu_head"] = mu_head
        z_lb = ctx.get("z_lookback")
        g_lb = ctx.get("gate_lb")
        if z_lb is not None and g_lb is not None:
            gf = g_lb.float().unsqueeze(-1)  # (..., 1, 1)
            mh_b = to_bands(mu_head.float(), self.num_bands)  # (..., nb, d_sub)
            lb_b = to_bands(z_lb.float(), self.num_bands)     # (..., nb, d_sub)
            if self.d_sub > 1:
                # Decouple PCA Channel 0 (`[..., :1]`, the 32-band clause-length / <PAD> envelope spectrum)
                # from Channels 1..31 (`[..., 1:]`, the 31 lexical channels per band) so lexical angle
                # differences never cross-contaminate or rescale Channel 0's physical length envelope.
                mh_0, lb_0 = mh_b[..., :1], lb_b[..., :1]  # (..., nb, 1)
                r_mh_0 = (mh_0.pow(2).mean(dim=-2, keepdim=True) + 1e-6).sqrt()
                r_lb_0 = (lb_0.pow(2).mean(dim=-2, keepdim=True) + 1e-6).sqrt()
                u_mh_0 = mh_0 / r_mh_0
                u_lb_0 = lb_0 / r_lb_0
                r_blend_0 = (1.0 - gf) * r_mh_0 + gf * r_lb_0
                u_blend_0 = F.normalize((1.0 - gf) * u_mh_0 + gf * u_lb_0, dim=-2, eps=1e-6) * math.sqrt(self.num_bands)
                mu_b_0 = u_blend_0 * r_blend_0  # (..., nb, 1)

                mh_lex, lb_lex = mh_b[..., 1:], lb_b[..., 1:]  # (..., nb, d_sub - 1)
                r_mh_lex = (mh_lex.pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
                r_lb_lex = (lb_lex.pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
                u_mh_lex = mh_lex / r_mh_lex
                u_lb_lex = lb_lex / r_lb_lex
                r_blend_lex = (1.0 - gf) * r_mh_lex + gf * r_lb_lex
                u_blend_lex = F.normalize((1.0 - gf) * u_mh_lex + gf * u_lb_lex, dim=-1, eps=1e-6) * math.sqrt(self.d_sub - 1)
                mu_b_lex = u_blend_lex * r_blend_lex  # (..., nb, d_sub - 1)

                mu = from_bands(torch.cat([mu_b_0, mu_b_lex], dim=-1))
            else:
                r_mh = (mh_b.pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
                r_lb = (lb_b.pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
                u_mh = mh_b / r_mh
                u_lb = lb_b / r_lb
                r_blend = (1.0 - gf) * r_mh + gf * r_lb
                u_blend = F.normalize((1.0 - gf) * u_mh + gf * u_lb, dim=-1, eps=1e-6) * math.sqrt(self.d_sub)
                mu = from_bands(u_blend * r_blend)
            if not self.training:
                # At inference/rollout, contract prior cone radius proportional to lookback copy confidence `g_lb`
                # so high-confidence context copies/relays (`g_lb > 0.85`) have `sigma ~ 0.10`.
                sigma = (sigma * (1.0 - 0.85 * g_lb.float())).clamp(min=self.prior_head.sigma_min)
        elif z_lb is not None:
            mu = mu_head + z_lb.float()
        else:
            mu = mu_head

        ctx_clauses = ctx.get("ctx_clauses")
        ctx_weights = ctx.get("ctx_weights")
        if ctx_clauses is None:
            last_cc = getattr(self.context_lookback, "_last_ctx_clauses", None)
            last_cw = getattr(self.context_lookback, "_last_ctx_weights", None)
            h0_ref = ctx["h0"]
            if last_cc is not None and last_cc.shape[0] == h0_ref.shape[0]:
                if h0_ref.dim() == 2:
                    ctx_clauses = last_cc[:, -1, :, :]
                    ctx_weights = last_cw[:, -1, :] if last_cw is not None else None
                elif h0_ref.dim() == 3 and last_cc.shape[1] >= h0_ref.shape[1]:
                    T_ref = h0_ref.shape[1]
                    ctx_clauses = last_cc[:, -T_ref:, :, :]
                    ctx_weights = last_cw[:, -T_ref:, :] if last_cw is not None else None
        self.flow_matcher._cached_sample_ctx = {
            "z_lookback": z_lb,
            "gate_lb": g_lb,
            "ctx_clauses": ctx_clauses,
            "ctx_weights": ctx_weights,
        }
        return mu, sigma

    # ------------------------------------------------------------------ training loss
    def compute_loss(
        self,
        z_discourse_seq: Optional[torch.Tensor] = None,
        target_valid_mask: Optional[torch.Tensor] = None,
        aux: Optional[torch.Tensor] = None,
        aux_valid: Optional[torch.Tensor] = None,
        stop_labels: Optional[torch.Tensor] = None,
        z_in_override: Optional[torch.Tensor] = None,
        sample_group: Optional[torch.Tensor] = None,
        return_metrics: bool = True,
        *,
        active_streams: Optional[Sequence[str]] = None,
        z_knowledge_seq: Optional[torch.Tensor] = None,
        z_goal_seq: Optional[torch.Tensor] = None,
        knowledge_seq: Optional[torch.Tensor] = None,
        goal_seq: Optional[torch.Tensor] = None,
        discourse_seq: Optional[torch.Tensor] = None,
        knowledge_valid: Optional[torch.Tensor] = None,
        goal_valid: Optional[torch.Tensor] = None,
        z_target: Optional[torch.Tensor] = None,
        stop_valid_mask: Optional[torch.Tensor] = None,
        group_names: Optional[Dict[int, str]] = None,
        profile_cb: Optional[Any] = None,
    ):
        """
        Right-Brain v3 training loss.
        """
        if z_discourse_seq is None:
            z_discourse_seq = discourse_seq
        if z_discourse_seq is None:
            raise ValueError("MultiStreamRightBrain.compute_loss requires `z_discourse_seq`.")
        if z_knowledge_seq is None:
            z_knowledge_seq = knowledge_seq
        if z_goal_seq is None:
            z_goal_seq = goal_seq

        ctx_kwargs = dict(
            active_streams=active_streams, z_knowledge_seq=z_knowledge_seq, z_goal_seq=z_goal_seq,
            knowledge_valid=knowledge_valid, goal_valid=goal_valid,
        )
        if z_target is not None:
            z1 = z_target.reshape(z_target.shape[0], 1, -1)
            valid = target_valid_mask.reshape(-1, 1) if target_valid_mask is not None else None
        else:
            if z_discourse_seq.shape[1] < 2:
                raise ValueError(
                    f"z_discourse_seq must have at least 2 clauses (lead + >=1 target), got shape {tuple(z_discourse_seq.shape)}"
                )
            z_in = z_in_override if z_in_override is not None else z_discourse_seq[:, :-1, :]
            z1 = z_discourse_seq[:, 1:, :]
            valid = target_valid_mask

        B, Tp, D = z1.shape
        nb = self.num_bands
        if valid is None:
            valid = torch.ones((B, Tp), device=z1.device, dtype=torch.bool)
        valid = valid.reshape(B, Tp).bool()

        mask_flat = valid.reshape(-1)
        empty_valid = not bool(mask_flat.any())
        sel = mask_flat if not empty_valid else torch.zeros_like(mask_flat).index_fill_(0, torch.zeros(1, dtype=torch.long, device=z1.device), True)
        z1_v = z1.reshape(-1, D)[sel]

        fm = self.flow_matcher
        if self.training and not empty_valid and int(fm.calib_steps.item()) < 50:
            fm.calibrate_z(z1_v)

        if z_target is not None:
            ctx = self.forward_context(z_discourse_seq, aux, aux_valid, **ctx_kwargs)
            ctx = {
                k: (
                    v[:, -1:]
                    if (
                        torch.is_tensor(v)
                        and k in ("h", "h0", "F_aux", "gates", "gate_max", "gate_sum", "z_lookback", "gate_lb", "peak_logit", "ctx_clauses", "ctx_weights")
                    )
                    else v
                )
                for k, v in ctx.items()
            }
        else:
            ctx = self.forward_context(z_in, aux, aux_valid, z_lookback_seq=z_discourse_seq[:, :-1, :], **ctx_kwargs)

        mu, sigma = self.predict_prior(ctx)  # (B, Tp, D), (B, Tp, nb) float32
        h = ctx["h"]
        stop_logits = self.stop_head(h).squeeze(-1).float()  # (B, Tp)
        if profile_cb is not None:
            profile_cb("mamba_prior")

        h_v = h.reshape(-1, D)[sel]
        mu_v = mu.reshape(-1, D)[sel]
        sig_v = sigma.reshape(-1, nb)[sel]
        z1n = fm.normalize(z1_v.float())

        z_lb_v = ctx["z_lookback"].reshape(-1, D)[sel] if ctx.get("z_lookback") is not None else None
        gate_lb_v = ctx["gate_lb"].reshape(-1, 1)[sel] if ctx.get("gate_lb") is not None else None
        if ctx.get("ctx_clauses") is not None:
            k_top = ctx["ctx_clauses"].shape[-2]
            ctx_clauses_v = ctx["ctx_clauses"].reshape(-1, k_top, D)[sel]
            ctx_weights_v = ctx["ctx_weights"].reshape(-1, k_top)[sel] if ctx.get("ctx_weights") is not None else None
        else:
            ctx_clauses_v = None
            ctx_weights_v = None

        mu_b = to_bands(mu_v.float(), nb)
        z1_b = to_bands(z1n.float(), nb)
        cos_b = F.cosine_similarity(mu_b.detach(), z1_b.detach(), dim=-1, eps=1e-6).clamp(min=0.0).unsqueeze(-1)
        r_z1_b = (z1_b.detach().pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
        mu_unit_b = F.normalize(mu_b, dim=-1, eps=1e-6) * math.sqrt(self.d_sub)
        mu_for_nll = from_bands(mu_unit_b.detach() * r_z1_b * cos_b)

        nll_raw = band_gaussian_nll(z1n, mu_for_nll, sig_v, nb).mean()
        mu_polar_loss, _ = fm.compute_polar_band_loss(mu_v, z1n)
        mu_slot_loss, _ = fm.compute_spatial_slot_loss(mu_v, z1n)
        mu_cart_mse = F.mse_loss(mu_v.float(), z1n.float())
        prior_nll = nll_raw + 0.5 * mu_polar_loss + 0.5 * mu_slot_loss + 0.15 * mu_cart_mse
        if "mu_head" in ctx:
            mu_head_v = ctx["mu_head"].reshape(-1, D)[sel]
            head_polar_loss, _ = fm.compute_polar_band_loss(mu_head_v, z1n)
            head_slot_loss, _ = fm.compute_spatial_slot_loss(mu_head_v, z1n)
            prior_nll = prior_nll + 0.30 * (head_polar_loss + head_slot_loss)

        gate_lb_loss = prior_nll * 0.0
        if z_target is None and ctx.get("peak_logit") is not None and Tp >= 2:
            z_prev_clean = fm.normalize(z_discourse_seq[:, :-1, :].reshape(-1, D).float()).reshape(B, Tp, D).detach()
            z1_clean = fm.normalize(z1.reshape(-1, D).float()).reshape(B, Tp, D).detach()
            u_prev = F.normalize(z_prev_clean, dim=-1, eps=1e-6)
            u_targ = F.normalize(z1_clean, dim=-1, eps=1e-6)
            sim_oracle = torch.bmm(u_targ, u_prev.transpose(1, 2))  # (B, Tp, Tp)
            strict_prev = torch.tril(torch.ones((Tp, Tp), device=z1.device, dtype=torch.bool), diagonal=-1)
            prev_valid = z_discourse_seq[:, :-1, :].abs().sum(dim=-1) > 1e-6  # (B, Tp)
            key_allow_oracle = strict_prev.unsqueeze(0) & prev_valid.unsqueeze(1)  # (B, Tp, Tp)
            if Tp >= 3:
                j0_slots_o = (fm.reconstruct_spatial_slots(z_prev_clean[:, 0]).norm(dim=-1) > 0.20).sum(dim=-1)
                j0_hdr_o = ((j0_slots_o <= 7) & prev_valid[:, 0] & prev_valid[:, 1]).view(B, 1)
                key_allow_oracle = key_allow_oracle.clone()
                key_allow_oracle[:, 2:, 0] = key_allow_oracle[:, 2:, 0] & (~j0_hdr_o)
            sim_masked = sim_oracle.masked_fill(~key_allow_oracle, -1.0)
            max_ctx_cos, oracle_idx = sim_masked.max(dim=-1, keepdim=True)  # (B, Tp, 1)
            y_gate = ((max_ctx_cos - 0.55) / 0.25).clamp(0.0, 1.0).detach()  # (B, Tp, 1)
            has_prev_mask = key_allow_oracle.any(dim=-1)  # (B, Tp)
            valid_t1 = (valid & has_prev_mask).unsqueeze(-1).float()  # (B, Tp, 1)
            rel_logits = ctx.get("clause_rel_logits")
            if rel_logits is not None and rel_logits.shape == (B, Tp, Tp):
                pos_logit = torch.gather(rel_logits.float(), -1, oracle_idx)  # (B, Tp, 1)
                # Also detect sub-clause character span overlap on non-extractive rows (y_gate == 0)
                x_prev_o = fm.reconstruct_spatial_slots(z_prev_clean)  # (B, Tp, 32, d_sub)
                x_targ_o = fm.reconstruct_spatial_slots(z1_clean)      # (B, Tp, 32, d_sub)
                u_xp = F.normalize(x_prev_o, dim=-1, eps=1e-6)
                u_xt = F.normalize(x_targ_o, dim=-1, eps=1e-6)
                targ_act = (x_targ_o.norm(dim=-1) > 0.15).float()
                prev_act = (x_prev_o.norm(dim=-1) > 0.15).float()
                slot_sim_any = (
                    torch.einsum("btld,bjcd->btjlc", u_xt, u_xp)
                    * targ_act.unsqueeze(2).unsqueeze(-1)
                    * prev_act.unsqueeze(1).unsqueeze(-2)
                )  # (B, Tp, Tp, 32, 32)
                slot_sim_best = slot_sim_any.amax(dim=-1)  # (B, Tp, Tp, 32)
                span_sim = (
                    slot_sim_best.topk(k=min(6, self.num_bands), dim=-1).values.mean(dim=-1)
                    .masked_fill(~key_allow_oracle, -1.0)
                )  # (B, Tp, Tp)
                max_span_sim, span_idx = span_sim.max(dim=-1, keepdim=True)  # (B, Tp, 1)
                has_span = (y_gate <= 0.0) & (max_span_sim > 0.65)

                eff_peak_logit = torch.where(y_gate > 0.0, pos_logit, ctx["peak_logit"].float())
                # On non-extractive span-borrowing rows, cap peak_logit <= -2.5 (so gate_lb < 0.08 < 0.15)
                # while keeping span_idx as the clear Top-1 context clause over non-matching clauses!
                span_pos_logit = torch.gather(rel_logits.float(), -1, span_idx)
                span_loss_pos = F.smooth_l1_loss(span_pos_logit, torch.full_like(span_pos_logit, -2.5), reduction="none")
                span_neg_allow = key_allow_oracle & (span_sim < 0.50)
                has_span_neg = span_neg_allow.any(dim=-1, keepdim=True).float()
                span_neg_peak = rel_logits.float().masked_fill(~span_neg_allow, -1e4).amax(dim=-1, keepdim=True)
                span_neg_peak = torch.where(has_span_neg > 0.5, span_neg_peak, torch.full_like(span_neg_peak, -6.0))
                span_loss_neg = F.softplus(span_neg_peak + 4.5) * has_span_neg

                gate_bce = F.binary_cross_entropy_with_logits(eff_peak_logit, y_gate, reduction="none")
                neg_allow = key_allow_oracle & (sim_oracle < 0.45)
                has_neg = neg_allow.any(dim=-1, keepdim=True).float()
                neg_peak = rel_logits.float().masked_fill(~neg_allow, -1e4).amax(dim=-1, keepdim=True)
                neg_peak = torch.where(has_neg > 0.5, neg_peak, torch.zeros_like(neg_peak))
                neg_bce = F.binary_cross_entropy_with_logits(neg_peak, torch.zeros_like(neg_peak), reduction="none") * has_neg * (y_gate > 0.0).float()
                gate_bce = torch.where(
                    has_span,
                    0.5 * gate_bce + span_loss_pos + span_loss_neg,
                    gate_bce + 0.5 * neg_bce,
                )
            else:
                gate_bce = F.binary_cross_entropy_with_logits(ctx["peak_logit"].float(), y_gate, reduction="none")
            gate_lb_loss = (gate_bce * valid_t1).sum() / valid_t1.sum().clamp(min=1.0)
            prior_nll = prior_nll + 0.40 * gate_lb_loss

        # K-branch soft-competition flow loss (mu / sigma detached inside).
        flow_loss, fstats = fm.compute_branch_loss(
            z1n,
            h_v,
            mu_v,
            sig_v,
            z_lookback=z_lb_v,
            gate_lb=gate_lb_v,
            ctx_clauses=ctx_clauses_v,
            ctx_weights=ctx_weights_v,
        )
        if profile_cb is not None:
            profile_cb("flow")

        # Stop head BCE with pos_weight balancing.
        stop_loss = stop_logits.sum() * 0.0
        stop_acc = None
        stop_recall = None
        if stop_labels is not None:
            lbl = stop_labels.reshape(B, Tp).float()
            if stop_valid_mask is not None:
                vf = (stop_valid_mask.reshape(B, Tp).bool() & valid).float()
            else:
                vf = valid.float()
            n_pos = (lbl * vf).sum()
            n_neg = ((1.0 - lbl) * vf).sum()
            pos_weight = (n_neg / n_pos.clamp(min=1.0)).clamp(min=1.0, max=10.0).detach()
            bce = F.binary_cross_entropy_with_logits(stop_logits, lbl, pos_weight=pos_weight, reduction="none")
            n_valid = vf.sum()
            stop_loss = (bce * vf).sum() / n_valid.clamp(min=1.0)
            with torch.no_grad():
                if n_valid.item() > 0:
                    correct = ((stop_logits > 0).float() == (lbl > 0.5).float()).float()
                    stop_acc = float(((correct * vf).sum() / n_valid).item())
                    pos_vf = (lbl > 0.5).float() * vf
                    if pos_vf.sum().item() > 0:
                        stop_recall = float(((correct * pos_vf).sum() / pos_vf.sum()).item())

        total = flow_loss + self.prior_weight * prior_nll + self.stop_weight * stop_loss
        if empty_valid:
            total = total * 0.0

        if not return_metrics:
            return total

        with torch.no_grad():
            L = self.sem_bands
            sig_d = sig_v.detach()
            sigma_band_mean = sig_d.mean(dim=0)  # (nb,)
            gates = ctx["gates"].detach()
            gate_lb_t = ctx.get("gate_lb")
            sv = ctx.get("stream_valid")
            metrics: Dict[str, Any] = {
                "total_loss": float(total.detach().item()),
                "flow_loss": float(flow_loss.detach().item()),
                "vel_loss": float(fstats["vel_loss"].item()),
                "cos_loss": float(fstats["cos_loss"].item()),
                "slot_loss": float(fstats["slot_loss"].item()),
                "slot_cos_loss": float(fstats["slot_cos_loss"].item()),
                "slot_diff_loss": float(fstats["slot_diff_loss"].item()),
                "slot_valid_loss": float(fstats["slot_valid_loss"].item()),
                "thought_cos": float(fstats["thought_cos"].item()),
                "end_mse": float(fstats["end_mse"].item()),
                "win_loss": float(fstats["win_loss"].item()),
                "other_loss": float(fstats["other_loss"].item()),
                "prior_nll": float(prior_nll.detach().item()),
                "gate_lb_loss": float(gate_lb_loss.detach().item()),
                "gate_lb_mean": float(gate_lb_t.detach()[valid].mean().item()) if (gate_lb_t is not None and not empty_valid) else 0.0,
                "stop_loss": float(stop_loss.detach().item()),
                "stop_acc": stop_acc,
                "stop_recall": stop_recall,
                "sigma_sem": float(sig_d[:, :L].mean().item()),
                "sigma_syn": float(sig_d[:, L:].mean().item()) if L < nb else 0.0,
                "sigma_band": [float(x) for x in sigma_band_mean.tolist()],
                "winner_hist": [int(x) for x in fstats["winner_hist"].tolist()],
                "branch_diversity": float(fstats["branch_diversity"].item()),
                "alpha_aux": float(torch.tanh(self.alpha_aux.detach()).item()),
                "n_streams": int(gates.shape[-1]),
                "gate_mean": float(gates[valid].mean().item()) if (gates.shape[-1] > 0 and not empty_valid) else 0.0,
                "n_valid_rows": int(mask_flat.sum().item()),
                "empty_valid": empty_valid,
            }
            if sv is not None and gates.shape[-1] > 0 and not empty_valid:
                gv = gates[valid]  # (n, N)
                svv = sv.unsqueeze(1).expand(B, Tp, -1)[valid]
                metrics["gate_valid_mean"] = float((gv.sum() / svv.float().sum().clamp(min=1.0)).item())
            if sample_group is not None and not empty_valid:
                grp = sample_group.reshape(B, 1).expand(B, Tp).reshape(-1)[sel]
                names = self.GROUP_NAMES if group_names is None else group_names
                for gid, gname in names.items():
                    m = grp == int(gid)
                    if bool(m.any()):
                        metrics[f"sigma_sem_{gname}"] = float(sig_d[m][:, :L].mean().item())
                        metrics[f"sigma_syn_{gname}"] = float(sig_d[m][:, L:].mean().item()) if L < nb else 0.0
                        metrics[f"n_rows_{gname}"] = int(m.sum().item())
            metrics["n_stop_rows"] = int(vf.sum().item()) if stop_labels is not None else 0
            # Tensor views for callers that prefer device tensors (old #027 keys)
            metrics["loss_vel"] = fstats["vel_loss"]
            metrics["loss_cos"] = fstats["cos_loss"]
            metrics["cos_sim"] = fstats["thought_cos"]
        return total, metrics

    # ------------------------------------------------------------------ sampling / rollout
    @torch.no_grad()
    def sample_next_clauses(
        self,
        z_in: torch.Tensor,
        aux: Optional[torch.Tensor] = None,
        aux_valid: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        ode_steps: int = 10,
        *,
        active_streams: Optional[Sequence[str]] = None,
        z_knowledge_seq: Optional[torch.Tensor] = None,
        z_goal_seq: Optional[torch.Tensor] = None,
        renorm: bool = True,
    ) -> torch.Tensor:
        """
        Teacher-forced one-step-ahead sampling for scheduled self-feeding:
        `z_in: (B, T', D)` -> `z_hat: (B, T', D)` in z_clause (DENORMALIZED) space, where
        z_hat[:, t] is a sample of the clause following z_in[:, :t+1].
        """
        ctx = self.forward_context(
            z_in, aux, aux_valid, active_streams=active_streams,
            z_knowledge_seq=z_knowledge_seq, z_goal_seq=z_goal_seq,
        )
        mu, sigma = self.predict_prior(ctx)
        eff_temp = min(float(temperature), 0.35)
        z_hat = self.flow_matcher.sample(
            ctx["h"], mu=mu, sigma_band=sigma, steps=ode_steps, temperature=eff_temp, renorm=renorm,
        )
        return z_hat.to(dtype=z_in.dtype)

    def step(
        self,
        z_t: torch.Tensor,
        states: Optional[Dict[str, Any]] = None,
        h_aux: Optional[torch.Tensor] = None,
        stream_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Advances the main stream by ONE clause `z_t: (B, D)`, applies `context_lookback.step`
        over the accumulated clause history, and fuses pre-encoded companions.
        Returns `(ctx, new_states)` where ctx has h/h0/F_aux/gates/gate_max/gate_sum of shape (B, ·).
        """
        states = {} if states is None else states
        new_states: Dict[str, Any] = dict(states)
        z_t_in = self._normalize_clause_input(z_t)
        h0_raw_t, new_states["main"] = self.main_stream.step(z_t_in, states.get("main"))
        prev_h = states.get("history_h")
        prev_z = states.get("history_z")
        if prev_h is not None and prev_z is not None:
            hist_h = torch.cat([prev_h.to(dtype=h0_raw_t.dtype), h0_raw_t.unsqueeze(1)], dim=1)
            hist_z = torch.cat([prev_z.to(dtype=z_t_in.dtype), z_t_in.unsqueeze(1)], dim=1)
        else:
            hist_h = h0_raw_t.unsqueeze(1)
            hist_z = z_t_in.unsqueeze(1)
        new_states["history_h"] = hist_h.detach()
        new_states["history_z"] = hist_z.detach()
        if hist_z.shape[1] >= 2:
            j0_slots = (self.flow_matcher.reconstruct_spatial_slots(hist_z[:, 0]).norm(dim=-1) > 0.20).sum(dim=-1)
            j0_valid = hist_z[:, 0].abs().sum(dim=-1) > 1e-6
            j1_valid = hist_z[:, 1].abs().sum(dim=-1) > 1e-6
            j0_header_mask = (j0_slots <= 7) & j0_valid & j1_valid
        else:
            j0_header_mask = None
        h0_t, z_lb_t, gate_lb_t = self.context_lookback.step(
            h0_raw_t, z_t_in, hist_h, hist_z, return_delta=True, j0_header_mask=j0_header_mask
        )
        ctx = self._fuse(h0_t.unsqueeze(1), h_aux, stream_valid)
        ctx = {k: v.squeeze(1) for k, v in ctx.items()}
        ctx["z_lookback"] = z_lb_t
        ctx["gate_lb"] = gate_lb_t
        last_cc = getattr(self.context_lookback, "_last_ctx_clauses", None)
        last_cw = getattr(self.context_lookback, "_last_ctx_weights", None)
        ctx["ctx_clauses"] = last_cc[:, -1, :, :] if last_cc is not None else None
        ctx["ctx_weights"] = last_cw[:, -1, :] if last_cw is not None else None
        return ctx, new_states

    @torch.no_grad()
    def step_clause(
        self,
        z_curr: torch.Tensor,
        states: Optional[Dict[str, Any]] = None,
        aux: Optional[torch.Tensor] = None,
        aux_valid: Optional[torch.Tensor] = None,
        ode_steps: int = 10,
        temperature: float = 0.8,
        return_all: bool = False,
        *,
        active_streams: Optional[Sequence[str]] = None,
        refresh_aux: bool = False,
        z_knowledge: Optional[torch.Tensor] = None,
        z_goal: Optional[torch.Tensor] = None,
        knowledge_valid: Optional[torch.Tensor] = None,
        goal_valid: Optional[torch.Tensor] = None,
        renorm: bool = True,
    ):
        """
        Autoregressive single-clause step for rollout:
          1. Encodes companions ONCE (cached as states['h_aux'] / states['stream_valid']; pass
             `refresh_aux=True` to re-encode, e.g. when per-clause selection changes the pool).
          2. Steps `main_stream` with `z_curr: (B, D)` (or last clause of (B, T, D)).
          3. Prior head -> (mu, sigma_band); samples `z_thought` from the adaptive cone.
        Returns `(z_thought, new_states)` or, with `return_all`, `(z_thought, h_t, new_states, info)` where
        info = {stop_prob (B,), sigma_band (B, nb), gates (B, N), mu (B, D) normalized}.
        """
        states = {} if states is None else states
        new_states: Dict[str, Any] = dict(states)
        z_t = z_curr[:, -1, :] if z_curr.dim() == 3 else z_curr

        if refresh_aux or not states.get("aux_encoded", False):
            aux_r, aux_valid_r = self._resolve_aux(
                aux, aux_valid, active_streams, z_knowledge, z_goal, knowledge_valid, goal_valid
            )
            h_aux, stream_valid = self.encode_aux(aux_r, aux_valid_r)
            new_states["h_aux"] = h_aux
            new_states["stream_valid"] = stream_valid
            new_states["aux_encoded"] = True
        h_aux = new_states.get("h_aux")
        stream_valid = new_states.get("stream_valid")

        ctx, new_states = self.step(z_t, new_states, h_aux, stream_valid)
        mu, sigma = self.predict_prior(ctx)
        h_t = ctx["h"]
        z_thought = self.flow_matcher.sample(
            h_t, mu=mu, sigma_band=sigma, steps=ode_steps, temperature=temperature, renorm=renorm,
        ).to(dtype=z_t.dtype)
        if return_all:
            info = {
                "stop_prob": torch.sigmoid(self.stop_head(h_t).float()).squeeze(-1),
                "sigma_band": sigma,
                "gates": ctx["gates"],
                "mu": mu,
            }
            return z_thought, h_t, new_states, info
        return z_thought, new_states

    @torch.no_grad()
    def sample_thought(
        self,
        discourse_seq: torch.Tensor,
        aux: Optional[torch.Tensor] = None,
        aux_valid: Optional[torch.Tensor] = None,
        steps: int = 10,
        temperature: float = 1.0,
        *,
        active_streams: Optional[Sequence[str]] = None,
        knowledge_seq: Optional[torch.Tensor] = None,
        goal_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples the clause following `discourse_seq: (B, T, D)`. Returns `(z_thought (B, D), h_T (B, D))`."""
        ctx = self.forward_context(
            discourse_seq, aux, aux_valid, active_streams=active_streams,
            z_knowledge_seq=knowledge_seq, z_goal_seq=goal_seq,
        )
        last = {
            k: v[:, -1]
            for k, v in ctx.items()
            if k in ("h", "h0", "F_aux", "gates", "gate_max", "gate_sum", "z_lookback", "gate_lb", "ctx_clauses", "ctx_weights") and v is not None
        }
        mu, sigma = self.predict_prior(last)
        z_thought = self.flow_matcher.sample(last["h"], mu=mu, sigma_band=sigma, steps=steps, temperature=temperature)
        return z_thought, last["h"]


class KnowledgeSelector(nn.Module):
    """
    Independent Post-Thought Knowledge & Symbol Selector (#025 / #027):
      - Scores candidate knowledge/context clauses in either 2D HiPPO clause-vector space
        `z_candidates: (B, N_cand, D)` or 3D token-embedding space `cand_tokens: (B, N_cand, L_c, D)`
        against query vector `z_query: (B, D)`.
      - Masks out padded empty candidate slots via `cand_valid_mask: (B, N_cand)` during both
        supervised BCE + ranking margin training (`compute_selection_loss`) and inference.
      - Provides `select_active_clause_attractors` to preserve contiguous character spans within
        selected clauses for `LeftBrainCrystallizer`'s 3-gram `OrthogonalAttractor` and diagonal span relay.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        d_clause: Optional[int] = None,
        d_proj: int = 256,
    ):
        super().__init__()
        if d_clause is not None:
            hidden_dim = int(d_clause)
        self.hidden_dim = hidden_dim
        self.d_clause = hidden_dim
        self.d_proj = int(d_proj)

        self.probe_norm = nn.LayerNorm(hidden_dim)
        self.probe_proj = nn.Linear(hidden_dim, hidden_dim)
        self.goal_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.cand_norm = nn.LayerNorm(hidden_dim)
        self.cand_proj = nn.Linear(hidden_dim, hidden_dim)

        # Low-rank semantic bilinear projection head for 2D clause vector matching (#027)
        self.query_sub_proj = nn.Sequential(
            nn.Linear(hidden_dim, self.d_proj),
            nn.SiLU(),
            nn.Linear(self.d_proj, self.d_proj),
        )
        self.cand_sub_proj = nn.Sequential(
            nn.Linear(hidden_dim, self.d_proj),
            nn.SiLU(),
            nn.Linear(self.d_proj, self.d_proj),
        )

        self.logit_scale = nn.Parameter(torch.tensor(8.0))
        self.bias = nn.Parameter(torch.tensor(-0.5))

        with torch.no_grad():
            self.probe_proj.weight.copy_(torch.eye(hidden_dim))
            self.probe_proj.bias.zero_()
            self.cand_proj.weight.copy_(torch.eye(hidden_dim))
            self.cand_proj.bias.zero_()
            nn.init.normal_(self.goal_proj.weight, mean=0.0, std=0.02)

        # [Feature Addition - Task #033: learnable sink ("do not copy") head]
        # 1. Purpose: the #031 left-brain diagnosis showed irrelevant attractors HURT (B < C), so the selector
        #    must be able to say "nothing in the pool is worth copying" -> the left brain then runs with
        #    `k_attractors=None` (condition C). Labels: `sink_label` = no candidate reaches the IDF rare-span
        #    overlap threshold (dataset #032).
        # 2. Method: sink_logit = MLP([probe feature (d_sink), pooled candidate-similarity statistics]).
        #    The statistics are [max combined sim, mean combined sim, max full-space sim, max low-rank sim,
        #    log1p(#valid candidates) / log(33)] over the VALID candidates (all 0 when there is none), so the
        #    head sees both "what am I looking for" (probe) and "how good is the best match" (pool).
        # 3. Caveats: new parameters `sink_probe.*` / `sink_head.*`. Old #029 `knowledge_selector.pt` files
        #    lack them; a load_state_dict pre-hook fills the missing sink keys with the current (fresh) values
        #    so strict loading of #029 checkpoints keeps working (sink is then untrained!).
        self.d_sink = 128
        self.num_sink_stats = 5
        self.sink_probe = nn.Sequential(nn.Linear(hidden_dim, self.d_sink), nn.SiLU())
        self.sink_head = nn.Sequential(
            nn.Linear(self.d_sink + self.num_sink_stats, self.d_sink),
            nn.SiLU(),
            nn.Linear(self.d_sink, 1),
        )
        with torch.no_grad():
            nn.init.normal_(self.sink_head[-1].weight, mean=0.0, std=0.02)
            self.sink_head[-1].bias.zero_()
        self.register_buffer("z_mean", torch.zeros(hidden_dim, dtype=torch.float32))
        self.register_buffer("z_std", torch.ones(hidden_dim, dtype=torch.float32))
        self.register_buffer("calib_steps", torch.tensor(0, dtype=torch.long))
        self._register_load_state_dict_pre_hook(self._fill_missing_sink_keys)

    _SINK_PREFIXES = ("sink_probe.", "sink_head.")
    _CALIB_BUFFERS = ("z_mean", "z_std", "calib_steps")

    def _fill_missing_sink_keys(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """#033 backward compatibility: injects fresh sink parameters & calib buffers into older selector state dicts."""
        own = None
        filled = 0
        for name, tensor in self.named_parameters():
            if not name.startswith(self._SINK_PREFIXES):
                continue
            key = prefix + name
            if key not in state_dict:
                if own is None:
                    own = {k: v for k, v in self.state_dict().items()}
                state_dict[key] = own[name].detach().clone()
                filled += 1
        for name in self._CALIB_BUFFERS:
            key = prefix + name
            if key not in state_dict:
                if own is None:
                    own = {k: v for k, v in self.state_dict().items()}
                state_dict[key] = own[name].detach().clone()
            elif name in ("z_mean", "z_std") and state_dict[key].shape != (self.hidden_dim,):
                state_dict[key] = state_dict[key].reshape(-1).expand(self.hidden_dim).clone()
        if filled:
            print(f"ℹ️ KnowledgeSelector: checkpoint has no sink head ({filled} tensors) — using fresh (untrained) sink weights.")

    @torch.no_grad()
    def calibrate_z(
        self,
        z_sample: torch.Tensor,
        max_calib_steps: int = 50,
        momentum: float = 0.1,
    ) -> bool:
        step = int(self.calib_steps.item())
        if step >= max_calib_steps:
            return False
        z_float = z_sample.detach().float().reshape(-1, self.hidden_dim)
        valid = z_float.abs().sum(dim=-1) > 1e-6
        z_valid = z_float[valid]
        if z_valid.numel() == 0:
            return False
        batch_mean = z_valid.mean(dim=0)
        if z_valid.shape[0] >= 2:
            batch_std = z_valid.std(dim=0, unbiased=False).clamp(min=1e-3)
        else:
            s = z_valid.std().clamp(min=1e-3) if z_valid.numel() > 1 else torch.tensor(1.0, device=z_valid.device)
            batch_std = s.expand(self.hidden_dim).clone()
        if step == 0:
            self.z_mean.copy_(batch_mean)
            self.z_std.copy_(batch_std)
        else:
            self.z_mean.lerp_(batch_mean, momentum)
            self.z_std.lerp_(batch_std, momentum)
        self.calib_steps.add_(1)
        return True

    def normalize_z(self, z: torch.Tensor) -> torch.Tensor:
        nonzero = z.abs().sum(dim=-1, keepdim=True) > 1e-6
        mean = self.z_mean.to(device=z.device, dtype=z.dtype)
        std = self.z_std.to(device=z.device, dtype=z.dtype).clamp(min=1e-3)
        return torch.where(nonzero, (z - mean) / std, z)

    # ------------------------------------------------------------------ #033 shared projections
    def _project_probe(self, z_probe: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """z_probe (..., D) -> (q_normed (..., D), q_full unit (..., D), q_sub unit (..., d_proj))."""
        q_normed = self.probe_norm(self.normalize_z(z_probe))
        q_full = F.normalize(self.probe_proj(q_normed), dim=-1, eps=1e-6)
        q_sub = F.normalize(self.query_sub_proj(q_normed), dim=-1, eps=1e-6)
        return q_normed, q_full, q_sub

    def _project_cand(self, z_cand: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """z_cand (..., D) -> (k_full unit (..., D), k_sub unit (..., d_proj))."""
        c_normed = self.cand_norm(self.normalize_z(z_cand))
        k_full = F.normalize(self.cand_proj(c_normed), dim=-1, eps=1e-6)
        k_sub = F.normalize(self.cand_sub_proj(c_normed), dim=-1, eps=1e-6)
        return k_full, k_sub

    def _scores_from_sims(self, sim_full: torch.Tensor, sim_sub: torch.Tensor) -> torch.Tensor:
        """Same scoring rule as `score_clause_candidates` (kept bit-compatible with #027/#029)."""
        scale = self.logit_scale.clamp(min=1.0, max=30.0)
        return (0.5 * sim_full + 0.5 * sim_sub) * scale + self.bias

    def _sink_from_sims(
        self,
        q_normed: torch.Tensor,
        sim_full: torch.Tensor,
        sim_sub: torch.Tensor,
        cand_valid: torch.Tensor,
    ) -> torch.Tensor:
        """
        q_normed: (N, D); sim_full / sim_sub: (N, C); cand_valid: (N, C) bool -> sink_logit (N,) float32.
        Pooled statistics are taken over valid candidates only (0 if a row has none).
        """
        v = cand_valid.bool()
        vf = v.float()
        n_valid = vf.sum(dim=-1)  # (N,)
        has = n_valid > 0
        sf = sim_full.float()
        ss = sim_sub.float()
        sc = 0.5 * sf + 0.5 * ss
        neg = torch.full_like(sc, -2.0)
        max_c = torch.where(v, sc, neg).amax(dim=-1)
        max_f = torch.where(v, sf, neg).amax(dim=-1)
        max_s = torch.where(v, ss, neg).amax(dim=-1)
        mean_c = (sc * vf).sum(dim=-1) / n_valid.clamp(min=1.0)
        zero = torch.zeros_like(max_c)
        stats = torch.stack(
            [
                torch.where(has, max_c, zero),
                torch.where(has, mean_c, zero),
                torch.where(has, max_f, zero),
                torch.where(has, max_s, zero),
                torch.log1p(n_valid) / math.log(33.0),
            ],
            dim=-1,
        )  # (N, 5)
        feat = self.sink_probe(q_normed)
        x = torch.cat([feat, stats.to(dtype=feat.dtype)], dim=-1)
        return self.sink_head(x).squeeze(-1).float()

    def score_with_sink(
        self,
        z_probe: torch.Tensor,
        z_cand: torch.Tensor,
        cand_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        #033: per-row candidate scores + sink ("copy nothing") logit.
          z_probe: (N, D); z_cand: (N, C, D); cand_valid: (N, C) bool (default all valid).
        Returns scores (N, C) float32 raw logits (invalid -> -1e4) and sink_logit (N,) float32.
        """
        N, C, _ = z_cand.shape
        if cand_valid is None:
            cand_valid = torch.ones((N, C), device=z_cand.device, dtype=torch.bool)
        cand_valid = cand_valid.bool()
        q_normed, q_full, q_sub = self._project_probe(z_probe)
        k_full, k_sub = self._project_cand(z_cand)
        sim_full = torch.einsum("nd,ncd->nc", q_full, k_full)
        sim_sub = torch.einsum("nd,ncd->nc", q_sub, k_sub)
        scores = self._scores_from_sims(sim_full, sim_sub).float().masked_fill(~cand_valid, -1e4)
        sink_logit = self._sink_from_sims(q_normed, sim_full, sim_sub, cand_valid)
        return scores, sink_logit

    def compute_context_selection_loss(
        self,
        z_main: torch.Tensor,
        z_probe: torch.Tensor,
        ctx_rel: torch.Tensor,
        ctx_rel_valid: torch.Tensor,
        sink_label: torch.Tensor,
        main_valid: torch.Tensor,
        *,
        top_m: int = 3,
        pos_threshold: float = 0.5,
        max_pos_weight: float = 5.0,
        sink_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        #033 self-supervised context-clause selection (staged plan §3).
          z_main: (B, T, D) clause vectors = candidate pool; z_probe: (B, T, D) per-row probes
          (GT target clause + noise, or right-brain thoughts); ctx_rel: (B, T, T) soft relevance of
          candidate row j for target row t (only j < t used); ctx_rel_valid / main_valid: (B, T) bool;
          sink_label: (B, T) 1 = no candidate worth copying.
        Candidates of row t = valid rows j < t (causal). Rows are supervised iff ctx_rel_valid & main_valid
        & (>= 1 candidate). Loss = soft-label BCE on candidates (pos_weight = n_neg / n_pos clamped to
        [1, max_pos_weight]) + sink_weight * BCE(sink) (pos_weight balanced, clamped to [0.2, 5]).
        Pair projections are computed once per sample (O(B T D^2)) and compared with an einsum
        (O(B T^2 D)) instead of expanding the pool per probe.
        Metrics (rows with sink_label == 0 for ranking): precision@1, recall@M (positives = rel >=
        pos_threshold, or the row-max candidate if none passes), sink accuracy, predicted / true sink rate.
        """
        B, T, D = z_main.shape
        device = z_main.device
        mv = main_valid.bool()
        if self.training and bool(mv.any()) and int(self.calib_steps.item()) < 50:
            self.calibrate_z(z_main[mv])
        q_normed, q_full, q_sub = self._project_probe(z_probe)
        k_full, k_sub = self._project_cand(z_main)
        sim_full = torch.einsum("btd,bjd->btj", q_full, k_full)
        sim_sub = torch.einsum("btd,bjd->btj", q_sub, k_sub)
        scores = self._scores_from_sims(sim_full, sim_sub).float()  # (B, T, T)

        causal = torch.ones((T, T), device=device, dtype=torch.bool).tril(diagonal=-1)  # j < t
        cand_valid = causal.unsqueeze(0) & mv.unsqueeze(1)  # (B, T, T)
        row_valid = ctx_rel_valid.bool() & mv & cand_valid.any(dim=-1)  # (B, T)
        pair_valid = row_valid.unsqueeze(-1) & cand_valid
        pf = pair_valid.float()

        targets = ctx_rel.float().clamp(0.0, 1.0) * pf
        s_clean = torch.where(pair_valid, scores, torch.zeros_like(scores))
        n_pos = targets.sum()
        n_neg = ((1.0 - targets) * pf).sum()
        pos_weight = (n_neg / n_pos.clamp(min=1e-3)).clamp(min=1.0, max=float(max_pos_weight)).detach()
        bce = F.binary_cross_entropy_with_logits(s_clean, targets, pos_weight=pos_weight, reduction="none")
        cand_loss = (bce * pf).sum() / pf.sum().clamp(min=1.0)

        sink_logit = self._sink_from_sims(
            q_normed.reshape(B * T, -1), sim_full.reshape(B * T, T), sim_sub.reshape(B * T, T),
            cand_valid.reshape(B * T, T),
        ).reshape(B, T)
        sl = sink_label.float().clamp(0.0, 1.0)
        rf = row_valid.float()
        s_pos = (sl * rf).sum()
        s_neg = ((1.0 - sl) * rf).sum()
        sink_pw = (s_neg / s_pos.clamp(min=1.0)).clamp(min=0.2, max=5.0).detach()
        sink_bce = F.binary_cross_entropy_with_logits(sink_logit, sl, pos_weight=sink_pw, reduction="none")
        sink_loss = (sink_bce * rf).sum() / rf.sum().clamp(min=1.0)

        loss = cand_loss + float(sink_weight) * sink_loss
        n_rows = int(row_valid.sum().item())
        if n_rows == 0:
            loss = (scores.sum() + sink_logit.sum()) * 0.0

        with torch.no_grad():
            metrics: Dict[str, Any] = {
                "sel_loss": float(loss.detach().item()),
                "cand_loss": float(cand_loss.detach().item()),
                "sink_loss": float(sink_loss.detach().item()),
                "n_rows": n_rows,
                "n_pairs": int(pair_valid.sum().item()),
                "pos_weight": float(pos_weight.item()),
                "precision_at_1": None,
                "recall_at_m": None,
                "sink_acc": None,
                "sink_rate_pred": None,
                "sink_rate_true": None,
                "n_rank_rows": 0,
            }
            if n_rows > 0:
                sink_pred = (torch.sigmoid(sink_logit) > 0.5)
                sink_true = sl > 0.5
                metrics["sink_acc"] = float(((sink_pred == sink_true) & row_valid).float().sum().item() / n_rows)
                metrics["sink_rate_pred"] = float((sink_pred & row_valid).float().sum().item() / n_rows)
                metrics["sink_rate_true"] = float((sink_true & row_valid).float().sum().item() / n_rows)
                rank_rows = row_valid & ~sink_true  # (B, T)
                n_rank = int(rank_rows.sum().item())
                metrics["n_rank_rows"] = n_rank
                if n_rank > 0:
                    rel = ctx_rel.float().masked_fill(~cand_valid, -1.0)[rank_rows]  # (n, T)
                    sc = scores.masked_fill(~cand_valid, -1e4)[rank_rows]  # (n, T)
                    cv = cand_valid[rank_rows]
                    row_max = rel.amax(dim=-1, keepdim=True)
                    pos = cv & ((rel >= float(pos_threshold)) | ((rel >= row_max) & (row_max > 0)))
                    M = max(1, min(int(top_m), T))
                    top_idx = sc.topk(M, dim=-1).indices  # (n, M)
                    top_valid = torch.gather(cv, 1, top_idx)
                    hit = torch.gather(pos, 1, top_idx) & top_valid  # (n, M)
                    p1 = hit[:, 0].float()
                    n_pos_row = pos.float().sum(dim=-1)
                    rec = hit.float().sum(dim=-1) / n_pos_row.clamp(min=1.0)
                    has_pos = n_pos_row > 0
                    if bool(has_pos.any()):
                        metrics["precision_at_1"] = float(p1[has_pos].mean().item())
                        metrics["recall_at_m"] = float(rec[has_pos].mean().item())
        return loss, metrics

    @torch.no_grad()
    def select_context_attractors(
        self,
        z_probe: torch.Tensor,
        z_cand: torch.Tensor,
        w_cand: torch.Tensor,
        cand_valid: Optional[torch.Tensor] = None,
        top_m: int = 3,
        min_score: float = 0.25,
        sink_threshold: float = 0.5,
        pad_token_id: int = 0,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        #033 sink-aware attractor selection for ONE target clause per sample.
          z_probe: (B, D); z_cand: (B, C, D); w_cand: (B, C, L) token ids; cand_valid: (B, C).
        A sample gets NO attractors (use_attr=False, all-pad row, k_mask all False) when
        sigmoid(sink_logit) > sink_threshold, or no valid candidate has sigmoid(score) >= min_score.
        Otherwise the top-M candidates with sigmoid(score) >= min_score are packed in chronological
        (index) order into `w_attractors (B, M*L)` + `k_mask = w != pad`.
        Returns (None, None, info) if no sample uses attractors. info = {use_attr (B,) bool,
        scores (B, C) sigmoid gates (invalid=0), sink_prob (B,), top_idx (B, M), keep (B, M)}.
        Callers MUST route rows with use_attr=False through `k_attractors=None`
        (see `crystallize_with_optional_attractors`).
        """
        B, C, L = w_cand.shape
        device = w_cand.device
        if cand_valid is None:
            cand_valid = torch.ones((B, C), device=device, dtype=torch.bool)
        cand_valid = cand_valid.bool()
        if C == 0:
            info = {
                "use_attr": torch.zeros((B,), device=device, dtype=torch.bool),
                "scores": torch.zeros((B, 0), device=device),
                "sink_prob": torch.ones((B,), device=device),
                "top_idx": torch.zeros((B, 0), device=device, dtype=torch.long),
                "keep": torch.zeros((B, 0), device=device, dtype=torch.bool),
            }
            return None, None, info
        scores, sink_logit = self.score_with_sink(z_probe, z_cand, cand_valid)
        gates = torch.sigmoid(scores).masked_fill(~cand_valid, 0.0)
        sink_prob = torch.sigmoid(sink_logit)
        M = max(1, min(int(top_m), C))
        top_g, top_idx = gates.topk(M, dim=-1)
        keep = (top_g >= float(min_score)) & torch.gather(cand_valid, 1, top_idx)
        use_attr = keep.any(dim=-1) & (sink_prob <= float(sink_threshold))
        keep = keep & use_attr.unsqueeze(-1)
        # chronological order among the selected clauses (unselected slots sorted to the end)
        order_key = torch.where(keep, top_idx, top_idx + C)
        order = order_key.argsort(dim=-1)
        top_idx = torch.gather(top_idx, 1, order)
        keep = torch.gather(keep, 1, order)
        info = {"use_attr": use_attr, "scores": gates, "sink_prob": sink_prob, "top_idx": top_idx, "keep": keep}
        if not bool(use_attr.any()):
            return None, None, info
        sel_w = torch.gather(w_cand, 1, top_idx.unsqueeze(-1).expand(B, M, L))
        sel_w = torch.where(keep.unsqueeze(-1), sel_w, torch.full_like(sel_w, int(pad_token_id)))
        w_attr = sel_w.reshape(B, M * L)
        k_mask = w_attr != int(pad_token_id)
        return w_attr, k_mask, info

    def score_clause_candidates(
        self,
        z_query: torch.Tensor,
        z_candidates: torch.Tensor,
        cand_valid_mask: Optional[torch.Tensor] = None,
        z_goal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scores 2D clause candidates `z_candidates: (B, N_cand, D)` against `z_query: (B, D)`.
        Returns:
          scores: (B, N_cand) raw logits (with invalid candidates masked to -1e4)
          gates:  (B, N_cand) sigmoid relevance probabilities in [0, 1] (with invalid candidates zeroed)
        """
        if z_query.dim() == 3:
            z_query = z_query.mean(dim=1)

        probe_in = z_query + (0.25 * self.goal_proj(z_goal) if z_goal is not None else 0.0)
        q_normed = self.probe_norm(probe_in)
        c_normed = self.cand_norm(z_candidates)

        q_full = F.normalize(self.probe_proj(q_normed), dim=-1, eps=1e-6)  # (B, D)
        k_full = F.normalize(self.cand_proj(c_normed), dim=-1, eps=1e-6)   # (B, N_cand, D)
        sim_full = torch.einsum("bd,bnd->bn", q_full, k_full)

        q_sub = F.normalize(self.query_sub_proj(q_normed), dim=-1, eps=1e-6)  # (B, d_proj)
        k_sub = F.normalize(self.cand_sub_proj(c_normed), dim=-1, eps=1e-6)   # (B, N_cand, d_proj)
        sim_sub = torch.einsum("bd,bnd->bn", q_sub, k_sub)

        scale = self.logit_scale.clamp(min=1.0, max=30.0)
        scores = (0.5 * sim_full + 0.5 * sim_sub) * scale + self.bias
        gates = torch.sigmoid(scores)

        if cand_valid_mask is not None:
            valid_b = cand_valid_mask.bool()
            scores = scores.masked_fill(~valid_b, -1e4)
            gates = gates.masked_fill(~valid_b, 0.0)

        return scores, gates

    def score_candidates(
        self,
        z_thought: torch.Tensor,
        cand_tokens: torch.Tensor,
        cand_mask: Optional[torch.Tensor] = None,
        z_goal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scores candidate clauses given `z_thought: (B, D)` and either:
          - 3D clause vectors `cand_tokens: (B, N_cand, D)`, or
          - 4D token embeddings `cand_tokens: (B, N_cand, L_c, D)`.
        Returns:
          scores: (B, N_cand) raw logits
          gates: (B, N_cand) sigmoid relevance probabilities in (0, 1)
        """
        if cand_tokens.dim() == 3:
            cand_valid_mask = cand_mask.bool() if (cand_mask is not None and cand_mask.dim() == 2) else None
            return self.score_clause_candidates(
                z_query=z_thought,
                z_candidates=cand_tokens,
                cand_valid_mask=cand_valid_mask,
                z_goal=z_goal,
            )

        B, N_cand, L_c, D = cand_tokens.shape
        probe_in = z_thought + (0.25 * self.goal_proj(z_goal) if z_goal is not None else 0.0)
        q = F.normalize(self.probe_proj(self.probe_norm(probe_in)), dim=-1, eps=1e-6)  # (B, D)

        k_tok = F.normalize(self.cand_proj(self.cand_norm(cand_tokens)), dim=-1, eps=1e-6)  # (B, N_cand, L_c, D)

        # 1. Max token-level cosine resonance: (B, N_cand, L_c) -> (B, N_cand)
        tok_sim = torch.einsum("bd,bnld->bnl", q, k_tok)
        if cand_mask is not None:
            tok_sim = tok_sim.masked_fill(~cand_mask, -1e4)
        max_tok_sim = tok_sim.max(dim=-1).values  # (B, N_cand)

        # 2. Clause-level mean pooled cosine resonance: (B, N_cand)
        if cand_mask is not None:
            mask_f = cand_mask.unsqueeze(-1).to(dtype=k_tok.dtype)
            denom = mask_f.sum(dim=2).clamp(min=1.0)
            cand_mean = F.normalize((k_tok * mask_f).sum(dim=2) / denom, dim=-1, eps=1e-6)
        else:
            cand_mean = F.normalize(k_tok.mean(dim=2), dim=-1, eps=1e-6)
        mean_sim = torch.einsum("bd,bnd->bn", q, cand_mean)

        scale = self.logit_scale.clamp(min=1.0, max=30.0)
        scores = (0.65 * max_tok_sim + 0.35 * mean_sim) * scale + self.bias
        gates = torch.sigmoid(scores)
        return scores, gates

    def compute_selection_loss(
        self,
        z_query: torch.Tensor,
        z_candidates: Optional[torch.Tensor] = None,
        pos_mask: Optional[torch.Tensor] = None,
        cand_valid_mask: Optional[torch.Tensor] = None,
        valid_cand_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes masked weighted binary cross-entropy + contrastive ranking margin loss pushing
        positive knowledge clauses K+ (`pos_mask == 1`) above negative distractors K- (`pos_mask == 0`).
        Supports both:
          - `#027` signature: `compute_selection_loss(z_query, z_candidates, pos_mask, cand_valid_mask=...)`
          - `#025` signature: `compute_selection_loss(scores, pos_mask, valid_cand_mask=...)`
        """
        mask = cand_valid_mask if cand_valid_mask is not None else valid_cand_mask

        # Detect legacy #025 call `compute_selection_loss(scores, pos_mask, valid_cand_mask)`
        if pos_mask is None and z_candidates is not None and z_query.dim() == 2 and z_candidates.dim() == 2:
            scores = z_query
            targets = z_candidates.float()
        else:
            if z_candidates is None or pos_mask is None:
                raise ValueError(
                    "KnowledgeSelector.compute_selection_loss requires (z_query, z_candidates, pos_mask)."
                )
            if z_candidates.dim() == 3:
                scores, _ = self.score_clause_candidates(
                    z_query=z_query, z_candidates=z_candidates, cand_valid_mask=mask
                )
            else:
                scores, _ = self.score_candidates(z_thought=z_query, cand_tokens=z_candidates)
            targets = pos_mask.float()

        s_f = scores.float()
        if mask is not None:
            valid_b = mask.bool()
            valid_f = valid_b.float()
            n_pos = (targets * valid_f).sum().clamp(min=1.0)
            n_neg = ((1.0 - targets) * valid_f).sum().clamp(min=1.0)
            pos_weight = (n_neg / n_pos).clamp(min=1.0, max=5.0)
            # Replace masked-out logits with 0 before BCE so -1e4 masked slots never create large numbers
            s_clean = torch.where(valid_b, s_f, torch.zeros_like(s_f))
            bce = F.binary_cross_entropy_with_logits(
                s_clean, targets, pos_weight=pos_weight, reduction="none"
            )
            bce_loss = (bce * valid_f).sum() / valid_f.sum().clamp(min=1.0)
        else:
            valid_b = None
            bce = F.binary_cross_entropy_with_logits(s_f, targets, reduction="none")
            bce_loss = bce.mean()

        # Contrastive margin between hardest positive and hardest valid negative per sample
        pos_bool = (targets > 0.5) if valid_b is None else ((targets > 0.5) & valid_b)
        neg_bool = (~pos_bool) if valid_b is None else ((targets <= 0.5) & valid_b)
        has_both = pos_bool.any(dim=-1) & neg_bool.any(dim=-1)
        if has_both.any():
            pos_min = s_f.masked_fill(~pos_bool, 1e4).min(dim=-1).values
            neg_max = s_f.masked_fill(~neg_bool, -1e4).max(dim=-1).values
            margin = F.relu(1.0 - (pos_min - neg_max))
            margin_loss = margin[has_both].mean()
            return bce_loss + 0.5 * margin_loss

        return bce_loss

    # [Feature Addition - Task #029: Per-Clause Selection Loss]
    # 1. Purpose: v2 moves KnowledgeSelector "after the Right Brain, per clause": every target clause step t
    #    uses its own probe (true target clause + small noise during training, z_thought at inference) to
    #    pick that clause's left-brain attractors. Labels are soft bigram-overlap pseudo-labels (T, C).
    # 2. Method: flattens probes to (B*T, D), reuses `score_clause_candidates` against the per-sample
    #    candidate pool expanded to (B*T, C, D), and applies masked soft-label BCE with pos_weight.
    # 3. Caveats: cost scales with B*T*C*D^2 (candidate projections recomputed per probe); labels are
    #    pseudo-labels with low recall for paraphrased answers. The sample-level `compute_selection_loss`
    #    (goal probe) is kept as an auxiliary loss.
    def compute_clause_selection_loss(
        self,
        z_probe: torch.Tensor,
        z_candidates: torch.Tensor,
        rel_labels: torch.Tensor,
        cand_valid_mask: Optional[torch.Tensor] = None,
        row_valid: Optional[torch.Tensor] = None,
        max_pos_weight: float = 5.0,
    ) -> torch.Tensor:
        """
        Per-clause soft-label selection loss.
          z_probe: (B, T, D) per-clause probes; z_candidates: (B, C, D);
          rel_labels: (B, T, C) soft relevance in [0, 1]; cand_valid_mask: (B, C); row_valid: (B, T).
        Returns a scalar masked BCE (zero with intact graph if no valid (row, cand) pair).
        """
        B, T, D = z_probe.shape
        C = z_candidates.shape[1]
        device = z_probe.device
        if cand_valid_mask is None:
            cand_valid_mask = torch.ones((B, C), device=device, dtype=torch.bool)
        if row_valid is None:
            row_valid = torch.ones((B, T), device=device, dtype=torch.bool)
        if C == 0 or T == 0:
            return z_probe.sum() * 0.0

        q_flat = z_probe.reshape(B * T, D)
        c_flat = z_candidates.unsqueeze(1).expand(B, T, C, z_candidates.shape[-1]).reshape(B * T, C, -1)
        cv_flat = cand_valid_mask.bool().unsqueeze(1).expand(B, T, C).reshape(B * T, C)
        scores, _ = self.score_clause_candidates(z_query=q_flat, z_candidates=c_flat, cand_valid_mask=cv_flat)

        pair_valid = (row_valid.bool().reshape(B * T, 1) & cv_flat)  # (B*T, C)
        vf = pair_valid.float()
        targets = rel_labels.reshape(B * T, C).float().clamp(0.0, 1.0) * vf
        s_f = scores.float()
        s_clean = torch.where(pair_valid, s_f, torch.zeros_like(s_f))
        n_pos = (targets * vf).sum()
        n_neg = ((1.0 - targets) * vf).sum()
        pos_weight = (n_neg / n_pos.clamp(min=1e-3)).clamp(min=1.0, max=float(max_pos_weight)).detach()
        bce = F.binary_cross_entropy_with_logits(s_clean, targets, pos_weight=pos_weight, reduction="none")
        return (bce * vf).sum() / vf.sum().clamp(min=1.0)

    @torch.no_grad()
    def select_active_clause_attractors(
        self,
        z_query: torch.Tensor,
        z_candidates: torch.Tensor,
        w_candidates: torch.Tensor,
        w_goal: Optional[torch.Tensor] = None,
        cand_valid_mask: Optional[torch.Tensor] = None,
        top_m_clauses: int = 3,
        min_score: float = 0.20,
        pad_token_id: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Span-Preserving Clause Attractor Selector (#027):
          1. Scores `z_candidates: (B, N_cand, D)` against `z_query: (B, D)`, masking out invalid
             candidates where `cand_valid_mask == False`.
          2. Selects the top `M = min(top_m_clauses, N_cand)` candidate clauses per batch item.
          3. Zeroes out (pads with `pad_token_id`) clauses whose score `< min_score`
             (unless all are `< min_score`, in which case the single top-1 clause is kept).
          4. Concatenates `[w_goal (if provided), selected_w_candidates]` along the sequence dimension
             into `w_attractors: (B, L_total * L_clause)` along with `k_mask = (w_attractors != pad_token_id)`
             and `top_scores: (B, M)`.
        """
        B, N_cand, L_clause = w_candidates.shape
        M = min(max(1, int(top_m_clauses)), N_cand)

        _, gates = self.score_clause_candidates(
            z_query=z_query,
            z_candidates=z_candidates,
            cand_valid_mask=cand_valid_mask,
        )

        top_scores, top_idx = torch.topk(gates, k=M, dim=-1)  # (B, M)
        gather_idx = top_idx.unsqueeze(-1).expand(B, M, L_clause)
        selected_w_cand = torch.gather(w_candidates, dim=1, index=gather_idx)  # (B, M, L_clause)

        keep_clause = top_scores >= float(min_score)  # (B, M)
        keep_clause[:, 0] = True  # Always keep at least top-1 clause if all < min_score
        if cand_valid_mask is not None:
            top_valid = torch.gather(cand_valid_mask.bool(), dim=1, index=top_idx)
            keep_clause = keep_clause & top_valid
            has_any_valid = cand_valid_mask.bool().any(dim=-1)
            keep_clause[:, 0] = keep_clause[:, 0] | has_any_valid

        pad_fill = torch.full_like(selected_w_cand, int(pad_token_id))
        selected_w_cand = torch.where(keep_clause.unsqueeze(-1), selected_w_cand, pad_fill)
        w_cand_flat = selected_w_cand.reshape(B, M * L_clause)

        if w_goal is not None:
            w_goal_flat = w_goal.reshape(B, -1) if w_goal.dim() == 3 else w_goal
            w_attractors = torch.cat([w_goal_flat, w_cand_flat], dim=1)
        else:
            w_attractors = w_cand_flat

        k_mask = (w_attractors != int(pad_token_id))
        return w_attractors, k_mask, top_scores

    @torch.no_grad()
    def select_active_knowledge_vectors(
        self,
        z_query: torch.Tensor,
        z_candidates: torch.Tensor,
        cand_valid_mask: Optional[torch.Tensor] = None,
        top_m_clauses: int = 3,
        min_score: float = 0.20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selects the top `M = min(top_m_clauses, N_cand)` 2D candidate clause vectors `z_knowledge_active: (B, M, D)`
        for the Right Brain's Knowledge Stream Mamba, zeroing out clauses with score `< min_score`
        (while always preserving at least the top-1 clause).
        """
        B, N_cand, D = z_candidates.shape
        M = min(max(1, int(top_m_clauses)), N_cand)

        _, gates = self.score_clause_candidates(
            z_query=z_query,
            z_candidates=z_candidates,
            cand_valid_mask=cand_valid_mask,
        )
        top_scores, top_idx = torch.topk(gates, k=M, dim=-1)  # (B, M)
        gather_idx = top_idx.unsqueeze(-1).expand(B, M, D)
        selected_z = torch.gather(z_candidates, dim=1, index=gather_idx)  # (B, M, D)

        keep_clause = top_scores >= float(min_score)
        keep_clause[:, 0] = True
        if cand_valid_mask is not None:
            top_valid = torch.gather(cand_valid_mask.bool(), dim=1, index=top_idx)
            keep_clause = keep_clause & top_valid

        selected_z = selected_z * keep_clause.unsqueeze(-1).to(dtype=selected_z.dtype)
        return selected_z, top_scores

    def forward(
        self,
        z_thought: torch.Tensor,
        cand_tokens: torch.Tensor,
        cand_mask: Optional[torch.Tensor] = None,
        z_goal: Optional[torch.Tensor] = None,
        top_k: int = 2,
        threshold: float = 0.25,
        soft_gate_values: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Selects the top-`k` relevant candidate clauses and packs their raw `TinyCharEncoderCUDA`
        character embeddings into `A_active: (B, top_k * L_c, D)` and `A_mask: (B, top_k * L_c)`
        for direct consumption by `LeftBrainCrystallizer`.
        """
        if cand_tokens.dim() == 3:
            B_flat, L_total, D_flat = cand_tokens.shape
            chunk_len = min(32, L_total)
            n_chunks = max(1, L_total // chunk_len)
            trimmed_len = n_chunks * chunk_len
            cand_tokens = cand_tokens[:, :trimmed_len, :].reshape(B_flat, n_chunks, chunk_len, D_flat)
            if cand_mask is not None:
                cand_mask = cand_mask[:, :trimmed_len].reshape(B_flat, n_chunks, chunk_len)

        B, N_cand, L_c, D = cand_tokens.shape
        if cand_mask is None:
            cand_mask = torch.ones((B, N_cand, L_c), device=cand_tokens.device, dtype=torch.bool)

        scores, gates = self.score_candidates(z_thought, cand_tokens, cand_mask=cand_mask, z_goal=z_goal)
        k_keep = min(max(1, top_k), N_cand)

        # Preserve original chronological order among the selected top-k clauses
        _, top_indices = torch.topk(scores, k=k_keep, dim=-1)
        top_indices_sorted, _ = torch.sort(top_indices, dim=-1)  # (B, k_keep)

        idx_tok = top_indices_sorted.unsqueeze(-1).unsqueeze(-1).expand(B, k_keep, L_c, D)
        idx_msk = top_indices_sorted.unsqueeze(-1).expand(B, k_keep, L_c)
        selected_tokens = torch.gather(cand_tokens, dim=1, index=idx_tok)  # (B, k_keep, L_c, D)
        selected_mask = torch.gather(cand_mask, dim=1, index=idx_msk)  # (B, k_keep, L_c)
        selected_gates = torch.gather(gates, dim=1, index=top_indices_sorted)  # (B, k_keep)

        # Suppress distracted clauses below threshold when at least the best clause is valid
        above_thresh = selected_gates >= threshold
        keep_clause = above_thresh.clone()
        keep_clause[:, 0] = True  # Always keep at least 1 clause so mask is never all-False
        selected_mask = selected_mask & keep_clause.unsqueeze(-1)

        if soft_gate_values:
            selected_tokens = selected_tokens * selected_gates.unsqueeze(-1).unsqueeze(-1).to(dtype=selected_tokens.dtype)

        A_active = selected_tokens.reshape(B, k_keep * L_c, D)
        A_mask = selected_mask.reshape(B, k_keep * L_c)
        return A_active, A_mask, scores, gates


# [Feature Addition - Task #033: mixed "with / without attractors" left-brain decoding]
# 1. Problem: with the sink-aware selector some samples of a batch get attractors and others must run
#    with `k_attractors=None` (#031: irrelevant attractors hurt). Passing an all-False `k_mask` row to
#    `OrthogonalAttractor` is NOT safe in general: every key is masked to -1e4 and multiplied by the
#    logit scale (up to 35) -> -3.5e5 overflows to -inf under fp16 autocast -> softmax over all -inf = NaN
#    (NaN * snap-gate 0 is still NaN). In fp32 / bf16 it happens to yield force == 0, but that depends on
#    the gate saturating, so we never rely on it.
# 2. Method: split the batch. Rows with (use_attr & k_mask.any()) run with their attractors, all other
#    rows run with `k_attractors=None`; BOTH groups start from the SAME per-row initial noise `x0`
#    (generated once for the full batch, or caller-supplied), and outputs are scattered back.
#    The LeftBrainCrystallizer is row-independent (convs / per-sample attention), so a split run equals
#    running each group on its own.
# 3. Caveats: two forward passes per ODE step when the batch is mixed (cost ~ unchanged overall because
#    each pass only sees its own rows). Returns DENORMALIZED x (`unet.denormalize`), like
#    `crystallize_clause_euler_ode` in the #029 trainer. Duck-typed: works with any module exposing
#    `forward(x, t, h_potential, k_attractors, k_mask)`, `denormalize`, and `hidden_dim`.
@torch.no_grad()
def crystallize_with_optional_attractors(
    unet: nn.Module,
    z: torch.Tensor,
    x_attr_norm: Optional[torch.Tensor] = None,
    k_mask: Optional[torch.Tensor] = None,
    use_attr: Optional[torch.Tensor] = None,
    steps: int = 10,
    seq_len: int = 32,
    x0: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Euler-integrates the frozen left brain for clause vectors `z: (B, D)`.
      x_attr_norm: (B, La, D) normalized attractor embeddings or None; k_mask: (B, La) bool or None;
      use_attr: (B,) bool (default: all rows that have >= 1 valid attractor token);
      x0: (B, seq_len, D) initial noise (default randn, optionally from `generator`).
    Returns x_gen (B, seq_len, D) in the encoder (denormalized) space.
    """
    B, D = z.shape
    device = z.device
    dtype = z.dtype
    if x0 is None:
        x0 = torch.randn((B, seq_len, D), device=device, dtype=torch.float32, generator=generator)
    x0 = x0.to(device=device, dtype=dtype)

    if x_attr_norm is None:
        rows_attr = torch.zeros((B,), device=device, dtype=torch.bool)
    else:
        has_tok = k_mask.bool().any(dim=-1) if k_mask is not None else torch.ones((B,), device=device, dtype=torch.bool)
        rows_attr = has_tok if use_attr is None else (use_attr.bool() & has_tok)

    def _integrate(idx: torch.Tensor, with_attr: bool) -> torch.Tensor:
        x = x0[idx].clone()
        zc = z[idx]
        ka = x_attr_norm[idx].to(dtype=dtype) if with_attr else None
        km = (k_mask[idx].bool() if k_mask is not None else None) if with_attr else None
        n = int(idx.numel())
        dt = 1.0 / float(max(int(steps), 1))
        for i in range(max(int(steps), 1)):
            t_tensor = torch.full((n,), i * dt, device=device, dtype=torch.float32)
            v = unet(x, t_tensor, h_potential=zc, k_attractors=ka, k_mask=km)
            x = x + v * dt
        return x

    out = torch.empty_like(x0)
    idx_attr = torch.nonzero(rows_attr, as_tuple=False).squeeze(-1)
    idx_none = torch.nonzero(~rows_attr, as_tuple=False).squeeze(-1)
    if idx_attr.numel() > 0:
        out[idx_attr] = _integrate(idx_attr, True).to(dtype=out.dtype)
    if idx_none.numel() > 0:
        out[idx_none] = _integrate(idx_none, False).to(dtype=out.dtype)
    return unet.denormalize(out)


if __name__ == "__main__":
    torch.manual_seed(42)
    rb = MultiStreamRightBrain(hidden_dim=1024, num_mamba_layers=2, num_flow_layers=2)
    fm = rb.flow_matcher
    z_test = torch.randn(4, 1024)
    fm.calibrate_z(z_test)
    z_norm = fm.normalize(z_test)
    x_slot = fm.reconstruct_spatial_slots(z_norm)
    z_rec = fm.from_spatial_slots(x_slot)
    v_slot = fm.velocity_to_spatial_slots(z_norm)
    v_rec = fm.velocity_from_spatial_slots(v_slot)
    err_x = (z_rec - z_norm).abs().max().item()
    err_v = (v_rec - z_norm).abs().max().item()
    assert err_x < 1e-5 and err_v < 1e-5, f"Isomorphism error too large: {err_x}, {err_v}"
    z_seq = torch.randn(2, 5, 1024)
    loss, metrics = rb.compute_loss(z_seq, return_metrics=True)
    loss.backward()
    rb.eval()
    z_hat, h_last = rb.sample_thought(z_seq[:, :-1])
    print(
        f"[Self-Test PASS] flow_version={int(fm.flow_version.item())} "
        f"iso_err=({err_x:.2e}, {err_v:.2e}) loss={metrics['total_loss']:.4f} "
        f"z_hat={tuple(z_hat.shape)}"
    )

