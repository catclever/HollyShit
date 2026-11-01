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


# [Bug Fix & Manifold Calibration Documentation - Task #027]
# 1. Problem Fixed:
#    `HiPPOOracleCUDA` clause vectors `z_clause` can have a coordinate scale (`Std ~ 0.289` or `1.0`)
#    that differs from the standard unit Gaussian prior `z_0 ~ N(0, I)` (`Std = 1.0`) used by `ThoughtFlowMatcher`,
#    and padded empty clause transition steps (`t -> t+1`) previously polluted flow matching loss.
# 2. Resolution Logic:
#    Added persistent buffers (`z_mean`, `z_std`, `calib_steps`) and online EMA calibration `calibrate_z`
#    (first 50 training steps) + `normalize` / `denormalize` to `ThoughtFlowMatcher`.
#    In `compute_loss`, invalid/padded transition steps are filtered via `valid_mask`, targets `z_1` are
#    auto-calibrated and normalized before OT-CFM velocity matching, and in `sample`, the integrated
#    vector is denormalized back to the exact `z_clause` coordinate space expected by `LeftBrainCrystallizer`.
# 3. Caveats for Subsequent Development:
#    `z_mean`, `z_std`, and `calib_steps` are registered buffers saved in `right_brain.pt`. Do not
#    manually re-scale `z_thought` outside `ThoughtFlowMatcher.sample`.
class ThoughtFlowMatcher(nn.Module):
    """
    2D Conditional OT-CFM Velocity Predictor in Clause-Vector Space `(B, D)`.
    Given the Right Brain's Mamba discourse state `cond` (`h_t: (B, D)`), transports Gaussian prior
    `z_0 ~ N(0, I_D)` at tau=0 to the normalized target clause vector `z_1_norm: (B, D)` at tau=1,
    and denormalizes sampled `z_thought` back to the target clause manifold.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        d_clause: Optional[int] = None,
    ):
        super().__init__()
        if d_clause is not None:
            hidden_dim = int(d_clause)
        self.hidden_dim = hidden_dim
        self.d_clause = hidden_dim

        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.in_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.blocks = nn.ModuleList([
            AdaLN2DResMLPBlock(hidden_dim=hidden_dim, mlp_ratio=2)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.normal_(self.final_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.final_proj.bias)

        # In-Model Manifold Auto-Calibration Buffers (#027)
        self.register_buffer("z_mean", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("z_std", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("calib_steps", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def calibrate_z(
        self,
        z_sample: torch.Tensor,
        max_calib_steps: int = 50,
        momentum: float = 0.05,
    ) -> bool:
        """
        Online EMA Auto-Calibration for 2D clause vectors `z_sample` over the first `max_calib_steps` steps.
        """
        step = int(self.calib_steps.item())
        if step >= max_calib_steps:
            return False

        z_float = z_sample.detach().float()
        if z_float.numel() == 0:
            return False

        batch_mean = z_float.mean()
        batch_std = (
            z_float.std().clamp(min=1e-5)
            if z_float.numel() > 1
            else torch.tensor(1.0, device=z_float.device, dtype=torch.float32)
        )

        if step == 0:
            self.z_mean.copy_(batch_mean)
            self.z_std.copy_(batch_std)
        else:
            self.z_mean.lerp_(batch_mean, momentum)
            self.z_std.lerp_(batch_std, momentum)

        self.calib_steps.add_(1)
        return True

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        """Normalizes clause vector `z` from `N(z_mean, z_std^2)` to unit manifold `N(0, 1)`."""
        mean = self.z_mean.to(device=z.device, dtype=z.dtype)
        std = self.z_std.to(device=z.device, dtype=z.dtype).clamp(min=1e-5)
        return (z - mean) / std

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        """Denormalizes unit-manifold vector `z` back to `N(z_mean, z_std^2)` clause coordinate space."""
        mean = self.z_mean.to(device=z.device, dtype=z.dtype)
        std = self.z_std.to(device=z.device, dtype=z.dtype).clamp(min=1e-5)
        return z * std + mean

    def forward(self, z_tau: torch.Tensor, tau: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        """
        Predicts 2D velocity field `v_pred: (B, D)` at flow time `tau: (B,)` conditioned on `h_t: (B, D)`.
        """
        orig_shape = z_tau.shape
        if z_tau.dim() == 3:
            B, T, D = orig_shape
            z_tau = z_tau.reshape(B * T, D)
            h_t = h_t.reshape(B * T, D)
            if tau.numel() == B:
                tau = tau.unsqueeze(1).expand(B, T).reshape(B * T)
            elif tau.numel() == B * T:
                tau = tau.reshape(B * T)

        t_emb = self.t_embedder(tau).to(dtype=z_tau.dtype)
        cond = self.cond_proj(torch.cat([h_t, t_emb], dim=-1))
        h = self.in_proj(torch.cat([z_tau, h_t], dim=-1))
        for block in self.blocks:
            h = block(h, cond)
        v_pred = (h_t - z_tau) + self.final_proj(self.final_norm(h))

        if len(orig_shape) == 3:
            v_pred = v_pred.reshape(orig_shape)
        return v_pred

    def compute_loss(
        self,
        z_1: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
        *,
        h_t: Optional[torch.Tensor] = None,
        z_target: Optional[torch.Tensor] = None,
    ):
        """
        Computes Conditional Optimal Transport Flow Matching (OT-CFM) loss between target clause
        vector(s) `z_1` and Right-Brain Mamba condition `cond` (`h_t`), with optional `valid_mask`
        filtering and automatic manifold calibration.
        """
        if z_1 is None and z_target is not None:
            z_1 = z_target
        if cond is None and h_t is not None:
            cond = h_t
        if z_1 is None or cond is None:
            raise ValueError("ThoughtFlowMatcher.compute_loss requires `z_1` and `cond` (or `h_t` and `z_target`).")

        D = z_1.shape[-1]
        empty_valid = False
        if valid_mask is not None:
            mask_bool = valid_mask.reshape(-1).bool()
            if mask_bool.any():
                z_1 = z_1.reshape(-1, D)[mask_bool]
                cond = cond.reshape(-1, D)[mask_bool]
            else:
                empty_valid = True
                z_1 = z_1.reshape(-1, D)[:1]
                cond = cond.reshape(-1, D)[:1]
        elif z_1.dim() > 2:
            z_1 = z_1.reshape(-1, D)
            cond = cond.reshape(-1, D)

        if self.training and not empty_valid and int(self.calib_steps.item()) < 50:
            self.calibrate_z(z_1)

        z_1_norm = self.normalize(z_1)
        N_valid = z_1_norm.shape[0]
        device = z_1_norm.device
        dtype = z_1_norm.dtype

        z_0 = torch.randn_like(z_1_norm)
        t = torch.rand((N_valid,), device=device, dtype=torch.float32).clamp(min=1e-4, max=1.0 - 1e-4)
        t_col = t.unsqueeze(-1).to(dtype=dtype)

        z_t = (1.0 - t_col) * z_0 + t_col * z_1_norm
        v_target = z_1_norm - z_0

        v_pred = self.forward(z_t, t, cond)
        loss_mse = F.mse_loss(v_pred.float(), v_target.float())

        z_1_hat = z_t.float() + (1.0 - t.unsqueeze(-1)) * v_pred.float()
        cos_sim = F.cosine_similarity(z_1_hat, z_1_norm.float(), dim=-1)
        cos_sim_mean = cos_sim.mean()
        loss_cos = 1.0 - cos_sim_mean

        loss = loss_mse + 0.25 * loss_cos
        if empty_valid:
            loss = loss * 0.0

        if return_metrics:
            return loss, {
                "vel_loss": float(loss_mse.detach().item()),
                "cos_loss": float(loss_cos.detach().item()),
                "thought_cos": float(cos_sim_mean.detach().item()),
                "loss_vel": loss_mse.detach(),
                "loss_cos": loss_cos.detach(),
                "cos_sim": cos_sim_mean.detach(),
            }
        return loss

    @torch.no_grad()
    def sample(
        self,
        cond: Optional[torch.Tensor] = None,
        steps: int = 10,
        temperature: float = 1.0,
        *,
        h_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Integrates the 2D ODE from `tau = 0` to `tau = 1` in `steps` Euler steps in normalized space
        and returns `self.denormalize(z)` so the sampled `z_thought: (B, D)` is in the calibrated
        target clause coordinate space expected by `LeftBrainCrystallizer` and `ClauseMambaStream`.
        """
        if cond is None and h_t is not None:
            cond = h_t
        if cond is None:
            raise ValueError("ThoughtFlowMatcher.sample requires `cond` (or `h_t`).")

        orig_shape = cond.shape
        if cond.dim() == 3:
            cond_2d = cond.reshape(-1, self.d_clause)
        else:
            cond_2d = cond

        B = cond_2d.shape[0]
        z = torch.randn((B, self.d_clause), device=cond_2d.device, dtype=cond_2d.dtype) * float(temperature)
        dt = 1.0 / float(max(steps, 1))
        for i in range(steps):
            tau_val = i / float(max(steps, 1))
            tau = torch.full((B,), tau_val, device=cond_2d.device, dtype=torch.float32)
            v_pred = self.forward(z, tau, cond_2d)
            z = z + v_pred * dt

        z_unit = z / z.float().std(dim=-1, keepdim=True).clamp(min=1e-5).to(dtype=z.dtype)
        z_out = self.denormalize(z_unit)
        if len(orig_shape) == 3:
            z_out = z_out.reshape(orig_shape)
        return z_out


class MultiStreamRightBrain(nn.Module):
    """
    Paired Multi-Stream Mamba + ThoughtFlowMatcher Right Brain (#025 / #027):
      - Builds all 3 Mamba streams upfront:
        1. `discourse_stream`: Main clause progression Mamba (Stream 1)
        2. `knowledge_stream`: Knowledge resonance Mamba (Stream 2)
        3. `goal_stream`: Global question / AST outline goal Mamba (Stream 3)
      - Fuses Stream 2 and Stream 3 into Stream 1 via Gram-Schmidt orthogonal projection
        (`_project_orthogonal`) gated by zero-initialized scalar parameters `alpha_k = 0` and `alpha_g = 0`.
      - Supports curriculum switching between:
        - `active_streams=("discourse",)` (Stage 1: Single-Stream First fast training)
        - `active_streams=("discourse", "knowledge", "goal")` (Stage 2: Full Multi-Stream mode)
        At initialization (`alpha_k == 0`, `alpha_g == 0`), both modes produce bit-for-bit identical `h_t`.
      - Paired with `self.flow_matcher = ThoughtFlowMatcher(d_clause)` to sample `z_thought: (B, D)`.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        d_state: int = 16,
        num_mamba_layers: int = 2,
        num_flow_layers: int = 4,
        *,
        d_clause: Optional[int] = None,
        d_mamba: Optional[int] = None,
        n_layers_discourse: Optional[int] = None,
        n_layers_knowledge: Optional[int] = None,
        n_layers_goal: Optional[int] = None,
        active_streams: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        if d_clause is not None:
            hidden_dim = int(d_clause)
        self.hidden_dim = hidden_dim
        self.d_clause = hidden_dim
        self.d_mamba = int(d_mamba) if d_mamba is not None else hidden_dim

        l_disc = int(n_layers_discourse) if n_layers_discourse is not None else num_mamba_layers
        l_know = int(n_layers_knowledge) if n_layers_knowledge is not None else num_mamba_layers
        l_goal = int(n_layers_goal) if n_layers_goal is not None else num_mamba_layers

        if active_streams is None:
            self.active_streams: Tuple[str, ...] = ("discourse", "knowledge", "goal")
        elif isinstance(active_streams, str):
            self.active_streams = tuple(s.strip() for s in active_streams.split(",") if s.strip())
        else:
            self.active_streams = tuple(active_streams)

        # 3 Parallel Clause-Level Mamba Streams
        self.discourse_stream = ClauseMambaStream(
            hidden_dim=self.d_mamba,
            d_state=d_state,
            num_layers=l_disc,
            input_dim=self.d_clause,
        )
        self.knowledge_stream = ClauseMambaStream(
            hidden_dim=self.d_mamba,
            d_state=d_state,
            num_layers=l_know,
            input_dim=self.d_clause,
        )
        self.goal_stream = ClauseMambaStream(
            hidden_dim=self.d_mamba,
            d_state=d_state,
            num_layers=l_goal,
            input_dim=self.d_clause,
        )

        # Zero-initialized orthogonal fusion gates (guarantees bit-for-bit identity at Step 0)
        self.alpha_k = nn.Parameter(torch.zeros(1))
        self.alpha_g = nn.Parameter(torch.zeros(1))

        # Paired 2D Clause-Vector Conditional Flow Matcher
        self.flow_matcher = ThoughtFlowMatcher(hidden_dim=self.d_clause, num_layers=num_flow_layers)

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

    def _fuse_streams(
        self,
        h_d: torch.Tensor,
        h_k: Optional[torch.Tensor] = None,
        h_g: Optional[torch.Tensor] = None,
        active_streams: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        streams = self.active_streams if active_streams is None else active_streams
        h_fused = h_d
        if "knowledge" in streams and h_k is not None:
            h_k_orth = self._project_orthogonal(h_k, h_d)
            h_fused = h_fused + torch.tanh(self.alpha_k).to(dtype=h_d.dtype) * h_k_orth
        if "goal" in streams and h_g is not None:
            h_g_orth = self._project_orthogonal(h_g, h_d)
            h_fused = h_fused + torch.tanh(self.alpha_g).to(dtype=h_d.dtype) * h_g_orth
        return h_fused

    def step(
        self,
        s_discourse_t: torch.Tensor,
        s_knowledge_t: Optional[torch.Tensor] = None,
        s_goal_t: Optional[torch.Tensor] = None,
        states: Optional[Dict[str, List[torch.Tensor]]] = None,
        active_streams: Optional[Sequence[str]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """
        Advances the Right Brain by 1 punctuation clause step (`1 step of Mamba = 1 punctuation clause`).
        """
        streams = self.active_streams if active_streams is None else active_streams
        if states is None:
            states = {}
        new_states: Dict[str, List[torch.Tensor]] = {}

        h_d, new_states["discourse"] = self.discourse_stream.step(
            s_discourse_t, states.get("discourse")
        )

        h_k = None
        if "knowledge" in streams and s_knowledge_t is not None:
            h_k, new_states["knowledge"] = self.knowledge_stream.step(
                s_knowledge_t, states.get("knowledge")
            )

        h_g = None
        if "goal" in streams and s_goal_t is not None:
            h_g, new_states["goal"] = self.goal_stream.step(
                s_goal_t, states.get("goal")
            )

        h_t = self._fuse_streams(h_d, h_k=h_k, h_g=h_g, active_streams=streams)
        return h_t, new_states

    @torch.no_grad()
    def step_clause(
        self,
        z_curr: torch.Tensor,
        states: Optional[Dict[str, Any]] = None,
        z_knowledge: Optional[torch.Tensor] = None,
        z_goal: Optional[torch.Tensor] = None,
        active_streams: Optional[Sequence[str]] = None,
        ode_steps: int = 10,
        temperature: float = 0.8,
        return_all: bool = False,
    ):
        """
        Autoregressive single-clause Right-Brain step for Stage 3 Dual-Brain Rollout:
          1. Updates `discourse_stream` state with current clause vector `z_curr: (B, D)`.
          2. Encodes/steps `knowledge_stream` and `goal_stream` (caching 3D sequence states across steps).
          3. Fuses active streams via orthogonal projection (`_fuse_streams`).
          4. Samples the next clause's thought vector `z_thought: (B, D)` via `self.flow_matcher.sample(...)`.
        Returns `(z_thought, new_states)` (or `(z_thought, h_t, new_states)` if `return_all=True`).
        """
        streams = self.active_streams if active_streams is None else active_streams
        if states is None:
            states = {}
        new_states: Dict[str, Any] = dict(states)

        if z_curr.dim() == 3:
            z_curr_2d = z_curr[:, -1, :]
        else:
            z_curr_2d = z_curr

        h_d, new_states["discourse"] = self.discourse_stream.step(
            z_curr_2d, states.get("discourse")
        )

        h_k = None
        if "knowledge" in streams:
            if "knowledge_h" in states and states["knowledge_h"] is not None:
                h_k = states["knowledge_h"]
            elif z_knowledge is not None:
                if z_knowledge.dim() == 3:
                    h_k_seq, new_states["knowledge"] = self.knowledge_stream(
                        z_knowledge, states=states.get("knowledge"), return_state=True
                    )
                    h_k = h_k_seq[:, -1, :]
                    new_states["knowledge_h"] = h_k
                else:
                    h_k, new_states["knowledge"] = self.knowledge_stream.step(
                        z_knowledge, states.get("knowledge")
                    )

        h_g = None
        if "goal" in streams:
            if "goal_h" in states and states["goal_h"] is not None:
                h_g = states["goal_h"]
            elif z_goal is not None:
                if z_goal.dim() == 3:
                    h_g_seq, new_states["goal"] = self.goal_stream(
                        z_goal, states=states.get("goal"), return_state=True
                    )
                    h_g = h_g_seq[:, -1, :]
                    new_states["goal_h"] = h_g
                else:
                    h_g, new_states["goal"] = self.goal_stream.step(
                        z_goal, states.get("goal")
                    )

        h_t = self._fuse_streams(h_d, h_k=h_k, h_g=h_g, active_streams=streams)
        z_thought = self.flow_matcher.sample(h_t, steps=ode_steps, temperature=temperature)
        if return_all:
            return z_thought, h_t, new_states
        return z_thought, new_states

    def forward_context(
        self,
        discourse_seq: torch.Tensor,
        knowledge_seq: Optional[torch.Tensor] = None,
        goal_seq: Optional[torch.Tensor] = None,
        active_streams: Optional[Sequence[str]] = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Processes a sequence of clause vectors `(B, T, D)` across the active Mamba streams
        and returns either the latest fused context vector `h_T: (B, D)` (`return_sequence=False`)
        or the full fused sequence `h_seq: (B, T, D)` (`return_sequence=True`).
        """
        streams = self.active_streams if active_streams is None else active_streams
        h_d_seq = self.discourse_stream(discourse_seq)

        h_k_seq = None
        if "knowledge" in streams and knowledge_seq is not None:
            if knowledge_seq.dim() == 2:
                knowledge_seq = knowledge_seq.unsqueeze(1)
            h_k_raw = self.knowledge_stream(knowledge_seq)
            if h_k_raw.shape[1] == h_d_seq.shape[1]:
                h_k_seq = h_k_raw
            else:
                h_k_seq = h_k_raw[:, -1:, :].expand(-1, h_d_seq.shape[1], -1)

        h_g_seq = None
        if "goal" in streams and goal_seq is not None:
            if goal_seq.dim() == 2:
                goal_seq = goal_seq.unsqueeze(1)
            h_g_raw = self.goal_stream(goal_seq)
            if h_g_raw.shape[1] == h_d_seq.shape[1]:
                h_g_seq = h_g_raw
            else:
                h_g_seq = h_g_raw[:, -1:, :].expand(-1, h_d_seq.shape[1], -1)

        h_fused_seq = self._fuse_streams(
            h_d_seq, h_k=h_k_seq, h_g=h_g_seq, active_streams=streams
        )
        if return_sequence:
            return h_fused_seq
        return h_fused_seq[:, -1, :]

    def compute_loss(
        self,
        z_discourse_seq: Optional[torch.Tensor] = None,
        z_knowledge_seq: Optional[torch.Tensor] = None,
        z_goal_seq: Optional[torch.Tensor] = None,
        target_valid_mask: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
        *,
        discourse_seq: Optional[torch.Tensor] = None,
        z_target: Optional[torch.Tensor] = None,
        knowledge_seq: Optional[torch.Tensor] = None,
        goal_seq: Optional[torch.Tensor] = None,
        active_streams: Optional[Sequence[str]] = None,
    ):
        """
        Computes Right-Brain Mamba + ThoughtFlowMatcher loss:
          - Multi-clause trajectory mode (#027):
            Given `z_discourse_seq: (B, T, D)` (`T >= 2`), predicts `z_1 = z_discourse_seq[:, 1:, :]`
            from Mamba context `cond = forward_context(z_discourse_seq[:, :-1, :], ...)` masked by
            `target_valid_mask: (B, T - 1)`.
          - Single-target legacy mode (#025):
            Given `discourse_seq: (B, T, D)` and 2D `z_target: (B, D)`, predicts `z_target` from `h_T`.
        """
        if z_discourse_seq is None and discourse_seq is not None:
            z_discourse_seq = discourse_seq
        if z_knowledge_seq is None and knowledge_seq is not None:
            z_knowledge_seq = knowledge_seq
        if z_goal_seq is None and goal_seq is not None:
            z_goal_seq = goal_seq

        if z_discourse_seq is None:
            raise ValueError("MultiStreamRightBrain.compute_loss requires `z_discourse_seq`.")

        streams = self.active_streams if active_streams is None else active_streams

        # Backward compatibility check for #025 positional call `compute_loss(discourse_seq, z_target)`
        if z_target is None and z_knowledge_seq is not None and z_knowledge_seq.dim() == 2 and z_discourse_seq.dim() == 3:
            z_target = z_knowledge_seq
            z_knowledge_seq = None

        if z_target is not None:
            h_t = self.forward_context(
                discourse_seq=z_discourse_seq,
                knowledge_seq=z_knowledge_seq,
                goal_seq=z_goal_seq,
                active_streams=streams,
                return_sequence=False,
            )
            return self.flow_matcher.compute_loss(
                z_1=z_target,
                cond=h_t,
                valid_mask=target_valid_mask,
                return_metrics=return_metrics,
            )

        if z_discourse_seq.shape[1] < 2:
            raise ValueError(
                f"z_discourse_seq must have at least 2 clauses (lead + >=1 target), got shape {tuple(z_discourse_seq.shape)}"
            )

        z_in = z_discourse_seq[:, :-1, :]  # (B, T - 1, D)
        z_1 = z_discourse_seq[:, 1:, :]    # (B, T - 1, D)

        cond_seq = self.forward_context(
            discourse_seq=z_in,
            knowledge_seq=z_knowledge_seq,
            goal_seq=z_goal_seq,
            active_streams=streams,
            return_sequence=True,
        )
        return self.flow_matcher.compute_loss(
            z_1=z_1,
            cond=cond_seq,
            valid_mask=target_valid_mask,
            return_metrics=return_metrics,
        )

    @torch.no_grad()
    def sample_thought(
        self,
        discourse_seq: torch.Tensor,
        knowledge_seq: Optional[torch.Tensor] = None,
        goal_seq: Optional[torch.Tensor] = None,
        active_streams: Optional[Sequence[str]] = None,
        steps: int = 10,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the active Mamba streams over `discourse_seq` to obtain `h_t: (B, D)` and integrates
        `ThoughtFlowMatcher` to sample the next clause's thought vector `z_thought: (B, D)`.
        Returns `(z_thought, h_t)`.
        """
        streams = self.active_streams if active_streams is None else active_streams
        h_t = self.forward_context(
            discourse_seq=discourse_seq,
            knowledge_seq=knowledge_seq,
            goal_seq=goal_seq,
            active_streams=streams,
            return_sequence=False,
        )
        z_thought = self.flow_matcher.sample(h_t, steps=steps, temperature=temperature)
        return z_thought, h_t


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
