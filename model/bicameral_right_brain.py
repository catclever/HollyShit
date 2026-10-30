import math
from typing import Optional, Tuple, Sequence, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.flow_unet_cuda import TimestepEmbedder


class ClauseMambaBlock(nn.Module):
    """
    Pure-PyTorch GPU-friendly Selective State-Space Model (Mamba) block operating on
    punctuation-clause sequences `(B, T, D)` where each discrete time step `t` is
    1 punctuation clause (`10-28` chars packed via HiPPOOracleCUDA into R^D).

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
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        d_state: int = 16,
        num_layers: int = 2,
        expand: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            ClauseMambaBlock(hidden_dim=hidden_dim, d_state=d_state, expand=expand)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

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
        h = s_t
        for layer, st in zip(self.layers, states):
            h, next_st = layer.step(h, st)
            new_states.append(next_st)
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
        h = s_seq
        for layer, st in zip(self.layers, states):
            h, next_st = layer(h, state=st, return_state=True)
            new_states.append(next_st)
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


class ThoughtFlowMatcher(nn.Module):
    """
    2D Conditional OT-CFM Velocity Predictor in Clause-Vector Space `(B, D)`.
    Given the Right Brain's Mamba discourse state `h_t: (B, D)`, transports Gaussian prior
    `z_0 ~ N(0, I_D)` at tau=0 to the next clause's HiPPO sentence vector `z_target: (B, D)` at tau=1,
    explicitly resolving the "one context -> multiple valid next clauses" averaged-vector blur problem.
    """

    def __init__(self, hidden_dim: int = 1024, num_layers: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
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

    def forward(self, z_tau: torch.Tensor, tau: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        """
        Predicts 2D velocity field `v_pred: (B, D)` at flow time `tau: (B,)` conditioned on `h_t: (B, D)`.
        """
        t_emb = self.t_embedder(tau).to(dtype=z_tau.dtype)
        cond = self.cond_proj(torch.cat([h_t, t_emb], dim=-1))
        h = self.in_proj(torch.cat([z_tau, h_t], dim=-1))
        for block in self.blocks:
            h = block(h, cond)
        v_pred = (h_t - z_tau) + self.final_proj(self.final_norm(h))
        return v_pred

    def compute_loss(
        self,
        h_t: torch.Tensor,
        z_target: torch.Tensor,
        return_metrics: bool = False,
    ):
        """
        Computes Conditional Optimal Transport Flow Matching (OT-CFM) loss between Mamba state `h_t: (B, D)`
        and target clause vector `z_target: (B, D)`.
        """
        B, _ = z_target.shape
        device = z_target.device
        dtype = z_target.dtype

        z_0 = torch.randn_like(z_target)
        tau = torch.rand((B,), device=device, dtype=torch.float32).clamp(min=1e-4, max=1.0 - 1e-4)
        tau_col = tau.unsqueeze(-1).to(dtype=dtype)

        z_tau = (1.0 - tau_col) * z_0 + tau_col * z_target
        v_target = z_target - z_0

        v_pred = self.forward(z_tau, tau, h_t)
        loss_vel = F.mse_loss(v_pred.float(), v_target.float())

        z_1_est = z_tau.float() + (1.0 - tau.unsqueeze(-1)) * v_pred.float()
        cos_sim = F.cosine_similarity(z_1_est, z_target.float(), dim=-1).mean()
        loss_cos = 1.0 - cos_sim

        total_loss = loss_vel + 0.25 * loss_cos
        if return_metrics:
            return total_loss, {
                "loss_vel": loss_vel.detach(),
                "loss_cos": loss_cos.detach(),
                "cos_sim": cos_sim.detach(),
            }
        return total_loss

    @torch.no_grad()
    def sample(
        self,
        h_t: torch.Tensor,
        steps: int = 8,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Integrates the 2D ODE from `tau = 0` to `tau = 1` in `steps` Euler steps to sample a sharp,
        unit-variance clause thought vector `z_thought: (B, D)` conditioned on `h_t: (B, D)`.
        """
        B, D = h_t.shape
        z = torch.randn((B, D), device=h_t.device, dtype=h_t.dtype) * float(temperature)
        dt = 1.0 / float(max(steps, 1))
        for i in range(steps):
            tau_val = i / float(max(steps, 1))
            tau = torch.full((B,), tau_val, device=h_t.device, dtype=torch.float32)
            v_pred = self.forward(z, tau, h_t)
            z = z + v_pred * dt

        z_std = z.float().std(dim=-1, keepdim=True).clamp(min=1e-5).to(dtype=z.dtype)
        return z / z_std


class MultiStreamRightBrain(nn.Module):
    """
    Paired Multi-Stream Mamba + ThoughtFlowMatcher Right Brain (#025):
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
      - Paired with `self.flow_matcher = ThoughtFlowMatcher(hidden_dim)` to sample `z_thought: (B, D)`.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        d_state: int = 16,
        num_mamba_layers: int = 2,
        num_flow_layers: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 3 Parallel Clause-Level Mamba Streams
        self.discourse_stream = ClauseMambaStream(
            hidden_dim=hidden_dim, d_state=d_state, num_layers=num_mamba_layers
        )
        self.knowledge_stream = ClauseMambaStream(
            hidden_dim=hidden_dim, d_state=d_state, num_layers=num_mamba_layers
        )
        self.goal_stream = ClauseMambaStream(
            hidden_dim=hidden_dim, d_state=d_state, num_layers=num_mamba_layers
        )

        # Zero-initialized orthogonal fusion gates (guarantees bit-for-bit identity at Step 0)
        self.alpha_k = nn.Parameter(torch.zeros(1))
        self.alpha_g = nn.Parameter(torch.zeros(1))

        # Paired 2D Clause-Vector Conditional Flow Matcher
        self.flow_matcher = ThoughtFlowMatcher(hidden_dim=hidden_dim, num_layers=num_flow_layers)

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
        active_streams: Sequence[str] = ("discourse",),
    ) -> torch.Tensor:
        h_fused = h_d
        if "knowledge" in active_streams and h_k is not None:
            h_k_orth = self._project_orthogonal(h_k, h_d)
            h_fused = h_fused + torch.tanh(self.alpha_k).to(dtype=h_d.dtype) * h_k_orth
        if "goal" in active_streams and h_g is not None:
            h_g_orth = self._project_orthogonal(h_g, h_d)
            h_fused = h_fused + torch.tanh(self.alpha_g).to(dtype=h_d.dtype) * h_g_orth
        return h_fused

    def step(
        self,
        s_discourse_t: torch.Tensor,
        s_knowledge_t: Optional[torch.Tensor] = None,
        s_goal_t: Optional[torch.Tensor] = None,
        states: Optional[Dict[str, List[torch.Tensor]]] = None,
        active_streams: Sequence[str] = ("discourse",),
    ) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """
        Advances the Right Brain by 1 punctuation clause step (`1 step of Mamba = 1 punctuation clause`).
        """
        if states is None:
            states = {}
        new_states: Dict[str, List[torch.Tensor]] = {}

        h_d, new_states["discourse"] = self.discourse_stream.step(
            s_discourse_t, states.get("discourse")
        )

        h_k = None
        if "knowledge" in active_streams and s_knowledge_t is not None:
            h_k, new_states["knowledge"] = self.knowledge_stream.step(
                s_knowledge_t, states.get("knowledge")
            )

        h_g = None
        if "goal" in active_streams and s_goal_t is not None:
            h_g, new_states["goal"] = self.goal_stream.step(
                s_goal_t, states.get("goal")
            )

        h_t = self._fuse_streams(h_d, h_k=h_k, h_g=h_g, active_streams=active_streams)
        return h_t, new_states

    def forward_context(
        self,
        discourse_seq: torch.Tensor,
        knowledge_seq: Optional[torch.Tensor] = None,
        goal_seq: Optional[torch.Tensor] = None,
        active_streams: Sequence[str] = ("discourse",),
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Processes a sequence of clause vectors `(B, T, D)` across the active Mamba streams
        and returns either the latest fused context vector `h_T: (B, D)` (`return_sequence=False`)
        or the full fused sequence `h_seq: (B, T, D)` (`return_sequence=True`).
        """
        h_d_seq = self.discourse_stream(discourse_seq)

        h_k_seq = None
        if "knowledge" in active_streams and knowledge_seq is not None:
            h_k_seq = self.knowledge_stream(knowledge_seq)
            if h_k_seq.shape[1] == 1 and h_d_seq.shape[1] > 1:
                h_k_seq = h_k_seq.expand(-1, h_d_seq.shape[1], -1)

        h_g_seq = None
        if "goal" in active_streams and goal_seq is not None:
            h_g_seq = self.goal_stream(goal_seq)
            if h_g_seq.shape[1] == 1 and h_d_seq.shape[1] > 1:
                h_g_seq = h_g_seq.expand(-1, h_d_seq.shape[1], -1)

        h_fused_seq = self._fuse_streams(
            h_d_seq, h_k=h_k_seq, h_g=h_g_seq, active_streams=active_streams
        )
        if return_sequence:
            return h_fused_seq
        return h_fused_seq[:, -1, :]

    def compute_loss(
        self,
        discourse_seq: torch.Tensor,
        z_target: torch.Tensor,
        knowledge_seq: Optional[torch.Tensor] = None,
        goal_seq: Optional[torch.Tensor] = None,
        active_streams: Sequence[str] = ("discourse",),
        return_metrics: bool = False,
    ):
        """
        Computes Right-Brain Mamba + ThoughtFlowMatcher loss for predicting `z_target: (B, D)`
        from preceding clause sequence `discourse_seq: (B, T, D)`.
        """
        h_t = self.forward_context(
            discourse_seq=discourse_seq,
            knowledge_seq=knowledge_seq,
            goal_seq=goal_seq,
            active_streams=active_streams,
            return_sequence=False,
        )
        return self.flow_matcher.compute_loss(h_t, z_target, return_metrics=return_metrics)

    @torch.no_grad()
    def sample_thought(
        self,
        discourse_seq: torch.Tensor,
        knowledge_seq: Optional[torch.Tensor] = None,
        goal_seq: Optional[torch.Tensor] = None,
        active_streams: Sequence[str] = ("discourse",),
        steps: int = 8,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the active Mamba streams over `discourse_seq` to obtain `h_t: (B, D)` and integrates
        `ThoughtFlowMatcher` to sample the next clause's thought vector `z_thought: (B, D)`.
        Returns `(z_thought, h_t)`.
        """
        h_t = self.forward_context(
            discourse_seq=discourse_seq,
            knowledge_seq=knowledge_seq,
            goal_seq=goal_seq,
            active_streams=active_streams,
            return_sequence=False,
        )
        z_thought = self.flow_matcher.sample(h_t, steps=steps, temperature=temperature)
        return z_thought, h_t


class KnowledgeSelector(nn.Module):
    """
    Independent Post-Thought Knowledge & Symbol Selector (#025):
      - Mounted after the Right Brain (`MultiStreamRightBrain`).
      - Uses the newly generated clause thought vector `z_thought: (B, D)` (and optional `z_goal: (B, D)`)
        as a retrieval probe against candidate knowledge/code/history clauses `cand_tokens: (B, N_cand, L_c, D)`.
      - Combines clause-level semantic resonance and max token-level lexical/symbol resonance to score
        each candidate clause `i in {1..N_cand}` with gate `g_i = sigmoid(score_i) in (0, 1)`.
      - Supports supervised/contrastive training against positive knowledge `K+` vs negative distractors `K-`
        via `compute_selection_loss(scores, pos_mask)`, and extracts the top-k active character embedding pool
        `A_active: (B, L_act, D)` and `A_mask: (B, L_act)` for `LeftBrainCrystallizer`.
    """

    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.probe_norm = nn.LayerNorm(hidden_dim)
        self.probe_proj = nn.Linear(hidden_dim, hidden_dim)
        self.goal_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.cand_norm = nn.LayerNorm(hidden_dim)
        self.cand_proj = nn.Linear(hidden_dim, hidden_dim)

        self.logit_scale = nn.Parameter(torch.tensor(8.0))
        self.bias = nn.Parameter(torch.tensor(-0.5))

        with torch.no_grad():
            self.probe_proj.weight.copy_(torch.eye(hidden_dim))
            self.probe_proj.bias.zero_()
            self.cand_proj.weight.copy_(torch.eye(hidden_dim))
            self.cand_proj.bias.zero_()
            nn.init.normal_(self.goal_proj.weight, mean=0.0, std=0.02)

    def score_candidates(
        self,
        z_thought: torch.Tensor,
        cand_tokens: torch.Tensor,
        cand_mask: Optional[torch.Tensor] = None,
        z_goal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scores candidate clauses given `z_thought: (B, D)` and `cand_tokens: (B, N_cand, L_c, D)`.
        Returns:
          scores: (B, N_cand) raw logits
          gates: (B, N_cand) sigmoid relevance probabilities in (0, 1)
        """
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

    @staticmethod
    def compute_selection_loss(
        scores: torch.Tensor,
        pos_mask: torch.Tensor,
        valid_cand_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes binary cross-entropy + ranking margin loss pushing K+ (`pos_mask == 1`) above
        unrelated distractors K- (`pos_mask == 0`).
        """
        targets = pos_mask.float()
        bce = F.binary_cross_entropy_with_logits(scores.float(), targets, reduction="none")
        if valid_cand_mask is not None:
            valid_f = valid_cand_mask.float()
            bce_loss = (bce * valid_f).sum() / valid_f.sum().clamp(min=1.0)
        else:
            bce_loss = bce.mean()

        # Contrastive margin between hardest positive and hardest negative per sample
        pos_bool = pos_mask.bool()
        neg_bool = ~pos_bool if valid_cand_mask is None else (~pos_bool & valid_cand_mask.bool())
        has_both = pos_bool.any(dim=-1) & neg_bool.any(dim=-1)
        if has_both.any():
            s_f = scores.float()
            pos_min = s_f.masked_fill(~pos_bool, 1e4).min(dim=-1).values
            neg_max = s_f.masked_fill(~neg_bool, -1e4).max(dim=-1).values
            margin = F.relu(1.0 - (pos_min - neg_max))
            margin_loss = margin[has_both].mean()
            return bce_loss + 0.5 * margin_loss

        return bce_loss

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
            # Auto-reshape flat token pool (B, L_total, D) into 1-clause or 32-token chunks
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
