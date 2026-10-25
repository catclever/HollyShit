import math
from typing import Optional, Tuple, Dict, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Macro-level Orthogonal Attractor with <PAD> mask support.
    Reuses the exact ReLU(inner_product) physics of FlowMatching1DUnet.OrthogonalAttractor:
      - Normalizes query trajectory X and knowledge pool K via LayerNorm(D).
      - Computes inner-product scores = (X_norm @ K_norm^T) / sqrt(D).
      - Masks <PAD> knowledge tokens to 0.0 before ReLU.
      - Applies ReLU (NO Softmax!) so irrelevant/orthogonal knowledge yields exact 0.0 force.
      - Superimposes the resulting force onto X via OrthogonalFieldSuperposition.
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
    ) -> torch.Tensor:
        """
        Computes the raw orthogonal pull force (B, L_q, D) without adding it to x.
        Guarantees exact zero force when k is None, k_mask is all False, or inner products <= 0.
        """
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
            k_masked = k

        x_norm = self.norm_x(x)
        k_norm = self.norm_k(k_masked)

        # Zero out norm_k at <PAD> positions in case LayerNorm bias is non-zero
        if k_mask_bool is not None:
            k_norm = k_norm * mask_f

        # Inner product activation: (B, L_q, L_k)
        scores = torch.bmm(x_norm, k_norm.transpose(1, 2)) * self.scale

        if k_mask_bool is not None:
            scores = scores.masked_fill(~k_mask_bool.unsqueeze(1), 0.0)
            has_any = k_mask_bool.any(dim=1).view(B, 1, 1).to(dtype=scores.dtype)
            scores = scores * has_any

        # Snap-to-grid ReLU threshold (NO Softmax -> preserves zero-pollution orthogonality)
        activation = torch.relu(scores)

        # Normalize by sqrt of active knowledge count to stabilize force across variable stream counts
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
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        force = self.compute_force(x, k, k_mask=k_mask)
        out = self.superposition(x, force)
        if return_force:
            return out, force
        return out


class TimeWarpedLegendreExpander(nn.Module):
    """
    Expands K_beats (e.g., 32) macroscopic narrative beats into a continuous 3D potential curve
    H_potential of arbitrary canvas length L_out (e.g., 256, 512, 768) with zero hardcoded L weights.

    Implements Dual-Level Elastic Time-Warping:
      1. Global Effective Horizon (length_head / rho_override):
         Predicts active ratio rho in [0.04, 1.0] -> effective token horizon L_eff = rho * L_out.
      2. Local Beat Pacing (pacing_head):
         Predicts positive beat duration weights delta_s_k -> normalized p_k (sum_k p_k = 1).
         Constructs a differentiable piecewise-linear cumulative time-warping curve s_warped(t)
         over [0, L_eff], evaluates K_beats Legendre polynomials P_k(2*s_warped(t) - 1) via
         Bonnet's recursion + localized Gaussian beat routing, applies a smooth <EOS> horizon
         gate, adds 1D sinusoidal positional embeddings, and refines with a residual 1D conv block.
    """
    def __init__(self, d_model: int = 1024, num_beats: int = 32):
        super().__init__()
        self.d_model = d_model
        self.num_beats = num_beats

        # 1. Global Effective Horizon Predictor: h_summary (B, D) -> rho_pred in [0.04, 1.0]
        self.length_head = nn.Sequential(
            nn.Linear(d_model, max(d_model // 4, 64)),
            nn.SiLU(),
            nn.Linear(max(d_model // 4, 64), 1),
        )

        # 2. Local Beat Pacing Predictor: C_beats (B, K, D) -> delta_s_k > 0
        self.pacing_head = nn.Sequential(
            nn.Linear(d_model, max(d_model // 4, 64)),
            nn.SiLU(),
            nn.Linear(max(d_model // 4, 64), 1),
        )
        # Initialize pacing head to start near uniform beat durations
        nn.init.zeros_(self.pacing_head[-1].weight)
        nn.init.zeros_(self.pacing_head[-1].bias)

        # 3. Learned <PAD>/<EOS> asymptotic tail state beyond L_eff
        self.pad_tail_emb = nn.Parameter(torch.zeros(1, 1, d_model))

        # 4. Lightweight 1D sharpening conv block (local grammatical transition refinement)
        self.sharpen_norm = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        self._pos_cache = {}

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

    def _eval_legendre_basis(self, tau: torch.Tensor, degree: int) -> torch.Tensor:
        """
        Evaluates orthogonal Legendre polynomials P_0(tau) .. P_{degree-1}(tau) on tau in [-1, 1]
        via Bonnet's recursion:
            k P_k(tau) = (2k - 1) * tau * P_{k-1}(tau) - (k - 1) * P_{k-2}(tau)
        Args:
            tau: (B, L_out) in [-1, 1]
            degree: K_beats
        Returns:
            P: (B, L_out, K_beats) scaled by 1 / sqrt(K_beats)
        """
        B, L_out = tau.shape
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
            h_summary: (B, D)
            target_len: int (dynamic output canvas length L_out)
            rho_override: Optional (B,) tensor or float in (0, 1] for manual verbosity or hint conditioning
        Returns:
            h_potential: (B, target_len, D)
            aux_dict: dict with rho_pred, rho_used, l_eff, pacing_weights, beat_centers
        """
        B, K, D = c_beats.shape
        device = c_beats.device
        dtype = c_beats.dtype
        L_out = int(target_len)

        # 1. Global Horizon Prediction (rho_pred in [0.04, 1.0])
        rho_pred = torch.sigmoid(self.length_head(h_summary)).squeeze(-1).clamp(min=0.04, max=1.0)
        if rho_override is not None:
            if isinstance(rho_override, (int, float)):
                rho_used = torch.full((B,), float(rho_override), device=device, dtype=dtype)
            else:
                rho_used = rho_override.to(device=device, dtype=dtype).view(B)
            rho_used = rho_used.clamp(min=0.04, max=1.0)
        else:
            rho_used = rho_pred

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

        # Exact differentiable piecewise-linear cumulative time-warping:
        # Maps each beat interval [low_edges_k, cum_p_k] of width p_k onto [k/K, (k+1)/K]
        local_frac = ((s_grid.unsqueeze(-1) - low_edges.unsqueeze(1)) / p_weights.unsqueeze(1).clamp(min=1e-6)).clamp(0.0, 1.0)
        s_warped = local_frac.mean(dim=-1)  # (B, L_out) in [0, 1]
        tau = 2.0 * s_warped - 1.0          # (B, L_out) in [-1, 1]

        # Evaluate K-order Legendre basis on warped coordinate tau(t)
        p_legendre = self._eval_legendre_basis(tau, K)  # (B, L_out, K)

        # Soft Gaussian beat routing centered at u_k with width sigma_k = p_k
        sigma = p_weights.unsqueeze(1).clamp(min=0.5 / K)  # (B, 1, K)
        dist_sq = (s_grid.unsqueeze(-1) - u_centers.unsqueeze(1)) ** 2
        w_gauss = torch.softmax(-0.5 * dist_sq / (sigma ** 2), dim=-1)  # (B, L_out, K)

        # Combined Time-Warped Basis: (B, L_out, K) @ (B, K, D) -> (B, L_out, D)
        b_warped = (p_legendre + w_gauss).to(dtype=dtype)
        h_raw = torch.bmm(b_warped, c_beats)  # (B, L_out, D)

        # 4. Smooth <EOS> Horizon Tail Gate + 1D Sharpening Conv
        horizon_gate = torch.sigmoid((l_eff - t_grid) / 2.0).unsqueeze(-1).to(dtype=dtype)  # (B, L_out, 1)
        pad_tail = self.pad_tail_emb.to(dtype=dtype)
        h_gated = horizon_gate * h_raw + (1.0 - horizon_gate) * pad_tail

        pos_emb = self._get_pos_emb(L_out, device, dtype)
        h_pos = h_gated + 0.1 * pos_emb

        # Residual 1D Conv refinement
        h_conv_in = self.sharpen_norm(h_pos).transpose(1, 2)  # (B, D, L_out)
        h_conv = self.conv2(self.act(self.conv1(h_conv_in))).transpose(1, 2)
        h_potential = h_pos + h_conv

        aux_dict = {
            "rho_pred": rho_pred,
            "rho_used": rho_used,
            "l_eff": l_eff.squeeze(-1),
            "pacing_weights": p_weights,
            "beat_centers": u_centers,
            "horizon_gate": horizon_gate.squeeze(-1),
        }
        return h_potential, aux_dict


class MultiStreamMambaPlanner(nn.Module):
    """
    Phase 5 Unified Left-Brain Architecture (Multi-Stream Mamba Planner).
    Reads variable-length dialogue context + M parallel knowledge streams, rolls out
    K=32 future narrative beats with layer-wise OrthogonalKnowledgeResonance +
    OrthogonalFieldSuperposition, and expands them into a dynamic 3D Legendre potential
    curve H_potential in R^{B x L_out x D} via TimeWarpedLegendreExpander.
    """
    def __init__(
        self,
        d_model: int = 1024,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 6,
        num_beats: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_beats = num_beats

        try:
            from mamba_ssm import Mamba
            self.layers = nn.ModuleList([
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                ) for _ in range(num_layers)
            ])
            self.use_mamba_ssm = True
        except ImportError:
            print("Warning: mamba_ssm not found. Falling back to Transformer blocks for local testing.")
            nhead = 8 if d_model % 8 == 0 else max(1, math.gcd(d_model, 8))
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * expand,
                    batch_first=True,
                ) for _ in range(num_layers)
            ])
            self.use_mamba_ssm = False

        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        # Multi-stream orthogonal knowledge resonance modules per layer
        self.ctx_know_resonance = nn.ModuleList([
            OrthogonalKnowledgeResonance(d_model, init_gate=0.1) for _ in range(num_layers)
        ])
        self.beat_ctx_resonance = nn.ModuleList([
            OrthogonalKnowledgeResonance(d_model, init_gate=0.1) for _ in range(num_layers)
        ])
        self.beat_know_resonance = nn.ModuleList([
            OrthogonalKnowledgeResonance(d_model, init_gate=0.1) for _ in range(num_layers)
        ])

        # Global verbosity hint field projection (1 -> D) + orthogonal superposition
        self.verbosity_field_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.verbosity_superposition = OrthogonalFieldSuperposition(d_model, init_gate=0.1)

        # K=32 learnable future narrative beat queries
        self.beat_queries = nn.Parameter(torch.randn(num_beats, d_model) * 0.02)

        self.norm_f = nn.LayerNorm(d_model)
        self.beat_norm = nn.LayerNorm(d_model)
        self.z_proj = nn.Linear(d_model, d_model)

        # Elastic Time-Warped Legendre Expander
        self.expander = TimeWarpedLegendreExpander(d_model=d_model, num_beats=num_beats)

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
            k_pool = x_knowledge.reshape(B, M * L_k, D)
        elif x_knowledge.dim() == 3:
            k_pool = x_knowledge
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

        h = x_context
        # Inject global verbosity field when rho_override is provided
        if rho_override is not None:
            if isinstance(rho_override, (int, float)):
                rho_t = torch.full((B, 1, 1), float(rho_override), device=h.device, dtype=h.dtype)
            else:
                rho_t = rho_override.to(device=h.device, dtype=h.dtype).view(B, 1, 1)
            v_field = self.verbosity_field_proj(rho_t)  # (B, 1, D)
            h = self.verbosity_superposition(h, v_field)

        if valid_mask is not None:
            mask_f = valid_mask.unsqueeze(-1).to(dtype=h.dtype)
            h = h * mask_f
        else:
            mask_f = None

        # 1. Context Scan with Multi-Stream Knowledge Resonance
        for i, layer in enumerate(self.layers):
            h = h + layer(self.layer_norms[i](h))
            if mask_f is not None:
                h = h * mask_f
            if k_pool is not None:
                h = self.ctx_know_resonance[i](h, k_pool, k_mask)
                if mask_f is not None:
                    h = h * mask_f

        h = self.norm_f(h)
        if mask_f is not None:
            h = h * mask_f

        # Extract causal summary at the exact last valid non-PAD token index per sample (zero <PAD> gap)
        if valid_mask is not None:
            last_idx = (valid_mask.long().sum(dim=1) - 1).clamp(min=0)
            gather_idx = last_idx.view(B, 1, 1).expand(B, 1, D)
            last_hidden_state = torch.gather(h, dim=1, index=gather_idx).squeeze(1)
        else:
            last_hidden_state = h[:, -1, :]

        h_summary = self.z_proj(last_hidden_state)  # (B, D)

        if not return_sequence and not return_aux:
            if keepdim:
                return h_summary.unsqueeze(1)
            return h_summary

        # 2. Future 32-Beat Rollout initialized immediately after last valid context state
        beats = self.beat_queries.unsqueeze(0).to(dtype=h.dtype) + h_summary.unsqueeze(1)  # (B, K_beats, D)

        for i, layer in enumerate(self.layers):
            beats = beats + layer(self.layer_norms[i](beats))
            # Cross-resonate with valid context tokens so beats recall specific prompt entities
            beats = self.beat_ctx_resonance[i](beats, h, valid_mask)
            # Cross-resonate with multi-stream external knowledge pool
            if k_pool is not None:
                beats = self.beat_know_resonance[i](beats, k_pool, k_mask)

        beats = self.beat_norm(beats)

        # 3. Dynamic Time-Warped Legendre Expansion -> (B, out_len, D)
        h_potential, aux_dict = self.expander(
            c_beats=beats,
            h_summary=h_summary,
            target_len=out_len,
            rho_override=rho_override,
        )

        if not return_sequence:
            out = h_summary.unsqueeze(1) if keepdim else h_summary
            return (out, aux_dict) if return_aux else out

        if return_aux:
            return h_potential, aux_dict
        return h_potential


# Backward-compatible alias
MambaReader = MultiStreamMambaPlanner
