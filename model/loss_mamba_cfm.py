from typing import Optional, Tuple, Dict, Union
import torch
import torch.nn.functional as F


def mamba_cfm_loss(
    mamba,
    hippo_oracle,
    x_context: torch.Tensor,
    x_target: torch.Tensor,
    unet=None,
    context_mask: Optional[torch.Tensor] = None,
    target_mask: Optional[torch.Tensor] = None,
    x_knowledge: Optional[torch.Tensor] = None,
    knowledge_mask: Optional[torch.Tensor] = None,
    pad_loss_weight: float = 0.05,
    know_drop_prob: float = 0.15,
    verbosity_hint_prob: float = 0.85,
    t: Optional[torch.Tensor] = None,
    return_sequence: bool = True,
    return_metrics: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """
    Phase 5 Closed-Loop Multi-Stream Mamba-HiPPO OT-CFM Loss:
    Trains MultiStreamMambaPlanner in a 100% continuous two-stage closed loop (NO auxiliary CE):
      1. Manifold scale normalization for context, target, and multi-stream knowledge.
      2. Optimal Transport Conditional Flow Matching (OT-CFM) trajectory construction:
           t ~ U(0.05, 0.95)
           X_0 ~ N(0, I)
           X_t = t * H_target + (1 - t) * X_0
           V_target = H_target - X_0
      3. Curriculum knowledge dropout (know_drop_prob = 0.15) and target horizon alignment
         (verbosity_hint_prob = 0.85) with strict h_summary_clean gradient isolation.
      4. Explicit FP32 multi-objective continuous loss with t^2-weighted potential loss:
           w_pot(t) = 3.0 * max(t, 0.1)^2   (E[3*t^2] = 1.0, suppresses t -> 0 unconditional mean pull by 30x)
           total_loss = loss_potential + 0.5 * loss_cfm + 0.25 * loss_macro + 0.5 * loss_len + 0.01 * loss_pacing
    """
    B, L_t, D = x_target.shape
    device = x_target.device

    # 1. Manifold Scale Alignment & HiPPO Oracle Target Computation
    with torch.no_grad():
        if unet is not None:
            if hasattr(unet, "manifold_calibrated"):
                is_calibrated = bool(unet.manifold_calibrated.item())
            elif hasattr(unet, "calib_steps"):
                is_calibrated = int(unet.calib_steps.item()) > 0
            else:
                is_calibrated = True

            if not is_calibrated:
                unet.calibrate_manifold(x_target, valid_mask=target_mask)

            x_context_norm = unet.normalize(x_context)
            x_target_norm = unet.normalize(x_target)
            x_knowledge_norm = unet.normalize(x_knowledge) if x_knowledge is not None else None
        else:
            x_context_norm = x_context
            x_target_norm = x_target
            x_knowledge_norm = x_knowledge

        h_target = hippo_oracle(x_target_norm, return_sequence=return_sequence)

        # Auto-calibrate <PAD> tail attractor buffer on first pass if uninitialized and <PAD> tokens exist
        if (
            hasattr(mamba, "expander")
            and hasattr(mamba.expander, "pad_tail_emb")
            and target_mask is not None
            and (~target_mask.bool()).any()
        ):
            if float(mamba.expander.pad_tail_emb.detach().abs().max().item()) < 1e-7:
                pad_positions = ~target_mask.bool()
                pad_mean = x_target_norm[pad_positions].detach().float().mean(dim=0, keepdim=True).unsqueeze(0)
                mamba.expander.pad_tail_emb.data.copy_(pad_mean.to(dtype=mamba.expander.pad_tail_emb.dtype))

    # 2. Curriculum Knowledge Dropout (prevents blind reliance on auxiliary streams)
    eff_knowledge_mask = knowledge_mask
    if x_knowledge_norm is not None:
        if eff_knowledge_mask is None:
            eff_knowledge_mask = torch.ones(
                x_knowledge_norm.shape[:-1], dtype=torch.bool, device=device
            )
        else:
            eff_knowledge_mask = eff_knowledge_mask.bool().clone()

        if mamba.training and know_drop_prob > 0.0:
            drop_samples = torch.rand(B, device=device) < know_drop_prob
            if drop_samples.any():
                eff_knowledge_mask[drop_samples] = False

    # 3. True Target Verbosity Ratio & 85% Target Span Alignment
    if target_mask is not None:
        rho_true = (target_mask.float().sum(dim=1) / float(max(L_t, 1))).clamp(min=0.04, max=1.0)
    else:
        rho_true = torch.ones(B, device=device, dtype=torch.float32)

    rho_override = None
    if mamba.training and verbosity_hint_prob > 0.0:
        if verbosity_hint_prob >= 1.0 or torch.rand(1, device=device).item() < verbosity_hint_prob:
            rho_override = rho_true

    # 4. Optimal Transport Conditional Flow Matching (OT-CFM) State & Target Velocity across t in [0.05, 0.95]
    with torch.no_grad():
        if t is None:
            t_used = 0.05 + 0.90 * torch.rand(B, device=device, dtype=torch.float32)
        else:
            t_used = t.reshape(B).to(device=device, dtype=torch.float32)
        x_0 = torch.randn_like(h_target)
        if h_target.dim() == 3:
            t_expand = t_used.view(B, 1, 1).to(dtype=h_target.dtype)
        else:
            t_expand = t_used.view(B, 1).to(dtype=h_target.dtype)
        x_t = t_expand * h_target + (1.0 - t_expand) * x_0
        v_target = h_target - x_0

    # 5. Mamba Closed-Loop Forward Prediction (supports Two-Stage (X_t, t) and legacy MambaReader)
    aux = None
    v_mamba = None
    try:
        out = mamba(
            x_context_norm,
            valid_mask=context_mask,
            x_knowledge=x_knowledge_norm,
            knowledge_mask=eff_knowledge_mask,
            target_len=L_t,
            rho_override=rho_override,
            x_t=x_t if h_target.dim() == 3 else None,
            t=t_used,
            return_sequence=return_sequence,
            return_aux=True,
            return_velocity=True,
        )
        if isinstance(out, tuple):
            if len(out) == 3:
                h_pred, v_mamba, aux = out
            elif len(out) == 2:
                h_pred, aux = out
                if isinstance(aux, dict) and "v_mamba" in aux:
                    v_mamba = aux["v_mamba"]
            else:
                h_pred = out[0]
        else:
            h_pred = out
    except TypeError:
        try:
            out = mamba(
                x_context_norm,
                valid_mask=context_mask,
                x_knowledge=x_knowledge_norm,
                knowledge_mask=eff_knowledge_mask,
                target_len=L_t,
                rho_override=rho_override,
                return_sequence=return_sequence,
                return_aux=True,
            )
            if isinstance(out, tuple):
                h_pred, aux = out
            else:
                h_pred = out
        except TypeError:
            h_pred = mamba(x_context_norm, valid_mask=context_mask, return_sequence=return_sequence)

    if h_pred.dim() == 2 and h_target.dim() == 3:
        h_pred = h_pred.unsqueeze(1)
    elif h_pred.dim() == 3 and h_target.dim() == 2:
        h_pred = h_pred[:, -1, :]

    # 6. Explicit FP32 Multi-Objective Loss Computation
    with torch.amp.autocast(device_type=device.type, enabled=False):
        t_f = t_used.float()
        w_pot_sample = 3.0 * (t_f.clamp(min=0.1) ** 2)  # (B,)

        if target_mask is not None and h_target.dim() == 3:
            mask_f = target_mask.float().unsqueeze(-1)  # (B, L_t, 1)
            weights = mask_f + (1.0 - mask_f) * float(pad_loss_weight)
            denom = (weights.sum() * D).clamp(min=1.0)
            sample_denom = (weights.sum(dim=(1, 2)) * D).clamp(min=1.0)  # (B,)
        else:
            weights = None
            denom = None
            sample_denom = None

        # (a) Refined Potential Trajectory MSE Loss (t^2-weighted + unweighted raw per-sample tracking)
        sq_err_pot = (h_pred.float() - h_target.float()) ** 2
        if weights is not None and sq_err_pot.dim() == 3:
            w_pot_3d = w_pot_sample.view(B, 1, 1)
            loss_potential = (sq_err_pot * weights * w_pot_3d).sum() / denom
            loss_potential_raw = (sq_err_pot * weights).sum() / denom
            per_sample_pot_mse = (sq_err_pot * weights).sum(dim=(1, 2)) / sample_denom
            per_sample_pot_weighted = per_sample_pot_mse * w_pot_sample
        else:
            w_pot_nd = w_pot_sample.view(B, *([1] * (sq_err_pot.dim() - 1)))
            loss_potential = torch.mean(sq_err_pot * w_pot_nd)
            loss_potential_raw = torch.mean(sq_err_pot)
            per_sample_pot_mse = sq_err_pot.reshape(B, -1).mean(dim=1)
            per_sample_pot_weighted = per_sample_pot_mse * w_pot_sample

        # (b) Instantaneous OT-CFM Velocity Matching Loss (weighted by (1 - t)^2 for unit gradient scale with 1/(1 - t) v_base)
        if v_mamba is not None and v_mamba.shape == v_target.shape:
            cfm_t_scale = (1.0 - t_f).clamp(min=0.05) ** 2
            if v_mamba.dim() == 3:
                cfm_t_scale = cfm_t_scale.view(B, 1, 1)
            else:
                cfm_t_scale = cfm_t_scale.view(B, 1)
            sq_err_cfm = ((v_mamba.float() - v_target.float()) ** 2) * cfm_t_scale
            if weights is not None and sq_err_cfm.dim() == 3:
                loss_cfm = (sq_err_cfm * weights).sum() / denom
                per_sample_cfm = (sq_err_cfm * weights).sum(dim=(1, 2)) / sample_denom
            else:
                loss_cfm = torch.mean(sq_err_cfm)
                per_sample_cfm = sq_err_cfm.reshape(B, -1).mean(dim=1)
        else:
            loss_cfm = torch.zeros((), device=device, dtype=torch.float32)
            per_sample_cfm = torch.zeros(B, device=device, dtype=torch.float32)

        # (c) Macroscopic Potential Anchor Loss (keeps Stage 1 encode_macro sharp at t -> 0)
        if aux is not None and "h_macro_potential" in aux and aux["h_macro_potential"].shape == h_target.shape:
            h_macro_pot = aux["h_macro_potential"].float()
            sq_err_macro = (h_macro_pot - h_target.float()) ** 2
            if weights is not None and sq_err_macro.dim() == 3:
                loss_macro = (sq_err_macro * weights).sum() / denom
                per_sample_macro = (sq_err_macro * weights).sum(dim=(1, 2)) / sample_denom
            else:
                loss_macro = torch.mean(sq_err_macro)
                per_sample_macro = sq_err_macro.reshape(B, -1).mean(dim=1)
        else:
            loss_macro = torch.zeros((), device=device, dtype=torch.float32)
            per_sample_macro = torch.zeros(B, device=device, dtype=torch.float32)

        # (d) Global Horizon Length Prediction Loss (Smooth L1 in FP32)
        if aux is not None and "rho_pred" in aux:
            rho_pred = aux["rho_pred"].float()
            loss_len = F.smooth_l1_loss(rho_pred, rho_true.float(), beta=0.02)
            per_sample_len = F.smooth_l1_loss(rho_pred, rho_true.float(), beta=0.02, reduction="none").reshape(B)
        else:
            rho_pred = rho_true.float()
            loss_len = torch.zeros((), device=device, dtype=torch.float32)
            per_sample_len = torch.zeros(B, device=device, dtype=torch.float32)

        # (e) Beat Pacing Anti-Collapse Regularization
        if aux is not None and "pacing_weights" in aux:
            pacing_w = aux["pacing_weights"].float()
            k_beats = pacing_w.shape[-1]
            dev = (pacing_w * float(k_beats) - 1.0).abs()
            loss_pacing = (torch.relu(dev - 0.5) ** 2).mean()
            per_sample_pacing = (torch.relu(dev - 0.5) ** 2).mean(dim=-1)
        else:
            loss_pacing = torch.zeros((), device=device, dtype=torch.float32)
            per_sample_pacing = torch.zeros(B, device=device, dtype=torch.float32)

        total_loss = (
            loss_potential
            + 0.5 * loss_cfm
            + 0.25 * loss_macro
            + 0.5 * loss_len
            + 0.01 * loss_pacing
        )

        per_sample_total_loss = (
            per_sample_pot_weighted
            + 0.5 * per_sample_cfm
            + 0.25 * per_sample_macro
            + 0.5 * per_sample_len
            + 0.01 * per_sample_pacing
        )

    if return_metrics:
        with torch.no_grad():
            t_det = t_used.detach().float()
            pot_det = per_sample_pot_mse.detach().float()

            mask_low = t_det < 0.3
            mask_mid = (t_det >= 0.3) & (t_det < 0.7)
            mask_high = t_det >= 0.7

            nan_val = torch.tensor(float("nan"), device=device, dtype=torch.float32)
            pot_t_low = pot_det[mask_low].mean() if mask_low.any() else nan_val
            pot_t_mid = pot_det[mask_mid].mean() if mask_mid.any() else nan_val
            pot_t_high = pot_det[mask_high].mean() if mask_high.any() else nan_val

            if aux is not None and "copy_prob" in aux and aux["copy_prob"] is not None:
                copy_prob_mean = aux["copy_prob"].detach().float().mean()
            else:
                copy_prob_mean = torch.zeros((), device=device, dtype=torch.float32)

        metrics = {
            "loss_total": total_loss.detach(),
            "loss_potential": loss_potential.detach(),
            "loss_potential_raw": loss_potential_raw.detach(),
            "loss_cfm": loss_cfm.detach(),
            "loss_macro": loss_macro.detach(),
            "loss_len": loss_len.detach(),
            "loss_pacing": loss_pacing.detach(),
            "rho_pred_mean": rho_pred.detach().float().mean(),
            "rho_true_mean": rho_true.detach().float().mean(),
            "pot_t_low": pot_t_low,
            "pot_t_mid": pot_t_mid,
            "pot_t_high": pot_t_high,
            "copy_prob_mean": copy_prob_mean,
            "per_sample_loss": per_sample_total_loss.detach(),
            "per_sample_pot": per_sample_pot_mse.detach(),
        }
        return total_loss, metrics

    return total_loss
