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
    verbosity_hint_prob: float = 0.50,
    return_sequence: bool = True,
    return_metrics: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """
    Phase 5 Multi-Stream Mamba-HiPPO CFM Loss:
    Trains MultiStreamMambaPlanner to predict the 3D Hilbert space potential trajectory
    in the normalized Flow Matching manifold, with:
      1. Manifold scale normalization for context, target, and multi-stream knowledge.
      2. Curriculum knowledge dropout (know_drop_prob = 0.15).
      3. Verbosity hint conditioning (verbosity_hint_prob = 0.50) to break one-to-many length ambiguity.
      4. Mask-weighted potential MSE + global length prediction loss (0.1 * SmoothL1)
         + lightweight beat pacing anti-collapse regularization (0.01 * loss_pacing).
    """
    B, L_t, D = x_target.shape

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

    # 2. Curriculum Knowledge Dropout (prevents blind reliance on auxiliary streams)
    eff_knowledge_mask = knowledge_mask
    if x_knowledge_norm is not None:
        if eff_knowledge_mask is None:
            eff_knowledge_mask = torch.ones(
                x_knowledge_norm.shape[:-1], dtype=torch.bool, device=x_target.device
            )
        else:
            eff_knowledge_mask = eff_knowledge_mask.bool().clone()

        if mamba.training and know_drop_prob > 0.0:
            drop_samples = torch.rand(B, device=x_target.device) < know_drop_prob
            if drop_samples.any():
                eff_knowledge_mask[drop_samples] = False

    # 3. True Target Verbosity Ratio & 50% Verbosity Hint Conditioning
    if target_mask is not None:
        rho_true = (target_mask.float().sum(dim=1) / float(max(L_t, 1))).clamp(min=0.04, max=1.0)
    else:
        rho_true = torch.ones(B, device=x_target.device, dtype=torch.float32)

    rho_override = None
    if mamba.training and verbosity_hint_prob > 0.0:
        if torch.rand(1, device=x_target.device).item() < verbosity_hint_prob:
            rho_override = rho_true

    # 4. Mamba Forward Prediction (supports both MultiStreamMambaPlanner and legacy MambaReader)
    aux = None
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

    # 5. Mask-Weighted Potential Trajectory MSE Loss
    sq_err = (h_pred - h_target) ** 2
    if target_mask is not None and sq_err.dim() == 3:
        mask_f = target_mask.to(dtype=sq_err.dtype).unsqueeze(-1)  # (B, L_t, 1)
        weights = mask_f + (1.0 - mask_f) * pad_loss_weight
        loss_potential = (sq_err * weights).sum() / (weights.sum() * D).clamp(min=1.0)
    else:
        loss_potential = torch.mean(sq_err)

    # 6. Auxiliary Horizon Length Loss & Beat Pacing Regularization
    if aux is not None and "rho_pred" in aux:
        rho_pred = aux["rho_pred"]
        loss_len = F.smooth_l1_loss(rho_pred.float(), rho_true.float(), beta=0.05).to(dtype=loss_potential.dtype)
    else:
        rho_pred = rho_true
        loss_len = torch.zeros((), device=x_target.device, dtype=loss_potential.dtype)

    if aux is not None and "pacing_weights" in aux:
        pacing_w = aux["pacing_weights"]
        k_beats = pacing_w.shape[-1]
        # Penalize only extreme beat collapse/explosion outside [0.5/K, 1.5/K]
        dev = (pacing_w.float() * float(k_beats) - 1.0).abs()
        loss_pacing = (torch.relu(dev - 0.5) ** 2).mean().to(dtype=loss_potential.dtype)
    else:
        loss_pacing = torch.zeros((), device=x_target.device, dtype=loss_potential.dtype)

    total_loss = loss_potential + 0.1 * loss_len + 0.01 * loss_pacing

    if return_metrics:
        metrics = {
            "loss_total": total_loss.detach(),
            "loss_potential": loss_potential.detach(),
            "loss_len": loss_len.detach(),
            "loss_pacing": loss_pacing.detach(),
            "rho_pred_mean": rho_pred.detach().float().mean(),
            "rho_true_mean": rho_true.detach().float().mean(),
        }
        return total_loss, metrics

    return total_loss
