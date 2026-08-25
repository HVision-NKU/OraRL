"""Pure helpers for OraRL post-selection moment correction."""

import math
from dataclasses import dataclass

import torch


SIGMA_OP_FALLBACK_THRESHOLD = 1e-3


@dataclass(frozen=True)
class PostSelectionReference:
    """Pre-selection on-policy statistics for one rollout group."""

    on_policy_rms: float
    sigma_op: float
    on_policy_rows: int = 0


def balance_post_selection_group(
    active_advantages: torch.Tensor,
    is_oracle_row: torch.Tensor,
    *,
    reference: PostSelectionReference,
    recenter: bool,
    rms_match: bool,
    rms_min_scale: float = 0.25,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Recenter one selected group and optionally match its pre-selection RMS.

    The active group contains the policy rows retained by OraRL selection and
    one detached oracle row. If recentering would make the oracle advantage
    negative, the vector is projected onto ``sum(A)=0, A_oracle>=0`` by setting
    the oracle to zero and distributing the correction evenly over policy rows.
    RMS matching can only downscale.
    """

    if active_advantages.ndim != 1:
        raise ValueError(
            "active_advantages must be one-dimensional, got "
            f"{tuple(active_advantages.shape)}."
        )
    if (
        is_oracle_row.ndim != 1
        or is_oracle_row.shape != active_advantages.shape
    ):
        raise ValueError(
            "is_oracle_row must match active_advantages, got "
            f"{tuple(is_oracle_row.shape)} and {tuple(active_advantages.shape)}."
        )
    if active_advantages.numel() == 0:
        raise ValueError("active_advantages must not be empty.")
    if not bool(torch.isfinite(active_advantages).all()):
        raise ValueError("active_advantages contains non-finite values.")
    if rms_match and not recenter:
        raise ValueError("rms_match requires recenter.")
    if not 0.0 <= rms_min_scale <= 1.0:
        raise ValueError(
            f"rms_min_scale must be in [0, 1], got {rms_min_scale}."
        )
    if (
        not math.isfinite(reference.on_policy_rms)
        or not math.isfinite(reference.sigma_op)
        or reference.on_policy_rms < 0.0
        or reference.sigma_op < 0.0
    ):
        raise ValueError(
            "reference RMS and sigma_op must be finite and non-negative."
        )

    oracle_mask = is_oracle_row.to(
        device=active_advantages.device,
        dtype=torch.bool,
    )
    oracle_rows = int(oracle_mask.sum().item())
    if oracle_rows > 1:
        raise ValueError(
            "post-selection correction supports at most one oracle row, "
            f"got {oracle_rows}."
        )
    policy_rows = int((~oracle_mask).sum().item())
    if oracle_rows == 1 and policy_rows == 0:
        raise ValueError(
            "post-selection correction requires at least one active policy row "
            "alongside the oracle row."
        )

    active_before = active_advantages
    mean_before = active_before.mean()
    rms_before = torch.sqrt(torch.mean(active_before.square()))
    active_after = active_before.clone()
    oracle_projection = active_before.new_zeros(())

    if recenter:
        active_after -= mean_before
        if (
            oracle_rows == 1
            and float(active_after[oracle_mask].item()) < 0.0
        ):
            policy_mask = ~oracle_mask
            correction = -active_after[oracle_mask]
            active_after[oracle_mask] = 0.0
            active_after[policy_mask] -= correction / float(policy_rows)
            oracle_projection = active_before.new_ones(())

    rms_scale = active_before.new_ones(())
    sigma_fallback = reference.sigma_op < SIGMA_OP_FALLBACK_THRESHOLD
    if rms_match:
        active_rms = torch.sqrt(torch.mean(active_after.square()))
        if float(active_rms.item()) > eps and not sigma_fallback:
            target_rms = active_after.new_tensor(reference.on_policy_rms)
            rms_scale = torch.clamp(
                target_rms / (active_rms + eps),
                min=float(rms_min_scale),
                max=1.0,
            )
            active_after *= rms_scale

    mean_after = active_after.mean()
    rms_after = torch.sqrt(torch.mean(active_after.square()))
    oracle_after = (
        float(active_after[oracle_mask].item())
        if oracle_rows == 1
        else 0.0
    )
    metrics = {
        "active_mean_before": float(mean_before.item()),
        "active_rms_before": float(rms_before.item()),
        "on_policy_rms": float(reference.on_policy_rms),
        "rms_scale": float(rms_scale.item()),
        "oracle_sign_projection": float(oracle_projection.item()),
        "sigma_op_fallback": float(sigma_fallback),
        "active_mean_after": float(mean_after.item()),
        "active_rms_after": float(rms_after.item()),
        "oracle_advantage_after": oracle_after,
        "active_rows": float(active_after.numel()),
        "active_policy_rows": float((~oracle_mask).sum().item()),
        "oracle_rows": float(oracle_rows),
    }
    return active_after, metrics
