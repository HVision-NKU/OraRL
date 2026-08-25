# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2026 The OraRL Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pre-selection references and post-selection moment correction.

The projection and RMS-downscale construction is derived from the Apache-2.0
EasyR1/verl implementation.
"""

from __future__ import annotations

from typing import Any

import torch

from ._utils import (
    batch_reward_tokens,
    boolean_mask,
    broadcast_sequence_values,
    group_rows,
    sample_std,
    sequence_advantages,
    sequence_rewards,
    validate_floating_tensor,
)
from .config import CorrectionConfig, PostSelectionReference


def _resolved_config(
    config: CorrectionConfig | None,
    rms_match: bool | None,
    rms_min_scale: float | None,
    sigma_policy_floor: float | None,
    eps: float | None,
) -> CorrectionConfig:
    base = CorrectionConfig() if config is None else config
    return CorrectionConfig(
        rms_match=base.rms_match if rms_match is None else rms_match,
        rms_min_scale=(base.rms_min_scale if rms_min_scale is None else rms_min_scale),
        sigma_policy_floor=(
            base.sigma_policy_floor if sigma_policy_floor is None else sigma_policy_floor
        ),
        eps=base.eps if eps is None else eps,
    )


@torch.no_grad()
def capture_pre_selection_references(
    data: Any,
    *,
    reward_key: str | None = None,
    group_key: str = "uid",
    oracle_key: str = "is_oracle_row",
) -> dict[Any, PostSelectionReference]:
    """Capture policy RMS and reward spread before rows are selected."""

    required = ("advantages", "response_mask")
    missing = [key for key in required if key not in data.batch]
    if missing:
        raise ValueError(f"reference capture is missing batch keys: {missing}.")
    if group_key not in data.non_tensor_batch:
        raise ValueError(f"reference capture requires {group_key!r}.")
    if oracle_key not in data.non_tensor_batch:
        raise ValueError(f"reference capture requires {oracle_key!r}.")

    advantages = data.batch["advantages"]
    response_mask = data.batch["response_mask"]
    sequence_values = sequence_advantages(advantages, response_mask).float()
    reward_tokens = batch_reward_tokens(data, reward_key)
    reward_values, _ = sequence_rewards(reward_tokens, response_mask)
    reward_values = reward_values.to(
        device=sequence_values.device,
        dtype=torch.float32,
    )
    grouped = group_rows(
        data.non_tensor_batch[group_key],
        sequence_values.numel(),
    )
    oracle = boolean_mask(
        data.non_tensor_batch[oracle_key],
        sequence_values.numel(),
        device=sequence_values.device,
        name=oracle_key,
    )

    references: dict[Any, PostSelectionReference] = {}
    for group_id, rows in grouped.items():
        row_index = torch.tensor(
            rows,
            dtype=torch.long,
            device=sequence_values.device,
        )
        group_oracle = oracle.index_select(0, row_index)
        if int(group_oracle.sum().item()) != 1:
            raise ValueError(f"group {group_id!r} requires exactly one oracle row.")
        policy = ~group_oracle
        policy_advantages = sequence_values.index_select(0, row_index)[policy]
        policy_rewards = reward_values.index_select(0, row_index)[policy]
        references[group_id] = PostSelectionReference(
            policy_rms=float(torch.sqrt(torch.mean(policy_advantages.square())).item()),
            sigma_policy=float(sample_std(policy_rewards).item()),
            policy_rows=int(policy.sum().item()),
        )
    return references


build_pre_selection_references = capture_pre_selection_references


@torch.no_grad()
def correct_post_selection_group(
    active_advantages: torch.Tensor,
    is_oracle_row: torch.Tensor,
    *,
    reference: PostSelectionReference,
    config: CorrectionConfig | None = None,
    rms_match: bool | None = None,
    rms_min_scale: float | None = None,
    sigma_policy_floor: float | None = None,
    eps: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Zero-center one selected group and optionally downscale its RMS."""

    cfg = _resolved_config(
        config,
        rms_match,
        rms_min_scale,
        sigma_policy_floor,
        eps,
    )
    validate_floating_tensor("active_advantages", active_advantages)
    if active_advantages.ndim != 1 or active_advantages.numel() == 0:
        raise ValueError("active_advantages must be a non-empty vector.")
    oracle = boolean_mask(
        is_oracle_row,
        active_advantages.numel(),
        device=active_advantages.device,
        name="is_oracle_row",
    )
    oracle_rows = int(oracle.sum().item())
    if oracle_rows != 1:
        raise ValueError(f"post-selection correction requires one oracle row, got {oracle_rows}.")
    policy = ~oracle
    policy_rows = int(policy.sum().item())
    if policy_rows < 1:
        raise ValueError("post-selection correction requires a policy row.")

    before = active_advantages.detach()
    mean_before = before.mean()
    rms_before = torch.sqrt(torch.mean(before.square()))
    corrected = before - mean_before
    projected = False
    if float(corrected[oracle].item()) < 0.0:
        correction = -corrected[oracle].squeeze(0)
        corrected[oracle] = 0.0
        corrected[policy] -= correction / policy_rows
        projected = True

    small_sigma_fallback = reference.sigma_policy < cfg.sigma_policy_floor
    rms_scale = corrected.new_ones(())
    if cfg.rms_match and not small_sigma_fallback:
        active_rms = torch.sqrt(torch.mean(corrected.square()))
        if float(active_rms.item()) > cfg.eps:
            target = corrected.new_tensor(reference.policy_rms)
            rms_scale = torch.clamp(
                target / (active_rms + cfg.eps),
                min=cfg.rms_min_scale,
                max=1.0,
            )
            corrected = corrected * rms_scale

    mean_after = corrected.mean()
    rms_after = torch.sqrt(torch.mean(corrected.square()))
    metrics = {
        "active_mean_before": float(mean_before.item()),
        "active_rms_before": float(rms_before.item()),
        "policy_rms_reference": reference.policy_rms,
        "rms_scale": float(rms_scale.item()),
        "oracle_sign_projection": float(projected),
        "small_sigma_fallback": float(small_sigma_fallback),
        "active_mean_after": float(mean_after.item()),
        "active_rms_after": float(rms_after.item()),
        "oracle_advantage_after": float(corrected[oracle].item()),
        "active_rows": float(corrected.numel()),
        "active_policy_rows": float(policy_rows),
    }
    return corrected, metrics


balance_post_selection_group = correct_post_selection_group


@torch.no_grad()
def apply_post_selection_correction(
    data: Any,
    references: dict[Any, PostSelectionReference],
    *,
    config: CorrectionConfig | None = None,
    rms_match: bool | None = None,
    rms_min_scale: float | None = None,
    sigma_policy_floor: float | None = None,
    eps: float | None = None,
    group_key: str = "uid",
    oracle_key: str = "is_oracle_row",
) -> dict[str, float]:
    """Apply post-selection correction to every DataProto-like group."""

    cfg = _resolved_config(
        config,
        rms_match,
        rms_min_scale,
        sigma_policy_floor,
        eps,
    )
    if "advantages" not in data.batch or "response_mask" not in data.batch:
        raise ValueError("correction requires advantages and response_mask.")
    advantages = data.batch["advantages"]
    response_mask = data.batch["response_mask"]
    sequence_values = sequence_advantages(advantages, response_mask).float()
    grouped = group_rows(
        data.non_tensor_batch[group_key],
        sequence_values.numel(),
    )
    oracle = boolean_mask(
        data.non_tensor_batch[oracle_key],
        sequence_values.numel(),
        device=sequence_values.device,
        name=oracle_key,
    )

    updated = advantages.detach().clone()
    collected: dict[str, list[float]] = {}
    for group_id, rows in grouped.items():
        if group_id not in references:
            raise ValueError(f"missing pre-selection reference for {group_id!r}.")
        row_index = torch.tensor(
            rows,
            dtype=torch.long,
            device=sequence_values.device,
        )
        corrected, group_metrics = correct_post_selection_group(
            sequence_values.index_select(0, row_index),
            oracle.index_select(0, row_index),
            reference=references[group_id],
            config=cfg,
        )
        updated.index_copy_(
            0,
            row_index.to(device=updated.device),
            broadcast_sequence_values(
                corrected.to(device=updated.device),
                response_mask.index_select(
                    0,
                    row_index.to(device=response_mask.device),
                ),
                dtype=advantages.dtype,
            ),
        )
        for key, value in group_metrics.items():
            collected.setdefault(key, []).append(value)

    data.batch["advantages"] = updated
    metrics: dict[str, float] = {
        "orarl/post_selection_groups": float(len(grouped)),
        "orarl/post_selection_rms_match": float(cfg.rms_match),
    }
    for key, values in collected.items():
        metrics[f"orarl/post_selection_{key}"] = sum(values) / len(values)
    return metrics


apply_post_selection_advantage_correction = apply_post_selection_correction
