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
"""Strict sign-balanced rollout selection.

The physical batch-slicing pattern is derived from the Apache-2.0 EasyR1/verl
trainer. The selector here is limited to the OraRL paper recipe.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from ._utils import (
    boolean_mask,
    copy_and_refresh_meta,
    group_rows,
    sequence_advantages,
)
from .config import SelectionConfig


def _resolved_config(
    config: SelectionConfig | None,
    keep_per_group: int | None,
    positive_quota: int | None,
    negative_quota: int | None,
    world_size: int | None,
    n_rollouts: int | None,
    prune_ratio: float | None,
) -> SelectionConfig:
    base = SelectionConfig() if config is None else config
    if (n_rollouts is None) != (prune_ratio is None):
        raise ValueError("n_rollouts and prune_ratio must be provided together.")
    derived_keep: int | None = None
    if n_rollouts is not None and prune_ratio is not None:
        if n_rollouts <= 1:
            raise ValueError("n_rollouts must be greater than one.")
        if not math.isfinite(prune_ratio) or not 0.0 <= prune_ratio < 1.0:
            raise ValueError("prune_ratio must be finite and in [0, 1).")
        derived_keep = int(n_rollouts * (1.0 - prune_ratio))
        if keep_per_group is not None and keep_per_group != derived_keep:
            raise ValueError("keep_per_group disagrees with the n_rollouts/prune_ratio budget.")
    resolved_keep = (
        derived_keep
        if derived_keep is not None
        else base.keep_per_group
        if keep_per_group is None
        else keep_per_group
    )
    return SelectionConfig(
        keep_per_group=resolved_keep,
        positive_quota=(base.positive_quota if positive_quota is None else positive_quota),
        negative_quota=(base.negative_quota if negative_quota is None else negative_quota),
        world_size=base.world_size if world_size is None else world_size,
    )


def _rank_by_magnitude(rows: list[int], scores: torch.Tensor) -> list[int]:
    return sorted(
        rows,
        key=lambda row: (-abs(float(scores[row].item())), row),
    )


@torch.no_grad()
def select_sign_balanced_rollouts(
    data: Any,
    keep_per_group: int | None = None,
    positive_quota: int | None = None,
    negative_quota: int | None = None,
    world_size: int | None = None,
    *,
    config: SelectionConfig | None = None,
    n_rollouts: int | None = None,
    prune_ratio: float | None = None,
    group_key: str = "uid",
    oracle_key: str = "is_oracle_row",
) -> tuple[Any, dict[str, float]]:
    """Select exactly one oracle plus strict positive/negative policy quotas.

    A short sign bucket is filled from the opposite sign by descending
    magnitude, followed by zero-advantage rows. The selected indices physically
    slice the DataProto-like object before the actor update.
    """

    cfg = _resolved_config(
        config,
        keep_per_group,
        positive_quota,
        negative_quota,
        world_size,
        n_rollouts,
        prune_ratio,
    )
    if "advantages" not in data.batch or "response_mask" not in data.batch:
        raise ValueError("selection requires advantages and response_mask.")
    if group_key not in data.non_tensor_batch:
        raise ValueError(f"selection requires non_tensor_batch[{group_key!r}].")
    if oracle_key not in data.non_tensor_batch:
        raise ValueError(f"selection requires non_tensor_batch[{oracle_key!r}].")

    advantages = data.batch["advantages"]
    response_mask = data.batch["response_mask"]
    signed_scores = sequence_advantages(advantages, response_mask)
    total_rows = signed_scores.numel()
    grouped = group_rows(data.non_tensor_batch[group_key], total_rows)
    oracle = boolean_mask(
        data.non_tensor_batch[oracle_key],
        total_rows,
        device=signed_scores.device,
        name=oracle_key,
    )

    selected_rows: list[int] = []
    positive_kept = 0
    negative_kept = 0
    zero_kept = 0
    cross_sign_fallback = 0
    zero_fallback = 0

    for group_id, rows in grouped.items():
        if len(rows) < cfg.keep_per_group:
            raise ValueError(
                f"group {group_id!r} has {len(rows)} rows, fewer than the "
                f"keep budget {cfg.keep_per_group}."
            )
        oracle_rows = [row for row in rows if bool(oracle[row])]
        if len(oracle_rows) != 1:
            raise ValueError(
                f"group {group_id!r} requires exactly one oracle row, got {len(oracle_rows)}."
            )
        oracle_row = oracle_rows[0]
        candidates = [row for row in rows if row != oracle_row]
        positive = _rank_by_magnitude(
            [row for row in candidates if float(signed_scores[row].item()) > 0.0],
            signed_scores,
        )
        negative = _rank_by_magnitude(
            [row for row in candidates if float(signed_scores[row].item()) < 0.0],
            signed_scores,
        )
        zeros = [row for row in candidates if float(signed_scores[row].item()) == 0.0]

        chosen = positive[: cfg.positive_quota] + negative[: cfg.negative_quota]
        chosen_set = set(chosen)
        policy_budget = cfg.keep_per_group - 1
        remaining = policy_budget - len(chosen)

        if remaining > 0:
            opposite_sign_surplus = _rank_by_magnitude(
                [row for row in positive + negative if row not in chosen_set],
                signed_scores,
            )
            cross_fill = opposite_sign_surplus[:remaining]
            chosen.extend(cross_fill)
            chosen_set.update(cross_fill)
            cross_sign_fallback += len(cross_fill)
            remaining -= len(cross_fill)

        if remaining > 0:
            zero_fill = [row for row in zeros if row not in chosen_set][:remaining]
            chosen.extend(zero_fill)
            chosen_set.update(zero_fill)
            zero_fallback += len(zero_fill)
            remaining -= len(zero_fill)

        if remaining > 0:
            final_fill = [row for row in candidates if row not in chosen_set][:remaining]
            chosen.extend(final_fill)
            remaining -= len(final_fill)
        if remaining != 0 or len(chosen) != policy_budget:
            raise RuntimeError(f"group {group_id!r} could not satisfy its keep budget.")

        for row in chosen:
            score = float(signed_scores[row].item())
            positive_kept += int(score > 0.0)
            negative_kept += int(score < 0.0)
            zero_kept += int(score == 0.0)
        selected_rows.extend([oracle_row, *chosen])

    selected_rows.sort()
    if len(selected_rows) % cfg.world_size != 0:
        raise RuntimeError(
            f"selected batch size {len(selected_rows)} is not divisible by "
            f"world_size {cfg.world_size}."
        )

    selected = data[selected_rows]
    copy_and_refresh_meta(selected)
    selected_index = torch.tensor(
        selected_rows,
        dtype=torch.long,
        device=signed_scores.device,
    )
    kept_abs = signed_scores.index_select(0, selected_index).abs()
    selected_set = set(selected_rows)
    dropped_rows = [row for row in range(total_rows) if row not in selected_set]
    metrics: dict[str, float] = {
        "orarl/selection_groups": float(len(grouped)),
        "orarl/keep_per_group": float(cfg.keep_per_group),
        "orarl/kept_rows": float(len(selected_rows)),
        "orarl/dropped_rows": float(total_rows - len(selected_rows)),
        "orarl/effective_keep_ratio": len(selected_rows) / float(total_rows),
        "orarl/oracle_rows_forced": float(len(grouped)),
        "orarl/positive_policy_rows_kept": float(positive_kept),
        "orarl/negative_policy_rows_kept": float(negative_kept),
        "orarl/zero_policy_rows_kept": float(zero_kept),
        "orarl/cross_sign_fallback_rows": float(cross_sign_fallback),
        "orarl/zero_fallback_rows": float(zero_fallback),
        "orarl/abs_advantage_kept_mean": float(kept_abs.mean().item()),
    }
    if prune_ratio is not None:
        metrics["orarl/requested_prune_ratio"] = prune_ratio
    if dropped_rows:
        dropped_index = torch.tensor(
            dropped_rows,
            dtype=torch.long,
            device=signed_scores.device,
        )
        metrics["orarl/abs_advantage_dropped_mean"] = float(
            signed_scores.index_select(0, dropped_index).abs().mean().item()
        )
    return selected, metrics
