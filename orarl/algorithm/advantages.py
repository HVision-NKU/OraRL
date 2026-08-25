# Copyright 2022 The HuggingFace Team
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
"""Pure advantage construction for vanilla GRPO and OraRL.

The grouped baseline and token broadcasting behavior are derived from the
Apache-2.0 EasyR1/verl implementation. OraRL keeps the annotation outside the
policy baseline and treats its advantage as a detached scalar.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
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
from .config import (
    DetachedOracleConfig,
    DirectionalGainConfig,
    GRPOConfig,
    OraRLConfig,
)

DIRECTIONAL_GAIN_MIN = 1.0
DIRECTIONAL_GAIN_MAX = 4.0


@torch.no_grad()
def vanilla_grpo_advantage(
    rewards: torch.Tensor,
    *,
    config: GRPOConfig | None = None,
) -> torch.Tensor:
    """Compute vanilla GRPO advantage for one rollout group."""

    cfg = GRPOConfig() if config is None else config
    validate_floating_tensor("rewards", rewards)
    if rewards.ndim != 1:
        raise ValueError("rewards must be one-dimensional.")
    if rewards.numel() < 2:
        raise ValueError("vanilla GRPO requires at least two rollouts per group.")
    centered = rewards.detach() - rewards.detach().mean()
    if cfg.normalize:
        centered = centered / (sample_std(rewards.detach()) + cfg.eps)
    return centered


@torch.no_grad()
def compute_grpo_advantages(
    rewards: torch.Tensor,
    group_ids: Sequence[Any] | np.ndarray | torch.Tensor,
    response_mask: torch.Tensor | None = None,
    *,
    config: GRPOConfig | None = None,
) -> torch.Tensor:
    """Compute grouped vanilla GRPO advantages.

    One-dimensional input produces one scalar per row. Token-level input
    produces a matrix with each group scalar broadcast over valid tokens.
    """

    cfg = GRPOConfig() if config is None else config
    scores, token_input = sequence_rewards(rewards, response_mask)
    grouped = group_rows(group_ids, scores.numel())
    output = torch.empty_like(scores)
    for rows in grouped.values():
        row_index = torch.tensor(rows, dtype=torch.long, device=scores.device)
        output.index_copy_(
            0,
            row_index,
            vanilla_grpo_advantage(scores.index_select(0, row_index), config=cfg),
        )
    if not token_input:
        return output
    assert response_mask is not None
    return broadcast_sequence_values(output, response_mask, dtype=rewards.dtype)


@torch.no_grad()
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Sequence[Any] | np.ndarray | torch.Tensor,
    eps: float = 1e-6,
    *,
    scale_rewards: bool = True,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility wrapper matching the common trainer estimator contract."""

    advantages = compute_grpo_advantages(
        token_level_rewards,
        index,
        response_mask,
        config=GRPOConfig(normalize=scale_rewards, eps=eps),
    )
    return advantages, advantages


@torch.no_grad()
def raw_center_on_policy(
    rewards: torch.Tensor,
    is_oracle_row: Sequence[bool] | np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Raw-center policy rewards while assigning zero to the oracle row."""

    validate_floating_tensor("rewards", rewards)
    if rewards.ndim != 1:
        raise ValueError("rewards must be one-dimensional.")
    oracle = boolean_mask(
        is_oracle_row,
        rewards.numel(),
        device=rewards.device,
        name="is_oracle_row",
    )
    oracle_rows = int(oracle.sum().item())
    if oracle_rows != 1:
        raise ValueError(f"each OraRL group requires one oracle row, got {oracle_rows}.")
    policy = ~oracle
    if not bool(policy.any()):
        raise ValueError("each OraRL group requires at least one policy row.")
    output = torch.zeros_like(rewards)
    policy_rewards = rewards.detach()[policy]
    output[policy] = policy_rewards - policy_rewards.mean()
    return output


@torch.no_grad()
def compute_directional_gain(
    sigma_all: torch.Tensor | float,
    sigma_policy: torch.Tensor | float,
    *,
    config: DirectionalGainConfig | None = None,
) -> torch.Tensor:
    """Compute ``clip((sigma_all / (sigma_policy + eps))**gamma, 1, 4)``."""

    cfg = DirectionalGainConfig() if config is None else config
    if isinstance(sigma_all, torch.Tensor):
        all_value = sigma_all.detach()
    else:
        all_value = torch.tensor(float(sigma_all), dtype=torch.float32)
    if all_value.ndim != 0:
        raise ValueError("sigma_all must be scalar.")
    policy_value = torch.as_tensor(
        sigma_policy,
        dtype=all_value.dtype,
        device=all_value.device,
    ).detach()
    if policy_value.ndim != 0:
        raise ValueError("sigma_policy must be scalar.")
    if (
        not bool(torch.isfinite(all_value))
        or not bool(torch.isfinite(policy_value))
        or float(all_value.item()) < 0.0
        or float(policy_value.item()) < 0.0
    ):
        raise ValueError("standard deviations must be finite and non-negative.")
    ratio = all_value / (policy_value + cfg.eps)
    return torch.clamp(
        ratio.pow(cfg.gamma),
        min=DIRECTIONAL_GAIN_MIN,
        max=DIRECTIONAL_GAIN_MAX,
    )


directional_gain = compute_directional_gain


@torch.no_grad()
def transform_directional_utility(
    centered_policy_advantages: torch.Tensor,
    gain: torch.Tensor | float,
) -> torch.Tensor:
    """Amplify positive policy utility only, then restore zero mean."""

    validate_floating_tensor(
        "centered_policy_advantages",
        centered_policy_advantages,
    )
    if centered_policy_advantages.ndim != 1:
        raise ValueError("centered_policy_advantages must be one-dimensional.")
    if centered_policy_advantages.numel() == 0:
        raise ValueError("centered_policy_advantages must not be empty.")
    gain_value = torch.as_tensor(
        gain,
        dtype=centered_policy_advantages.dtype,
        device=centered_policy_advantages.device,
    ).detach()
    if gain_value.ndim != 0 or not bool(torch.isfinite(gain_value)):
        raise ValueError("gain must be a finite scalar.")
    if float(gain_value.item()) < 1.0:
        raise ValueError("gain must be at least one.")
    base = centered_policy_advantages.detach()
    transformed = torch.where(base > 0, base * gain_value, base)
    return transformed - transformed.mean()


def _orarl_policy_group(
    rewards: torch.Tensor,
    oracle: torch.Tensor,
    config: DirectionalGainConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    centered = raw_center_on_policy(rewards, oracle)
    policy = ~oracle
    gain = compute_directional_gain(
        sample_std(rewards),
        sample_std(rewards[policy]),
        config=config,
    )
    output = torch.zeros_like(rewards)
    output[policy] = transform_directional_utility(centered[policy], gain)
    return output, gain


@torch.no_grad()
def compute_orarl_policy_advantages(
    rewards: torch.Tensor,
    group_ids: Sequence[Any] | np.ndarray | torch.Tensor,
    is_oracle_row: Sequence[bool] | np.ndarray | torch.Tensor,
    response_mask: torch.Tensor | None = None,
    *,
    config: DirectionalGainConfig | None = None,
    metrics_out: dict[str, float] | None = None,
) -> torch.Tensor:
    """Apply raw policy centering and directional gain to grouped rewards."""

    cfg = DirectionalGainConfig() if config is None else config
    scores, token_input = sequence_rewards(rewards, response_mask)
    oracle = boolean_mask(
        is_oracle_row,
        scores.numel(),
        device=scores.device,
        name="is_oracle_row",
    )
    grouped = group_rows(group_ids, scores.numel())
    output = torch.empty_like(scores)
    gains: list[float] = []
    for rows in grouped.values():
        row_index = torch.tensor(rows, dtype=torch.long, device=scores.device)
        group_output, gain = _orarl_policy_group(
            scores.index_select(0, row_index),
            oracle.index_select(0, row_index),
            cfg,
        )
        output.index_copy_(0, row_index, group_output)
        gains.append(float(gain.item()))
    if metrics_out is not None and gains:
        metrics_out["orarl/directional_gain_mean"] = sum(gains) / len(gains)
        metrics_out["orarl/directional_gain_max"] = max(gains)
        metrics_out["orarl/groups"] = float(len(gains))
    if not token_input:
        return output
    assert response_mask is not None
    return broadcast_sequence_values(output, response_mask, dtype=rewards.dtype)


@torch.no_grad()
def normalized_reward_gap(
    oracle_reward: torch.Tensor | float,
    policy_rewards: torch.Tensor,
    *,
    beta: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return the clipped, task-scale-normalized oracle reward gap."""

    validate_floating_tensor("policy_rewards", policy_rewards)
    if policy_rewards.ndim != 1 or policy_rewards.numel() == 0:
        raise ValueError("policy_rewards must be a non-empty vector.")
    if beta <= 0.0 or eps <= 0.0:
        raise ValueError("beta and eps must be positive.")
    oracle = torch.as_tensor(
        oracle_reward,
        dtype=policy_rewards.dtype,
        device=policy_rewards.device,
    ).detach()
    if oracle.ndim != 0 or not bool(torch.isfinite(oracle)):
        raise ValueError("oracle_reward must be a finite scalar.")
    gap = (oracle - policy_rewards.detach().mean()) / (oracle + eps)
    return torch.clamp(gap, min=0.0, max=1.0).pow(beta)


def _detached_oracle_details(
    oracle_reward: torch.Tensor | float,
    policy_rewards: torch.Tensor,
    policy_advantages: torch.Tensor,
    config: DetachedOracleConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    validate_floating_tensor("policy_advantages", policy_advantages)
    if policy_advantages.ndim != 1 or policy_advantages.numel() == 0:
        raise ValueError("policy_advantages must be a non-empty vector.")
    if policy_advantages.shape != policy_rewards.shape:
        raise ValueError("policy rewards and advantages must have matching shapes.")

    weight = normalized_reward_gap(
        oracle_reward,
        policy_rewards,
        beta=config.gap_beta,
        eps=config.eps,
    )
    raw = weight * config.scale
    anchor = raw
    positives = policy_advantages.detach()[policy_advantages.detach() > 0]
    best_positive = positives.max() if positives.numel() > 0 else policy_advantages.new_zeros(())
    cap = policy_advantages.new_tensor(float("inf"))
    cap_applied = False
    if config.match_best_ratio is not None:
        proposed = (
            config.match_best_ratio * best_positive
            if float(best_positive.item()) > 0.0
            else policy_advantages.new_tensor(config.match_best_min)
        )
        cap = torch.clamp(
            proposed,
            min=config.match_best_min,
            max=config.match_best_max,
        )
        anchor = torch.minimum(raw, cap)
        cap_applied = float(anchor.item()) < float(raw.item())
    details = {
        "weight": float(weight.item()),
        "raw": float(raw.item()),
        "anchor": float(anchor.item()),
        "best_positive": float(best_positive.item()),
        "cap": float(cap.item()),
        "cap_applied": float(cap_applied),
    }
    return anchor.detach(), details


@torch.no_grad()
def detached_oracle_advantage(
    oracle_reward: torch.Tensor | float,
    policy_rewards: torch.Tensor,
    policy_advantages: torch.Tensor,
    *,
    config: DetachedOracleConfig | None = None,
) -> torch.Tensor:
    """Construct a detached oracle advantage with an optional positive cap."""

    cfg = DetachedOracleConfig() if config is None else config
    anchor, _ = _detached_oracle_details(
        oracle_reward,
        policy_rewards,
        policy_advantages,
        cfg,
    )
    return anchor


def _write_oracle_metrics(
    metrics_out: dict[str, float],
    details: list[dict[str, float]],
) -> None:
    if not details:
        return
    count = len(details)
    for source, target in (
        ("weight", "orarl/oracle_gap_weight_mean"),
        ("raw", "orarl/oracle_advantage_raw_mean"),
        ("anchor", "orarl/oracle_advantage_mean"),
        ("best_positive", "orarl/best_positive_advantage_mean"),
        ("cap_applied", "orarl/oracle_cap_applied_fraction"),
    ):
        metrics_out[target] = sum(item[source] for item in details) / count
    finite_caps = [item["cap"] for item in details if np.isfinite(item["cap"])]
    if finite_caps:
        metrics_out["orarl/oracle_cap_mean"] = sum(finite_caps) / len(finite_caps)


def _compute_orarl_from_scores(
    policy_scores: torch.Tensor,
    oracle_scores: torch.Tensor,
    grouped: dict[Any, list[int]],
    oracle: torch.Tensor,
    config: OraRLConfig,
    metrics_out: dict[str, float] | None,
) -> torch.Tensor:
    output = torch.empty_like(policy_scores)
    gains: list[float] = []
    details: list[dict[str, float]] = []
    for rows in grouped.values():
        row_index = torch.tensor(rows, dtype=torch.long, device=policy_scores.device)
        group_oracle = oracle.index_select(0, row_index)
        group_policy_output, gain = _orarl_policy_group(
            policy_scores.index_select(0, row_index),
            group_oracle,
            config.directional_gain,
        )
        policy = ~group_oracle
        raw_group_scores = oracle_scores.index_select(0, row_index)
        anchor, group_details = _detached_oracle_details(
            raw_group_scores[group_oracle].squeeze(0),
            raw_group_scores[policy],
            group_policy_output[policy],
            config.detached_oracle,
        )
        group_policy_output[group_oracle] = anchor
        output.index_copy_(0, row_index, group_policy_output)
        gains.append(float(gain.item()))
        details.append(group_details)

    if metrics_out is not None and gains:
        metrics_out["orarl/directional_gain_mean"] = sum(gains) / len(gains)
        metrics_out["orarl/directional_gain_max"] = max(gains)
        metrics_out["orarl/groups"] = float(len(gains))
        _write_oracle_metrics(metrics_out, details)
    return output


@torch.no_grad()
def compute_orarl_advantages(
    rewards: torch.Tensor,
    group_ids: Sequence[Any] | np.ndarray | torch.Tensor,
    is_oracle_row: Sequence[bool] | np.ndarray | torch.Tensor,
    response_mask: torch.Tensor | None = None,
    *,
    config: OraRLConfig | None = None,
    metrics_out: dict[str, float] | None = None,
) -> torch.Tensor:
    """Compute raw-centered, directionally transformed OraRL advantages."""

    cfg = OraRLConfig() if config is None else config
    scores, token_input = sequence_rewards(rewards, response_mask)
    oracle = boolean_mask(
        is_oracle_row,
        scores.numel(),
        device=scores.device,
        name="is_oracle_row",
    )
    grouped = group_rows(group_ids, scores.numel())
    output = _compute_orarl_from_scores(
        scores,
        scores,
        grouped,
        oracle,
        cfg,
        metrics_out,
    )
    if not token_input:
        return output
    assert response_mask is not None
    return broadcast_sequence_values(output, response_mask, dtype=rewards.dtype)


@torch.no_grad()
def apply_grpo_advantages(
    data: Any,
    *,
    config: GRPOConfig | None = None,
    reward_key: str | None = "token_level_rewards",
    group_key: str = "uid",
) -> dict[str, float]:
    """Write vanilla GRPO advantages into a DataProto-like batch."""

    rewards = batch_reward_tokens(data, reward_key)
    response_mask = data.batch["response_mask"]
    group_ids = data.non_tensor_batch[group_key]
    advantages = compute_grpo_advantages(
        rewards,
        group_ids,
        response_mask,
        config=config,
    )
    data.batch["advantages"] = advantages
    data.batch["returns"] = advantages
    return {
        "grpo/groups": float(len(group_rows(group_ids, advantages.shape[0]))),
    }


@torch.no_grad()
def apply_detached_oracle_advantage(
    data: Any,
    *,
    config: DetachedOracleConfig | None = None,
    reward_key: str | None = None,
    group_key: str = "uid",
    oracle_key: str = "is_oracle_row",
) -> dict[str, float]:
    """Overwrite oracle rows in a DataProto-like batch with detached anchors."""

    cfg = DetachedOracleConfig() if config is None else config
    advantages = data.batch["advantages"]
    response_mask = data.batch["response_mask"]
    sequence_values = sequence_advantages(advantages, response_mask)
    reward_tokens = batch_reward_tokens(data, reward_key)
    reward_values, _ = sequence_rewards(reward_tokens, response_mask)
    grouped = group_rows(data.non_tensor_batch[group_key], advantages.shape[0])
    oracle = boolean_mask(
        data.non_tensor_batch[oracle_key],
        advantages.shape[0],
        device=advantages.device,
        name=oracle_key,
    )
    updated = advantages.detach().clone()
    details: list[dict[str, float]] = []
    for rows in grouped.values():
        row_index = torch.tensor(rows, dtype=torch.long, device=advantages.device)
        group_oracle = oracle.index_select(0, row_index)
        if int(group_oracle.sum().item()) != 1:
            raise ValueError("each OraRL group requires exactly one oracle row.")
        policy = ~group_oracle
        group_rewards = reward_values.index_select(0, row_index)
        group_advantages = sequence_values.index_select(0, row_index)
        anchor, group_details = _detached_oracle_details(
            group_rewards[group_oracle].squeeze(0),
            group_rewards[policy],
            group_advantages[policy],
            cfg,
        )
        oracle_global = row_index[group_oracle]
        updated.index_copy_(
            0,
            oracle_global,
            broadcast_sequence_values(
                anchor.reshape(1),
                response_mask.index_select(0, oracle_global),
                dtype=advantages.dtype,
            ),
        )
        details.append(group_details)
    data.batch["advantages"] = updated
    metrics: dict[str, float] = {"orarl/oracle_groups": float(len(details))}
    _write_oracle_metrics(metrics, details)
    return metrics


@torch.no_grad()
def apply_orarl_advantages(
    data: Any,
    *,
    config: OraRLConfig | None = None,
    policy_reward_key: str | None = "token_level_rewards",
    oracle_reward_key: str | None = None,
    group_key: str = "uid",
    oracle_key: str = "is_oracle_row",
) -> dict[str, float]:
    """Write the complete pre-selection OraRL advantage into a batch."""

    cfg = OraRLConfig() if config is None else config
    response_mask = data.batch["response_mask"]
    policy_tokens = batch_reward_tokens(data, policy_reward_key)
    oracle_tokens = batch_reward_tokens(data, oracle_reward_key)
    policy_scores, _ = sequence_rewards(policy_tokens, response_mask)
    oracle_scores, _ = sequence_rewards(oracle_tokens, response_mask)
    group_ids = data.non_tensor_batch[group_key]
    grouped = group_rows(group_ids, policy_scores.numel())
    oracle = boolean_mask(
        data.non_tensor_batch[oracle_key],
        policy_scores.numel(),
        device=policy_scores.device,
        name=oracle_key,
    )
    metrics: dict[str, float] = {}
    sequence_output = _compute_orarl_from_scores(
        policy_scores,
        oracle_scores.to(
            device=policy_scores.device,
            dtype=policy_scores.dtype,
        ),
        grouped,
        oracle,
        cfg,
        metrics,
    )
    advantages = broadcast_sequence_values(
        sequence_output,
        response_mask,
        dtype=policy_tokens.dtype,
    )
    data.batch["advantages"] = advantages
    data.batch["returns"] = advantages
    return metrics
