# Copyright 2022 The HuggingFace Team
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import torch_functional as VF


if TYPE_CHECKING:
    from .config import AlgorithmConfig


class KLController(ABC):
    kl_coef: float
    """KL coefficient."""

    @abstractmethod
    def update(self, current_kl: float, n_steps: int):
        """Update kl_coef according to current KL."""
        ...


class AdaptiveKLController(KLController):
    """Adaptive KL controller described in: https://arxiv.org/pdf/1909.08593.pdf

    Copied from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L54"""

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: float):
        self.kl_coef = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.kl_coef *= mult


class FixedKLController(KLController):
    """Fixed KL controller.

    Copeid from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L72"""

    def __init__(self, init_kl_coef: float):
        self.kl_coef = init_kl_coef

    def update(self, current_kl: float, n_steps: int):
        pass


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    GRPO_PASSK = "grpo_passk"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"


ADV_ESTIMATOR_MAP: dict[str, Any] = {}


def get_kl_controller(algorithm_config: "AlgorithmConfig") -> KLController:
    """Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L319"""
    if algorithm_config.kl_type == "fixed":
        kl_ctrl = FixedKLController(init_kl_coef=algorithm_config.kl_coef)
    elif algorithm_config.kl_type == "adaptive":
        assert algorithm_config.kl_horizon > 0, f"horizon must be larger than 0. Got {algorithm_config.kl_horizon}."
        kl_ctrl = AdaptiveKLController(
            init_kl_coef=algorithm_config.kl_coef,
            target_kl=algorithm_config.kl_target,
            horizon=algorithm_config.kl_horizon,
        )
    else:
        raise ValueError(f"Unknown kl type: {algorithm_config.kl_type}.")

    return kl_ctrl


def register_adv_estimator(name: AdvantageEstimator):
    """Decorator to register a advantage estimator function with a given name."""

    def decorator(fn):
        wrapped_fn = torch.no_grad()(fn)
        ADV_ESTIMATOR_MAP[getattr(name, "value", name)] = wrapped_fn
        return wrapped_fn

    return decorator


def compute_advantage_return(name: AdvantageEstimator, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute advantage and return for a given advantage estimator."""
    return ADV_ESTIMATOR_MAP[getattr(name, "value", name)](**kwargs)


@register_adv_estimator(AdvantageEstimator.GAE)
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapted from https://github.com/huggingface/trl/blob/v0.16.0/trl/trainer/ppo_trainer.py#L513

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). The token after eos tokens have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    nextvalues = 0
    lastgaelam = 0
    advantages_reversed = []
    gen_len = token_level_rewards.shape[-1]
    for t in reversed(range(gen_len)):
        delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
        gaelam = delta + gamma * lam * lastgaelam

        if response_mask[:, t]:  # skip values and TD-error on observation tokens
            nextvalues = values[:, t]
            lastgaelam = gaelam

        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    advantages = VF.masked_whiten(advantages, response_mask)
    return advantages, returns


@register_adv_estimator(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward (with only one scalar reward for each response).

    ``scale_rewards=True`` uses standard GRPO whitening:
        A_i = (r_i - μ) / (σ + ε)
    ``scale_rewards=False`` uses raw-centered advantages:
        A_i = r_i - μ

    Oracle-aware statistics (active when ``is_oracle_row`` is supplied and
    ``oracle_excluded_baseline=True``):

        Both μ and σ come from the on-policy rollouts only. Excluding the
        oracle from μ keeps the baseline action-independent and an honest
        estimate of E_{y~π_θ}[R]; a perfect-reward oracle inside μ would push
        the best on-policy rollout to a negative advantage on hard prompts and
        corrupt any sign-aware selector. Excluding it from σ stops it from
        inflating the denominator and shrinking on-policy gradients.

        On-policy σ is numerically unsafe on its own: when every on-policy
        reward collapses to one value (typical when a shaping floor swallows
        all failures), σ_op is exactly 0 and the oracle advantage diverges.
        Groups with σ_op < SIGMA_OP_FALLBACK_THRESHOLD therefore fall back to
        the full-group σ, which stays positive because the oracle is a
        high-reward outlier.

    Args:
        token_level_rewards, response_mask, index, eps: standard.
        is_oracle_row (kwargs): np.ndarray[bool] of shape (bs,); True marks the
            annotation-derived oracle row.
        oracle_excluded_baseline (kwargs): build μ and σ from on-policy rows.
        directional_gain (kwargs): scale on-policy utilities by the clipped
            oracle-gap ratio (σ_g/σ_op)^γ.
        directional_gain_recenter (kwargs): after a positive-only gain,
            subtract the transformed on-policy group mean, preserving the
            directional preference while restoring a zero-mean group.
    """
    is_oracle_row = kwargs.get("is_oracle_row", None)
    scale_rewards = bool(kwargs.get("scale_rewards", True))
    exclude_oracle = bool(kwargs.get("oracle_excluded_baseline", False))
    # The oracle-gap gain multiplies the base advantage by
    # clip((σ_g/σ_op)^γ, 1, 4). σ_g includes the oracle and therefore measures
    # how far it lies outside the on-policy reward distribution, so the gain
    # grows exactly on the groups the policy has not solved. It requires the
    # oracle-excluded baseline μ_op; σ_op stays available even in raw mode.
    use_gain = bool(kwargs.get("directional_gain", False))
    gain_gamma = float(kwargs.get("directional_gain_gamma", 0.25))
    gain_pos_only = bool(kwargs.get("directional_gain_positive_only", False))
    gain_recenter = bool(kwargs.get("directional_gain_recenter", False))
    if use_gain:
        exclude_oracle = True
    # Cap (σ_g/σ_op)^γ so a near-degenerate σ_op cannot blow the gain up.
    GAIN_MAX: float = 4.0

    scores = token_level_rewards.sum(dim=-1)
    bsz = scores.shape[0]

    # Build two per-group score lists in a single pass:
    #   id2score_full = ALL rows (σ_op fallback source, and vanilla GRPO
    #                   mean/std when exclude_oracle=False).
    #   id2score_op   = on-policy only (μ and σ source when exclude_oracle).
    id2score_full: dict[Any, list[torch.Tensor]] = defaultdict(list)
    id2score_op: dict[Any, list[torch.Tensor]] = defaultdict(list)
    for i in range(bsz):
        id2score_full[index[i]].append(scores[i])
        if exclude_oracle and is_oracle_row is not None and bool(is_oracle_row[i]):
            continue
        id2score_op[index[i]].append(scores[i])

    # 1e-3 is far below any realistic reward std (≥ ~0.05 even on very hard
    # prompts), so the σ_g fallback only fires on truly degenerate groups.
    SIGMA_OP_FALLBACK_THRESHOLD: float = 1e-3
    sigma_op_fallback_count = 0

    id2mean: dict[Any, torch.Tensor] = {}
    id2std: dict[Any, torch.Tensor] = {}
    # Per-group gain g_idx = clip((σ_g/σ_op)^γ, ≤ GAIN_MAX), applied to
    # on-policy rows only. Empty unless use_gain.
    id2gain: dict[Any, torch.Tensor] = {}
    for idx, full_lst in id2score_full.items():
        if not exclude_oracle:
            assert len(full_lst) > 1, "GRPO needs rollout.n > 1."
            id2mean[idx] = torch.mean(torch.tensor(full_lst))
            id2std[idx] = torch.std(torch.tensor(full_lst))
            continue

        op_lst = id2score_op.get(idx, [])
        # ---- Mean source: on-policy ONLY (with edge-case fallbacks) ----
        if len(op_lst) == 0:
            # Pathological: the group holds only the oracle row. The append
            # pipeline keeps all n on-policy rollouts, so this cannot happen.
            id2mean[idx] = torch.zeros((), dtype=scores.dtype, device=scores.device)
        elif len(op_lst) == 1:
            # Single on-policy row → its own mean → its own advantage ≈ 0.
            id2mean[idx] = op_lst[0]
        else:
            id2mean[idx] = torch.mean(torch.tensor(op_lst))

        # ---- Std sources ----
        # σ_g (full group, oracle included) is always computed: it is the
        # degenerate-group fallback and also feeds the oracle-gap gain. σ_op
        # (on-policy only) is the base whitening std.
        sigma_g = (
            torch.std(torch.tensor(full_lst))
            if len(full_lst) >= 2
            else torch.zeros((), dtype=scores.dtype, device=scores.device)
        )
        sigma_op = torch.std(torch.tensor(op_lst)) if len(op_lst) >= 2 else None
        op_degenerate = sigma_op is None or float(sigma_op.item()) < SIGMA_OP_FALLBACK_THRESHOLD

        if op_degenerate:
            id2std[idx] = sigma_g
            sigma_op_fallback_count += 1
        else:
            id2std[idx] = sigma_op

        # ---- Oracle-gap gain (σ_g/σ_op)^γ ----
        # Scaled mode neutralizes the gain when σ_op degenerates, because its
        # whitening denominator has already fallen back to σ_g. Raw mode can
        # follow the clipped-gain formula even for tiny σ_op: eps and the upper
        # cap keep g finite, while the unwhitened centered reward still tends to
        # zero with σ_op.
        if use_gain:
            if op_degenerate and scale_rewards:
                id2gain[idx] = torch.ones((), dtype=scores.dtype, device=scores.device)
            else:
                sigma_op_for_gain = (
                    sigma_op
                    if sigma_op is not None
                    else torch.zeros((), dtype=scores.dtype, device=scores.device)
                )
                ratio = sigma_g / (sigma_op_for_gain + eps)
                id2gain[idx] = torch.clamp(
                    ratio ** gain_gamma,
                    min=1.0,
                    max=GAIN_MAX,
                )

    gain_on_policy = 0
    gain_amplified = 0
    for i in range(bsz):
        m = id2mean.get(index[i])
        s = id2std.get(index[i])
        if m is None:
            continue
        centered_reward = scores[i] - m
        adv = centered_reward / (s + eps) if scale_rewards else centered_reward
        if use_gain:
            g = id2gain.get(index[i], None)
            # Amplify ON-POLICY rows only. The oracle row keeps its raw
            # (r_oracle-μ_op)/σ_op here; it is overwritten downstream by the
            # detached anchor. When positive_only, amplify only improving
            # (adv > 0, toward-oracle) rows; negatives stay at base scale.
            if g is not None and not (is_oracle_row is not None and bool(is_oracle_row[i])):
                gain_on_policy += 1
                if (not gain_pos_only) or (adv > 0):
                    adv = adv * g
                    gain_amplified += 1
        scores[i] = adv

    # Positive-only gain is an asymmetric utility transform. Without this
    # second baseline, each on-policy group has a positive sum
    #   (g - 1) * sum(max(A_base, 0)),
    # which stacks an unconditional positive bias on top of oracle anchoring and
    # sign-balanced selection. Re-centering preserves ranking and the boosted
    # positive-vs-negative margin while restoring mean_op(A)=0.
    recenter_shifts: list[float] = []
    if use_gain and gain_recenter:
        id2transformed_op: dict[Any, list[torch.Tensor]] = defaultdict(list)
        for i in range(bsz):
            if is_oracle_row is not None and bool(is_oracle_row[i]):
                continue
            id2transformed_op[index[i]].append(scores[i])

        id2recenter_shift: dict[Any, torch.Tensor] = {}
        for idx, transformed in id2transformed_op.items():
            if transformed:
                shift = torch.mean(torch.stack(transformed))
                id2recenter_shift[idx] = shift
                recenter_shifts.append(abs(float(shift.item())))

        for i in range(bsz):
            if is_oracle_row is not None and bool(is_oracle_row[i]):
                continue
            shift = id2recenter_shift.get(index[i])
            if shift is not None:
                scores[i] = scores[i] - shift

    # Stash the fallback count on a side-channel kwarg if the caller passed an
    # empty dict for telemetry; this is opt-in to keep the function API clean.
    telemetry = kwargs.get("_telemetry_out", None)
    if telemetry is not None:
        telemetry["scale_rewards"] = float(scale_rewards)
        telemetry["sigma_op_fallback_count"] = sigma_op_fallback_count
        if use_gain and id2gain:
            gains = [float(g.item()) for g in id2gain.values()]
            telemetry["directional_gain_mean"] = sum(gains) / len(gains)
            telemetry["directional_gain_max"] = max(gains)
            telemetry["directional_gain_positive_only"] = float(gain_pos_only)
            telemetry["directional_gain_recenter"] = float(gain_recenter)
            if recenter_shifts:
                telemetry["directional_gain_recenter_abs_shift_mean"] = (
                    sum(recenter_shifts) / len(recenter_shifts)
                )
                telemetry["directional_gain_recenter_abs_shift_max"] = max(recenter_shifts)
            if gain_on_policy > 0:
                telemetry["directional_gain_amplified_frac"] = (
                    gain_amplified / gain_on_policy
                )

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_estimator(AdvantageEstimator.GRPO_PASSK)
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            shape: (bs,)
        eps: `(float)`
            epsilon value to avoid division by zero

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)
    advantages = torch.zeros_like(scores)
    id2score = defaultdict(list)
    id2indices = defaultdict(list)

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
        id2indices[index[i]].append(i)

    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        rewards = torch.tensor(id2score[idx])
        topk, topk_idx = torch.topk(rewards, k=2)
        r_max, r_second_max = topk[0], topk[1]
        i_max = id2indices[idx][topk_idx[0]]
        advantages[i_max] = (r_max - r_second_max) / (torch.std(torch.tensor(id2score[idx])) + eps)

    returns = advantages.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_estimator(AdvantageEstimator.RLOO)
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            shape: (bs,)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2sum = {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        id2sum[idx] = torch.sum(torch.tensor(id2score[idx]))

    for i in range(bsz):
        sample_num = len(id2score[index[i]])
        assert sample_num > 1, "RLOO needs rollout.n > 1."
        baseline = (id2sum[index[i]] - scores[i]) / (sample_num - 1)
        scores[i] = scores[i] - baseline

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_estimator(AdvantageEstimator.REINFORCE_PLUS_PLUS)
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    returns = torch.zeros_like(token_level_rewards)
    running_return = 0
    for t in reversed(range(token_level_rewards.shape[1])):
        running_return = token_level_rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
        # Reset after EOS
        running_return = running_return * response_mask[:, t]

    advantages = VF.masked_whiten(returns, response_mask)
    return advantages, returns


@register_adv_estimator(AdvantageEstimator.REMAX)
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    advantages = (token_level_rewards.sum(dim=-1) - reward_baselines) * response_mask
    returns = (token_level_rewards * response_mask).flip(dims=(-1,)).cumsum(dim=-1).flip(dims=(-1,))
    return advantages, returns


def compute_rewards(
    token_level_scores: torch.Tensor,
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_ratio: float,
) -> torch.Tensor:
    kl = log_probs - ref_log_probs
    return token_level_scores - kl * kl_ratio


def average_loss(
    values: torch.Tensor, mask: torch.Tensor, mode: Literal["token", "seq"], eps: float = 1e-8
) -> torch.Tensor:
    """Average the policy loss.

    Args:
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        mask: `(torch.Tensor)`
            shape: (bs, response_length)
        mode: `(Literal["token", "seq"])`
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means
        eps: `(float)`
            epsilon value

    Returns:
        loss: `a scalar torch.Tensor`
    """
    if mode == "token":
        return VF.masked_mean(values, mask, eps=eps)
    elif mode == "seq":
        return ((values * mask).sum(-1) / (mask.sum(-1) + eps)).mean()
    else:
        raise NotImplementedError(f"Unknown mode: {mode}.")


def compute_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_dual: float,
    loss_type: Literal["default", "gspo", "gspo_token", "cispo"],
    loss_avg_mode: Literal["token", "seq"],
    **kwargs,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the clipped policy objective and related metrics for PPO.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L568

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        clip_ratio_low: (float)
            The lower clip range used in PPO. See https://arxiv.org/abs/1707.06347
        clip_ratio_high: (float)
            The higher clip range used in DAPO. See https://arxiv.org/pdf/2503.14476
        clip_ratio_dual: (float)
            The dual clip range used in Dual-clip PPO. See https://arxiv.org/pdf/1912.09729
        loss_avg_mode: (Literal["token", "seq"])
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac_higher: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a higher value
        pg_clipfrac_lower: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a lower value
        ppo_kl: (float)
            a float number indicating the mean KL divergence between the old policy and the new policy
        entropy_loss: (float)
            a float number indicating the mean entropy loss

    """
    negative_approx_kl = log_probs - old_log_probs
    if loss_type in ["gspo", "gspo_token"]:
        # compute sequence-level importance ratio
        negative_approx_kl_in_seq = VF.masked_mean(negative_approx_kl, response_mask, dim=-1)
        # combined ratio at token level
        if loss_type == "gspo_token":
            log_importance_ratio = negative_approx_kl_in_seq.detach().unsqueeze(-1) + log_probs - log_probs.detach()
        else:
            log_importance_ratio = negative_approx_kl_in_seq.unsqueeze(-1) * response_mask
    else:
        log_importance_ratio = negative_approx_kl

    # clamp the ratio before exp to avoid nan grad
    # see: https://github.com/pytorch/pytorch/issues/10729
    ratio = torch.exp(torch.clamp(log_importance_ratio, -20.0, 20.0))
    clipped_ratio = torch.exp(
        torch.clamp(log_importance_ratio, np.log(1.0 - clip_ratio_low), np.log(1.0 + clip_ratio_high))
    )

    # pg metrics
    metrics = {"ppo_kl": -negative_approx_kl}
    # use negative log probs as an estimator of entropy loss
    metrics["entropy_loss"] = average_loss(-log_probs, response_mask, mode=loss_avg_mode)

    if loss_type == "cispo":
        final_pg_loss = -advantages * log_probs * clipped_ratio.detach()
    else:
        pg_loss = -advantages * ratio  # -ratio * A
        pg_loss2 = -advantages * clipped_ratio  # -clip(ratio, 1-clip_low, 1+clip_high) * A
        pg_loss3 = -advantages * clip_ratio_dual  # -clip_dual * A

        clipped_pg_loss_higher = torch.max(pg_loss, pg_loss2)  # clip if pg_loss < pg_loss2
        metrics["pg_clipfrac_higher"] = (pg_loss < pg_loss2).float()
        clipped_pg_loss_lower = torch.min(clipped_pg_loss_higher, pg_loss3)  # clip if pg_loss > pg_loss3 and adv < 0
        final_pg_loss = torch.where(advantages < 0, clipped_pg_loss_lower, clipped_pg_loss_higher)
        metrics["pg_clipfrac_lower"] = (clipped_pg_loss_higher > pg_loss3).float() * (advantages < 0).float()

    final_pg_loss = average_loss(final_pg_loss, response_mask, mode=loss_avg_mode)
    metrics = {k: VF.masked_mean(v, response_mask).detach().item() for k, v in metrics.items()}
    return final_pg_loss, metrics


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_avg_mode: Literal["token", "seq"],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the value loss.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L556

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange_value: (float)
            The clip range for value net used in PPO. See https://arxiv.org/abs/1707.06347
        loss_avg_mode: (Literal["token", "seq"])
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped
        vpred_mean: a float
            The mean of predicted values

    """
    vpredclipped = torch.clamp(vpreds, values - cliprange_value, values + cliprange_value)
    vf_loss1 = torch.square(vpreds - returns)
    vf_loss2 = torch.square(vpredclipped - returns)
    clipped_vf_losses = torch.max(vf_loss1, vf_loss2)  # clip if vf_loss1 < vf_loss2
    vf_loss = 0.5 * average_loss(clipped_vf_losses, response_mask, mode=loss_avg_mode)
    metrics = {
        "vf_clipfrac": VF.masked_mean((vf_loss1 < vf_loss2).float(), response_mask).detach().item(),
        "vpred_mean": VF.masked_mean(vpreds, response_mask).detach().item(),
    }
    return vf_loss, metrics


def compute_kl(
    log_probs: torch.FloatTensor,
    ref_log_probs: torch.FloatTensor,
    kl_penalty: Literal["kl", "abs", "mse", "low_var_kl", "full"],
) -> torch.Tensor:
    """Compute KL divergence given log_probs and ref_log_probs.

    Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L1150

    Args:
        log_probs: torch.Tensor
        ref_log_probs: torch.Tensor
        kl_penalty: str ("kl", "abs", "mse", "low_var_kl", "full")

    Returns:
        kl_div: torch.Tensor

    """
    log_probs, ref_log_probs = log_probs.float(), ref_log_probs.float()
    if kl_penalty == "kl":
        return log_probs - ref_log_probs

    if kl_penalty == "abs":
        return (log_probs - ref_log_probs).abs()

    if kl_penalty == "mse":
        return 0.5 * (log_probs - ref_log_probs).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # URL http://joschu.net/blog/kl-approx.html
    if kl_penalty == "low_var_kl":
        # For numerical stability
        kl = (ref_log_probs - log_probs).clamp(-20.0, 20.0)
        kld = (kl.exp() - kl - 1).contiguous()
        return torch.clamp(kld, min=-10.0, max=10.0)

    if kl_penalty == "full":
        return F.kl_div(ref_log_probs, log_probs, log_target=True, reduction="none").sum(-1)

    raise NotImplementedError(f"Unknown KL penalty: {kl_penalty}.")
