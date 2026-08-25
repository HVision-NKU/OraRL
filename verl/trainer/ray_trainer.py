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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import importlib
import json
import math
import os
import time
import uuid
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt, thin_out_old_ckpts
from ..utils.logger import Tracker
from ..utils.multimodal_contract import validate_multi_modal_data_contract
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .orarl_post_selection import (
    PostSelectionReference,
    balance_post_selection_group,
)
from .orarl_selection import select_orarl_rollouts
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from .orarl_config import (
    is_orarl,
    validate_algorithm,
)


def _disable_tqdm() -> bool:
    return os.getenv("VERL_DISABLE_TQDM", "0") == "1"


def _print_step_summary_enabled() -> bool:
    return os.getenv("VERL_PRINT_STEP_SUMMARY", "1") == "1"


def _skip_old_log_probs_enabled() -> bool:
    return os.getenv("VERL_SKIP_OLD_LOGPROBS", "0") == "1"


class _NoOpProgress:
    def update(self, *args, **kwargs) -> None:
        pass


def _fmt_metric(value: Any, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes}m{seconds:02d}s"


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def _load_dotted_callable(spec: str):
    """Resolve a 'pkg.mod:attr' string into the callable it points to."""
    if ":" not in spec:
        raise ValueError(
            f"Expected a 'module.path:function' style spec, got {spec!r}. "
            "Example: 'orarl.rewards:build_oracle_response_from_ground_truth'."
        )
    module_path, attr = spec.split(":", 1)
    module = importlib.import_module(module_path)
    try:
        fn = getattr(module, attr)
    except AttributeError as err:
        raise AttributeError(f"Module {module_path!r} has no attribute {attr!r}.") from err
    if not callable(fn):
        raise TypeError(f"{spec!r} resolved to a non-callable object: {type(fn).__name__}")
    return fn


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards."""
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    scale_rewards: bool = True,
    directional_gain: bool = False,
    directional_gain_gamma: float = 0.25,
    directional_gain_positive_only: bool = False,
    directional_gain_recenter: bool = False,
):
    """Compute advantage estimates for policy optimization.

    ``scale_rewards=False`` selects raw-centered GRPO advantages ``r-mean``
    without per-group standard-deviation whitening.

    When ``is_oracle_row`` exists in ``data.non_tensor_batch`` the GRPO-family
    estimator keeps the oracle out of the baseline: the per-group mean is built
    over on-policy rollouts only, so a perfect-reward oracle cannot suppress a
    genuine on-policy winner. The standard deviation is also on-policy only, so
    the oracle cannot inflate it and shrink on-policy gradients. Has no effect
    for non-GRPO estimators or for runs without oracle rows.
    """
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
        "scale_rewards": bool(scale_rewards),
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    telemetry: dict[str, Any] = {}
    is_oracle_row = data.non_tensor_batch.get("is_oracle_row", None)
    if is_oracle_row is not None:
        adv_inputs["is_oracle_row"] = is_oracle_row
        adv_inputs["oracle_excluded_baseline"] = True
        if directional_gain:
            adv_inputs["directional_gain"] = True
            adv_inputs["directional_gain_gamma"] = float(directional_gain_gamma)
            adv_inputs["directional_gain_positive_only"] = bool(
                directional_gain_positive_only
            )
            adv_inputs["directional_gain_recenter"] = bool(directional_gain_recenter)
        adv_inputs["_telemetry_out"] = telemetry

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    if telemetry:
        # Stash so the training loop can merge into per-step metrics.
        data.meta_info = dict(data.meta_info)
        data.meta_info["_advantage_telemetry"] = telemetry
    return data


@torch.no_grad()
def compute_oracle_advantage_diagnostics(
    data: DataProto,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Per-step diagnostics for the oracle-excluded advantage baseline.

    Returns ``orarl/*`` metrics quantifying how much the oracle row would
    contaminate a vanilla GRPO baseline and how much the oracle-excluded
    statistics recover. No-op without oracle rows (returns {}).

    Metrics emitted (per-batch averages over groups unless noted):

    Per-group statistics (reward space):
      - orarl/mu_op_mean         mean of μ_op (on-policy mean) across groups
      - orarl/mu_g_mean          mean of μ_g  (full-group mean)
      - orarl/sigma_op_mean      mean of σ_op
      - orarl/sigma_g_mean       mean of σ_g
      - orarl/sigma_ratio_mean   mean of σ_g / σ_op (>1 ↔ oracle far outside
                                the on-policy reward distribution)
      - orarl/oracle_mean_contamination
                                |μ_g - μ_op| / σ_op — how far the oracle would
                                shift the baseline; high values flag the groups
                                where excluding it matters most

    Recovery counts, computed against the current ``data.batch['advantages']``:
      - orarl/sign_flip_count    on-policy rows whose advantage sign differs
                                from what vanilla GRPO (μ_g, σ_g) would give —
                                the rows the oracle-excluded baseline rescued
      - orarl/sign_flip_rate     sign_flip_count / total_op_rows
      - orarl/groups_with_no_op_pos_baseline
                                groups where vanilla GRPO would leave every
                                on-policy advantage ≤ 0, so a sign-balanced
                                selector would have no positive policy row

    Magnitudes (post-transform advantages):
      - orarl/oracle_adv_mag_mean / oracle_adv_mag_max   |A_oracle|
      - orarl/op_adv_pos_mean   mean |A_op| over on-policy positives
      - orarl/op_adv_neg_mean   mean |A_op| over on-policy negatives

    Use these to confirm (a) |A_oracle| stays moderate (e.g. < 5) so PPO
    clipping does not eat the oracle signal, and (b) sign_flip_rate is
    non-trivial, which means the oracle-excluded baseline is doing real work.
    """
    is_oracle_row_np = data.non_tensor_batch.get("is_oracle_row", None)
    if is_oracle_row_np is None:
        return {}

    response_mask = data.batch["response_mask"]
    token_level_rewards = data.batch["token_level_rewards"]
    advantages = data.batch["advantages"]

    scores = (token_level_rewards * response_mask).sum(dim=-1)  # (bsz,)
    mask_sum = response_mask.sum(dim=-1).clamp(min=1)
    adv_signed = (advantages * response_mask).sum(dim=-1) / mask_sum  # (bsz,)

    bsz = scores.shape[0]
    is_oracle = torch.tensor(
        [bool(is_oracle_row_np[i]) for i in range(bsz)],
        dtype=torch.bool, device=scores.device,
    )

    uids = data.non_tensor_batch["uid"]
    uid_to_idx: dict[Any, list[int]] = defaultdict(list)
    for i, uid in enumerate(uids):
        uid_to_idx[uid].append(i)

    mu_op_acc, mu_g_acc = [], []
    sigma_op_acc, sigma_g_acc = [], []
    # Robust ratio aggregation: collect ratios ONLY from groups with non-trivial
    # sigma_op (above SIGMA_FLOOR), so a degenerate group with σ_op≈0 doesn't
    # blow up the per-batch average. Also report the unweighted mean(σ_g) /
    # mean(σ_op) as a stable companion.
    SIGMA_FLOOR = 1e-3
    contam_acc, sigma_ratio_acc = [], []
    degenerate_op_groups = 0
    sign_flip = 0
    total_op = 0
    no_op_pos_groups = 0
    n_groups = 0
    oracle_mags, op_pos_mags, op_neg_mags = [], [], []

    for indices in uid_to_idx.values():
        if len(indices) < 2:
            continue
        idx_t = torch.tensor(indices, dtype=torch.long, device=scores.device)
        g_scores = scores.index_select(0, idx_t)
        g_adv = adv_signed.index_select(0, idx_t)
        g_oracle_mask = is_oracle.index_select(0, idx_t)
        op_mask = ~g_oracle_mask

        if op_mask.sum() < 1:
            continue

        op_scores = g_scores[op_mask]
        oracle_scores = g_scores[g_oracle_mask]

        mu_op = op_scores.mean()
        mu_g = g_scores.mean()
        sigma_op = (op_scores.std(unbiased=False)
                    if op_scores.numel() > 1 else torch.zeros_like(mu_op))
        sigma_g = g_scores.std(unbiased=False)

        mu_op_acc.append(mu_op.item())
        mu_g_acc.append(mu_g.item())
        sigma_op_v = float(sigma_op.item())
        sigma_g_v = float(sigma_g.item())
        sigma_op_acc.append(sigma_op_v)
        sigma_g_acc.append(sigma_g_v)
        if sigma_op_v < SIGMA_FLOOR:
            # σ_op too tiny → ratio/contamination divergent; skip from
            # per-group ratio averages but still count it.
            degenerate_op_groups += 1
        else:
            contam_acc.append(abs(float(mu_g.item()) - float(mu_op.item())) / sigma_op_v)
            sigma_ratio_acc.append(sigma_g_v / sigma_op_v)

        # Hypothetical baseline advantages (always full-group μ, σ) for the
        # on-policy rows — used to count sign-flips vs the actual variant.
        baseline_op_adv = (op_scores - mu_g) / (sigma_g + eps)
        actual_op_adv = g_adv[op_mask]

        sign_flips_mask = (
            ((actual_op_adv > 0) & (baseline_op_adv < 0))
            | ((actual_op_adv < 0) & (baseline_op_adv > 0))
        )
        sign_flip += int(sign_flips_mask.sum().item())
        total_op += int(op_mask.sum().item())

        if (baseline_op_adv <= 0).all():
            no_op_pos_groups += 1

        op_abs = actual_op_adv.abs()
        op_pos_mags.extend(op_abs[actual_op_adv > 0].tolist())
        op_neg_mags.extend(op_abs[actual_op_adv < 0].tolist())

        if g_oracle_mask.any():
            oracle_mags.extend(g_adv[g_oracle_mask].abs().tolist())

        n_groups += 1

    if n_groups == 0:
        return {}

    mean_sigma_op = sum(sigma_op_acc) / n_groups if sigma_op_acc else 0.0
    mean_sigma_g = sum(sigma_g_acc) / n_groups if sigma_g_acc else 0.0
    mean_mu_op = sum(mu_op_acc) / n_groups if mu_op_acc else 0.0
    mean_mu_g = sum(mu_g_acc) / n_groups if mu_g_acc else 0.0

    out: dict[str, float] = {
        "orarl/mu_op_mean": mean_mu_op,
        "orarl/mu_g_mean": mean_mu_g,
        "orarl/sigma_op_mean": mean_sigma_op,
        "orarl/sigma_g_mean": mean_sigma_g,
        # Two ratio summaries — different aggregation:
        #  - *_per_group: average of per-group ratios (more sensitive but
        #    requires non-degenerate σ_op; we already filter < SIGMA_FLOOR).
        #  - *_of_means:  ratio of per-batch averages (always finite, more
        #    stable for swanlab plots; use this if per_group is noisy).
        "orarl/sigma_ratio_per_group": (
            sum(sigma_ratio_acc) / max(len(sigma_ratio_acc), 1)
            if sigma_ratio_acc else 0.0
        ),
        "orarl/sigma_ratio_of_means": (
            mean_sigma_g / mean_sigma_op if mean_sigma_op > 0 else 0.0
        ),
        "orarl/oracle_mean_contamination_per_group": (
            sum(contam_acc) / max(len(contam_acc), 1) if contam_acc else 0.0
        ),
        "orarl/oracle_mean_contamination_of_means": (
            abs(mean_mu_g - mean_mu_op) / mean_sigma_op if mean_sigma_op > 0 else 0.0
        ),
        "orarl/degenerate_op_groups": float(degenerate_op_groups),
        "orarl/sign_flip_count": float(sign_flip),
        "orarl/sign_flip_rate": sign_flip / max(total_op, 1),
        "orarl/groups_with_no_op_pos_baseline": float(no_op_pos_groups),
        "orarl/n_groups": float(n_groups),
    }
    if oracle_mags:
        out["orarl/oracle_adv_mag_mean"] = sum(oracle_mags) / len(oracle_mags)
        out["orarl/oracle_adv_mag_max"] = max(oracle_mags)
    if op_pos_mags:
        out["orarl/op_adv_pos_mean"] = sum(op_pos_mags) / len(op_pos_mags)
        out["orarl/op_adv_pos_count"] = float(len(op_pos_mags))
    if op_neg_mags:
        out["orarl/op_adv_neg_mean"] = sum(op_neg_mags) / len(op_neg_mags)
        out["orarl/op_adv_neg_count"] = float(len(op_neg_mags))

    return out


@torch.no_grad()
def apply_detached_oracle_advantage(
    data: DataProto,
    scale: float = 1.0,
    beta: float = 1.0,
    directional_gain_gamma: float = 0.0,
    directional_gain_max: float = 4.0,
    match_best_ratio: float = 0.0,
    match_best_min: float = 0.5,
    match_best_max: float = 2.0,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Overwrite each oracle row's advantage with a detached positive anchor.

        A_oracle_raw = scale * w_g * gain_g
        w_g    = clip((r_oracle - mean_op) / (r_oracle + eps), 0, 1) ** beta
        gain_g = clip((σ_g / σ_op) ** directional_gain_gamma, ≤ directional_gain_max)
               = 1                                              (γ=0)

    Optional best-on-policy matching (``match_best_ratio > 0``) keeps the
    detached oracle row from dominating the contrastive policy rows:

        cap_g    = clip(match_best_ratio * max(A_op positive), match_best_min, match_best_max)
        A_oracle = min(A_oracle_raw, cap_g)

    The cap comes from the already-normalized, post-gain on-policy advantages in
    the same group. When a group has no positive on-policy advantage,
    ``match_best_min`` acts as a small bootstrap anchor rather than suppressing
    the oracle to zero.

    ``r_oracle`` is the oracle row's reward and ``mean_op`` the group's on-policy
    mean reward (both summed ``token_level_scores``). Setting rather than scaling
    the advantage means the oracle never reaches the policy rows through the
    whitening denominator: combined with the on-policy-only σ, a high-reward
    oracle can no longer inflate σ and shrink on-policy gradients.

    The anchor stays positive with a magnitude that tracks how far on-policy is
    from the oracle along two axes:
      * reward gap ``w_g`` — →0 on solved groups, →1 on hard groups;
      * variance gap ``gain_g`` — the same oracle-gap factor ``(σ_g/σ_op)^γ``
        that scales the on-policy toward-oracle push. A positive
        ``directional_gain_gamma`` puts the anchor on that same pre-cap scale.
        σ_g is the full-group std, σ_op the on-policy std; gain_g falls back to
        1 on a degenerate σ_op group.

    Mutates ``data.batch['advantages']`` in place, oracle rows only. No-op
    without ``is_oracle_row``. Emits per-step and per-task telemetry.
    """
    SIGMA_OP_FALLBACK_THRESHOLD: float = 1e-3
    is_oracle_row_np = data.non_tensor_batch.get("is_oracle_row", None)
    if is_oracle_row_np is None:
        return {}

    # Prefer the pure reward (pre-KL) for gating; fall back to the KL-adjusted one.
    if "token_level_scores" in data.batch:
        reward_tok = data.batch["token_level_scores"]
    else:
        reward_tok = data.batch["token_level_rewards"]
    advantages = data.batch["advantages"]
    response_mask = data.batch["response_mask"]

    scores = (reward_tok * response_mask).sum(dim=-1)  # (bsz,)
    bsz = scores.shape[0]
    device = scores.device
    is_oracle = torch.tensor(
        [bool(is_oracle_row_np[i]) for i in range(bsz)], dtype=torch.bool, device=device
    )

    uids = data.non_tensor_batch["uid"]
    problem_types = data.non_tensor_batch.get("problem_type", None)
    uid_to_idx: dict[Any, list[int]] = defaultdict(list)
    for i, uid in enumerate(uids):
        uid_to_idx[uid].append(i)

    w_acc: list[float] = []
    a_oracle_acc: list[float] = []
    gain_acc: list[float] = []
    raw_a_oracle_acc: list[float] = []
    best_op_adv_acc: list[float] = []
    cap_acc: list[float] = []
    cap_applied = 0
    per_task_w: dict[str, list[float]] = defaultdict(list)
    n_oracle_groups = 0
    use_gain = directional_gain_gamma > 0.0
    use_best_match = match_best_ratio > 0.0

    for indices in uid_to_idx.values():
        idx_t = torch.tensor(indices, dtype=torch.long, device=device)
        g_oracle = is_oracle.index_select(0, idx_t)
        if not bool(g_oracle.any()):
            continue
        op_mask = ~g_oracle
        if int(op_mask.sum().item()) < 1:
            continue
        g_scores = scores.index_select(0, idx_t)
        r_oracle = float(g_scores[g_oracle].max().item())
        op_scores = g_scores[op_mask]
        mean_op = float(op_scores.mean().item())
        w = (r_oracle - mean_op) / (r_oracle + eps)
        w = max(0.0, min(1.0, w))
        if beta != 1.0:
            w = float(w ** beta)
        # Directional gain (σ_g/σ_op)^γ — the SAME factor that scales the
        # on-policy toward-oracle push, so the anchor rides on the same scale.
        gain = 1.0
        if use_gain and int(op_mask.sum().item()) >= 2:
            sigma_op = float(op_scores.std().item())
            sigma_g = float(g_scores.std().item()) if g_scores.numel() >= 2 else 0.0
            if sigma_op >= SIGMA_OP_FALLBACK_THRESHOLD:
                gain = max(
                    1.0,
                    min(
                        (sigma_g / (sigma_op + eps)) ** directional_gain_gamma,
                        directional_gain_max,
                    ),
                )
        raw_a_oracle = float(scale) * w * gain
        a_oracle = raw_a_oracle
        if use_best_match:
            group_advantages = advantages.index_select(0, idx_t)
            group_response_mask = response_mask.index_select(0, idx_t)
            op_advantages = group_advantages[op_mask]
            op_response_mask = group_response_mask[op_mask].to(group_advantages.dtype)
            op_lengths = op_response_mask.sum(dim=-1).clamp_min(1.0)
            op_sequence_advantages = (op_advantages * op_response_mask).sum(dim=-1) / op_lengths
            positive_advantages = op_sequence_advantages[op_sequence_advantages > 0]
            best_op_adv = (
                float(positive_advantages.max().item())
                if positive_advantages.numel() > 0
                else 0.0
            )
            target_cap = (
                float(match_best_ratio) * best_op_adv
                if best_op_adv > 0.0
                else float(match_best_min)
            )
            target_cap = max(float(match_best_min), min(float(match_best_max), target_cap))
            a_oracle = min(raw_a_oracle, target_cap)
            best_op_adv_acc.append(best_op_adv)
            cap_acc.append(target_cap)
            cap_applied += int(a_oracle < raw_a_oracle)
        for j in indices:
            if bool(is_oracle[j]):
                advantages[j] = a_oracle * response_mask[j].to(advantages.dtype)
        w_acc.append(w)
        a_oracle_acc.append(a_oracle)
        raw_a_oracle_acc.append(raw_a_oracle)
        gain_acc.append(gain)
        n_oracle_groups += 1
        if problem_types is not None:
            per_task_w[str(problem_types[indices[0]])].append(w)

    if n_oracle_groups == 0:
        return {}

    data.batch["advantages"] = advantages

    out: dict[str, float] = {
        "orarl/detached_w_mean": float(sum(w_acc) / len(w_acc)),
        "orarl/detached_a_oracle_mean": float(sum(a_oracle_acc) / len(a_oracle_acc)),
        "orarl/detached_a_oracle_raw_mean": float(sum(raw_a_oracle_acc) / len(raw_a_oracle_acc)),
        "orarl/detached_gain_mean": float(sum(gain_acc) / len(gain_acc)),
        "orarl/detached_groups": float(n_oracle_groups),
    }
    if use_best_match:
        out.update({
            "orarl/detached_best_op_adv_mean": float(
                sum(best_op_adv_acc) / len(best_op_adv_acc)
            ),
            "orarl/detached_match_cap_mean": float(sum(cap_acc) / len(cap_acc)),
            "orarl/detached_match_cap_applied_frac": float(cap_applied) / float(n_oracle_groups),
        })
    for task, ws in per_task_w.items():
        if ws:
            out[f"orarl/detached_w_mean_by_task/{task}"] = float(sum(ws) / len(ws))
    return out


# Per-row scalar metric keys to mirror from reward_metrics into
# `batch.non_tensor_batch` so downstream consumers can read them at the rollout
# level. ``iou_raw`` is the un-shaped IoU, the only reward component strictly
# aligned with the evaluation metric.
_ROW_LEVEL_REWARD_KEYS_TO_PROPAGATE: tuple[str, ...] = ("iou_raw",)


def _propagate_per_row_reward_metrics(
    batch: DataProto,
    reward_metrics: dict[str, list[float]],
    keys: tuple[str, ...] = _ROW_LEVEL_REWARD_KEYS_TO_PROPAGATE,
) -> None:
    """Write selected per-row reward components into ``batch.non_tensor_batch``.

    ``reward_metrics`` is a dict-of-lists in row order (length == bsz). For
    keys present in ``reward_metrics`` we materialize an ndarray on the
    batch's non_tensor_batch so subsequent batch slicing and selection
    naturally keep the per-row alignment. Idempotent (skips keys already
    present).
    """
    bsz = len(batch)
    for key in keys:
        if key in batch.non_tensor_batch:
            continue
        values = reward_metrics.get(key)
        if values is None:
            continue
        if len(values) != bsz:
            continue
        arr = np.asarray(values, dtype=np.float32)
        # Replace NaN/Inf with 0 so downstream torch ops never explode.
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        batch.non_tensor_batch[key] = arr


def _sequence_advantage_scores(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Collapse token advantages without making selection response-length biased.

    GRPO broadcasts one scalar advantage over every valid response token. A
    masked sum therefore multiplies that scalar by the response length and can
    rank a weaker, longer completion above a stronger, shorter completion.
    Masked mean recovers the original scalar and remains well-defined for
    estimators whose token advantages are not uniform.
    """
    if advantages.shape != response_mask.shape:
        raise ValueError(
            "advantages and response_mask must have identical shapes, got "
            f"{tuple(advantages.shape)} and {tuple(response_mask.shape)}."
        )
    return VF.masked_mean(advantages, response_mask, dim=-1)


@torch.no_grad()
def build_orarl_post_selection_references(
    data: DataProto,
) -> dict[Any, PostSelectionReference]:
    """Snapshot all policy rows before OraRL selection drops any of them."""
    required_batch_keys = ("advantages", "response_mask")
    missing = [key for key in required_batch_keys if key not in data.batch]
    if missing or "uid" not in data.non_tensor_batch:
        raise ValueError(
            "post-selection balance requires advantages, response_mask, and uid; "
            f"missing={missing + ([] if 'uid' in data.non_tensor_batch else ['uid'])}."
        )

    if "token_level_scores" in data.batch:
        reward_tokens = data.batch["token_level_scores"]
    elif "token_level_rewards" in data.batch:
        reward_tokens = data.batch["token_level_rewards"]
    else:
        raise ValueError(
            "post-selection balance requires token_level_scores or token_level_rewards."
        )

    advantages = data.batch["advantages"]
    response_mask = data.batch["response_mask"]
    sequence_advantages = _sequence_advantage_scores(
        advantages,
        response_mask,
    ).detach().float()
    sequence_rewards = (
        reward_tokens * response_mask.to(reward_tokens.dtype)
    ).sum(dim=-1).detach().float()
    uids = np.asarray(data.non_tensor_batch["uid"], dtype=object)
    is_oracle = np.asarray(
        data.non_tensor_batch.get(
            "is_oracle_row",
            np.zeros(sequence_advantages.shape[0], dtype=bool),
        ),
        dtype=bool,
    )
    if len(uids) != sequence_advantages.shape[0] or len(is_oracle) != len(uids):
        raise ValueError("uid and oracle flags must align with the batch rows.")

    uid_to_rows: dict[Any, list[int]] = defaultdict(list)
    for row, uid in enumerate(uids):
        uid_to_rows[uid].append(row)

    references: dict[Any, PostSelectionReference] = {}
    device = sequence_advantages.device
    for uid, rows in uid_to_rows.items():
        op_rows = [row for row in rows if not bool(is_oracle[row])]
        if not op_rows:
            continue
        op_idx = torch.tensor(op_rows, dtype=torch.long, device=device)
        op_advantages = sequence_advantages.index_select(0, op_idx)
        op_rewards = sequence_rewards.index_select(0, op_idx)
        on_policy_rms = float(torch.sqrt(torch.mean(op_advantages.square())).item())
        sigma_op = (
            float(op_rewards.std().item())
            if op_rewards.numel() >= 2
            else 0.0
        )
        references[uid] = PostSelectionReference(
            on_policy_rms=on_policy_rms,
            sigma_op=sigma_op,
            on_policy_rows=len(op_rows),
        )
    return references


@torch.no_grad()
def apply_orarl_post_selection_advantage_balance(
    data: DataProto,
    references: dict[Any, PostSelectionReference],
    *,
    recenter: bool,
    rms_match: bool,
    rms_min_scale: float,
) -> dict[str, float]:
    """Correct each OraRL-selected group before the actor backward pass."""
    if not recenter and not rms_match:
        return {}

    is_oracle_np = data.non_tensor_batch.get("is_oracle_row")
    if is_oracle_np is None:
        return {
            "orarl/post_selection_enabled": 1.0,
            "orarl/post_selection_recenter": float(recenter),
            "orarl/post_selection_rms_match": float(rms_match),
            "orarl/post_selection_groups": 0.0,
            "orarl/post_selection_groups_skipped": 0.0,
        }

    advantages = data.batch["advantages"]
    response_mask = data.batch["response_mask"]
    sequence_advantages = _sequence_advantage_scores(
        advantages,
        response_mask,
    ).detach().float()
    uids = np.asarray(data.non_tensor_batch["uid"], dtype=object)
    is_oracle = np.asarray(is_oracle_np, dtype=bool)
    if len(uids) != sequence_advantages.shape[0] or len(is_oracle) != len(uids):
        raise ValueError("uid and oracle flags must align with selected batch rows.")

    uid_to_rows: dict[Any, list[int]] = defaultdict(list)
    for row, uid in enumerate(uids):
        uid_to_rows[uid].append(row)

    metric_values: dict[str, list[float]] = defaultdict(list)
    processed_groups = 0
    skipped_groups = 0
    device = sequence_advantages.device
    for uid, rows in uid_to_rows.items():
        reference = references.get(uid)
        group_oracle = np.asarray(
            [bool(is_oracle[row]) for row in rows],
            dtype=bool,
        )
        oracle_rows = int(group_oracle.sum())
        if oracle_rows > 1:
            raise ValueError(
                f"post-selection group {uid!r} contains {oracle_rows} oracle "
                "rows; expected at most one."
            )
        no_actual_selection = (
            reference is not None
            and reference.on_policy_rows > 0
            and len(rows) >= reference.on_policy_rows + 1
        )
        if reference is None or oracle_rows == 0 or no_actual_selection:
            skipped_groups += 1
            continue

        idx_t = torch.tensor(rows, dtype=torch.long, device=device)
        active_advantages = sequence_advantages.index_select(0, idx_t)
        active_is_oracle = torch.tensor(
            group_oracle,
            dtype=torch.bool,
            device=device,
        )
        balanced, group_metrics = balance_post_selection_group(
            active_advantages,
            active_is_oracle,
            reference=reference,
            recenter=recenter,
            rms_match=rms_match,
            rms_min_scale=rms_min_scale,
        )

        balanced_tokens = (
            balanced.to(dtype=advantages.dtype).unsqueeze(-1)
            * response_mask.index_select(0, idx_t).to(advantages.dtype)
        )
        advantages.index_copy_(
            0,
            idx_t.to(device=advantages.device),
            balanced_tokens.to(device=advantages.device),
        )
        for key, value in group_metrics.items():
            metric_values[key].append(value)
        processed_groups += 1

    data.batch["advantages"] = advantages
    metrics: dict[str, float] = {
        "orarl/post_selection_enabled": 1.0,
        "orarl/post_selection_recenter": float(recenter),
        "orarl/post_selection_rms_match": float(rms_match),
        "orarl/post_selection_groups": float(processed_groups),
        "orarl/post_selection_groups_skipped": float(skipped_groups),
    }
    for key, values in metric_values.items():
        if values:
            metrics[f"orarl/post_selection_{key}"] = float(
                sum(values) / len(values)
            )
    return metrics


def _rollout_token_diversity_metrics(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    n_rollouts: int,
) -> dict[str, float]:
    """Measure whether an ``n``-sample rollout actually explores distinct text."""
    if responses.shape != response_mask.shape:
        raise ValueError("responses and response_mask must have identical shapes.")
    if n_rollouts <= 1 or responses.shape[0] % n_rollouts != 0:
        return {}

    group_unique_fractions: list[float] = []
    all_identical = 0
    pairwise_agreements: list[float] = []
    length_stds: list[float] = []
    for start in range(0, responses.shape[0], n_rollouts):
        group_ids = responses[start : start + n_rollouts]
        group_mask = response_mask[start : start + n_rollouts].bool()
        sequences = [
            tuple(group_ids[i][group_mask[i]].detach().cpu().tolist())
            for i in range(n_rollouts)
        ]
        unique_count = len(set(sequences))
        group_unique_fractions.append(unique_count / n_rollouts)
        all_identical += int(unique_count == 1)

        lengths = group_mask.sum(dim=-1).to(dtype=torch.float32)
        length_stds.append(float(lengths.std(unbiased=False).item()))
        for i in range(n_rollouts):
            for j in range(i + 1, n_rollouts):
                common = group_mask[i] & group_mask[j]
                if common.any():
                    agreement = (group_ids[i][common] == group_ids[j][common]).float().mean()
                    pairwise_agreements.append(float(agreement.item()))

    num_groups = len(group_unique_fractions)
    return {
        "rollout/exact_unique_fraction": float(sum(group_unique_fractions) / num_groups),
        "rollout/all_identical_group_fraction": float(all_identical / num_groups),
        "rollout/pairwise_token_agreement_mean": float(
            sum(pairwise_agreements) / max(1, len(pairwise_agreements))
        ),
        "rollout/response_length_std_mean": float(sum(length_stds) / num_groups),
    }


def _on_policy_reward_diversity_metrics(data: DataProto) -> dict[str, float]:
    """Report useful reward spread per prompt, excluding injected oracle rows."""
    if "token_level_scores" not in data.batch or "uid" not in data.non_tensor_batch:
        return {}
    scores = data.batch["token_level_scores"].sum(dim=-1).detach().cpu().float().tolist()
    uids = np.asarray(data.non_tensor_batch["uid"], dtype=object)
    is_oracle = np.asarray(
        data.non_tensor_batch.get("is_oracle_row", np.zeros(len(scores), dtype=bool)),
        dtype=bool,
    )
    grouped: dict[Any, list[float]] = defaultdict(list)
    for uid, score, oracle_row in zip(uids, scores, is_oracle):
        if not oracle_row:
            grouped[uid].append(float(score))

    stds: list[float] = []
    unique_fractions: list[float] = []
    for group_scores in grouped.values():
        if len(group_scores) <= 1:
            continue
        score_tensor = torch.tensor(group_scores, dtype=torch.float32)
        stds.append(float(score_tensor.std(unbiased=False).item()))
        unique_fractions.append(
            len({round(score, 7) for score in group_scores}) / len(group_scores)
        )
    if not stds:
        return {}
    return {
        "rollout/reward_group_std_mean": float(sum(stds) / len(stds)),
        "rollout/zero_reward_std_group_fraction": float(
            sum(std <= 1e-8 for std in stds) / len(stds)
        ),
        "rollout/reward_unique_fraction": float(
            sum(unique_fractions) / len(unique_fractions)
        ),
    }


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[AutoRewardManager] = None,
        val_reward_fn: Optional[AutoRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self._is_orarl = is_orarl(config)
        validate_algorithm(config)

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        # Best-train ckpt tracking (separate from rolling save_limit window).
        # See TrainerConfig.keep_best_train_ckpt for rationale. The buffer is a
        # FIFO of recent metric values; we save when the smoothed value strictly
        # improves and we have already passed best_train_min_step.
        self._best_train_window: deque[float] = deque(
            maxlen=max(1, int(config.trainer.best_train_smooth_window))
        )
        self._best_train_score: float = float("-inf")
        self._best_train_step: int = 0

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        # generate_sequences uses DP_COMPUTE_PROTO, which splits the prompt
        # batch across every actor-rollout worker before rollout.n fan-out.
        # Validate the dataloader's actual batch size here instead of failing
        # after all Ray workers and rollout engines have been initialized.
        rollout_world_size = config.trainer.n_gpus_per_node * config.trainer.nnodes
        rollout_dispatch_batch_size = (
            config.data.mini_rollout_batch_size
            if config.data.mini_rollout_batch_size is not None
            else config.data.rollout_batch_size
        )
        if rollout_dispatch_batch_size % rollout_world_size != 0:
            batch_size_key = (
                "data.mini_rollout_batch_size"
                if config.data.mini_rollout_batch_size is not None
                else "data.rollout_batch_size"
            )
            raise ValueError(
                f"{batch_size_key}={rollout_dispatch_batch_size} must be divisible by "
                f"the actor-rollout world size {rollout_world_size} "
                f"(trainer.nnodes={config.trainer.nnodes} * "
                f"trainer.n_gpus_per_node={config.trainer.n_gpus_per_node}). "
                "The prompt batch is sharded before worker.rollout.n is applied."
            )

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        _scale_rewards = bool(getattr(config.algorithm, "scale_rewards", True))
        _adv_est = getattr(
            config.algorithm.adv_estimator,
            "value",
            config.algorithm.adv_estimator,
        )
        if _adv_est == "grpo":
            if _scale_rewards:
                print("[GRPO] Scaled advantages ON: A=(r-mean)/(std+eps).")
            else:
                print("[GRPO] Raw-centered advantages ON: A=r-mean (no std whitening).")

        self._oracle_builder = None
        if self._is_orarl:
            if getattr(config.worker.rollout, "calculate_log_probs", False):
                raise ValueError(
                    "OraRL oracle rows require worker.rollout.calculate_log_probs=false "
                    "so old log probabilities are recomputed for their actual tokens."
                )
            self._oracle_builder = _load_dotted_callable(
                config.algorithm.oracle_builder
            )
            print(
                "[OraRL] enabled | append annotation-as-rollout | "
                f"directional_gain_gamma={config.algorithm.directional_gain_gamma} | "
                "detached oracle | strict sign-balanced selection | "
                "post-selection moment correction"
            )

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        rollout_rows = config.data.rollout_batch_size * config.worker.rollout.n
        actor_global_batch_size = config.worker.actor.global_batch_size
        self.skip_old_log_probs = (
            _skip_old_log_probs_enabled()
            and config.worker.actor.ppo_epochs == 1
            and actor_global_batch_size in {config.data.rollout_batch_size, rollout_rows}
        )
        if _skip_old_log_probs_enabled() and not self.skip_old_log_probs:
            print(
                "[trainer] VERL_SKIP_OLD_LOGPROBS=1 requested but disabled for this run: "
                "requires actor.ppo_epochs=1 and one actor global mini-batch per rollout step."
            )
        elif self.skip_old_log_probs:
            print(
                "[trainer] VERL_SKIP_OLD_LOGPROBS=1: skipping FSDP old_log_probs recompute; "
                "actor update will use its first forward log_probs.detach() as old_log_probs. "
                "Reference-policy KL, if enabled, is still computed separately."
            )

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor, rollout and ref
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        # Only advance best_global_step on steps that actually ran validation.
        # The condition must match the one guarding _validate() in the main loop.
        # Otherwise val_reward_score stays at its initial 0.0 while
        # best_val_reward_score starts at -1.0, so 0.0 > -1.0 always holds and
        # the first save is pinned as "best" forever. remove_obsolete_ckpt would
        # then protect the oldest checkpoint and evict recent ones instead — with
        # save_limit=2 the disk keeps the oldest plus the newest rather than the
        # two most recent.
        validation_ran_this_step = (
            self.val_reward_fn is not None
            and self.config.trainer.val_freq > 0
            and self.global_step % self.config.trainer.val_freq == 0
        )
        if validation_ran_this_step and self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

        # Keep optimizer/extra_state/dataloader only for the newest checkpoint and
        # thin older ones down to weights (huggingface/ + model_*.pt).
        if (
            self.config.trainer.keep_optim_only_latest
            and not self.config.trainer.save_model_only
        ):
            thin_out_old_ckpts(
                self.config.trainer.save_checkpoint_path,
                keep_full_step=self.global_step,
            )

    def _maybe_save_best_train_checkpoint(self, metrics: dict) -> None:
        """Save model-only ckpt when smoothed train metric hits a new best.

        Independent of save_freq / save_limit:
          - Lives under {save_checkpoint_path}/best_train/global_step_{N}/actor/
          - Always model-only (no optim/dataloader): for downstream eval only.
          - Old best is removed atomically when a new best is found, so the
            best_train/ subdir holds at most ONE checkpoint.

        Decision rule:
          smoothed = mean of last `best_train_smooth_window` values of
                     metrics[best_train_metric_key]
          if step >= best_train_min_step AND smoothed > self._best_train_score:
              save and update tracker
        """
        cfg = self.config.trainer
        if not getattr(cfg, "keep_best_train_ckpt", False):
            return

        metric_val = metrics.get(cfg.best_train_metric_key)
        if metric_val is None:
            return
        try:
            metric_val = float(metric_val)
        except (TypeError, ValueError):
            return

        self._best_train_window.append(metric_val)

        if self.global_step < int(cfg.best_train_min_step):
            return
        if len(self._best_train_window) < self._best_train_window.maxlen:
            # Wait until the smoothing buffer is full, otherwise early steps
            # have an artificially small window and bias toward early peaks.
            return

        smoothed = sum(self._best_train_window) / len(self._best_train_window)
        if smoothed <= self._best_train_score:
            return

        # New best: save model-only into a side directory.
        prev_best_step = self._best_train_step
        prev_best_score = self._best_train_score
        self._best_train_score = smoothed
        self._best_train_step = self.global_step

        best_root = os.path.join(cfg.save_checkpoint_path, "best_train")
        os.makedirs(best_root, exist_ok=True)
        new_dir = os.path.join(best_root, f"global_step_{self.global_step}")
        new_actor = os.path.join(new_dir, "actor")
        # Save current step (model only). Force save_model_only=True regardless
        # of the global flag so this side ckpt stays small.
        self.actor_rollout_ref_wg.save_checkpoint(new_actor, save_model_only=True)
        if self.use_critic:
            new_critic = os.path.join(new_dir, "critic")
            self.critic_wg.save_checkpoint(new_critic, save_model_only=True)

        # Tracker file lets downstream eval scripts auto-discover the best step.
        tracker = {
            "best_train_step": self._best_train_step,
            "best_train_metric_key": cfg.best_train_metric_key,
            "best_train_smoothed_value": round(self._best_train_score, 6),
            "best_train_smooth_window": self._best_train_window.maxlen,
            "best_train_actor_path": os.path.abspath(new_actor),
            "previous_best_step": prev_best_step,
            "previous_best_smoothed_value": (
                round(prev_best_score, 6) if prev_best_score != float("-inf") else None
            ),
        }
        with open(os.path.join(best_root, "best_train_tracker.json"), "w") as f:
            json.dump(tracker, f, ensure_ascii=False, indent=2)

        # Evict the previous best ckpt directory (we only keep one in best_train/).
        if prev_best_step and prev_best_step != self.global_step:
            old_dir = os.path.join(best_root, f"global_step_{prev_best_step}")
            if os.path.isdir(old_dir):
                import shutil
                try:
                    shutil.rmtree(old_dir)
                except OSError as exc:
                    print(f"[best-train ckpt] WARN: failed to remove {old_dir}: {exc}")

        prev_score_str = (
            f"{prev_best_score:.4f}" if prev_best_score != float("-inf") else "n/a"
        )
        print(
            f"[best-train ckpt] step={self.global_step} "
            f"smoothed_{cfg.best_train_metric_key}={self._best_train_score:.4f} "
            f"(prev best step={prev_best_step or 0}, value={prev_score_str}) "
            f"→ {new_actor}",
            flush=True,
        )

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _assert_multimodal_contract(self, data: DataProto, stage: str) -> None:
        if "multi_modal_data" not in data.non_tensor_batch:
            return

        problem_ids = data.non_tensor_batch.get("problem_id", None)
        uids = data.non_tensor_batch.get("uid", None)
        for idx, multi_modal_data in enumerate(data.non_tensor_batch["multi_modal_data"]):
            try:
                validate_multi_modal_data_contract(multi_modal_data)
            except Exception as exc:
                problem_id = None if problem_ids is None else problem_ids[idx]
                uid = None if uids is None else uids[idx]
                raise ValueError(
                    f"{stage}: invalid multi_modal_data at index={idx}, uid={uid}, problem_id={problem_id}: {exc}"
                ) from exc

    def _maybe_log_val_generations(
        self,
        inputs: list[str],
        outputs: list[str],
        labels: list[str],
        scores: list[float],
        problem_ids: list[Any],
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, label, score, problem_id) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores, problem_ids))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores, sample_problem_ids = [], [], [], [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["image_min_pixels"] = self.config.data.image_min_pixels
            test_gen_batch.meta_info["image_max_pixels"] = self.config.data.image_max_pixels
            test_gen_batch.meta_info["video_min_pixels"] = self.config.data.val_video_min_pixels
            test_gen_batch.meta_info["video_max_pixels"] = self.config.data.val_video_max_pixels
            test_gen_batch.meta_info["video_total_pixels"] = self.config.data.val_video_total_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.val_video_fps
            test_gen_batch.meta_info["video_max_frames"] = self.config.data.val_video_max_frames

            self._assert_multimodal_contract(test_gen_batch, stage="validate")
            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # Only pass fields needed by reward, excluding large multi_modal_data to reduce serialization
            val_reward_batch = test_batch.select(
                batch_keys=["responses", "response_mask"],
                non_tensor_batch_keys=[k for k in test_batch.non_tensor_batch if k != "multi_modal_data"],
            )
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(val_reward_batch))

            # store generations
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)
            if "problem_id" in test_batch.non_tensor_batch:
                sample_problem_ids.extend(test_batch.non_tensor_batch["problem_id"].tolist())
            else:
                sample_problem_ids.extend([None] * len(scores))

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(
            sample_inputs, sample_outputs, sample_labels, sample_scores, sample_problem_ids
        )
        if self.config.trainer.val_generations_to_log > 0 and sample_inputs:
            print("Sample problem_id:", sample_problem_ids[0])
            print("Sample prompt (with template):", sample_inputs[0])
            print("Sample response:", sample_outputs[0])
            print("Sample ground_truth:", sample_labels[0])
            print("Sample reward:", sample_scores[0])
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _inject_oracle_rollout_in_gen_output(
        self,
        gen_batch_output: DataProto,
        ground_truths: Optional[np.ndarray],
        extras: Optional[np.ndarray],
        n: int,
        replace_index: Optional[int] = None,
    ) -> int:
        """Build and mark annotation-derived oracle rollout rows."""
        if not self.config.algorithm.oracle_injection or self._oracle_builder is None:
            return 0
        if ground_truths is None or len(ground_truths) == 0:
            return 0

        # `replace_index` override lets callers pin the target slot regardless of
        # the config default (used by the append path, which calls this with n=1
        # and index 0 so every representative row becomes its group's oracle).
        if replace_index is None:
            replace_index = self.config.algorithm.oracle_replace_index
        if replace_index == -1:
            replace_index = n - 1

        responses = gen_batch_output.batch["responses"].clone()
        response_length = responses.size(1)
        device = responses.device
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = gen_batch_output.meta_info.get("eos_token_id", self.tokenizer.eos_token_id)
        # `get_response_mask` supports list[int]; single-eos tail append needs a scalar.
        eos_single = eos_token_id[0] if isinstance(eos_token_id, (list, tuple)) else eos_token_id

        is_oracle_row = np.zeros(responses.size(0), dtype=bool)
        replaced = 0
        for i, oracle_raw in enumerate(ground_truths):
            if oracle_raw is None:
                continue
            extra = extras[i] if extras is not None and i < len(extras) else {}
            if extra is None:
                extra = {}
            try:
                oracle_text = self._oracle_builder(str(oracle_raw), extra)
            except Exception as err:
                print(f"[OraRL] oracle_builder failed at idx={i} annotation={oracle_raw!r}: {err}")
                continue
            if not isinstance(oracle_text, str) or len(oracle_text.strip()) == 0:
                continue

            oracle_token_ids = self.tokenizer.encode(oracle_text, add_special_tokens=False)
            # Ensure the oracle response terminates with EOS so response_mask covers it.
            if eos_single is not None and (len(oracle_token_ids) == 0 or oracle_token_ids[-1] != eos_single):
                oracle_token_ids = oracle_token_ids + [eos_single]
            if len(oracle_token_ids) > response_length:
                # Keep the tail so EOS survives; pad head is unusual, just truncate front.
                oracle_token_ids = oracle_token_ids[-response_length:]

            oracle_tokens = VF.pad_2d_list_to_length(
                [oracle_token_ids],
                pad_token_id,
                max_length=response_length,
            ).to(device)

            replace_idx = i * n + replace_index
            responses[replace_idx] = oracle_tokens.squeeze(0)
            is_oracle_row[replace_idx] = True
            replaced += 1

        if replaced == 0:
            return 0

        prompts = gen_batch_output.batch["prompts"]
        prompt_length = prompts.size(-1)
        attention_mask = gen_batch_output.batch["attention_mask"]
        position_ids = gen_batch_output.batch["position_ids"]

        prompt_attention_mask = attention_mask[..., :prompt_length]
        prompt_position_ids = position_ids[..., :prompt_length]

        response_mask = VF.get_response_mask(responses, eos_token_id=eos_token_id, dtype=prompt_attention_mask.dtype)
        sequence_ids = torch.cat([prompts, responses], dim=-1)

        batch_size = responses.size(0)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if prompt_position_ids.ndim == 3:  # qwen2vl mrope: (batch_size, 4, seq_length)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(
                batch_size, prompt_position_ids.size(1), -1
            )

        response_position_ids = prompt_position_ids[..., -1:] + delta_position_id
        full_position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)
        full_attention_mask = torch.cat((prompt_attention_mask, response_mask), dim=-1)

        gen_batch_output.batch["responses"] = responses
        gen_batch_output.batch["input_ids"] = sequence_ids
        gen_batch_output.batch["response_mask"] = response_mask
        gen_batch_output.batch["position_ids"] = full_position_ids
        gen_batch_output.batch["attention_mask"] = full_attention_mask
        gen_batch_output.non_tensor_batch["is_oracle_row"] = is_oracle_row

        # When the rollout collected per-sequence mean logprobs, the values at
        # oracle-row positions still describe the ORIGINAL on-policy sample
        # (different tokens), so they are meaningless for ranking. Override them
        # with +inf so the oracle row always survives the probability pre-filter;
        # `selection_keep_oracle` force-keeps it in the reward-based stage.
        seq_lp = gen_batch_output.non_tensor_batch.get("seq_logprob_for_filter", None)
        if seq_lp is not None:
            seq_lp = np.asarray(seq_lp, dtype=np.float32, copy=True)
            seq_lp[is_oracle_row] = np.float32(np.inf)
            gen_batch_output.non_tensor_batch["seq_logprob_for_filter"] = seq_lp
        return replaced

    def _build_oracle_append_rows(
        self,
        gen_batch_output: DataProto,
        ground_truths: np.ndarray,
        extras: Optional[np.ndarray],
        n: int,
    ) -> Optional[DataProto]:
        """Build one extra oracle rollout per prompt for append mode.

        Uses the first on-policy rollout of every group as a template (it already
        carries the correct prompt tokens + multi_modal_data), then overwrites its
        response with the annotation-derived one by reusing
        ``_inject_oracle_rollout_in_gen_output`` with ``n=1`` and
        ``replace_index=0``, so every representative row becomes its group's
        oracle.

        The returned DataProto has ``len(ground_truths)`` rows (one per prompt),
        all flagged ``is_oracle_row=True``, with the SAME batch/non-tensor keys as
        ``gen_batch_output`` — so the caller can union it with the per-prompt
        metadata and ``DataProto.concat`` it onto the on-policy block, growing each
        group from ``n`` to ``n+1``. ``index_select`` copies the tensors, so
        overwriting the template never mutates the on-policy rows.

        Returns ``None`` if no oracle row could be built (empty batch or the
        builder returned nothing for every prompt), so the caller falls back to a
        plain GRPO group for this step.
        """
        num_prompts = len(ground_truths)
        if num_prompts == 0:
            return None
        rep_idx = np.array([i * n for i in range(num_prompts)], dtype=np.int64)
        oracle_rows = gen_batch_output.index_select(rep_idx)
        replaced = self._inject_oracle_rollout_in_gen_output(
            gen_batch_output=oracle_rows,
            ground_truths=ground_truths,
            extras=extras,
            n=1,
            replace_index=0,
        )
        if self._is_orarl and replaced != num_prompts:
            raise RuntimeError(
                "OraRL requires one valid oracle rollout per prompt, but the "
                f"configured builder produced {replaced} of {num_prompts}."
            )
        if replaced == 0:
            return None
        return oracle_rows

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        print("Start generating batch...")
        try:
            batch_dict = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.train_dataloader)
            batch_dict = next(self.data_iterator)

        meta_info = {
            "image_min_pixels": self.config.data.image_min_pixels,
            "image_max_pixels": self.config.data.image_max_pixels,
            "video_min_pixels": self.config.data.video_min_pixels,
            "video_max_pixels": self.config.data.video_max_pixels,
            "video_total_pixels": self.config.data.video_total_pixels,
            "video_fps": self.config.data.video_fps,
            "video_max_frames": self.config.data.video_max_frames,
        }
        new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
        new_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
        )

        gen_batch = new_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            meta_info_keys=[
                "image_min_pixels",
                "image_max_pixels",
                "video_min_pixels",
                "video_max_pixels",
                "video_total_pixels",
                "video_fps",
                "video_max_frames",
            ],
        )

        self._assert_multimodal_contract(gen_batch, stage="train")

        rollout_n = int(self.config.worker.rollout.n)
        rollout_backend = str(self.config.worker.rollout.name).lower()
        if rollout_backend in {"hf", "transformers"}:
            if rollout_n != self.actor_rollout_ref_wg.world_size:
                raise ValueError(
                    "Official HF rollout requires rollout.n == actor world size so "
                    "each rank contributes one completion per prompt; got "
                    f"n={rollout_n}, world_size={self.actor_rollout_ref_wg.world_size}."
                )
            # ONE_TO_ALL sends all unique prompts to every rank. Each rank runs
            # one HF generation per prompt; the custom collector transposes the
            # results into contiguous n-sized prompt groups.
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences_hf_official(
                gen_batch
            )
            # The HF workers deliberately omit this large object from their
            # outputs. Restore the driver's original once, then repeat by n
            # below; this avoids sending 64 duplicate 448-frame tensors back
            # through Ray.
            multi_modal_data = gen_batch.non_tensor_batch.get("multi_modal_data")
            if multi_modal_data is not None:
                new_batch.non_tensor_batch["multi_modal_data"] = multi_modal_data
        else:
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
        expected_rollout_rows = len(new_batch) * rollout_n
        if len(gen_batch_output) != expected_rollout_rows:
            raise ValueError(
                "Rollout output row count does not match prompt-group semantics: "
                f"got {len(gen_batch_output)}, expected {len(new_batch)} * "
                f"{rollout_n} = {expected_rollout_rows}."
            )
        metrics.update(
            _rollout_token_diversity_metrics(
                gen_batch_output.batch["responses"],
                gen_batch_output.batch["response_mask"],
                rollout_n,
            )
        )

        if self.config.algorithm.adv_estimator == "remax":
            gen_baseline_batch = deepcopy(gen_batch)
            gen_baseline_batch.meta_info["temperature"] = 0
            gen_baseline_batch.meta_info["n"] = 1
            gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

            new_batch = new_batch.union(gen_baseline_output)
            remax_reward_batch = new_batch.select(
                batch_keys=["responses", "response_mask"],
                non_tensor_batch_keys=[k for k in new_batch.non_tensor_batch if k != "multi_modal_data"],
            )
            reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(remax_reward_batch))
            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
            new_batch.batch["reward_baselines"] = reward_baseline_tensor
            del gen_baseline_batch, gen_baseline_output

        # Add an annotation-derived oracle response to each prompt's group.
        #   * replace mode: overwrite one on-policy rollout in place; group stays n.
        #   * append mode: keep all n on-policy rollouts and add the oracle as an
        #     extra row (group becomes n+1). We build the oracle rows here but defer
        #     the concat until the on-policy block has been assembled below.
        n = rollout_n
        oracle_mode = getattr(self.config.algorithm, "oracle_injection_mode", "replace")
        append_oracle = bool(self.config.algorithm.oracle_injection and oracle_mode == "append")
        oracle_append_rows: Optional[DataProto] = None
        if self.config.algorithm.oracle_injection:
            oracle_values = new_batch.non_tensor_batch.get("ground_truth", None)
            if oracle_values is None:
                raise KeyError(
                    "algorithm.oracle_injection=True but `ground_truth` is missing from the "
                    "batch non_tensor_batch. Check your dataset's answer_key."
                )
            # Oracle builders need the task family to avoid inferring it from
            # ambiguous labels such as a single option letter. Keep lightweight
            # row metadata, but never duplicate media tensors into this side
            # channel.
            oracle_extra_keys = [
                key
                for key in new_batch.non_tensor_batch
                if key not in {"ground_truth", "multi_modal_data", "raw_prompt_ids"}
            ]
            oracle_extras = np.asarray(
                [
                    {
                        key: new_batch.non_tensor_batch[key][row]
                        for key in oracle_extra_keys
                    }
                    for row in range(len(new_batch))
                ],
                dtype=object,
            )
            if append_oracle:
                oracle_append_rows = self._build_oracle_append_rows(
                    gen_batch_output,
                    oracle_values,
                    oracle_extras,
                    n,
                )
                metrics["orarl/oracle/groups_injected"] = int(
                    0 if oracle_append_rows is None else len(oracle_append_rows)
                )
                # Flag the on-policy block so `is_oracle_row` exists on every row
                # before the concat below (concat requires matching keys).
                gen_batch_output.non_tensor_batch["is_oracle_row"] = np.zeros(
                    len(gen_batch_output), dtype=bool
                )
            else:
                oracle_injected = self._inject_oracle_rollout_in_gen_output(
                    gen_batch_output=gen_batch_output,
                    ground_truths=oracle_values,
                    extras=oracle_extras,
                    n=n,
                )
                metrics["orarl/oracle/groups_injected"] = int(oracle_injected)

        # On-policy block: repeat the per-prompt metadata n times and attach the
        # generated rollouts. `repeat` returns a fresh DataProto, so `new_batch`
        # (the un-repeated per-prompt metadata) stays intact for the oracle rows.
        op_batch = new_batch.repeat(repeat_times=n, interleave=True)
        op_batch = op_batch.union(gen_batch_output)

        if oracle_append_rows is not None:
            # Attach per-prompt metadata (uid/ground_truth/problem_type/...) to the
            # oracle rows so each carries its group's uid, then concat onto the
            # block. Everything downstream (advantage, selection, diagnostics)
            # groups by uid, so the oracle row need not be physically adjacent.
            oracle_full = new_batch.union(oracle_append_rows)
            batch_out = DataProto.concat([op_batch, oracle_full])
            group_size = n + 1
        else:
            batch_out = op_batch
            group_size = n

        return batch_out[: self.config.data.rollout_batch_size * group_size]

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = _NoOpProgress() if _disable_tqdm() else tqdm(
            range(self.training_steps),
            desc="Running step",
            position=0,
        )
        val_metrics: Optional[dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        train_start_time = time.time()
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                self._balance_batch(batch, metrics=metrics)

                # compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # compute reward asynchronously so it can overlap with old-log-prob compute.
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_batch = batch.select(
                            batch_keys=["responses", "response_mask"],
                            non_tensor_batch_keys=[k for k in batch.non_tensor_batch if k != "multi_modal_data"],
                        )
                        reward_ref = self.reward_fn.compute_reward.remote(reward_batch)

                # recompute old_log_probs
                with timer("old", timing_raw):
                    # Actor update always needs the rollout temperature to compute response log-probs.
                    batch.meta_info["temperature"] = self.config.worker.rollout.temperature
                    if "rollout_log_probs" in batch.batch:
                        # Bypass mode: reuse the per-token logprobs vLLM already
                        # returned as old_log_probs. This skips one FSDP recompute
                        # and keeps the first PPO mini-batch ratio away from a
                        # constant 1, so clipping acts as a real trust region.
                        # vLLM applies log-softmax to logits/temperature, matching
                        # actor.compute_log_prob, so the values are interchangeable.
                        rollout_lp = batch.batch.pop("rollout_log_probs")
                        batch.batch["old_log_probs"] = rollout_lp.to(torch.float32)
                    elif self.skip_old_log_probs:
                        batch.meta_info["skip_old_log_probs"] = True
                        metrics["actor/old_log_probs_skipped"] = 1.0
                    else:
                        old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                        batch = batch.union(old_log_probs)

                # compute ref_log_probs
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                # compute values
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        # get token level scores asynchronously
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        _propagate_per_row_reward_metrics(batch, reward_metrics)
                        metrics.update(_on_policy_reward_diversity_metrics(batch))
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)

                    # apply kl penalty if available
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        # apply kl penalty to reward
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        scale_rewards=bool(self.config.algorithm.scale_rewards),
                        directional_gain=bool(
                            self.config.algorithm.directional_gain
                        ),
                        directional_gain_gamma=float(
                            self.config.algorithm.directional_gain_gamma
                        ),
                        directional_gain_positive_only=bool(
                            self.config.algorithm.directional_gain_positive_only
                        ),
                        directional_gain_recenter=bool(
                            self.config.algorithm.directional_gain_recenter
                        ),
                    )
                    # Report the raw per-group oracle-gap statistics before any
                    # anchoring rewrites the oracle advantage. No-op without
                    # oracle rows.
                    oracle_diag = compute_oracle_advantage_diagnostics(batch)
                    if oracle_diag:
                        metrics.update(oracle_diag)
                    adv_telemetry = batch.meta_info.pop("_advantage_telemetry", None)
                    if adv_telemetry:
                        for k, v in adv_telemetry.items():
                            metrics[f"orarl/{k}"] = float(v)

                    # OraRL stage 4: overwrite the oracle row with a detached
                    # positive anchor so it never enters the group mean/std.
                    if self.config.algorithm.detached_oracle_advantage:
                        # The released recipe keeps the directional gain
                        # policy-only, so the anchor is governed solely by its
                        # adaptive cap.
                        _det_gain_gamma = (
                            float(self.config.algorithm.directional_gain_gamma)
                            if (
                                self.config.algorithm.directional_gain
                                and self.config.algorithm.detached_oracle_use_directional_gain
                            )
                            else 0.0
                        )
                        det_metrics = apply_detached_oracle_advantage(
                            batch,
                            scale=float(
                                self.config.algorithm.detached_oracle_advantage_scale
                            ),
                            beta=float(self.config.algorithm.oracle_reward_gate_beta),
                            directional_gain_gamma=_det_gain_gamma,
                            match_best_ratio=float(
                                self.config.algorithm.detached_oracle_match_best_ratio
                            ),
                            match_best_min=float(
                                self.config.algorithm.detached_oracle_match_best_min
                            ),
                            match_best_max=float(
                                self.config.algorithm.detached_oracle_match_best_max
                            ),
                        )
                        if det_metrics:
                            metrics.update(det_metrics)

                # OraRL stage 5: retain the oracle plus a strict sign-balanced
                # subset of policy rows before the actor backward pass.
                if self._is_orarl and not self.use_critic:
                    full_score = batch.batch["token_level_scores"].sum(-1)
                    metrics["orarl/selection/full_batch_score_mean"] = float(
                        full_score.mean().item()
                    )
                    metrics["orarl/selection/full_batch_score_max"] = float(
                        full_score.max().item()
                    )
                    metrics["orarl/selection/full_batch_score_min"] = float(
                        full_score.min().item()
                    )
                    _post_recenter = bool(
                        self.config.algorithm.post_selection_recenter
                    )
                    _post_rms_match = bool(
                        self.config.algorithm.post_selection_rms_match
                    )
                    post_selection_references = (
                        build_orarl_post_selection_references(batch)
                        if _post_recenter or _post_rms_match
                        else {}
                    )
                    with timer("orarl_selection", timing_raw):
                        batch, selection_metrics = select_orarl_rollouts(
                            batch,
                            n_rollouts=self.config.worker.rollout.n,
                            prune_ratio=self.config.algorithm.selection_prune_ratio,
                            world_size=self.actor_rollout_ref_wg.world_size,
                            positive_quota=self.config.algorithm.selection_positive_quota,
                            negative_quota=self.config.algorithm.selection_negative_quota,
                        )
                    metrics.update(selection_metrics)
                    if _post_recenter or _post_rms_match:
                        with timer("post_selection_balance", timing_raw):
                            post_balance_metrics = (
                                apply_orarl_post_selection_advantage_balance(
                                    batch,
                                    post_selection_references,
                                    recenter=_post_recenter,
                                    rms_match=_post_rms_match,
                                    rms_min_scale=float(
                                        self.config.algorithm.post_selection_rms_min_scale
                                    ),
                                )
                            )
                        metrics.update(post_balance_metrics)
                    # Re-balance per-rank seqlen after pruning: the original
                    # balance partitioned all G rows uniformly, but dropping the
                    # low-|adv| rows skews the per-rank totals. Cheap, and keeps
                    # the actor update from being bottlenecked on a single rank.
                    if (
                        selection_metrics.get(
                            "orarl/selection/dropped_rows",
                            0.0,
                        )
                        > 0
                    ):
                        self._balance_batch(
                            batch,
                            metrics=metrics,
                            logging_prefix="orarl_selection_seqlen",
                        )

                # update critic
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # update actor
                if self.config.trainer.critic_warmup <= self.global_step:
                    metrics["actor/backward_rows"] = float(len(batch))
                    metrics["actor/backward_rows_per_prompt"] = float(
                        len(batch) / max(1, int(self.config.data.rollout_batch_size))
                    )
                    if self._is_orarl:
                        metrics["actor/oracle_injection_enabled"] = 1.0
                        metrics["actor/orarl_selection_enabled"] = 1.0
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            # Best-train ckpt: track every step (decoupled from save_freq) and
            # save model-only into a side dir when smoothed reward improves.
            with timer("save_best_train_checkpoint", timing_raw):
                self._maybe_save_best_train_checkpoint(metrics)

            if _print_step_summary_enabled():
                step_time = metrics.get("timing_s/step")
                elapsed = time.time() - train_start_time
                avg_step_time = elapsed / max(1, self.global_step)
                remaining_steps = max(0, self.training_steps - self.global_step)
                eta = avg_step_time * remaining_steps
                print(
                    "[TRAIN STEP] "
                    f"{self.global_step}/{self.training_steps} "
                    f"step_s={_fmt_metric(step_time, precision=1)} "
                    f"elapsed={_fmt_duration(elapsed)} "
                    f"eta={_fmt_duration(eta)}",
                    flush=True,
                )

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # perform validation after training (skip entirely when val_freq <= 0)
        if self.val_reward_fn is not None and self.config.trainer.val_freq > 0:
            if (
                val_metrics is None
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
