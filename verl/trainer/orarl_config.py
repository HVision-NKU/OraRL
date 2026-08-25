# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2026 OraRL contributors
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
"""Validation and serialization for the GRPO/OraRL algorithm surface.

The trainer accepts exactly two recipes. Every OraRL option is checked here,
before any worker is created, so an inconsistent batch layout or a missing
oracle builder fails immediately instead of half-way through the first step.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any


SUPPORTED_ALGORITHMS = frozenset({"grpo", "orarl"})

_ORACLE_CONFIG_PREFIXES = (
    "oracle_",
    "directional_gain",
    "detached_oracle_",
    "selection_",
    "post_selection_",
)


def algorithm_name(config: Any) -> str:
    """Return the normalized algorithm name."""

    return str(getattr(config.algorithm, "name", "grpo")).strip().lower()


def is_orarl(config: Any) -> bool:
    return algorithm_name(config) == "orarl"


def validate_algorithm(config: Any) -> None:
    """Validate GRPO/OraRL options before workers are initialized."""

    name = algorithm_name(config)
    if name not in SUPPORTED_ALGORITHMS:
        choices = ", ".join(sorted(SUPPORTED_ALGORITHMS))
        raise ValueError(f"algorithm.name must be one of {{{choices}}}, got {name!r}.")
    if name == "grpo":
        return

    algo = config.algorithm
    estimator = getattr(algo.adv_estimator, "value", algo.adv_estimator)
    if estimator != "grpo":
        raise ValueError("OraRL requires algorithm.adv_estimator=grpo.")
    if bool(getattr(algo, "scale_rewards", True)):
        raise ValueError("OraRL requires raw advantages: algorithm.scale_rewards=false.")
    if not bool(algo.oracle_injection):
        raise ValueError("OraRL requires algorithm.oracle_injection=true.")
    if str(algo.oracle_injection_mode) != "append":
        raise ValueError("OraRL requires algorithm.oracle_injection_mode=append.")
    if not str(algo.oracle_builder or "").strip():
        raise ValueError("OraRL requires algorithm.oracle_builder.")
    if int(config.worker.rollout.n) <= 1:
        raise ValueError("OraRL requires worker.rollout.n > 1.")
    if bool(getattr(config.worker.rollout, "calculate_log_probs", False)):
        raise ValueError(
            "OraRL oracle rows require worker.rollout.calculate_log_probs=false so "
            "old log probabilities are recomputed for their actual tokens."
        )

    prune_ratio = float(algo.selection_prune_ratio)
    if not math.isfinite(prune_ratio) or not 0.0 < prune_ratio < 1.0:
        raise ValueError("OraRL selection_prune_ratio must be in (0, 1).")
    n_rollouts = int(config.worker.rollout.n)
    keep_rows = max(1, int(n_rollouts * (1.0 - prune_ratio)))
    if not bool(algo.selection_keep_oracle):
        raise ValueError("OraRL requires the oracle rollout to be retained.")
    oracle_slot = 1
    positive = int(algo.selection_positive_quota)
    negative = int(algo.selection_negative_quota)
    if positive < 0 or negative < 0:
        raise ValueError("OraRL selection quotas must be non-negative.")
    if positive + negative + oracle_slot != keep_rows:
        raise ValueError(
            "OraRL selection quotas must fill the keep budget: "
            f"positive({positive}) + negative({negative}) + "
            f"oracle({oracle_slot}) != keep_rows({keep_rows})."
        )
    if not bool(algo.selection_strict_sign_balance):
        raise ValueError("OraRL requires strict sign-balanced selection.")

    nodes = int(config.trainer.nnodes)
    gpus_per_node = int(config.trainer.n_gpus_per_node)
    rollout_batch_size = int(config.data.rollout_batch_size)
    if nodes <= 0 or gpus_per_node <= 0 or rollout_batch_size <= 0:
        raise ValueError(
            "OraRL requires positive trainer node/GPU counts and rollout_batch_size."
        )
    world_size = nodes * gpus_per_node
    selected_batch = rollout_batch_size * keep_rows
    if selected_batch % world_size != 0:
        raise ValueError(
            "OraRL selected batch must divide the actor world size: "
            f"{selected_batch} rows for world_size={world_size}."
        )
    # The oracle is appended as an extra row, so the pre-selection batch is
    # rollout_batch_size*(n+1) and seqlen-balanced partitioning needs it to
    # shard evenly.
    preselection_batch = rollout_batch_size * (n_rollouts + 1)
    if preselection_batch % world_size != 0:
        raise ValueError(
            "OraRL append-oracle batch must divide the actor world size: "
            f"{preselection_batch} rows for world_size={world_size}."
        )

    if not bool(algo.directional_gain):
        raise ValueError("OraRL requires algorithm.directional_gain=true.")
    if not bool(algo.directional_gain_positive_only):
        raise ValueError("OraRL requires positive-only directional gain.")
    if not bool(algo.directional_gain_recenter):
        raise ValueError("OraRL requires post-gain policy re-centering.")
    gain_gamma = float(algo.directional_gain_gamma)
    if not math.isfinite(gain_gamma) or gain_gamma <= 0.0:
        raise ValueError("OraRL directional_gain_gamma must be finite and positive.")
    if not bool(algo.detached_oracle_advantage):
        raise ValueError("OraRL requires a detached oracle advantage.")
    detached_scale = float(algo.detached_oracle_advantage_scale)
    if not math.isfinite(detached_scale) or detached_scale <= 0.0:
        raise ValueError(
            "OraRL detached_oracle_advantage_scale must be finite and positive."
        )
    gap_beta = float(algo.oracle_reward_gate_beta)
    if not math.isfinite(gap_beta) or gap_beta <= 0.0:
        raise ValueError("OraRL oracle_reward_gate_beta must be finite and positive.")
    if bool(algo.detached_oracle_use_directional_gain):
        raise ValueError(
            "The released OraRL recipe keeps directional gain policy-only; "
            "set detached_oracle_use_directional_gain=false."
        )
    match_ratio = float(algo.detached_oracle_match_best_ratio)
    match_min = float(algo.detached_oracle_match_best_min)
    match_max = float(algo.detached_oracle_match_best_max)
    if (
        not all(math.isfinite(value) for value in (match_ratio, match_min, match_max))
        or match_ratio <= 0.0
        or match_min < 0.0
        or match_max < match_min
    ):
        raise ValueError(
            "OraRL oracle cap requires a finite positive ratio and "
            "finite 0 <= min <= max."
        )
    if not bool(algo.post_selection_recenter):
        raise ValueError("OraRL requires post-selection re-centering.")
    if not bool(algo.post_selection_rms_match):
        raise ValueError("OraRL requires post-selection RMS matching.")
    rms_min = float(algo.post_selection_rms_min_scale)
    if not 0.0 <= rms_min <= 1.0:
        raise ValueError("OraRL post_selection_rms_min_scale must be in [0, 1].")


def public_config_dict(config: Any) -> dict[str, Any]:
    """Serialize the config, dropping options the selected recipe never reads."""

    value = asdict(config)
    if algorithm_name(config) != "grpo":
        return value

    algorithm = value.get("algorithm", {})
    for key in tuple(algorithm):
        if key.startswith(_ORACLE_CONFIG_PREFIXES):
            algorithm.pop(key, None)
    value.get("worker", {}).get("actor", {}).pop("selection_prune_ratio", None)
    return value
