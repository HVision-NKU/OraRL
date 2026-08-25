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
"""Public, dependency-light OraRL algorithm surface."""

from .advantages import (
    DIRECTIONAL_GAIN_MAX,
    DIRECTIONAL_GAIN_MIN,
    apply_detached_oracle_advantage,
    apply_grpo_advantages,
    apply_orarl_advantages,
    compute_directional_gain,
    compute_grpo_advantages,
    compute_grpo_outcome_advantage,
    compute_orarl_advantages,
    compute_orarl_policy_advantages,
    detached_oracle_advantage,
    directional_gain,
    normalized_reward_gap,
    raw_center_on_policy,
    transform_directional_utility,
    vanilla_grpo_advantage,
)
from .config import (
    CorrectionConfig,
    DetachedOracleConfig,
    DirectionalGainConfig,
    GRPOConfig,
    OraRLConfig,
    PostSelectionReference,
    SelectionConfig,
)
from .correction import (
    apply_post_selection_advantage_correction,
    apply_post_selection_correction,
    balance_post_selection_group,
    build_pre_selection_references,
    capture_pre_selection_references,
    correct_post_selection_group,
)
from .selection import select_sign_balanced_rollouts

__all__ = [
    "DIRECTIONAL_GAIN_MAX",
    "DIRECTIONAL_GAIN_MIN",
    "CorrectionConfig",
    "DetachedOracleConfig",
    "DirectionalGainConfig",
    "GRPOConfig",
    "OraRLConfig",
    "PostSelectionReference",
    "SelectionConfig",
    "apply_detached_oracle_advantage",
    "apply_grpo_advantages",
    "apply_orarl_advantages",
    "apply_post_selection_advantage_correction",
    "apply_post_selection_correction",
    "balance_post_selection_group",
    "build_pre_selection_references",
    "capture_pre_selection_references",
    "compute_directional_gain",
    "compute_grpo_advantages",
    "compute_grpo_outcome_advantage",
    "compute_orarl_advantages",
    "compute_orarl_policy_advantages",
    "correct_post_selection_group",
    "detached_oracle_advantage",
    "directional_gain",
    "normalized_reward_gap",
    "raw_center_on_policy",
    "select_sign_balanced_rollouts",
    "transform_directional_utility",
    "vanilla_grpo_advantage",
]
