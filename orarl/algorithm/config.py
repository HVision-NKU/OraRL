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
"""Typed configuration for the dependency-light algorithm helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value}.")


def _nonnegative_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value}.")


@dataclass(frozen=True, slots=True)
class GRPOConfig:
    """Configuration for the unmodified group-relative baseline."""

    normalize: bool = True
    eps: float = 1e-6

    def __post_init__(self) -> None:
        _positive_finite("eps", self.eps)


@dataclass(frozen=True, slots=True)
class DirectionalGainConfig:
    """Configuration for OraRL's positive-direction utility transform."""

    gamma: float = 0.25
    eps: float = 1e-6

    def __post_init__(self) -> None:
        _nonnegative_finite("gamma", self.gamma)
        _positive_finite("eps", self.eps)


@dataclass(frozen=True, slots=True)
class DetachedOracleConfig:
    """Configuration for the detached annotation advantage."""

    scale: float = 2.0
    gap_beta: float = 2.0
    match_best_ratio: float | None = 1.2
    match_best_min: float = 0.05
    match_best_max: float = 1.0
    eps: float = 1e-6

    def __post_init__(self) -> None:
        _nonnegative_finite("scale", self.scale)
        _positive_finite("gap_beta", self.gap_beta)
        _positive_finite("eps", self.eps)
        _nonnegative_finite("match_best_min", self.match_best_min)
        _nonnegative_finite("match_best_max", self.match_best_max)
        if self.match_best_min > self.match_best_max:
            raise ValueError("match_best_min must not exceed match_best_max.")
        if self.match_best_ratio is not None:
            _positive_finite("match_best_ratio", self.match_best_ratio)


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    """Strict sign-balanced selection settings.

    The annotation row occupies one slot in ``keep_per_group``. The remaining
    slots are assigned to the positive and negative policy quotas.
    """

    keep_per_group: int = 4
    positive_quota: int = 1
    negative_quota: int = 2
    world_size: int = 1

    def __post_init__(self) -> None:
        if self.keep_per_group < 2:
            raise ValueError("keep_per_group must leave room for policy and oracle rows.")
        if self.positive_quota < 0 or self.negative_quota < 0:
            raise ValueError("selection quotas must be non-negative.")
        if self.positive_quota + self.negative_quota + 1 != self.keep_per_group:
            raise ValueError(
                "positive_quota + negative_quota + one oracle slot must equal keep_per_group."
            )
        if self.world_size <= 0:
            raise ValueError("world_size must be positive.")


@dataclass(frozen=True, slots=True)
class CorrectionConfig:
    """Settings for post-selection mean and RMS correction."""

    rms_match: bool = True
    rms_min_scale: float = 0.25
    sigma_policy_floor: float = 1e-3
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if not 0.0 <= self.rms_min_scale <= 1.0:
            raise ValueError("rms_min_scale must be in [0, 1].")
        _nonnegative_finite("sigma_policy_floor", self.sigma_policy_floor)
        _positive_finite("eps", self.eps)


@dataclass(frozen=True, slots=True)
class PostSelectionReference:
    """Pre-selection statistics for one rollout group."""

    policy_rms: float
    sigma_policy: float
    policy_rows: int

    def __post_init__(self) -> None:
        _nonnegative_finite("policy_rms", self.policy_rms)
        _nonnegative_finite("sigma_policy", self.sigma_policy)
        if self.policy_rows < 1:
            raise ValueError("policy_rows must be positive.")


@dataclass(frozen=True, slots=True)
class OraRLConfig:
    """Paper recipe defaults for the four numerical stages."""

    directional_gain: DirectionalGainConfig = field(default_factory=DirectionalGainConfig)
    detached_oracle: DetachedOracleConfig = field(default_factory=DetachedOracleConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
