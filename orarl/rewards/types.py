"""Typed interfaces shared by OraRL reward adapters."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Protocol, TypedDict


class TaskFamily(str, Enum):
    """The seven reward families supported by the public router."""

    TEMPORAL_GROUNDING = "temporal_grounding"
    TRACKING = "tracking"
    SEGMENTATION = "segmentation"
    SPATIAL_GROUNDING = "spatial_grounding"
    SPATIAL_TEMPORAL_GROUNDING = "spatial_temporal_grounding"
    SPATIAL_INTELLIGENCE = "spatial_intelligence"
    VIDEO_QA = "video_qa"

    def __str__(self) -> str:
        return self.value


class RewardSample(TypedDict, total=False):
    """Fields consumed by the router.

    Task adapters may consume additional dataset-specific fields.
    """

    task: str
    task_name: str
    problem_type: str
    task_source: str
    scoring_family: str
    question_type: str
    response: str
    ground_truth: Any
    answer: Any


class RewardMetrics(TypedDict, total=False):
    """Common metrics; task adapters may add numeric telemetry fields."""

    overall: float
    accuracy: float
    format: float


RewardBatch = List[Mapping[str, Any]]
RewardResult = Dict[str, float]


class RewardAdapter(Protocol):
    """Structural contract implemented by packaged and custom adapters."""

    def compute_score(
        self,
        batch: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> List[Mapping[str, Any]]:
        """Score one or more samples."""

    def build_oracle_response_from_ground_truth(
        self,
        ground_truth: Any,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Optional[str]:
        """Build the response text used for an oracle row."""


class RewardError(Exception):
    """Base class for reward routing failures."""


class UnknownTaskError(RewardError, ValueError):
    """Raised when a sample cannot be assigned to a supported task."""


class RewardContractError(RewardError, RuntimeError):
    """Raised when an adapter violates the unified reward contract."""
