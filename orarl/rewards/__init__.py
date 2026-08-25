"""Unified rewards and oracle builders for OraRL."""

from .router import (
    DEFAULT_MODULE_PATHS,
    REWARD_NAME,
    REWARD_TYPE,
    RewardRouter,
    RouterSettings,
    build_oracle_response,
    build_oracle_response_from_ground_truth,
    compute_reward,
    compute_score,
    infer_task_from_ground_truth,
    normalize_task_name,
)
from .types import (
    RewardAdapter,
    RewardBatch,
    RewardContractError,
    RewardError,
    RewardMetrics,
    RewardResult,
    RewardSample,
    TaskFamily,
    UnknownTaskError,
)

__all__ = [
    "DEFAULT_MODULE_PATHS",
    "REWARD_NAME",
    "REWARD_TYPE",
    "RewardAdapter",
    "RewardBatch",
    "RewardContractError",
    "RewardError",
    "RewardMetrics",
    "RewardResult",
    "RewardRouter",
    "RewardSample",
    "RouterSettings",
    "TaskFamily",
    "UnknownTaskError",
    "build_oracle_response",
    "build_oracle_response_from_ground_truth",
    "compute_reward",
    "compute_score",
    "infer_task_from_ground_truth",
    "normalize_task_name",
]
