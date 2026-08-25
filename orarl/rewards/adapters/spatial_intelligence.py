"""Built-in VSI reward: numerical MRA and exact categorical matching."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..types import RewardContractError
from ._common import answer_payload, canonical_answer, exact_answer_payload, ground_truth

REWARD_NAME = "spatial_intelligence"
REWARD_TYPE = "batch"

NUMERICAL_SUBTYPES = frozenset(
    {
        "object_abs_distance",
        "object_counting",
        "object_size_estimation",
        "room_size_estimation",
    }
)
MULTIPLE_CHOICE_SUBTYPES = frozenset(
    {
        "object_rel_distance",
        "route_planning",
        "obj_appearance_order",
    }
)
DIRECTION_PREFIX = "object_rel_direction"
MRA_THRESHOLDS = tuple(0.50 + 0.05 * index for index in range(10))

_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)")
_SCALAR_RE = re.compile(r"\A\s*[-+]?(?:\d+(?:\.\d+)?|\.\d+)\s*\Z")
_OPTION_RE = re.compile(r"\A\s*([A-D])\s*\Z", flags=re.IGNORECASE)


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _is_direction(subtype: str) -> bool:
    return subtype.startswith(DIRECTION_PREFIX)


def _subtype(item: Mapping[str, Any]) -> str:
    for field in (
        "problem_type",
        "question_type",
        "task_name",
        "task",
        "scoring_family",
    ):
        candidate = _slug(item.get(field))
        if (
            candidate in NUMERICAL_SUBTYPES
            or candidate in MULTIPLE_CHOICE_SUBTYPES
            or _is_direction(candidate)
        ):
            return candidate
    return ""


def _number(value: Any) -> float | None:
    text = answer_payload(value)
    match = _NUMBER_RE.search(text)
    if match is None:
        return None
    try:
        number = float(match.group(0))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _option(value: Any, *, relaxed: bool = False) -> str:
    payload = answer_payload(value)
    match = _OPTION_RE.fullmatch(payload)
    if match is None and relaxed:
        matches = re.findall(r"\b([A-D])\b", payload, flags=re.IGNORECASE)
        return matches[-1].upper() if matches else ""
    return match.group(1).upper() if match is not None else ""


def mean_relative_accuracy(
    prediction: float,
    target: float,
    thresholds: Sequence[float] = MRA_THRESHOLDS,
) -> float:
    """Average correctness over VSI confidence thresholds 0.50 through 0.95."""

    if target == 0.0 or not thresholds:
        return 0.0
    relative_error = abs(prediction - target) / abs(target)
    return sum(relative_error <= 1.0 - float(threshold) for threshold in thresholds) / len(
        thresholds
    )


def _thresholds(value: Any) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("mra_thresholds must be a non-empty numeric sequence.")
    try:
        thresholds = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError("mra_thresholds must contain numbers.") from error
    if not thresholds or any(
        not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in thresholds
    ):
        raise ValueError("mra_thresholds must contain values from zero to one.")
    return thresholds


def compute_score(
    batch: list[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, float]]:
    thresholds = _thresholds(kwargs.get("mra_thresholds", MRA_THRESHOLDS))
    results: list[dict[str, float]] = []
    for item in batch:
        subtype = _subtype(item)
        response_payload = exact_answer_payload(item.get("response"))
        target_value = ground_truth(item)

        if subtype in NUMERICAL_SUBTYPES:
            format_score = float(
                response_payload is not None and _SCALAR_RE.fullmatch(response_payload) is not None
            )
            prediction = _number(response_payload)
            target = _number(target_value)
            accuracy = (
                mean_relative_accuracy(prediction, target, thresholds)
                if prediction is not None and target is not None
                else 0.0
            )
            metric_name = "mra"
        elif subtype in MULTIPLE_CHOICE_SUBTYPES or _is_direction(subtype):
            prediction = _option(response_payload)
            target = _option(target_value, relaxed=True)
            format_score = float(bool(prediction))
            accuracy = float(bool(target) and prediction == target)
            metric_name = "acc"
        else:
            format_score = 0.0
            accuracy = 0.0
            metric_name = "accuracy"

        result = {
            "overall": float(accuracy * format_score),
            "accuracy": float(accuracy),
            "format": float(format_score),
        }
        result[metric_name] = float(accuracy)
        results.append(result)
    return results


def build_oracle_response_from_ground_truth(
    ground_truth: Any,
    extra: Mapping[str, Any] | None = None,
) -> str:
    payload = answer_payload(ground_truth).strip()
    if not payload:
        raise RewardContractError("Spatial-intelligence ground truth must contain an answer.")
    subtype = _subtype(extra or {})
    if subtype in NUMERICAL_SUBTYPES:
        number = _number(payload)
        if number is None:
            raise RewardContractError(
                "Numerical spatial-intelligence ground truth must contain a number."
            )
        payload = f"{number:g}"
    elif subtype in MULTIPLE_CHOICE_SUBTYPES or _is_direction(subtype):
        option = _option(payload, relaxed=True)
        if option:
            payload = option
    return canonical_answer(payload)
