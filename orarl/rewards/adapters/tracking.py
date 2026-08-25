"""Built-in tracking reward aligned with strict average overlap evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..types import RewardContractError
from ._common import (
    box_iou,
    canonical_answer,
    exact_answer_payload,
    ground_truth,
    normalize_boxes,
    parse_mapping,
)

REWARD_NAME = "tracking"
REWARD_TYPE = "batch"


def _boxes(value: Any) -> dict[str, list[float]]:
    payload = parse_mapping(value)
    return normalize_boxes(payload.get("boxes")) if payload is not None else {}


def _prediction(value: Any) -> tuple[dict[str, list[float]], float]:
    answer = exact_answer_payload(value)
    if answer is None:
        return {}, 0.0
    try:
        payload = json.loads(answer)
    except (TypeError, ValueError):
        return {}, 0.0
    if not isinstance(payload, Mapping) or not isinstance(payload.get("boxes"), Mapping):
        return {}, 0.0
    raw_boxes = payload["boxes"]
    boxes = normalize_boxes(raw_boxes)
    valid_shape = bool(boxes) and len(boxes) == len(raw_boxes)
    return boxes, float(valid_shape)


def strict_mean_iou(
    predicted_boxes: Mapping[str, Any],
    target_boxes: Mapping[str, Any],
) -> float:
    """Mean box IoU over every target frame; missing predictions contribute zero."""

    if not target_boxes:
        return 0.0
    total = sum(
        box_iou(predicted_boxes.get(frame), target_box)
        for frame, target_box in target_boxes.items()
    )
    return total / len(target_boxes)


def compute_score(
    batch: list[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, float]]:
    del kwargs
    results: list[dict[str, float]] = []
    for item in batch:
        predicted, format_score = _prediction(item.get("response"))
        target = _boxes(ground_truth(item))
        mean_iou = strict_mean_iou(predicted, target)
        coverage = sum(frame in predicted for frame in target) / len(target) if target else 0.0
        results.append(
            {
                "overall": float(mean_iou * format_score),
                "accuracy": float(mean_iou),
                "format": float(format_score),
                "miou": float(mean_iou),
                "coverage": float(coverage),
            }
        )
    return results


def build_oracle_response_from_ground_truth(
    ground_truth: Any,
    extra: Any = None,
) -> str:
    del extra
    boxes = _boxes(ground_truth)
    if not boxes:
        raise RewardContractError("Tracking ground truth must contain non-empty frame-keyed boxes.")
    return canonical_answer({"boxes": boxes})
