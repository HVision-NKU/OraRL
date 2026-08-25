"""Built-in spatial-temporal grounding reward aligned with strict evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..types import RewardContractError
from ._common import (
    box_iou,
    canonical_answer,
    exact_answer_payload,
    finite_float,
    ground_truth,
    interval_iou,
    normalize_box,
    normalize_boxes,
    parse_mapping,
)

REWARD_NAME = "spatial_temporal_grounding"
REWARD_TYPE = "batch"
TEMPORAL_IOU_WEIGHT = 0.2
STRICT_SPATIAL_IOU_WEIGHT = 0.8


def _time_span(value: Any) -> list[float] | None:
    if not (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2
    ):
        return None
    span = [finite_float(part) for part in value]
    if any(part is None for part in span):
        return None
    return [float(part) for part in span]


def _boxes(value: Any, time_span: Any) -> dict[str, list[float]]:
    boxes = normalize_boxes(value)
    if boxes or not isinstance(value, list):
        return boxes
    span = _time_span(time_span)
    if span is None:
        return {}
    start = int(round(span[0]))
    return {
        str(start + index): box
        for index, raw_box in enumerate(value)
        if (box := normalize_box(raw_box)) is not None
    }


def _payload(value: Any) -> tuple[list[float] | None, dict[str, list[float]]]:
    payload = parse_mapping(value)
    if payload is None:
        return None, {}
    span = _time_span(payload.get("time"))
    return span, _boxes(payload.get("boxes"), span)


def _prediction(
    value: Any,
) -> tuple[list[float] | None, dict[str, list[float]], float]:
    answer = exact_answer_payload(value)
    if answer is None:
        return None, {}, 0.0
    try:
        payload = json.loads(answer)
    except (TypeError, ValueError):
        return None, {}, 0.0
    if not isinstance(payload, Mapping):
        return None, {}, 0.0
    span = _time_span(payload.get("time"))
    raw_boxes = payload.get("boxes")
    boxes = normalize_boxes(raw_boxes)
    valid = (
        span is not None
        and span[0] <= span[1]
        and isinstance(raw_boxes, Mapping)
        and bool(boxes)
        and len(boxes) == len(raw_boxes)
    )
    return span, boxes, float(valid)


def strict_spatial_iou(
    predicted_boxes: Mapping[str, Any],
    target_boxes: Mapping[str, Any],
) -> float:
    """Average box IoU over every target frame, including missing-frame zeros."""

    if not target_boxes:
        return 0.0
    return sum(
        box_iou(predicted_boxes.get(frame), target_box)
        for frame, target_box in target_boxes.items()
    ) / len(target_boxes)


def compute_score(
    batch: list[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, float]]:
    temporal_weight = finite_float(kwargs.get("temporal_iou_weight", TEMPORAL_IOU_WEIGHT))
    if temporal_weight is None or not 0.0 <= temporal_weight <= 1.0:
        raise ValueError("temporal_iou_weight must be between zero and one.")
    spatial_weight = 1.0 - temporal_weight

    results: list[dict[str, float]] = []
    for item in batch:
        predicted_time, predicted_boxes, format_score = _prediction(item.get("response"))
        target_time, target_boxes = _payload(ground_truth(item))
        temporal_iou = interval_iou(predicted_time, target_time)
        spatial_iou = strict_spatial_iou(predicted_boxes, target_boxes)
        coverage = (
            sum(frame in predicted_boxes for frame in target_boxes) / len(target_boxes)
            if target_boxes
            else 0.0
        )
        accuracy = temporal_weight * temporal_iou + spatial_weight * spatial_iou
        results.append(
            {
                "overall": float(accuracy * format_score),
                "accuracy": float(accuracy),
                "format": float(format_score),
                "tiou": float(temporal_iou),
                "siou_strict": float(spatial_iou),
                "coverage": float(coverage),
            }
        )
    return results


def build_oracle_response_from_ground_truth(
    ground_truth: Any,
    extra: Any = None,
) -> str:
    del extra
    span, boxes = _payload(ground_truth)
    if span is None or not boxes:
        raise RewardContractError(
            "Spatial-temporal ground truth must contain time and frame-keyed boxes."
        )
    if span[0] > span[1]:
        span = [span[1], span[0]]
    return canonical_answer({"time": span, "boxes": boxes})
