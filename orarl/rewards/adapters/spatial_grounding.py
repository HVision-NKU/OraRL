"""Built-in single-box spatial-grounding reward."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from ..types import RewardContractError
from ._common import (
    answer_payload,
    box_iou,
    canonical_json,
    exact_answer_payload,
    final_response_text,
    normalize_box,
    parse_json,
    unfence,
)

REWARD_NAME = "spatial_grounding"
REWARD_TYPE = "batch"
CANONICAL_RESPONSE_FORMAT = "qwen_json"
ACCEPT_TAGGED_RESPONSE = True

_NUMBER = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
_POINT_BOX_RE = re.compile(
    rf"\(\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)"
    rf"\s*,?\s*\(\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)"
)
_JSON_FENCE_RE = re.compile(
    r"\A\s*```json\s*(.*?)\s*```\s*\Z",
    flags=re.DOTALL | re.IGNORECASE,
)


def _box(value: Any) -> list[float] | None:
    payload = parse_json(value)
    box = normalize_box(payload)
    if box is not None:
        return box

    text = answer_payload(value)
    match = _POINT_BOX_RE.search(text)
    if match is not None:
        return [float(number) for number in match.groups()]
    numbers = re.findall(_NUMBER, text)
    if len(numbers) >= 4:
        return [float(number) for number in numbers[-4:]]
    return None


def _native_box(value: Any) -> list[float] | None:
    match = _JSON_FENCE_RE.fullmatch(str(value or ""))
    if match is None:
        return None
    try:
        payload = json.loads(match.group(1))
    except (TypeError, ValueError):
        return None
    if not (
        isinstance(payload, list)
        and len(payload) == 1
        and isinstance(payload[0], Mapping)
        and "bbox_2d" in payload[0]
    ):
        return None
    return normalize_box(payload[0]["bbox_2d"])


def _prediction(
    response: Any,
    *,
    accept_tagged_response: bool,
) -> tuple[list[float] | None, float]:
    final_text = final_response_text(response)
    native = _native_box(final_text)
    if native is not None:
        return native, 1.0

    tagged = exact_answer_payload(response)
    if accept_tagged_response and tagged is not None:
        tagged_box = _box(tagged)
        if tagged_box is not None:
            return tagged_box, 1.0
    return _box(unfence(final_text)), 0.0


def compute_score(
    batch: list[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, float]]:
    accept_tagged = bool(kwargs.get("accept_tagged_response", ACCEPT_TAGGED_RESPONSE))
    results: list[dict[str, float]] = []
    for item in batch:
        prediction, format_score = _prediction(
            item.get("response"),
            accept_tagged_response=accept_tagged,
        )
        target = _box(
            item.get("ground_truth") if item.get("ground_truth") is not None else item.get("answer")
        )
        iou = box_iou(prediction, target)
        results.append(
            {
                "overall": float(iou * format_score),
                "iou": float(iou),
                "format": float(format_score),
            }
        )
    return results


def build_oracle_response_from_ground_truth(
    ground_truth: Any,
    extra: Any = None,
) -> str:
    del extra
    box = _box(ground_truth)
    if box is None:
        raise RewardContractError("Spatial-grounding ground truth must contain one bounding box.")
    payload = [{"bbox_2d": box, "label": ""}]
    return f"```json\n{canonical_json(payload)}\n```"
