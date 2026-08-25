"""Built-in temporal-grounding reward using temporal intersection-over-union."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..types import RewardContractError
from ._common import (
    canonical_answer,
    exact_answer_payload,
    finite_float,
    ground_truth,
    interval_iou,
    parse_json,
)

REWARD_NAME = "temporal_grounding"
REWARD_TYPE = "batch"
RESPONSE_SEPARATOR = "to"

_NUMBER = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
_SPAN_RE = re.compile(
    rf"\A\s*({_NUMBER})\s+(?:to|and)\s+({_NUMBER})\s*\Z",
    flags=re.IGNORECASE,
)
_LOOSE_SPAN_RE = re.compile(
    rf"({_NUMBER})\s*(?:to|and|[-–—])\s*({_NUMBER})",
    flags=re.IGNORECASE,
)


def _span(value: Any) -> list[float] | None:
    payload = parse_json(value)
    if isinstance(payload, Mapping):
        if "time" in payload:
            return _span(payload["time"])
        if "start" in payload and "end" in payload:
            return _span([payload["start"], payload["end"]])
    if (
        isinstance(payload, Sequence)
        and not isinstance(payload, (str, bytes))
        and len(payload) == 2
    ):
        numbers = [finite_float(item) for item in payload]
        if all(number is not None for number in numbers):
            return [float(number) for number in numbers]

    match = _LOOSE_SPAN_RE.search(str(value or ""))
    if match is None:
        return None
    numbers = [finite_float(part) for part in match.groups()]
    if any(number is None for number in numbers):
        return None
    return [float(number) for number in numbers]


def _prediction(value: Any) -> tuple[list[float] | None, float]:
    payload = exact_answer_payload(value)
    if payload is None:
        return None, 0.0
    match = _SPAN_RE.fullmatch(payload)
    if match is None:
        return None, 0.0
    span = [float(number) for number in match.groups()]
    if span[0] > span[1]:
        return None, 0.0
    return span, 1.0


def compute_score(
    batch: list[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, float]]:
    """Score canonical ``<answer>start to end</answer>`` responses."""

    del kwargs
    results: list[dict[str, float]] = []
    for item in batch:
        prediction, format_score = _prediction(item.get("response"))
        target = _span(ground_truth(item))
        iou = (
            interval_iou(prediction, target)
            if prediction is not None and target is not None
            else 0.0
        )
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
    """Build the canonical temporal response used by oracle injection."""

    del extra
    span = _span(ground_truth)
    if span is None:
        raise RewardContractError("Temporal-grounding ground truth must contain a start/end span.")
    start, end = span
    if start > end:
        start, end = end, start
    return canonical_answer(f"{start:g} {RESPONSE_SEPARATOR} {end:g}")
