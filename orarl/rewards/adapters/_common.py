"""Small parsing and geometry primitives shared by built-in adapters."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Optional

_ANSWER_RE = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    flags=re.DOTALL | re.IGNORECASE,
)
_CANONICAL_ANSWER_RE = re.compile(
    r"\A\s*(?:(?:<think>)?.*?</think>\s*)?"
    r"<answer>\s*(.*?)\s*</answer>\s*\Z",
    flags=re.DOTALL | re.IGNORECASE,
)
_FENCE_RE = re.compile(
    r"\A\s*```(?:json)?\s*(.*?)\s*```\s*\Z",
    flags=re.DOTALL | re.IGNORECASE,
)
_THINK_BLOCK_RE = re.compile(
    r"<think>.*?</think>",
    flags=re.DOTALL | re.IGNORECASE,
)


def finite_float(value: Any) -> Optional[float]:
    """Return a finite float, rejecting booleans and invalid values."""

    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def exact_answer_payload(value: Any) -> Optional[str]:
    """Extract a payload only from the canonical final-answer shape."""

    match = _CANONICAL_ANSWER_RE.fullmatch(str(value or ""))
    if match is None or not match.group(1).strip():
        return None
    return match.group(1).strip()


def answer_payload(value: Any) -> str:
    """Extract the final answer block when present, otherwise return text."""

    text = str(value or "").strip()
    matches = _ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else text


def final_response_text(value: Any) -> str:
    """Remove a complete reasoning block and an optional answer wrapper."""

    text = str(value or "").strip()
    text = _THINK_BLOCK_RE.sub("", text).strip()
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1].strip()
    matches = _ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else text


def unfence(value: Any) -> str:
    text = str(value or "").strip()
    match = _FENCE_RE.fullmatch(text)
    return match.group(1).strip() if match is not None else text


def parse_json(value: Any) -> Any:
    """Parse JSON from common answer/fence wrappers without executing code."""

    if isinstance(value, (Mapping, list, tuple)):
        return value
    text = unfence(answer_payload(value))
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        pass

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except (TypeError, ValueError):
            continue
        return parsed
    return None


def parse_mapping(value: Any) -> Optional[dict[str, Any]]:
    payload = parse_json(value)
    return dict(payload) if isinstance(payload, Mapping) else None


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def canonical_answer(value: Any) -> str:
    payload = value if isinstance(value, str) else canonical_json(value)
    return f"<answer>{payload.strip()}</answer>"


def normalize_box(value: Any, *, reorder: bool = False) -> Optional[list[float]]:
    """Normalize one ``[x1, y1, x2, y2]`` box."""

    if isinstance(value, Mapping):
        for key in ("bbox_2d", "bbox", "box", "boxes"):
            if key in value:
                box = normalize_box(value[key], reorder=reorder)
                if box is not None:
                    return box
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 4:
        box = [finite_float(coordinate) for coordinate in value]
        if all(coordinate is not None for coordinate in box):
            normalized = [float(coordinate) for coordinate in box]
            if reorder:
                normalized[0], normalized[2] = sorted((normalized[0], normalized[2]))
                normalized[1], normalized[3] = sorted((normalized[1], normalized[3]))
            return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            box = normalize_box(item, reorder=reorder)
            if box is not None:
                return box
    return None


def normalize_boxes(value: Any) -> dict[str, list[float]]:
    """Normalize frame-keyed boxes using the released integer-key contract."""

    if not isinstance(value, Mapping):
        return {}
    boxes: dict[str, list[float]] = {}
    for key, raw_box in value.items():
        key_number = finite_float(key)
        box = normalize_box(raw_box)
        if key_number is None or box is None or not key_number.is_integer():
            continue
        boxes[str(int(key_number))] = box
    return boxes


def box_iou(first: Any, second: Any) -> float:
    box_a = normalize_box(first)
    box_b = normalize_box(second)
    if box_a is None or box_b is None:
        return 0.0
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def interval_iou(first: Any, second: Any) -> float:
    if not (
        isinstance(first, Sequence)
        and not isinstance(first, (str, bytes))
        and len(first) == 2
        and isinstance(second, Sequence)
        and not isinstance(second, (str, bytes))
        and len(second) == 2
    ):
        return 0.0
    first_values = [finite_float(value) for value in first]
    second_values = [finite_float(value) for value in second]
    if any(value is None for value in first_values + second_values):
        return 0.0
    a0, a1 = sorted(float(value) for value in first_values)
    b0, b1 = sorted(float(value) for value in second_values)
    intersection = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return intersection / union if union > 0.0 else 0.0


def ground_truth(item: Mapping[str, Any]) -> Any:
    value = item.get("ground_truth")
    return item.get("answer") if value is None else value
