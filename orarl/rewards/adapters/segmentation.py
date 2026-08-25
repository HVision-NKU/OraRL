"""Built-in mask-aware segmentation reward with a deterministic proxy fallback."""

from __future__ import annotations

import ast
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..types import RewardContractError
from ._common import (
    box_iou,
    canonical_answer,
    exact_answer_payload,
    finite_float,
    ground_truth,
    normalize_box,
    parse_mapping,
)

REWARD_NAME = "segmentation"
REWARD_TYPE = "batch"

POINT_SIGMA = 50.0
TIME_TAU = 2.0
IMAGE_WEIGHTS = (0.50, 0.25, 0.25)
VIDEO_WEIGHTS = (0.35, 0.10, 0.40, 0.15)
REQUIRE_VIDEO_TIME = True
MASK_AWARE = True
MASK_POSITIVE_ZERO_CAP = 0.10
MASK_BOX_MISS_CAP = 0.20
MASK_BOX_MISS_IOU = 0.10
MASK_POINT_RADIUS = 3


def _mapping(value: Any) -> dict[str, Any] | None:
    mapping = parse_mapping(value)
    if mapping is not None or not isinstance(value, str):
        return mapping
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    return dict(parsed) if isinstance(parsed, Mapping) else None


def _box(value: Any) -> list[float] | None:
    return normalize_box(value, reorder=True)


def _points(value: Any) -> list[list[float]] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        return None
    points: list[list[float]] = []
    for point in value:
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) < 2:
            return None
        x = finite_float(point[0])
        y = finite_float(point[1])
        if x is None or y is None:
            return None
        points.append([x, y])
    return points


def assignment_similarity(
    prediction: list[list[float]] | None,
    target: list[list[float]] | None,
    *,
    sigma: float,
) -> float:
    """Optimal three-point assignment with Gaussian distance similarity."""

    if prediction is None or target is None or sigma <= 0.0:
        return 0.0
    best_distance = math.inf
    for permutation in itertools.permutations(range(3)):
        distance = sum(
            math.hypot(
                prediction[permutation[index]][0] - target[index][0],
                prediction[permutation[index]][1] - target[index][1],
            )
            for index in range(3)
        )
        best_distance = min(best_distance, distance)
    average_distance = best_distance / 3.0
    return math.exp(-(average_distance**2) / (2.0 * sigma**2))


def _decode_compressed_counts(value: str) -> list[int]:
    counts: list[int] = []
    position = 0
    while position < len(value):
        decoded = 0
        shift = 0
        more = True
        while more:
            if position >= len(value):
                raise ValueError("Truncated compressed RLE.")
            code = ord(value[position]) - 48
            position += 1
            decoded |= (code & 0x1F) << (5 * shift)
            more = bool(code & 0x20)
            if not more and code & 0x10:
                decoded |= -1 << (5 * (shift + 1))
            shift += 1
        if len(counts) > 2:
            decoded += counts[-2]
        counts.append(decoded)
    return counts


def decode_coco_rle(value: Any) -> np.ndarray | None:
    """Decode compressed or uncompressed COCO RLE without pycocotools."""

    if not isinstance(value, Mapping):
        return None
    size = value.get("size")
    if not isinstance(size, Sequence) or isinstance(size, (str, bytes)) or len(size) != 2:
        return None
    try:
        height, width = int(size[0]), int(size[1])
    except (TypeError, ValueError):
        return None
    if height <= 0 or width <= 0:
        return None

    raw_counts = value.get("counts")
    try:
        if isinstance(raw_counts, bytes):
            raw_counts = raw_counts.decode("ascii")
        if isinstance(raw_counts, str):
            counts = _decode_compressed_counts(raw_counts)
        elif isinstance(raw_counts, Sequence) and not isinstance(raw_counts, (str, bytes)):
            counts = [int(run) for run in raw_counts]
        else:
            return None
    except (TypeError, ValueError):
        return None
    if any(run < 0 for run in counts):
        return None

    flat = np.zeros(height * width, dtype=np.uint8)
    offset = 0
    foreground = False
    for run in counts:
        end = min(flat.size, offset + run)
        if foreground:
            flat[offset:end] = 1
        offset = end
        foreground = not foreground
        if offset >= flat.size:
            break
    return flat.reshape((width, height)).T.astype(bool)


def _segmentation_output(item: Mapping[str, Any]) -> dict[str, Any] | None:
    return _mapping(item.get("segmentation_output"))


def _rle_value(container: Mapping[Any, Any], key: Any) -> Any:
    if key in container:
        return container[key]
    text_key = str(key)
    if text_key in container:
        return container[text_key]
    for candidate, value in container.items():
        if str(candidate) == text_key:
            return value
    return None


def _metadata_number(
    item: Mapping[str, Any],
    segmentation_output: Mapping[str, Any],
    *names: str,
) -> float | None:
    sources: list[Any] = [item, segmentation_output]
    for key in ("metadata", "video_metadata", "image_metadata"):
        sources.extend(
            source.get(key) for source in (item, segmentation_output) if isinstance(source, Mapping)
        )
    for media_key in ("videos", "images"):
        media = item.get(media_key)
        if isinstance(media, list) and media:
            sources.append(media[0])
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for name in names:
            number = finite_float(source.get(name))
            if number is not None:
                return number
    return None


def _coordinate_size(
    item: Mapping[str, Any],
    mask: np.ndarray,
) -> tuple[int, int]:
    output = _segmentation_output(item) or {}
    sources: list[Any] = [item, output]
    for source in (item, output):
        if not isinstance(source, Mapping):
            continue
        sources.append(_mapping(source.get("resolution")))
        for key in ("metadata", "video_metadata", "image_metadata"):
            sources.append(source.get(key))
    for media_key in ("videos", "images"):
        media = item.get(media_key)
        if isinstance(media, list) and media:
            sources.append(media[0])
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        width = finite_float(source.get("width") or source.get("w"))
        height = finite_float(source.get("height") or source.get("h"))
        if width is not None and height is not None and width > 0 and height > 0:
            return int(round(width)), int(round(height))
    return int(mask.shape[1]), int(mask.shape[0])


def _image_mask(item: Mapping[str, Any]) -> np.ndarray | None:
    output = _segmentation_output(item)
    if output is None:
        return None
    if "counts" in output and "size" in output:
        return decode_coco_rle(output)
    for key in ("segmentation_rle", "rle", "mask", "masks"):
        candidate = output.get(key)
        if isinstance(candidate, Mapping) and {
            "counts",
            "size",
        }.issubset(candidate):
            return decode_coco_rle(candidate)
        if isinstance(candidate, Mapping) and candidate:
            first = next(iter(candidate.values()))
            if isinstance(first, Mapping):
                return decode_coco_rle(first)
    return None


def _video_mask(
    item: Mapping[str, Any],
    predicted_time: float,
) -> tuple[np.ndarray | None, int, int] | None:
    output = _segmentation_output(item)
    if output is None:
        return None
    rles = output.get("segmentation_rle") or output.get("rle") or output.get("masks")
    if not isinstance(rles, Mapping) or not rles:
        return None
    frames = output.get("frames")
    frame_keys = list(frames) if isinstance(frames, list) and frames else list(rles)
    if not frame_keys:
        return None

    fps = _metadata_number(item, output, "fps", "video_fps")
    numeric_keys = [finite_float(key) for key in frame_keys]
    if fps is not None and fps > 0.0:
        target_frame = predicted_time * fps
        if all(key is not None for key in numeric_keys):
            frame_index = min(
                range(len(frame_keys)),
                key=lambda index: abs(float(numeric_keys[index]) - target_frame),
            )
        else:
            frame_index = int(round(target_frame))
    else:
        duration = _metadata_number(item, output, "video_second", "duration", "duration_seconds")
        if duration is not None and duration > 0.0:
            frame_index = int(round(predicted_time / duration * max(len(frame_keys) - 1, 0)))
        elif all(key is not None for key in numeric_keys):
            frame_index = min(
                range(len(frame_keys)),
                key=lambda index: abs(float(numeric_keys[index]) - predicted_time),
            )
        else:
            frame_index = 0
    frame_index = max(0, min(frame_index, len(frame_keys) - 1))
    rle = _rle_value(rles, frame_keys[frame_index])
    return decode_coco_rle(rle), frame_index, len(frame_keys)


def _mask_box(
    mask: np.ndarray,
    coordinate_width: int,
    coordinate_height: int,
) -> list[float] | None:
    y_values, x_values = np.where(mask)
    if not len(x_values):
        return None
    mask_height, mask_width = mask.shape
    return [
        float(x_values.min()) * coordinate_width / mask_width,
        float(y_values.min()) * coordinate_height / mask_height,
        float(x_values.max() + 1) * coordinate_width / mask_width,
        float(y_values.max() + 1) * coordinate_height / mask_height,
    ]


def _denormalize_box(
    box: list[float] | None,
    coordinate_width: int,
    coordinate_height: int,
) -> list[float] | None:
    if box is None:
        return None
    return [
        box[0] * coordinate_width / 1000.0,
        box[1] * coordinate_height / 1000.0,
        box[2] * coordinate_width / 1000.0,
        box[3] * coordinate_height / 1000.0,
    ]


def _point_in_mask(
    mask: np.ndarray,
    point: Sequence[float],
    *,
    coordinate_width: int,
    coordinate_height: int,
    radius: int,
) -> bool:
    mask_height, mask_width = mask.shape
    coordinate_x = point[0] * coordinate_width / 1000.0
    coordinate_y = point[1] * coordinate_height / 1000.0
    x = int(round(coordinate_x * mask_width / coordinate_width))
    y = int(round(coordinate_y * mask_height / coordinate_height))
    if x < 0 or y < 0 or x >= mask_width or y >= mask_height:
        return False
    if mask[y, x]:
        return True
    if radius <= 0:
        return False
    return bool(
        np.any(
            mask[
                max(0, y - radius) : min(mask_height, y + radius + 1),
                max(0, x - radius) : min(mask_width, x + radius + 1),
            ]
        )
    )


def _point_ratio(
    mask: np.ndarray,
    points: list[list[float]] | None,
    *,
    inside: bool,
    coordinate_width: int,
    coordinate_height: int,
    radius: int,
) -> float:
    if points is None:
        return 0.0
    matches: list[bool] = []
    for point in points:
        point_is_inside = _point_in_mask(
            mask,
            point,
            coordinate_width=coordinate_width,
            coordinate_height=coordinate_height,
            radius=radius,
        )
        matches.append(point_is_inside if inside else not point_is_inside)
    return sum(matches) / len(matches)


def _weights(
    kwargs: Mapping[str, Any],
    prefix: str,
    defaults: tuple[float, ...],
) -> tuple[float, ...]:
    names = (
        ("box_weight", "positive_weight", "negative_weight")
        if prefix == "image"
        else ("box_weight", "time_weight", "positive_weight", "negative_weight")
    )
    values: list[float] = []
    for name, default in zip(names, defaults):
        value = finite_float(kwargs.get(f"{prefix}_{name}", default))
        if value is None or value < 0.0:
            raise ValueError(f"{prefix}_{name} must be a non-negative number.")
        values.append(value)
    return tuple(values)


def _modality(item: Mapping[str, Any], target: Mapping[str, Any] | None) -> str:
    data_type = str(item.get("data_type") or "").strip().lower()
    if data_type in {"image", "video"}:
        return data_type
    has_time = target is not None and finite_float(target.get("time")) is not None
    return "video" if has_time else "image"


def _mask_components(
    item: Mapping[str, Any],
    mask: np.ndarray,
    prediction: Mapping[str, Any],
    predicted_box: list[float] | None,
    *,
    weights: tuple[float, ...],
    video: bool,
    radius: int,
    positive_zero_cap: float,
    box_miss_cap: float,
    box_miss_iou: float,
) -> dict[str, float]:
    if not np.any(mask):
        return {
            "accuracy": 0.0,
            "mask_box_iou": 0.0,
            "mask_pos_inside": 0.0,
            "mask_neg_outside": 0.0,
        }
    coordinate_width, coordinate_height = _coordinate_size(item, mask)
    mask_box_iou = box_iou(
        _denormalize_box(
            predicted_box,
            coordinate_width,
            coordinate_height,
        ),
        _mask_box(mask, coordinate_width, coordinate_height),
    )
    positive_inside = _point_ratio(
        mask,
        _points(prediction.get("positive_points")),
        inside=True,
        coordinate_width=coordinate_width,
        coordinate_height=coordinate_height,
        radius=radius,
    )
    negative_outside = _point_ratio(
        mask,
        _points(prediction.get("negative_points")),
        inside=False,
        coordinate_width=coordinate_width,
        coordinate_height=coordinate_height,
        radius=radius,
    )
    if video:
        box_weight, time_weight, positive_weight, negative_weight = weights
        accuracy = (
            box_weight * mask_box_iou
            + time_weight
            + positive_weight * positive_inside
            + negative_weight * negative_outside
        )
    else:
        box_weight, positive_weight, negative_weight = weights
        accuracy = (
            box_weight * mask_box_iou
            + positive_weight * positive_inside
            + negative_weight * negative_outside
        )
    if positive_inside <= 0.0:
        accuracy = min(accuracy, positive_zero_cap)
    if mask_box_iou < box_miss_iou:
        accuracy = min(accuracy, box_miss_cap)
    return {
        "accuracy": max(0.0, min(1.0, accuracy)),
        "mask_box_iou": float(mask_box_iou),
        "mask_pos_inside": float(positive_inside),
        "mask_neg_outside": float(negative_outside),
    }


def compute_score(
    batch: list[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, float]]:
    sigma = finite_float(kwargs.get("point_sigma", POINT_SIGMA))
    time_tau = finite_float(kwargs.get("time_tau", TIME_TAU))
    if sigma is None or sigma <= 0.0:
        raise ValueError("point_sigma must be positive.")
    if time_tau is None or time_tau <= 0.0:
        raise ValueError("time_tau must be positive.")
    image_weights = _weights(kwargs, "image", IMAGE_WEIGHTS)
    video_weights = _weights(kwargs, "video", VIDEO_WEIGHTS)
    mask_aware = bool(kwargs.get("mask_aware", MASK_AWARE))
    require_video_time = bool(kwargs.get("require_video_time", REQUIRE_VIDEO_TIME))
    radius = int(kwargs.get("mask_point_radius", MASK_POINT_RADIUS))
    positive_zero_cap = float(kwargs.get("mask_positive_zero_cap", MASK_POSITIVE_ZERO_CAP))
    box_miss_cap = float(kwargs.get("mask_box_miss_cap", MASK_BOX_MISS_CAP))
    box_miss_iou = float(kwargs.get("mask_box_miss_iou", MASK_BOX_MISS_IOU))

    results: list[dict[str, float]] = []
    for item in batch:
        target = _mapping(ground_truth(item))
        response_payload = exact_answer_payload(item.get("response"))
        prediction = _mapping(response_payload)
        try:
            strict_prediction = json.loads(response_payload or "")
        except (TypeError, ValueError):
            strict_prediction = None
        predicted_box = _box(prediction.get("boxes")) if prediction is not None else None
        target_box = _box(target.get("boxes")) if target is not None else None
        predicted_positive = (
            _points(prediction.get("positive_points")) if prediction is not None else None
        )
        target_positive = _points(target.get("positive_points")) if target is not None else None
        predicted_negative = (
            _points(prediction.get("negative_points")) if prediction is not None else None
        )
        target_negative = _points(target.get("negative_points")) if target is not None else None
        modality = _modality(item, target)
        predicted_time = finite_float(prediction.get("time")) if prediction is not None else None
        target_time = finite_float(target.get("time")) if target is not None else None

        valid_structure = (
            prediction is not None
            and isinstance(strict_prediction, Mapping)
            and predicted_box is not None
            and predicted_positive is not None
            and predicted_negative is not None
            and (modality != "video" or not require_video_time or predicted_time is not None)
        )
        format_score = float(response_payload is not None and valid_structure)

        proxy_box_iou = box_iou(predicted_box, target_box)
        positive_similarity = assignment_similarity(
            predicted_positive, target_positive, sigma=sigma
        )
        negative_similarity = assignment_similarity(
            predicted_negative, target_negative, sigma=sigma
        )
        time_similarity = 0.0
        if predicted_time is not None and target_time is not None:
            time_similarity = math.exp(-abs(predicted_time - target_time) / time_tau)

        mask_score: dict[str, float] | None = None
        if mask_aware and prediction is not None:
            if modality == "video" and predicted_time is not None:
                selected = _video_mask(item, predicted_time)
                if selected is not None:
                    mask, _, _ = selected
                    if mask is None:
                        mask_score = {
                            "accuracy": 0.0,
                            "mask_box_iou": 0.0,
                            "mask_pos_inside": 0.0,
                            "mask_neg_outside": 0.0,
                        }
                    else:
                        mask_score = _mask_components(
                            item,
                            mask,
                            prediction,
                            predicted_box,
                            weights=video_weights,
                            video=True,
                            radius=radius,
                            positive_zero_cap=positive_zero_cap,
                            box_miss_cap=box_miss_cap,
                            box_miss_iou=box_miss_iou,
                        )
            elif modality == "image":
                mask = _image_mask(item)
                if mask is not None:
                    mask_score = _mask_components(
                        item,
                        mask,
                        prediction,
                        predicted_box,
                        weights=image_weights,
                        video=False,
                        radius=radius,
                        positive_zero_cap=positive_zero_cap,
                        box_miss_cap=box_miss_cap,
                        box_miss_iou=box_miss_iou,
                    )

        if mask_score is not None:
            accuracy = mask_score["accuracy"]
        elif modality == "video":
            box_weight, time_weight, positive_weight, negative_weight = video_weights
            accuracy = (
                box_weight * proxy_box_iou
                + time_weight * time_similarity
                + positive_weight * positive_similarity
                + negative_weight * negative_similarity
            )
        else:
            box_weight, positive_weight, negative_weight = image_weights
            accuracy = (
                box_weight * proxy_box_iou
                + positive_weight * positive_similarity
                + negative_weight * negative_similarity
            )
        if modality == "video" and require_video_time and predicted_time is None:
            accuracy = 0.0
        accuracy = max(0.0, min(1.0, accuracy))

        result = {
            "overall": float(accuracy * format_score),
            "accuracy": float(accuracy),
            "format": float(format_score),
            "box_iou": float(proxy_box_iou),
            "pos_sim": float(positive_similarity),
            "neg_sim": float(negative_similarity),
            "time_sim": float(time_similarity),
            "mask_aware_used": float(mask_score is not None),
        }
        if mask_score is not None:
            result.update(
                {
                    "mask_box_iou": mask_score["mask_box_iou"],
                    "mask_pos_inside": mask_score["mask_pos_inside"],
                    "mask_neg_outside": mask_score["mask_neg_outside"],
                }
            )
        results.append(result)
    return results


def build_oracle_response_from_ground_truth(
    ground_truth: Any,
    extra: Any = None,
) -> str:
    del extra
    payload = _mapping(ground_truth)
    if payload is None:
        raise RewardContractError("Segmentation ground truth must be a structured JSON object.")
    return canonical_answer(payload)
