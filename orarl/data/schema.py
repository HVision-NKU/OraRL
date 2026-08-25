"""Canonical record handling for portable OraRL data builds."""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Union

CANONICAL_FIELDS = (
    "problem",
    "answer",
    "images",
    "videos",
    "problem_type",
    "source",
)
MEDIA_PATH_KEYS = (
    "path",
    "video",
    "image",
    "video_path",
    "image_path",
    "file_name",
)
_REMOTE_REFERENCE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_MEDIA_MARKER_RE = re.compile(r"<(?:image|video)>", flags=re.IGNORECASE)


class SchemaError(ValueError):
    """Raised when an input row cannot satisfy the canonical schema."""


def normalize_text(value: Any) -> str:
    """Normalize text for identity comparisons, not for model presentation."""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = _MEDIA_MARKER_RE.sub(" ", text)
    return " ".join(text.split())


def is_remote_reference(value: str) -> bool:
    """Return whether a path asks for a remote resource."""

    return bool(_REMOTE_REFERENCE_RE.match(value.strip()))


def normalize_local_path(value: str, root: Optional[Union[str, Path]] = None) -> str:
    """Normalize a local media path and bind relative paths to ``root``."""

    raw = str(value).strip()
    if not raw:
        raise SchemaError("media path must be nonempty")
    if is_remote_reference(raw):
        raise SchemaError(f"remote media references are not supported: {raw}")

    path = Path(os.path.expanduser(raw))
    if not path.is_absolute() and root is not None:
        path = Path(root) / path
    return os.path.normpath(str(path))


def _message_content(record: Mapping[str, Any], role: str, reverse: bool = False) -> Any:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    values = reversed(messages) if reverse else messages
    for message in values:
        if not isinstance(message, Mapping):
            continue
        if str(message.get("role") or "").casefold() == role:
            content = message.get("content")
            if content is not None:
                return content
    return None


def _first_present(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


def _as_media_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _normalize_media_entry(
    value: Any,
    media_root: Optional[Union[str, Path]],
) -> Any:
    if isinstance(value, str):
        return normalize_local_path(value, media_root)
    if not isinstance(value, Mapping):
        return value

    normalized = dict(value)
    for key in MEDIA_PATH_KEYS:
        candidate = normalized.get(key)
        if isinstance(candidate, str) and candidate.strip():
            normalized[key] = normalize_local_path(candidate, media_root)
    return normalized


def media_entry_path(value: Any) -> Optional[str]:
    """Extract the local path represented by one canonical media entry."""

    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, Mapping):
        for key in MEDIA_PATH_KEYS:
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def media_paths(record: Mapping[str, Any]) -> list[str]:
    """Return de-duplicated image and video paths in stable field order."""

    paths: list[str] = []
    seen: set[str] = set()
    for field in ("images", "videos"):
        values = record.get(field)
        if not isinstance(values, list):
            continue
        for value in values:
            path = media_entry_path(value)
            if path is not None and path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def _answer_has_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return bool(value) and any(_answer_has_content(item) for item in value.values())
    if isinstance(value, Sequence):
        return bool(value) and any(_answer_has_content(item) for item in value)
    return False


def _task_oracle_has_content(record: Mapping[str, Any]) -> bool:
    if str(record.get("problem_type") or "").strip().casefold() != "segmentation":
        return False
    payload = record.get("task_payload")
    if not isinstance(payload, Mapping):
        return False
    output = payload.get("segmentation_output")
    if not isinstance(output, Mapping):
        return False
    if "counts" in output and "size" in output:
        return _answer_has_content(output["counts"]) and _answer_has_content(output["size"])
    return any(
        _answer_has_content(output.get(key))
        for key in ("segmentation_rle", "rle", "mask", "masks")
    )


def validation_errors(record: Any, require_media: bool = False) -> list[str]:
    """Collect canonical schema violations for one record."""

    if not isinstance(record, Mapping):
        return ["record must be a JSON object"]

    errors: list[str] = []
    for field in ("problem", "problem_type", "source"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} must be a nonempty string")

    if (
        "answer" not in record
        or (
            not _answer_has_content(record.get("answer"))
            and not _task_oracle_has_content(record)
        )
    ):
        errors.append("answer must contain a nonempty oracle label")

    for field in ("images", "videos"):
        value = record.get(field)
        if not isinstance(value, list):
            errors.append(f"{field} must be a list")
            continue
        for index, entry in enumerate(value):
            path = media_entry_path(entry)
            if path is None:
                errors.append(f"{field}[{index}] must contain a nonempty local path")
                continue
            if is_remote_reference(path):
                errors.append(f"{field}[{index}] must be a local path")
            elif require_media and not Path(path).is_file():
                errors.append(f"{field}[{index}] does not exist: {path}")

    if isinstance(record.get("images"), list) and isinstance(record.get("videos"), list):
        if not record["images"] and not record["videos"]:
            errors.append("at least one image or video is required")
    return errors


def validate_record(
    record: Any,
    require_media: bool = False,
    context: str = "record",
) -> Mapping[str, Any]:
    """Validate one canonical record and return it unchanged."""

    errors = validation_errors(record, require_media=require_media)
    if errors:
        raise SchemaError(f"{context}: " + "; ".join(errors))
    return record


def canonicalize_record(
    record: Mapping[str, Any],
    *,
    problem_type: Optional[str] = None,
    source: Optional[str] = None,
    family: Optional[str] = None,
    media_root: Optional[Union[str, Path]] = None,
    require_media: bool = False,
    context: str = "record",
) -> dict[str, Any]:
    """Convert common aliases to the canonical OraRL JSONL fields."""

    if not isinstance(record, Mapping):
        raise SchemaError(f"{context}: record must be a JSON object")

    problem = _first_present(record, ("problem", "question", "prompt"))
    if problem is None:
        problem = _message_content(record, "user")

    answer = _first_present(
        record,
        ("answer", "ground_truth", "oracle_label", "oracle_labels", "solution"),
    )
    if answer is None:
        answer = _message_content(record, "assistant", reverse=True)

    images = record.get("images")
    if images is None:
        images = _first_present(record, ("image", "image_path"))
    videos = record.get("videos")
    if videos is None:
        videos = _first_present(record, ("video", "video_path"))

    canonical = dict(record)
    canonical.update(
        {
            "problem": problem,
            "answer": answer,
            "images": [
                _normalize_media_entry(value, media_root) for value in _as_media_list(images)
            ],
            "videos": [
                _normalize_media_entry(value, media_root) for value in _as_media_list(videos)
            ],
            "problem_type": record.get("problem_type") or problem_type,
            "source": source or record.get("source"),
        }
    )
    if family is not None:
        canonical["family"] = family
    for alias in ("image", "video", "image_path", "video_path"):
        canonical.pop(alias, None)

    validate_record(canonical, require_media=require_media, context=context)
    return canonical


def iter_jsonl(path: Union[str, Path]) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file with useful line errors."""

    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SchemaError(f"{input_path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise SchemaError(f"{input_path}:{line_number}: expected a JSON object")
            yield record


def read_records(path: Union[str, Path]) -> list[dict[str, Any]]:
    """Read records from JSONL or a JSON list/object wrapper."""

    input_path = Path(path)
    if input_path.suffix.casefold() in {".jsonl", ".ndjson"}:
        return list(iter_jsonl(input_path))

    try:
        with input_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return list(iter_jsonl(input_path))

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = None
        for key in ("records", "data", "samples", "results"):
            if isinstance(payload.get(key), list):
                records = payload[key]
                break
        if records is None:
            records = [payload]
    else:
        raise SchemaError(f"{input_path}: expected a JSON object or list")

    invalid = [index for index, record in enumerate(records) if not isinstance(record, dict)]
    if invalid:
        raise SchemaError(f"{input_path}: records at indices {invalid[:5]} are not JSON objects")
    return list(records)
