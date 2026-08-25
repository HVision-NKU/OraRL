"""Dependency-light compatibility adapter for canonical OraRL evaluation rows."""

from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_CANONICAL_FIELDS = {
    "schema_version",
    "eval_task",
    "sample_id",
    "benchmark",
    "split",
    "problem",
    "answer",
    "images",
    "videos",
    "problem_type",
    "source",
}
_MEDIA_PATH_KEYS = (
    "path",
    "video",
    "image",
    "video_path",
    "image_path",
    "file_name",
)
_PATH_KEY_RE = re.compile(
    r"(?:^|_)(?:artifact|artifacts|cfg|checkpoint|ckpt|dir|directory|file|"
    r"files|image|images|media|path|paths|prefix|root|subtitle|"
    r"subtitles|tensor|tensors|video|videos)$"
)
_GROUND_TRUTH_KEYS = (
    "bbox",
    "bboxes",
    "boxes",
    "coordinates",
    "duration",
    "end",
    "frame",
    "frame_id",
    "frame_ids",
    "gt_bbox",
    "mask",
    "masks",
    "negative_points",
    "normalized_solution",
    "object",
    "positive_points",
    "query",
    "segment",
    "segmentation",
    "span",
    "spans",
    "start",
    "target",
    "time",
    "timestamp",
    "timestamps",
    "trajectory",
)
_ASSET_DIRECTORIES = frozenset({"artifacts", "media"})


class CanonicalDataError(ValueError):
    """Raised when a canonical row cannot be safely adapted."""


def is_canonical_row(record: Any) -> bool:
    """Return whether a mapping uses the canonical evaluation-row contract."""

    return (
        isinstance(record, Mapping)
        and record.get("schema_version") == 1
        and _CANONICAL_FIELDS.issubset(record)
    )


def _configured_root() -> Path | None:
    value = os.environ.get("ORARL_EVAL_DATA_ROOT", "").strip()
    if value:
        return Path(value).expanduser().resolve()
    manifest = os.environ.get("ORARL_EVAL_DATASETS_JSONL", "").strip()
    if manifest:
        return Path(manifest).expanduser().resolve().parent
    return None


def _configured_asset_root() -> Path | None:
    value = os.environ.get("ORARL_EVAL_ASSET_ROOT", "").strip()
    return Path(value).expanduser().resolve() if value else None


def _infer_root(input_path: Path) -> Path | None:
    configured = _configured_root()
    if configured is not None:
        return configured
    for candidate in (input_path.parent, *input_path.parents):
        if (candidate / "datasets.jsonl").is_file():
            return candidate.resolve()
    return None


def canonical_dataset_record(
    eval_task: str,
    split: str,
    root: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    """Return one canonical ``datasets.jsonl`` record for a task split."""

    repository_root = (
        Path(root).expanduser().resolve() if root is not None else _configured_root()
    )
    if repository_root is None:
        return None
    manifest = repository_root / "datasets.jsonl"
    if not manifest.is_file():
        return None

    normalized_split = str(split).strip().casefold().replace("-", "_")
    matches: list[dict[str, Any]] = []
    try:
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise CanonicalDataError(
                        f"{manifest}:{line_number}: invalid JSON: {error}"
                    ) from error
                if not isinstance(record, Mapping):
                    raise CanonicalDataError(
                        f"{manifest}:{line_number}: dataset record must be an object"
                    )
                record_task = str(record.get("task") or "").casefold()
                record_split = str(record.get("split") or "").casefold()
                if record_task == str(eval_task).casefold() and record_split == normalized_split:
                    matches.append(copy.deepcopy(dict(record)))
    except OSError as error:
        raise CanonicalDataError(f"cannot read canonical dataset manifest: {error}") from error
    if len(matches) > 1:
        raise CanonicalDataError(
            f"duplicate canonical dataset profile for {eval_task}/{normalized_split}"
        )
    return matches[0] if matches else None


def resolve_repository_path(
    value: str,
    root: str | os.PathLike[str],
    asset_root: str | os.PathLike[str] | None = None,
) -> str:
    """Resolve one repository-relative canonical path without allowing escape."""

    raw = str(value).strip()
    if not raw:
        raise CanonicalDataError("canonical repository path must be nonempty")
    repository_root = Path(root).expanduser().resolve()
    configured_asset_root = (
        Path(asset_root).expanduser().resolve()
        if asset_root is not None
        else _configured_asset_root()
    )
    candidate = Path(raw).expanduser()
    selected_root = repository_root
    if (
        not candidate.is_absolute()
        and candidate.parts
        and candidate.parts[0].casefold() in _ASSET_DIRECTORIES
        and configured_asset_root is not None
    ):
        selected_root = configured_asset_root
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (selected_root / candidate).resolve()
    )
    allowed_roots = [repository_root]
    if configured_asset_root is not None:
        allowed_roots.append(configured_asset_root)
    try:
        next(root for root in allowed_roots if resolved.is_relative_to(root))
    except StopIteration as error:
        raise CanonicalDataError(
            f"canonical path resolves outside configured evaluation roots: {raw}"
        ) from error
    return os.path.normpath(str(resolved))


def repository_relative_output_path(
    value: str,
    root: str | os.PathLike[str],
) -> str:
    """Return a ``./``-prefixed path when an asset is contained by ``root``."""

    raw = str(value).strip()
    if not raw or not str(root).strip():
        return raw
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        return raw
    repository_root = Path(root).expanduser().resolve()
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(repository_root)
    except ValueError:
        return raw
    return f"./{relative.as_posix()}"


def _is_path_key(value: object) -> bool:
    return isinstance(value, str) and bool(_PATH_KEY_RE.search(value.casefold()))


def _resolve_media_entry(value: Any, root: Path, asset_root: Path | None) -> Any:
    if isinstance(value, str):
        return resolve_repository_path(value, root, asset_root)
    if not isinstance(value, Mapping):
        return copy.deepcopy(value)
    resolved = copy.deepcopy(dict(value))
    for key in _MEDIA_PATH_KEYS:
        path = resolved.get(key)
        if isinstance(path, str) and path.strip():
            resolved[key] = resolve_repository_path(path, root, asset_root)
    return resolved


def _resolve_declared_paths(
    value: Any,
    root: Path,
    asset_root: Path | None,
    path_context: bool = False,
) -> Any:
    if isinstance(value, str):
        return (
            resolve_repository_path(value, root, asset_root)
            if path_context
            else value
        )
    if isinstance(value, Mapping):
        return {
            key: _resolve_declared_paths(
                item,
                root,
                asset_root,
                path_context=path_context or _is_path_key(key),
            )
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _resolve_declared_paths(
                item,
                root,
                asset_root,
                path_context=path_context,
            )
            for item in value
        ]
    return copy.deepcopy(value)


def _media_path(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in _MEDIA_PATH_KEYS:
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate:
                return candidate
    return ""


def _declared_path_values(value: Any, path_context: bool = False) -> list[str]:
    if isinstance(value, str):
        return [value] if path_context else []
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key, item in value.items():
            paths.extend(
                _declared_path_values(
                    item,
                    path_context=path_context or _is_path_key(key),
                )
            )
        return paths
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        paths = []
        for item in value:
            paths.extend(_declared_path_values(item, path_context=path_context))
        return paths
    return []


def adapt_canonical_row(
    record: Mapping[str, Any],
    root: str | os.PathLike[str] | None = None,
    asset_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Resolve canonical paths and add non-destructive legacy aliases."""

    if not is_canonical_row(record):
        return copy.deepcopy(dict(record))
    repository_root = (
        Path(root).expanduser().resolve() if root is not None else _configured_root()
    )
    if repository_root is None:
        raise CanonicalDataError(
            "canonical rows require ORARL_EVAL_DATA_ROOT or an explicit root"
        )
    resolved_asset_root = (
        Path(asset_root).expanduser().resolve()
        if asset_root is not None
        else _configured_asset_root()
    )

    adapted = copy.deepcopy(dict(record))
    for field in ("images", "videos", "subtitles"):
        values = adapted.get(field)
        if isinstance(values, list):
            adapted[field] = [
                _resolve_media_entry(value, repository_root, resolved_asset_root)
                for value in values
            ]
    for field in ("preprocessed", "task_payload", "metadata", "evaluation"):
        value = adapted.get(field)
        if isinstance(value, Mapping):
            adapted[field] = _resolve_declared_paths(
                value,
                repository_root,
                resolved_asset_root,
            )

    image_paths = [
        path for value in adapted.get("images", []) if (path := _media_path(value))
    ]
    video_paths = [
        path for value in adapted.get("videos", []) if (path := _media_path(value))
    ]
    subtitle_paths = [
        path
        for value in adapted.get("subtitles", [])
        if (path := _media_path(value))
    ]

    adapted.setdefault("id", adapted["sample_id"])
    adapted.setdefault("problem_id", adapted["sample_id"])
    adapted.setdefault("question_id", adapted["sample_id"])
    adapted.setdefault("question", adapted["problem"])
    adapted.setdefault("prompt", adapted["problem"])
    adapted.setdefault("ground_truth", adapted["answer"])
    adapted.setdefault("solution", adapted["answer"])
    adapted.setdefault("dataset", adapted["benchmark"])
    adapted.setdefault("options", copy.deepcopy(adapted.get("choices", [])))
    adapted.setdefault("image_list", image_paths)
    if image_paths:
        adapted.setdefault("image", image_paths[0])
        adapted.setdefault("image_path", image_paths[0])
    if video_paths:
        adapted.setdefault("video", video_paths[0])
        adapted.setdefault("video_path", video_paths[0])
    if video_paths or image_paths:
        adapted.setdefault("path", (video_paths or image_paths)[0])
    if subtitle_paths:
        adapted.setdefault("subtitle", subtitle_paths[0])
        adapted.setdefault("subtitle_path", subtitle_paths[0])

    metadata = adapted.get("metadata")
    if isinstance(metadata, Mapping):
        for key, value in metadata.items():
            adapted.setdefault(str(key), copy.deepcopy(value))
    payload = adapted.get("task_payload")
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            adapted.setdefault(str(key), copy.deepcopy(value))
    if isinstance(adapted.get("answer"), Mapping):
        for key in _GROUND_TRUTH_KEYS:
            if key in adapted["answer"]:
                adapted.setdefault(key, copy.deepcopy(adapted["answer"][key]))

    preprocessed = adapted.get("preprocessed")
    if isinstance(preprocessed, Mapping):
        for key, value in preprocessed.items():
            adapted.setdefault(str(key), copy.deepcopy(value))
        paths = _declared_path_values(preprocessed)
        if paths:
            adapted.setdefault("preprocessed_video", paths[0])
            adapted.setdefault("preprocessed_video_path", paths[0])

    if video_paths:
        adapted.setdefault("data_type", "video")
    elif image_paths:
        adapted.setdefault("data_type", "image")
    adapted.setdefault("question_type", adapted["problem_type"])
    adapted.setdefault("task_type", adapted["problem_type"])
    return adapted


def load_json_records(
    path: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str] | None = None,
    asset_root: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Read JSON/JSONL and adapt only rows that use the canonical contract."""

    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    if not text.strip():
        records: list[Any] = []
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            records = payload if isinstance(payload, list) else [payload]
    if not all(isinstance(record, Mapping) for record in records):
        raise CanonicalDataError(f"{input_path}: every record must be a JSON object")

    repository_root = (
        Path(root).expanduser().resolve()
        if root is not None
        else _infer_root(input_path)
    )
    return [
        adapt_canonical_row(record, repository_root, asset_root)
        if is_canonical_row(record)
        else copy.deepcopy(dict(record))
        for record in records
    ]


__all__ = [
    "CanonicalDataError",
    "adapt_canonical_row",
    "canonical_dataset_record",
    "is_canonical_row",
    "load_json_records",
    "resolve_repository_path",
]
