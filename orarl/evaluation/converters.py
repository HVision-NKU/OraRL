"""Benchmark adapters that produce portable canonical evaluation rows."""

from __future__ import annotations

import ast
import base64
import binascii
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orarl.data.schema import MEDIA_PATH_KEYS, read_records

from .layout import annotation_path, artifact_directory, media_directory, sha256_file
from .schema import EVALUATION_SCHEMA_VERSION, EvaluationSchemaError, validate_evaluation_rows
from .sources import GENERIC_ADAPTERS, EvaluationSource

_PATH_KEY_RE = re.compile(
    r"(?:^|_)(?:artifact|artifacts|dir|directory|file|files|image|images|media|"
    r"path|paths|root|subtitle|subtitles|tensor|tensors|video|videos)$"
)
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_IMAGE_SUFFIXES = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".webp"})
_VIDEO_SUFFIXES = frozenset({".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".webm"})
_SUBTITLE_SUFFIXES = (".srt", ".vtt", ".json", ".txt")
_ARTIFACT_SUFFIXES = (".npz", ".npy", ".safetensors", ".pt", ".pth", ".pkl")
_GROUND_TRUTH_ANSWER_ALIAS = "_".join(("gt", "answer"))
_GROUND_TRUTH_BBOX_ALIAS = "_".join(("gt", "bbox"))
_MINDCUBE_EXPECTED_GROUP_COUNTS = {
    "rotation": 200,
    "among": 600,
    "around": 250,
}
_REVSI_EXPECTED_GROUP_COUNTS = {
    "object_abs_distance": 1232,
    "object_counting": 1091,
    "object_rel_direction": 1387,
    "object_rel_distance": 1351,
    "object_size_estimation": 1134,
    "room_size_estimation": 302,
    "route_planning": 311,
}
_PAYLOAD_KEYS = frozenset(
    {
        "bbox",
        "bboxes",
        "boxes",
        "coordinates",
        "duration",
        "end",
        "frame",
        "frame_id",
        "frame_ids",
        _GROUND_TRUTH_BBOX_ALIAS,
        "mask",
        "masks",
        "normalized_solution",
        "object",
        "segment",
        "segmentation",
        "segmentation_output",
        "span",
        "spans",
        "start",
        "target",
        "timestamp",
        "timestamps",
        "trajectory",
    }
)
_BASE_CONSUMED_FIELDS = frozenset(
    {
        "answer",
        "answers",
        "bench",
        "benchmark",
        "caption",
        "choices",
        "data_source",
        "data_type",
        "dataset",
        "eval_task",
        "evaluation",
        "family",
        "file_name",
        "ground_truth",
        _GROUND_TRUTH_ANSWER_ALIAS,
        _GROUND_TRUTH_BBOX_ALIAS,
        "id",
        "image",
        "image_path",
        "images",
        "index",
        "input_prompt",
        "media",
        "media_path",
        "messages",
        "metadata",
        "normal_caption",
        "options",
        "oracle_label",
        "oracle_labels",
        "path",
        "preprocessed",
        "preprocessed_video",
        "problem",
        "problem_id",
        "problem_type",
        "prompt",
        "question",
        "question_id",
        "question_type",
        "sample_id",
        "schema_version",
        "solution",
        "source",
        "split",
        "subtitle_file",
        "subtitle_path",
        "subtitles",
        "task_payload",
        "task_type",
        "video",
        "video_path",
        "videos",
    }
)
_DROP = object()


class ConversionError(ValueError):
    """Raised when a benchmark source cannot become canonical rows."""


@dataclass(frozen=True)
class PlannedAsset:
    """One content-addressed output asset and its private local source."""

    repository_path: str
    sha256: str | None
    bytes: int
    kind: str
    benchmark: str
    license: str
    source_path: Path | None = None
    content: bytes | None = None
    missing: bool = False


@dataclass(frozen=True)
class ConvertedSource:
    """Canonical rows and assets produced for one source record."""

    source: EvaluationSource
    rows: tuple[dict[str, Any], ...]
    assets: tuple[PlannedAsset, ...]
    source_rows: int

    @property
    def missing_assets(self) -> tuple[PlannedAsset, ...]:
        return tuple(asset for asset in self.assets if asset.missing)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_suffix(path: Path) -> str:
    suffix = path.suffix.casefold()
    aliases = {".jpeg": ".jpg", ".tiff": ".tif"}
    suffix = aliases.get(suffix, suffix)
    if not re.fullmatch(r"\.[a-z0-9]{1,12}", suffix):
        return ""
    return suffix


def _image_suffix(content: bytes) -> str:
    if content.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if content.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if content.startswith(b"BM"):
        return ".bmp"
    if content[:4] in {b"II*\x00", b"MM\x00*"}:
        return ".tif"
    if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return ".webp"
    return ".bin"


def _is_path_key(key: object) -> bool:
    return isinstance(key, str) and bool(_PATH_KEY_RE.search(key.casefold()))


def _is_private_path(value: str, source: EvaluationSource) -> bool:
    stripped = value.strip()
    if (
        stripped.startswith(("/", "~/", "file://"))
        or _WINDOWS_ABSOLUTE_RE.match(stripped)
    ):
        return True
    private_roots = {
        str(source.annotation_input),
        str(source.annotation_input.parent),
        *(str(path) for path in source.media_roots.values()),
        *(str(path) for path in source.preprocessed_roots.values()),
    }
    return any(root and root in stripped for root in private_roots)


def _json_safe(value: Any, source: EvaluationSource) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _DROP
    if isinstance(value, str):
        return _DROP if _is_private_path(value, source) else value
    if isinstance(value, (bytes, bytearray, memoryview, Path)):
        return _DROP
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str) or _is_private_path(raw_key, source):
                continue
            safe = _json_safe(item, source)
            if safe is not _DROP:
                converted[raw_key] = safe
        return converted
    if isinstance(value, Sequence):
        converted_items = []
        for item in value:
            safe = _json_safe(item, source)
            if safe is not _DROP:
                converted_items.append(safe)
        return converted_items

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            scalar = item_method()
        except (TypeError, ValueError):
            scalar = value
        if scalar is not value:
            return _json_safe(scalar, source)
    list_method = getattr(value, "tolist", None)
    if callable(list_method):
        try:
            listed = list_method()
        except (TypeError, ValueError):
            listed = value
        if listed is not value:
            return _json_safe(listed, source)
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return str(isoformat())
        except (TypeError, ValueError):
            pass
    return _DROP


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    list_method = getattr(value, "tolist", None)
    if callable(list_method) and not isinstance(value, (str, bytes, bytearray)):
        try:
            value = list_method()
        except (TypeError, ValueError):
            pass
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


def _is_blank_path_reference(value: Any) -> bool:
    if not isinstance(value, (str, os.PathLike)):
        return False
    raw = os.fspath(value)
    return not raw.strip()


def _first(record: Mapping[str, Any], fields: Sequence[str]) -> Any:
    for field in fields:
        value = record.get(field)
        if value is not None and not _is_blank_path_reference(value):
            return value
    return None


def _message_content(
    record: Mapping[str, Any],
    role: str,
    *,
    reverse: bool = False,
) -> Any:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    candidates = reversed(messages) if reverse else messages
    for message in candidates:
        if not isinstance(message, Mapping):
            continue
        if str(message.get("role") or "").casefold() != role:
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text = "\n".join(
                str(item.get("text"))
                for item in content
                if isinstance(item, Mapping) and item.get("text") is not None
            ).strip()
            if text:
                return text
    return None


class _AssetPlanner:
    def __init__(
        self,
        source: EvaluationSource,
        *,
        hash_assets: bool = False,
    ):
        self.source = source
        self.hash_assets = hash_assets
        self._assets: dict[str, PlannedAsset] = {}
        self._source_paths: dict[tuple[str, str], str] = {}
        self._missing_paths: dict[tuple[str, str], str] = {}
        self._references: dict[tuple[str, str, str], tuple[str, Path]] = {}
        self._resolved_paths: dict[tuple[str, str], Path] = {}
        self._embedded_counts: dict[str, int] = {}

    @property
    def assets(self) -> tuple[PlannedAsset, ...]:
        return tuple(
            sorted(
                self._assets.values(),
                key=lambda asset: (asset.repository_path, asset.missing),
            )
        )

    def _destination(self, kind: str, digest: str, suffix: str) -> str:
        filename = digest + suffix
        if kind == "artifacts":
            return f"{artifact_directory(self.source.benchmark)}/{filename}"
        return f"{media_directory(self.source.benchmark, kind)}/{filename}"

    def _source_destination(self, kind: str, path: Path, root: Path) -> str:
        try:
            relative = path.relative_to(root.resolve())
        except ValueError:
            relative = Path(path.name)
        raw_name = "__".join(relative.parts)
        filename = re.sub(r"[^A-Za-z0-9._+-]+", "_", raw_name).strip("_")
        if not filename or filename in {".", ".."}:
            raise ConversionError(f"asset path has no portable filename: {path}")
        if len(filename.encode("utf-8")) > 220:
            suffix = _canonical_suffix(path)
            stem = filename[: -len(suffix)] if suffix and filename.endswith(suffix) else filename
            path_token = hashlib.blake2b(
                relative.as_posix().encode("utf-8"),
                digest_size=10,
            ).hexdigest()
            stem_limit = 220 - len(suffix.encode("utf-8")) - len(path_token) - 2
            filename = f"{stem[:stem_limit]}--{path_token}{suffix}"
        if kind == "artifacts":
            directory = artifact_directory(self.source.benchmark)
        else:
            directory = media_directory(self.source.benchmark, kind)
        return f"{directory}/{self.source.split}/{filename}"

    @staticmethod
    def _reference_key(
        reference: object,
        root: Path,
        kind: str,
    ) -> tuple[str, str, str]:
        if not isinstance(reference, (str, os.PathLike)):
            raise ConversionError("asset reference must be a local path string")
        raw_reference = os.fspath(reference).strip()
        return (
            kind,
            os.path.normcase(os.path.abspath(os.path.normpath(str(root)))),
            os.path.normcase(os.path.normpath(raw_reference)),
        )

    def resolve(self, reference: object, root: Path) -> Path:
        if not isinstance(reference, (str, os.PathLike)):
            raise ConversionError("asset reference must be a local path string")
        raw = os.fspath(reference).strip()
        if not raw:
            raise ConversionError("asset reference must be nonempty")
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", raw):
            raise ConversionError(f"remote asset references are unsupported: {raw}")
        resolved_key = (
            os.path.normcase(os.path.abspath(os.path.normpath(str(root)))),
            os.path.normcase(os.path.normpath(raw)),
        )
        cached = self._resolved_paths.get(resolved_key)
        if cached is not None:
            return cached

        candidate = Path(os.path.expanduser(raw))
        if candidate.is_absolute():
            candidates = [candidate]
        else:
            normalized = Path(os.path.normpath(raw))
            candidates = [root / normalized]
            if normalized.parts and normalized.parts[0].casefold() == root.name.casefold():
                candidates.append(root.parent / normalized)
            if normalized.parts and normalized.parts[0].casefold() == "evaluation":
                candidates.append(root.joinpath(*normalized.parts[1:]))
            candidates.extend(
                (
                    self.source.annotation.parent / normalized,
                    root / normalized.name,
                )
            )
        unique: list[Path] = []
        seen: set[str] = set()
        for path in candidates:
            normalized_path = Path(os.path.abspath(os.path.normpath(str(path))))
            key = os.path.normcase(str(normalized_path))
            if key not in seen:
                seen.add(key)
                unique.append(normalized_path)
        for path in unique:
            if path.is_file():
                self._resolved_paths[resolved_key] = path
                return path
        self._resolved_paths[resolved_key] = unique[0]
        return unique[0]

    def mapped_path(
        self,
        reference: object,
        root: Path,
        *,
        kind: str,
    ) -> str | None:
        reference_key = self._reference_key(reference, root, kind)
        cached = self._references.get(reference_key)
        if cached is not None:
            return cached[0]
        path = self.resolve(reference, root)
        mapped = self._source_paths.get((kind, os.path.normcase(str(path))))
        if mapped is not None:
            self._references[reference_key] = (mapped, path)
        return mapped

    def add_file(self, reference: object, *, kind: str, root: Path) -> tuple[str, Path]:
        reference_key = self._reference_key(reference, root, kind)
        cached = self._references.get(reference_key)
        if cached is not None:
            return cached

        path = self.resolve(reference, root)
        source_key = os.path.normcase(str(path))
        mapped = self._source_paths.get((kind, source_key))
        if mapped is not None:
            result = (mapped, path)
            self._references[reference_key] = result
            return result

        if not path.is_file():
            missing_key = (kind, source_key)
            repository_path = self._missing_paths.get(missing_key)
            if repository_path is None:
                token = hashlib.sha256(f"{kind}\0{source_key}".encode("utf-8")).hexdigest()
                repository_path = self._destination(
                    kind,
                    f"missing-{token}",
                    _canonical_suffix(path),
                )
                self._missing_paths[missing_key] = repository_path
                self._assets[repository_path] = PlannedAsset(
                    repository_path=repository_path,
                    sha256=None,
                    bytes=0,
                    kind=kind,
                    benchmark=self.source.benchmark,
                    license=self.source.license,
                    source_path=path,
                    missing=True,
                )
            self._source_paths[(kind, source_key)] = repository_path
            result = (repository_path, path)
            self._references[reference_key] = result
            return result

        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise ConversionError(f"asset source is not a regular file: {path}")
        byte_count = resolved.stat().st_size
        if self.hash_assets:
            digest = sha256_file(resolved)
            repository_path = self._destination(
                kind,
                digest,
                _canonical_suffix(resolved),
            )
        else:
            digest = None
            repository_path = self._source_destination(kind, resolved, root)
        asset = PlannedAsset(
            repository_path=repository_path,
            sha256=digest,
            bytes=byte_count,
            kind=kind,
            benchmark=self.source.benchmark,
            license=self.source.license,
            source_path=resolved,
        )
        previous = self._assets.get(repository_path)
        if previous is not None and (
            previous.sha256 != asset.sha256
            or previous.bytes != asset.bytes
            or (
                asset.sha256 is None
                and previous.source_path != asset.source_path
            )
        ):
            raise ConversionError(f"asset output path collision: {repository_path}")
        self._assets[repository_path] = asset
        self._source_paths[(kind, source_key)] = repository_path
        self._source_paths[
            (kind, os.path.normcase(str(resolved)))
        ] = repository_path
        result = (repository_path, resolved)
        self._references[reference_key] = result
        return result

    def add_bytes(
        self,
        content: bytes,
        *,
        kind: str,
        suffix: str = "",
    ) -> str:
        if self.hash_assets:
            digest = hashlib.sha256(content).hexdigest()
            repository_path = self._destination(kind, digest, suffix)
        else:
            digest = None
            index = self._embedded_counts.get(kind, 0) + 1
            self._embedded_counts[kind] = index
            filename = f"{self.source.split}__embedded_{index:08d}{suffix}"
            if kind == "artifacts":
                directory = artifact_directory(self.source.benchmark)
            else:
                directory = media_directory(self.source.benchmark, kind)
            repository_path = f"{directory}/{self.source.split}/{filename}"
        asset = PlannedAsset(
            repository_path=repository_path,
            sha256=digest,
            bytes=len(content),
            kind=kind,
            benchmark=self.source.benchmark,
            license=self.source.license,
            content=content,
        )
        previous = self._assets.get(repository_path)
        if previous is not None and (
            previous.sha256 != asset.sha256 or previous.bytes != asset.bytes
        ):
            raise ConversionError(f"asset output path collision: {repository_path}")
        self._assets[repository_path] = asset
        return repository_path


def _kind_for_key(key: str, default: str = "artifacts") -> str:
    folded = key.casefold()
    if "subtitle" in folded:
        return "subtitles"
    if "image" in folded:
        return "images"
    if "video" in folded:
        return "videos"
    return default


def _root_for_kind(source: EvaluationSource, kind: str, key: str = "") -> Path:
    if kind == "artifacts":
        return source.preprocessed_root(key)
    return source.media_root(kind)


def _stage_path_value(
    value: Any,
    *,
    key: str,
    source: EvaluationSource,
    planner: _AssetPlanner,
    default_kind: str = "artifacts",
    force_kind: str | None = None,
) -> Any:
    kind = force_kind or _kind_for_key(key, default_kind)
    root = _root_for_kind(source, kind, key)
    if isinstance(value, (str, os.PathLike)):
        if _is_blank_path_reference(value):
            return _DROP
        local_path = planner.resolve(value, root)
        annotation_sources = {
            os.path.normcase(str(source.annotation)),
            os.path.normcase(str(source.annotation.resolve())),
        }
        if os.path.normcase(str(local_path)) in annotation_sources:
            return annotation_path(source.benchmark, source.split)
        mapped = planner.mapped_path(value, root, kind=kind)
        if mapped is not None:
            return mapped
        repository_path, _local = planner.add_file(value, kind=kind, root=root)
        return repository_path
    if isinstance(value, Mapping):
        staged: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str) or _is_private_path(raw_key, source):
                continue
            converted = _stage_path_value(
                item,
                key=raw_key,
                source=source,
                planner=planner,
                default_kind=kind,
                force_kind=force_kind,
            )
            if converted is not _DROP:
                staged[raw_key] = converted
        return staged
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        staged_items = []
        for item in value:
            converted = _stage_path_value(
                item,
                key=key,
                source=source,
                planner=planner,
                default_kind=kind,
                force_kind=force_kind,
            )
            if converted is not _DROP:
                staged_items.append(converted)
        return staged_items
    raise ConversionError(f"{key} must contain a local path string or list")


def _stage_payload(
    value: Any,
    *,
    source: EvaluationSource,
    planner: _AssetPlanner,
    path_kind: str | None = None,
) -> Any:
    if isinstance(value, Mapping):
        staged: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str) or _is_private_path(raw_key, source):
                continue
            if _is_path_key(raw_key):
                converted = _stage_path_value(
                    item,
                    key=raw_key,
                    source=source,
                    planner=planner,
                    force_kind=path_kind,
                )
                if converted is not _DROP:
                    staged[raw_key] = converted
                continue
            nested = _stage_payload(
                item,
                source=source,
                planner=planner,
                path_kind=path_kind,
            )
            if nested is not _DROP:
                staged[raw_key] = nested
        return staged
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        staged_items = []
        for item in value:
            nested = _stage_payload(
                item,
                source=source,
                planner=planner,
                path_kind=path_kind,
            )
            if nested is not _DROP:
                staged_items.append(nested)
        return staged_items
    return _json_safe(value, source)


def _profile_without_paths(value: Any, source: EvaluationSource) -> Any:
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for raw_key, item in value.items():
            if (
                not isinstance(raw_key, str)
                or _is_private_path(raw_key, source)
                or _is_path_key(raw_key)
            ):
                continue
            safe = _profile_without_paths(item, source)
            if safe is not _DROP:
                converted[raw_key] = safe
        return converted
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        converted_items = []
        for item in value:
            safe = _profile_without_paths(item, source)
            if safe is not _DROP:
                converted_items.append(safe)
        return converted_items
    return _json_safe(value, source)


def _structured_answer_payload(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        return None
    matches = re.findall(
        r"<answer>\s*(.*?)\s*</answer>",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    candidate = matches[-1] if matches else value.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _stage_media_entry(
    entry: Any,
    *,
    kind: str,
    source: EvaluationSource,
    planner: _AssetPlanner,
) -> tuple[Any, Path]:
    root = source.media_root(kind)
    if isinstance(entry, (str, os.PathLike)):
        if _is_blank_path_reference(entry):
            raise ConversionError(f"{kind} entry contains an empty local path")
        repository_path, local_path = planner.add_file(entry, kind=kind, root=root)
        return repository_path, local_path
    if not isinstance(entry, Mapping):
        raise ConversionError(f"{kind} entry must be a path string or JSON object")

    staged: dict[str, Any] = {}
    local_path: Path | None = None
    for raw_key, value in entry.items():
        if not isinstance(raw_key, str) or _is_private_path(raw_key, source):
            continue
        if raw_key in MEDIA_PATH_KEYS and isinstance(value, (str, os.PathLike)):
            if _is_blank_path_reference(value):
                continue
            repository_path, candidate = planner.add_file(value, kind=kind, root=root)
            staged[raw_key] = repository_path
            if local_path is None:
                local_path = candidate
        else:
            safe = _json_safe(value, source)
            if safe is not _DROP:
                staged[raw_key] = safe
    if local_path is None:
        raise ConversionError(f"{kind} entry does not contain a local path")
    return staged, local_path


def _media_inputs(record: Mapping[str, Any], source: EvaluationSource) -> tuple[Any, Any]:
    images = _first(record, ("images", "image", "image_path"))
    videos = _first(record, ("videos", "video", "video_path"))
    generic = _first(record, ("path", "media_path", "media", "file_name"))
    if source.eval_task == "mvbench" and generic is not None:
        # MVBench keeps the upstream WebM filename in ``video`` while ``path``
        # points to the materialized MP4 used by its evaluator.
        videos = generic
    if generic is not None and images is None and videos is None:
        values = _as_list(generic)
        suffixes = {
            Path(str(value)).suffix.casefold()
            for value in values
            if isinstance(value, (str, os.PathLike))
        }
        data_type = str(record.get("data_type") or "").casefold()
        family = source.family.casefold()
        if suffixes and suffixes <= _IMAGE_SUFFIXES:
            images = generic
        elif (
            "image" in data_type
            or source.adapter == "spatial_grounding"
            or ("spatial_grounding" in family and "temporal" not in family)
        ):
            images = generic
        else:
            videos = generic
    return images, videos


def _stage_media(
    record: Mapping[str, Any],
    source: EvaluationSource,
    planner: _AssetPlanner,
) -> tuple[list[Any], list[Any], list[Path]]:
    raw_images, raw_videos = _media_inputs(record, source)
    images: list[Any] = []
    videos: list[Any] = []
    video_sources: list[Path] = []
    for entry in _as_list(raw_images):
        if _is_blank_path_reference(entry):
            continue
        staged, _local = _stage_media_entry(
            entry,
            kind="images",
            source=source,
            planner=planner,
        )
        images.append(staged)
    for entry in _as_list(raw_videos):
        if _is_blank_path_reference(entry):
            continue
        staged, local = _stage_media_entry(
            entry,
            kind="videos",
            source=source,
            planner=planner,
        )
        videos.append(staged)
        video_sources.append(local)
    return images, videos, video_sources


def _subtitle_input(record: Mapping[str, Any], source: EvaluationSource) -> tuple[Any, bool]:
    explicit = _first(record, ("subtitles", "subtitle_path", "subtitle_file"))
    if explicit is not None:
        return explicit, True
    subtitle = record.get("subtitle")
    if subtitle is None:
        return None, False
    values = _as_list(subtitle)
    if all(
        isinstance(value, str)
        and (
            Path(value).suffix.casefold() in _SUBTITLE_SUFFIXES
            or source.media_root("subtitles").joinpath(value).is_file()
        )
        for value in values
    ):
        return subtitle, True
    return None, False


def _derived_files(
    source_paths: Sequence[Path],
    root: Path,
    suffixes: Sequence[str],
    video_root: Path | None = None,
) -> list[Path]:
    derived: list[Path] = []
    seen: set[str] = set()
    for source_path in source_paths:
        relative = Path(source_path.name)
        if video_root is not None:
            try:
                relative = source_path.relative_to(video_root)
            except ValueError:
                pass
        found = False
        for suffix in suffixes:
            candidates = (
                root / relative.with_suffix(suffix),
                root / f"{source_path.stem}{suffix}",
            )
            for candidate in candidates:
                if not candidate.is_file():
                    continue
                key = os.path.normcase(str(candidate.resolve()))
                if key not in seen:
                    seen.add(key)
                    derived.append(candidate)
                found = True
                break
            if found:
                break
    return derived


def _stage_subtitles(
    record: Mapping[str, Any],
    source: EvaluationSource,
    planner: _AssetPlanner,
    video_sources: Sequence[Path],
) -> tuple[list[Any], bool]:
    raw, consumed_subtitle = _subtitle_input(record, source)
    subtitles: list[Any] = []
    if raw is not None:
        for entry in _as_list(raw):
            if _is_blank_path_reference(entry):
                continue
            staged, _local = _stage_media_entry(
                entry,
                kind="subtitles",
                source=source,
                planner=planner,
            )
            subtitles.append(staged)
    elif "subtitles" in source.media_roots:
        for path in _derived_files(
            video_sources,
            source.media_roots["subtitles"],
            _SUBTITLE_SUFFIXES,
            source.media_roots.get("videos", source.media_roots.get("default")),
        ):
            staged, _local = _stage_media_entry(
                path,
                kind="subtitles",
                source=source,
                planner=planner,
            )
            subtitles.append(staged)
    return subtitles, consumed_subtitle


def _stage_preprocessed(
    record: Mapping[str, Any],
    source: EvaluationSource,
    planner: _AssetPlanner,
    video_sources: Sequence[Path],
) -> dict[str, Any]:
    raw = record.get("preprocessed")
    staged: dict[str, Any] = {}
    if raw is not None:
        if isinstance(raw, Mapping):
            converted = _stage_payload(
                raw,
                source=source,
                planner=planner,
                path_kind="artifacts",
            )
            if isinstance(converted, Mapping):
                staged.update(converted)
        elif isinstance(raw, (str, os.PathLike)):
            converted = _stage_path_value(
                raw,
                key="path",
                source=source,
                planner=planner,
                force_kind="artifacts",
            )
            if converted is not _DROP:
                staged["path"] = converted
        else:
            raise ConversionError("preprocessed must be a JSON object or local path")

    explicit_fields = (
        "artifact_path",
        "preprocessed_path",
        "preprocessed_video",
        "preprocessed_video_path",
        "tensor_path",
    )
    for field in explicit_fields:
        if record.get(field) is not None:
            converted = _stage_path_value(
                record[field],
                key=field,
                source=source,
                planner=planner,
                force_kind="artifacts",
            )
            if converted is not _DROP:
                staged[field] = converted

    if not any(_is_path_key(key) for key in staged) and source.preprocessed_roots:
        root = source.preprocessed_root()
        derived = _derived_files(
            video_sources,
            root,
            _ARTIFACT_SUFFIXES,
            source.media_roots.get("videos", source.media_roots.get("default")),
        )
        if len(derived) == 1:
            staged["video_path"] = _stage_path_value(
                derived[0],
                key="video_path",
                source=source,
                planner=planner,
                force_kind="artifacts",
            )
        elif derived:
            staged["video_paths"] = _stage_path_value(
                derived,
                key="video_paths",
                source=source,
                planner=planner,
                force_kind="artifacts",
            )
    return staged


def _choices(value: Any, source: EvaluationSource) -> list[str]:
    if isinstance(value, Mapping):
        value = [value[key] for key in sorted(value, key=str)]
    choices = []
    for item in _as_list(value):
        safe = _json_safe(item, source)
        if safe is _DROP:
            continue
        text = str(safe).strip()
        if text:
            choices.append(text)
    return choices


def _sample_id(record: Mapping[str, Any], source: EvaluationSource, index: int) -> str:
    value = _first(
        record,
        ("sample_id", "question_id", "problem_id", "id", "uid", "index"),
    )
    if value is None:
        return f"{source.benchmark}-{index + 1:08d}"
    text = str(value).strip()
    if not text:
        raise ConversionError(f"source row {index + 1}: sample id must be nonempty")
    if _is_private_path(text, source):
        return f"{source.benchmark}-{index + 1:08d}"
    return text


def _canonical_row(
    record: Mapping[str, Any],
    source: EvaluationSource,
    planner: _AssetPlanner,
    index: int,
    *,
    images_override: list[Any] | None = None,
    videos_override: list[Any] | None = None,
) -> dict[str, Any]:
    if source.adapter == "spatial_grounding":
        problem_keys = (
            "expression",
            "normal_caption",
            "caption",
            "query",
            "problem",
            "question",
            "prompt",
            "input_prompt",
        )
        answer_keys = (
            "normalized_solution",
            "bbox",
            _GROUND_TRUTH_BBOX_ALIAS,
            "solution",
            "answer",
            "answers",
            "ground_truth",
            "oracle_label",
            "oracle_labels",
            _GROUND_TRUTH_ANSWER_ALIAS,
        )
    elif source.eval_task == "mvbench":
        problem_keys = (
            "problem",
            "question",
            "prompt",
            "input_prompt",
        )
        answer_keys = (
            "solution",
            "answer",
            "ground_truth",
            _GROUND_TRUTH_ANSWER_ALIAS,
        )
    else:
        problem_keys = (
            "problem",
            "question",
            "prompt",
            "input_prompt",
            "expression",
            "normal_caption",
            "caption",
            "query",
        )
        answer_keys = (
            "answer",
            "answers",
            "ground_truth",
            "oracle_label",
            "oracle_labels",
            "solution",
            _GROUND_TRUTH_ANSWER_ALIAS,
            "bbox",
            "normalized_solution",
            _GROUND_TRUTH_BBOX_ALIAS,
            "span",
        )
    problem = _first(
        record,
        problem_keys,
    )
    answer = _first(
        record,
        answer_keys,
    )
    if source.eval_task == "mvbench" and isinstance(answer, str):
        tagged_answer = re.fullmatch(
            r"\s*<answer>\s*([A-Za-z])\s*</answer>\s*",
            answer,
            flags=re.IGNORECASE,
        )
        if tagged_answer is not None:
            answer = tagged_answer.group(1).upper()
    if problem is None:
        problem = _message_content(record, "user")
    if answer is None:
        answer = _message_content(record, "assistant", reverse=True)
    safe_problem = _json_safe(problem, source)
    safe_answer = _json_safe(answer, source)
    if safe_problem is _DROP:
        safe_problem = None
    if safe_answer is _DROP:
        safe_answer = None
    if (
        source.eval_task == "segmentation"
        and isinstance(safe_answer, str)
        and not safe_answer.strip()
        and isinstance(record.get("segmentation_output"), Mapping)
    ):
        safe_answer = None

    if images_override is None or videos_override is None:
        staged_images, staged_videos, video_sources = _stage_media(record, source, planner)
        images = staged_images if images_override is None else images_override
        videos = staged_videos if videos_override is None else videos_override
    else:
        images = images_override
        videos = videos_override
        video_sources = []

    subtitles, consumed_subtitle = _stage_subtitles(
        record,
        source,
        planner,
        video_sources,
    )
    preprocessed = _stage_preprocessed(record, source, planner, video_sources)

    task_payload: dict[str, Any] = {}
    parsed_answer = _structured_answer_payload(answer)
    if parsed_answer is not None:
        staged_answer = _stage_payload(
            parsed_answer,
            source=source,
            planner=planner,
        )
        if isinstance(staged_answer, Mapping):
            task_payload.update(staged_answer)
    raw_payload = record.get("task_payload")
    if isinstance(raw_payload, Mapping):
        staged_payload = _stage_payload(raw_payload, source=source, planner=planner)
        if isinstance(staged_payload, Mapping):
            task_payload.update(staged_payload)
    metadata: dict[str, Any] = {}
    raw_metadata = record.get("metadata")
    if isinstance(raw_metadata, Mapping):
        staged_metadata = _stage_payload(raw_metadata, source=source, planner=planner)
        if isinstance(staged_metadata, Mapping):
            metadata.update(staged_metadata)

    consumed = set(_BASE_CONSUMED_FIELDS)
    consumed.update(
        {
            "artifact_path",
            "preprocessed_path",
            "preprocessed_video",
            "preprocessed_video_path",
            "tensor_path",
        }
    )
    if consumed_subtitle:
        consumed.add("subtitle")
    for identifier in (
        "question_id",
        "problem_id",
        "id",
        "uid",
        "index",
        "data_type",
        "input_prompt",
        "question_type",
        "task_type",
    ):
        if record.get(identifier) is not None:
            safe = _json_safe(record[identifier], source)
            if safe is not _DROP:
                metadata.setdefault(identifier, safe)

    for raw_key, value in record.items():
        if (
            not isinstance(raw_key, str)
            or _is_private_path(raw_key, source)
            or raw_key in consumed
        ):
            continue
        if raw_key in _PAYLOAD_KEYS or _is_path_key(raw_key):
            staged = (
                _stage_path_value(
                    value,
                    key=raw_key,
                    source=source,
                    planner=planner,
                )
                if _is_path_key(raw_key)
                else _stage_payload(value, source=source, planner=planner)
            )
            if staged is not _DROP:
                task_payload.setdefault(raw_key, staged)
        else:
            safe = _stage_payload(value, source=source, planner=planner)
            if safe is not _DROP:
                metadata.setdefault(raw_key, safe)

    row_evaluation = record.get("evaluation")
    evaluation = _profile_without_paths(source.evaluation, source)
    if not isinstance(evaluation, dict):
        evaluation = {}
    if isinstance(row_evaluation, Mapping):
        safe_evaluation = _profile_without_paths(row_evaluation, source)
        if isinstance(safe_evaluation, Mapping):
            evaluation.update(safe_evaluation)

    source_name = _first(record, ("source", "data_source", "dataset", "bench"))
    safe_source_name = _json_safe(source_name, source)
    if safe_source_name is _DROP or not str(safe_source_name or "").strip():
        safe_source_name = source.benchmark
    problem_type = _first(record, ("problem_type", "question_type", "task_type"))
    if (
        not isinstance(problem_type, str)
        or not problem_type.strip()
        or _is_private_path(problem_type, source)
    ):
        problem_type = source.eval_task

    row: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "eval_task": source.eval_task,
        "sample_id": _sample_id(record, source, index),
        "benchmark": source.benchmark,
        "split": source.split,
        "problem": safe_problem,
        "answer": safe_answer,
        "images": images,
        "videos": videos,
        "problem_type": problem_type,
        "source": str(safe_source_name),
        "family": source.family,
        "task_payload": task_payload,
        "evaluation": evaluation,
    }
    row_choices = _choices(_first(record, ("choices", "options")), source)
    if row_choices:
        row["choices"] = row_choices
    if subtitles:
        row["subtitles"] = subtitles
    if preprocessed:
        row["preprocessed"] = preprocessed
    if metadata:
        row["metadata"] = metadata
    return row


def _generic_records(
    source: EvaluationSource,
    planner: _AssetPlanner,
) -> tuple[list[dict[str, Any]], int]:
    try:
        records = read_records(source.annotation_input)
    except (OSError, ValueError) as error:
        raise ConversionError(f"{source.annotation_input}: {error}") from error
    if len(records) == 1:
        container = records[0]
        for field in ("annotations", "items", "questions"):
            nested = container.get(field)
            if isinstance(nested, list) and all(isinstance(item, Mapping) for item in nested):
                records = [dict(item) for item in nested]
                break
        else:
            row_fields = {
                "problem",
                "question",
                "prompt",
                "expression",
                "normal_caption",
                "caption",
            }
            if (
                container
                and not row_fields.intersection(container)
                and all(isinstance(item, Mapping) for item in container.values())
            ):
                flattened = []
                for key in sorted(container, key=str):
                    item = dict(container[key])
                    item.setdefault("id", key)
                    flattened.append(item)
                records = flattened
    rows = []
    for index, record in enumerate(records):
        try:
            rows.append(_canonical_row(record, source, planner, index))
        except ConversionError as error:
            raise ConversionError(f"source row {index + 1}: {error}") from error
    return rows, len(records)


def _decode_base64_images(value: Any, *, context: str) -> list[bytes]:
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = value
    else:
        parsed = value
    values = _as_list(parsed)
    images: list[bytes] = []
    for index, item in enumerate(values):
        if item is None or item == "":
            continue
        if isinstance(item, (bytes, bytearray, memoryview)):
            images.append(bytes(item))
            continue
        encoded = str(item).strip()
        if encoded.startswith("data:") and "," in encoded:
            encoded = encoded.split(",", 1)[1]
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ConversionError(f"{context}: image {index} is not valid base64") from error
        if not decoded:
            raise ConversionError(f"{context}: image {index} decoded to empty bytes")
        images.append(decoded)
    return images


def _mmsi_records(
    source: EvaluationSource,
    planner: _AssetPlanner,
) -> tuple[list[dict[str, Any]], int]:
    csv.field_size_limit(sys.maxsize)
    try:
        with source.annotation_input.open(
            "r",
            encoding="utf-8",
            errors="replace",
            newline="",
        ) as handle:
            records = list(csv.DictReader(handle, delimiter="\t"))
    except (csv.Error, OSError) as error:
        raise ConversionError(f"{source.annotation_input}: invalid MMSI TSV: {error}") from error
    if not records and source.expected_count:
        raise ConversionError(f"{source.annotation_input}: MMSI TSV has no source rows")

    rows = []
    for index, record in enumerate(records):
        image_bytes = _decode_base64_images(
            record.get("image"),
            context=f"{source.annotation_input}:{index + 2}",
        )
        image_paths = [
            planner.add_bytes(content, kind="images", suffix=_image_suffix(content))
            for content in image_bytes
        ]
        prepared = dict(record)
        prepared.pop("image", None)
        rows.append(
            _canonical_row(
                prepared,
                source,
                planner,
                index,
                images_override=image_paths,
                videos_override=[],
            )
        )
    return rows, len(records)


def _load_parquet(source: EvaluationSource) -> list[dict[str, Any]]:
    pandas_error: Exception | None = None
    try:
        import pandas as pd
    except ImportError as error:
        pandas_error = error
    else:
        try:
            return pd.read_parquet(source.annotation_input).to_dict("records")
        except ImportError as error:
            pandas_error = error

    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        dependency = "pandas with a parquet engine or pyarrow"
        raise ConversionError(f"parquet conversion requires {dependency}") from (
            pandas_error or error
        )
    try:
        return parquet.read_table(source.annotation_input).to_pylist()
    except (OSError, ValueError) as error:
        raise ConversionError(f"{source.annotation_input}: invalid parquet: {error}") from error


def _validate_official_revsi(
    records: Sequence[Mapping[str, Any]],
    source: EvaluationSource,
) -> None:
    expected_total = sum(_REVSI_EXPECTED_GROUP_COUNTS.values())
    if source.expected_count != expected_total:
        return
    counts = {group: 0 for group in _REVSI_EXPECTED_GROUP_COUNTS}
    unknown = 0
    for record in records:
        question_type = str(record.get("question_type") or "").strip()
        group = next(
            (
                prefix
                for prefix in (
                    "object_counting",
                    "object_rel_direction",
                    "object_rel_distance",
                    "room_size_estimation",
                )
                if question_type.startswith(prefix)
            ),
            question_type,
        )
        if group in counts:
            counts[group] += 1
        else:
            unknown += 1
    if counts != _REVSI_EXPECTED_GROUP_COUNTS or unknown:
        observed = {**counts, **({"unknown": unknown} if unknown else {})}
        raise ConversionError(
            "official ReVSI all-frame split must contain "
            f"{_REVSI_EXPECTED_GROUP_COUNTS}, got {observed}"
        )


def _revsi_records(
    source: EvaluationSource,
    planner: _AssetPlanner,
) -> tuple[list[dict[str, Any]], int]:
    records = _load_parquet(source)
    _validate_official_revsi(records, source)
    video_root = source.media_root("videos")
    rows = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ConversionError(
                f"{source.annotation_input}: parquet row {index} is not an object"
            )
        scene_id = str(record.get("scene_id") or "").strip()
        video_reference = _first(record, ("video_path", "video", "path"))
        if video_reference is None:
            if not scene_id:
                raise ConversionError(
                    f"{source.annotation_input}: parquet row {index} has no scene_id"
                )
            video_reference = f"{scene_id}.mp4"
        video_path, _local_path = planner.add_file(
            video_reference,
            kind="videos",
            root=video_root,
        )
        rows.append(
            _canonical_row(
                record,
                source,
                planner,
                index,
                images_override=[],
                videos_override=[video_path],
            )
        )
    return rows, len(records)


def _mindcube_image_values(record: Mapping[str, Any]) -> list[Any]:
    values = record.get("images")
    if values is not None:
        return _as_list(values)
    fields = sorted(
        (
            key
            for key in record
            if isinstance(key, str) and re.fullmatch(r"image_?\d+", key)
        ),
        key=lambda key: int(re.search(r"\d+", key).group()) if re.search(r"\d+", key) else 0,
    )
    return [record[field] for field in fields]


def _mindcube_image(
    value: Any,
    source: EvaluationSource,
    planner: _AssetPlanner,
) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        content = value.get("bytes")
        if isinstance(content, (bytes, bytearray, memoryview)):
            raw = bytes(content)
            return planner.add_bytes(raw, kind="images", suffix=_image_suffix(raw))
        path = value.get("path")
        if path:
            repository_path, _local = planner.add_file(
                path,
                kind="images",
                root=source.media_root("images"),
            )
            return repository_path
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return planner.add_bytes(raw, kind="images", suffix=_image_suffix(raw))
    if isinstance(value, (str, os.PathLike)):
        repository_path, _local = planner.add_file(
            value,
            kind="images",
            root=source.media_root("images"),
        )
        return repository_path
    save = getattr(value, "save", None)
    if callable(save):
        buffer = io.BytesIO()
        save(buffer, format="PNG")
        return planner.add_bytes(buffer.getvalue(), kind="images", suffix=".png")
    return None


def _mindcube_group(record: Mapping[str, Any]) -> str:
    for key in ("task", "setting"):
        value = str(record.get(key) or "").strip().casefold()
        if value in _MINDCUBE_EXPECTED_GROUP_COUNTS:
            return value
    sample_id = str(
        _first(record, ("id", "index", "sample_id")) or ""
    ).strip().casefold()
    match = re.match(r"^(rotation|among|around)(?:_|$)", sample_id)
    return match.group(1) if match else "unknown"


def _validate_official_mindcube(
    records: Sequence[Mapping[str, Any]],
    source: EvaluationSource,
) -> None:
    expected_total = sum(_MINDCUBE_EXPECTED_GROUP_COUNTS.values())
    if source.expected_count != expected_total:
        return
    counts = {group: 0 for group in _MINDCUBE_EXPECTED_GROUP_COUNTS}
    unknown = 0
    for record in records:
        group = _mindcube_group(record)
        if group in counts:
            counts[group] += 1
        else:
            unknown += 1
    if counts != _MINDCUBE_EXPECTED_GROUP_COUNTS or unknown:
        observed = {**counts, **({"unknown": unknown} if unknown else {})}
        raise ConversionError(
            "official MindCube-Tiny must contain "
            f"{_MINDCUBE_EXPECTED_GROUP_COUNTS}, got {observed}"
        )


def _mindcube_records(
    source: EvaluationSource,
    planner: _AssetPlanner,
) -> tuple[list[dict[str, Any]], int]:
    records = _load_parquet(source)
    _validate_official_mindcube(records, source)
    rows = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ConversionError(
                f"{source.annotation_input}: parquet row {index} is not an object"
            )
        image_paths = [
            path
            for value in _mindcube_image_values(record)
            if (path := _mindcube_image(value, source, planner)) is not None
        ]
        prepared = {
            key: value
            for key, value in record.items()
            if key != "images"
            and not (isinstance(key, str) and re.fullmatch(r"image_?\d+", key))
        }
        rows.append(
            _canonical_row(
                prepared,
                source,
                planner,
                index,
                images_override=image_paths,
                videos_override=[],
            )
        )
    return rows, len(records)


def _timelens_records(
    source: EvaluationSource,
    planner: _AssetPlanner,
) -> tuple[list[dict[str, Any]], int]:
    try:
        with source.annotation_input.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError) as error:
        raise ConversionError(
            f"{source.annotation_input}: invalid TimeLens JSON: {error}"
        ) from error

    if isinstance(payload, Mapping):
        nested = [(str(video_id), info) for video_id, info in payload.items()]
    elif isinstance(payload, list):
        nested = []
        for index, info in enumerate(payload):
            if not isinstance(info, Mapping):
                raise ConversionError(
                    f"{source.annotation_input}: TimeLens entry {index} is not an object"
                )
            video_id = _first(info, ("video_id", "video", "id", "name"))
            nested.append((str(video_id if video_id is not None else index), info))
    else:
        raise ConversionError(f"{source.annotation_input}: TimeLens JSON must be an object or list")

    rows: list[dict[str, Any]] = []
    for video_id, raw_info in nested:
        if not isinstance(raw_info, Mapping):
            raise ConversionError(
                f"{source.annotation_input}: TimeLens entry {video_id!r} is not an object"
            )
        queries = _as_list(_first(raw_info, ("queries", "sentences")))
        spans = _as_list(_first(raw_info, ("spans", "timestamps")))
        if (
            len(queries) == 1
            and len(spans) == 2
            and all(isinstance(value, (int, float)) for value in spans)
        ):
            spans = [spans]
        if len(queries) != len(spans):
            raise ConversionError(
                f"{source.annotation_input}: TimeLens entry {video_id!r} has "
                f"{len(queries)} queries but {len(spans)} spans"
            )
        video_reference = _first(raw_info, ("video_path", "path", "video"))
        if video_reference is None or str(video_reference) == video_id:
            video_reference = f"{video_id}.mp4"

        common_metadata = {
            key: value
            for key, value in raw_info.items()
            if key
            not in {
                "queries",
                "sentences",
                "spans",
                "timestamps",
                "video",
                "video_path",
                "path",
            }
        }
        for query_index, (query, span) in enumerate(zip(queries, spans)):
            query_text = re.sub(r"\s+", " ", str(query)).strip().strip(".")
            prepared = {
                "sample_id": f"{video_id}:{query_index:06d}",
                "problem": query_text,
                "answer": span,
                "videos": [video_reference],
                "problem_type": source.eval_task,
                "source": source.benchmark,
                "task_payload": {
                    "span": span,
                    **(
                        {"duration": raw_info["duration"]}
                        if raw_info.get("duration") is not None
                        else {}
                    ),
                },
                "metadata": {
                    "video_id": video_id,
                    "query_index": query_index,
                    **common_metadata,
                },
            }
            rows.append(_canonical_row(prepared, source, planner, len(rows)))
    return rows, len(nested)


def convert_source(
    source: EvaluationSource,
    *,
    hash_assets: bool = False,
) -> ConvertedSource:
    """Convert one resolved source record without writing any files."""

    planner = _AssetPlanner(source, hash_assets=hash_assets)
    try:
        if source.adapter in GENERIC_ADAPTERS:
            rows, source_rows = _generic_records(source, planner)
        elif source.adapter == "mmsi":
            rows, source_rows = _mmsi_records(source, planner)
        elif source.adapter == "mindcube":
            rows, source_rows = _mindcube_records(source, planner)
        elif source.adapter == "revsi":
            rows, source_rows = _revsi_records(source, planner)
        elif source.adapter == "timelens":
            rows, source_rows = _timelens_records(source, planner)
        else:
            raise ConversionError(f"unsupported adapter: {source.adapter}")

        rows = sorted(rows, key=lambda row: (str(row["sample_id"]), _json_bytes(row)))
        validate_evaluation_rows(
            rows,
            benchmark=source.benchmark,
            split=source.split,
            eval_task=source.eval_task,
            context=f"{source.benchmark}/{source.split}",
        )
    except EvaluationSchemaError as error:
        raise ConversionError(
            f"{source.benchmark}/{source.split}: {error}"
        ) from error
    except ConversionError as error:
        raise ConversionError(
            f"{source.benchmark}/{source.split}: {error}"
        ) from error
    except (TypeError, ValueError) as error:
        raise ConversionError(
            f"{source.benchmark}/{source.split}: conversion failed: {error}"
        ) from error
    return ConvertedSource(
        source=source,
        rows=tuple(rows),
        assets=planner.assets,
        source_rows=source_rows,
    )


def convert_sources(
    sources: Sequence[EvaluationSource],
    *,
    workers: int = 1,
    hash_assets: bool = False,
) -> list[ConvertedSource]:
    """Convert sources deterministically, optionally in parallel by split."""

    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ConversionError("workers must be a positive integer")
    ordered = sorted(sources, key=lambda item: (item.benchmark, item.split))
    if workers == 1 or len(ordered) <= 1:
        return [
            convert_source(source, hash_assets=hash_assets)
            for source in ordered
        ]
    with ThreadPoolExecutor(
        max_workers=min(workers, len(ordered)),
        thread_name_prefix="orarl-eval",
    ) as executor:
        return list(
            executor.map(
                lambda source: convert_source(
                    source,
                    hash_assets=hash_assets,
                ),
                ordered,
            )
        )


__all__ = [
    "ConversionError",
    "ConvertedSource",
    "PlannedAsset",
    "convert_source",
    "convert_sources",
]
