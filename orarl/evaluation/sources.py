"""Typed local source manifests for deterministic evaluation-data builds."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, Union
from urllib.parse import urlsplit

from .layout import LayoutError, validate_dataset_id

PathLike = Union[str, os.PathLike[str]]

GENERIC_ADAPTERS = frozenset(
    {
        "generic",
        "json",
        "video_qa",
        "one_thinker",
        "onethinker",
        "spatial_grounding",
    }
)
SOURCE_ADAPTERS = frozenset(
    {*GENERIC_ADAPTERS, "mmsi", "mindcube", "revsi", "timelens"}
)
_MEDIA_ROOT_KEYS = frozenset({"images", "videos", "subtitles", "default"})
_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_ENVIRONMENT_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class _RequiredEvaluationSourceRecord(TypedDict):
    benchmark: str
    eval_task: str
    split: str
    family: str
    adapter: str
    annotation_input: str
    expected_count: int | None
    license: str
    source_url: str
    redistribution_authorized: bool
    evaluation: dict[str, Any]


class EvaluationSourceRecord(_RequiredEvaluationSourceRecord, total=False):
    """One JSONL source record for a benchmark/split conversion."""

    annotation: str
    annotation_path: str
    input: str
    media_root: str
    media_roots: dict[str, str]
    image_root: str
    video_root: str
    subtitle_root: str
    artifact_root: str
    preprocessed_root: str
    preprocessed_roots: dict[str, str]
    preprocessing: dict[str, Any]
    legacy_environment: dict[str, Any]
    metadata: dict[str, Any]


class SourceManifestError(ValueError):
    """Raised when a local evaluation source manifest is invalid."""


@dataclass(frozen=True)
class EvaluationSource:
    """Resolved local source information that is never serialized to outputs."""

    benchmark: str
    eval_task: str
    split: str
    family: str
    adapter: str
    annotation_input: Path
    media_roots: Mapping[str, Path]
    preprocessed_roots: Mapping[str, Path]
    expected_count: int | None
    license: str
    source_url: str
    redistribution_authorized: bool
    evaluation: Mapping[str, Any]
    preprocessing: Mapping[str, Any]
    legacy_environment: Mapping[str, Any]
    metadata: Mapping[str, Any]
    manifest_index: int

    @property
    def annotation(self) -> Path:
        """Return the resolved annotation input path."""

        return self.annotation_input

    def media_root(self, kind: str) -> Path:
        """Return the most specific source root for a media kind."""

        return self.media_roots.get(kind, self.media_roots.get("default", self.annotation.parent))

    def preprocessed_root(self, key: str | None = None) -> Path:
        """Return a source artifact root selected by key, then by default."""

        if key:
            folded = key.casefold()
            for name, root in self.preprocessed_roots.items():
                if name != "default" and name.casefold() in folded:
                    return root
        if "default" in self.preprocessed_roots:
            return self.preprocessed_roots["default"]
        if self.preprocessed_roots:
            return self.preprocessed_roots[sorted(self.preprocessed_roots)[0]]
        return self.annotation.parent


def _local_path(base: Path, value: object, *, context: str) -> Path:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise SourceManifestError(f"{context} must be a nonempty local path string")
    if _URI_SCHEME_RE.match(value):
        raise SourceManifestError(f"{context} must be local, not a URL")
    candidate = Path(os.path.expanduser(value))
    if not candidate.is_absolute():
        candidate = base / candidate
    return Path(os.path.abspath(os.path.normpath(str(candidate))))


def _required_text(record: Mapping[str, Any], field: str, *, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise SourceManifestError(f"{context}.{field} must be a nonempty string")
    return value


def _mapping(record: Mapping[str, Any], field: str, *, context: str) -> dict[str, Any]:
    value = record.get(field)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SourceManifestError(f"{context}.{field} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _source_url(record: Mapping[str, Any], *, context: str) -> str:
    value = _required_text(record, "source_url", context=context)
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or any(character.isspace() for character in value)
    ):
        raise SourceManifestError(
            f"{context}.source_url must be an HTTP(S) URL without embedded credentials"
        )
    return value


def _annotation_value(record: Mapping[str, Any], *, context: str) -> object:
    supplied = [
        (field, record[field])
        for field in ("annotation_input", "annotation", "annotation_path", "input")
        if record.get(field) is not None
    ]
    if not supplied:
        raise SourceManifestError(f"{context}.annotation_input is required")
    spellings = {str(value) for _field, value in supplied}
    if len(spellings) != 1:
        fields = ", ".join(field for field, _value in supplied)
        raise SourceManifestError(f"{context} has conflicting annotation fields: {fields}")
    return supplied[0][1]


def _root_mapping(
    record: Mapping[str, Any],
    base: Path,
    *,
    context: str,
) -> tuple[dict[str, Path], dict[str, Path]]:
    media: dict[str, Path] = {}
    media_root = record.get("media_root")
    if media_root is not None:
        media["default"] = _local_path(base, media_root, context=f"{context}.media_root")

    raw_media = record.get("media_roots")
    if raw_media is not None:
        if not isinstance(raw_media, Mapping):
            raise SourceManifestError(f"{context}.media_roots must be a JSON object")
        for raw_kind, raw_path in raw_media.items():
            kind = str(raw_kind).casefold().removesuffix("_root")
            if kind in {"image", "video", "subtitle"}:
                kind += "s"
            if kind not in _MEDIA_ROOT_KEYS:
                choices = ", ".join(sorted(_MEDIA_ROOT_KEYS))
                raise SourceManifestError(
                    f"{context}.media_roots keys must be one of {{{choices}}}"
                )
            media[kind] = _local_path(
                base,
                raw_path,
                context=f"{context}.media_roots[{raw_kind!r}]",
            )

    for field, kind in (
        ("image_root", "images"),
        ("video_root", "videos"),
        ("subtitle_root", "subtitles"),
    ):
        if record.get(field) is not None:
            media[kind] = _local_path(base, record[field], context=f"{context}.{field}")

    preprocessed: dict[str, Path] = {}
    raw_preprocessed_root = record.get(
        "preprocessed_root",
        record.get("artifact_root"),
    )
    if raw_preprocessed_root is not None:
        preprocessed["default"] = _local_path(
            base,
            raw_preprocessed_root,
            context=f"{context}.preprocessed_root",
        )
    raw_preprocessed = record.get("preprocessed_roots")
    if raw_preprocessed is not None:
        if not isinstance(raw_preprocessed, Mapping):
            raise SourceManifestError(f"{context}.preprocessed_roots must be a JSON object")
        for raw_name, raw_path in raw_preprocessed.items():
            name = str(raw_name).strip().casefold()
            if not name:
                raise SourceManifestError(f"{context}.preprocessed_roots keys must be nonempty")
            preprocessed[name] = _local_path(
                base,
                raw_path,
                context=f"{context}.preprocessed_roots[{raw_name!r}]",
            )
    return media, preprocessed


def _source_from_record(
    record: Mapping[str, Any],
    manifest_path: Path,
    index: int,
) -> EvaluationSource:
    context = f"{manifest_path}:{index}"
    try:
        benchmark = validate_dataset_id(record.get("benchmark"), context="benchmark")
        eval_task = validate_dataset_id(record.get("eval_task"), context="eval_task")
        split = validate_dataset_id(record.get("split"), context="split")
        family = validate_dataset_id(record.get("family"), context="family")
    except LayoutError as error:
        raise SourceManifestError(f"{context}: {error}") from error

    adapter = _required_text(record, "adapter", context=context).casefold().replace("-", "_")
    if adapter not in SOURCE_ADAPTERS:
        choices = ", ".join(sorted(SOURCE_ADAPTERS))
        raise SourceManifestError(f"{context}.adapter must be one of {{{choices}}}")

    annotation = _local_path(
        manifest_path.parent,
        _annotation_value(record, context=context),
        context=f"{context}.annotation_input",
    )
    if not annotation.is_file():
        raise SourceManifestError(f"{context}.annotation_input does not exist: {annotation}")
    if annotation.suffix.casefold() in {".yaml", ".yml"}:
        raise SourceManifestError(f"{context}.annotation_input must not be YAML")

    expected_count = record.get("expected_count")
    if expected_count is not None and (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 0
    ):
        raise SourceManifestError(f"{context}.expected_count must be null or a nonnegative integer")

    redistribution = record.get("redistribution_authorized")
    if not isinstance(redistribution, bool):
        raise SourceManifestError(
            f"{context}.redistribution_authorized must be an explicit boolean"
        )

    evaluation = _mapping(record, "evaluation", context=context)
    if not evaluation:
        raise SourceManifestError(f"{context}.evaluation must be a nonempty JSON object")
    preprocessing = _mapping(record, "preprocessing", context=context)
    legacy_environment = _mapping(record, "legacy_environment", context=context)
    for name in legacy_environment:
        if not _ENVIRONMENT_NAME_RE.fullmatch(name):
            raise SourceManifestError(
                f"{context}.legacy_environment names must use uppercase letters, "
                "digits, and underscores"
            )
    metadata = _mapping(record, "metadata", context=context)
    media_roots, preprocessed_roots = _root_mapping(
        record,
        manifest_path.parent,
        context=context,
    )

    try:
        json.dumps(
            {
                "evaluation": evaluation,
                "preprocessing": preprocessing,
                "legacy_environment": legacy_environment,
                "metadata": metadata,
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise SourceManifestError(f"{context}: metadata must be strict JSON: {error}") from error

    license_name = _required_text(record, "license", context=context)
    if license_name.startswith(("/", "~/", "file://")) or re.match(
        r"^[A-Za-z]:[\\/]", license_name
    ):
        raise SourceManifestError(f"{context}.license must not contain a local path")

    return EvaluationSource(
        benchmark=benchmark,
        eval_task=eval_task,
        split=split,
        family=family,
        adapter=adapter,
        annotation_input=annotation,
        media_roots=media_roots,
        preprocessed_roots=preprocessed_roots,
        expected_count=expected_count,
        license=license_name,
        source_url=_source_url(record, context=context),
        redistribution_authorized=redistribution,
        evaluation=evaluation,
        preprocessing=preprocessing,
        legacy_environment=legacy_environment,
        metadata=metadata,
        manifest_index=index,
    )


def load_source_manifest(path: PathLike) -> list[EvaluationSource]:
    """Load a local JSONL source manifest, resolving only local input paths."""

    manifest_path = Path(path).expanduser()
    if manifest_path.suffix.casefold() not in {".jsonl", ".ndjson"}:
        raise SourceManifestError("evaluation source manifest must be JSONL, not YAML or JSON")
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise SourceManifestError(f"source manifest does not exist: {manifest_path}")

    sources: list[EvaluationSource] = []
    datasets: dict[tuple[str, str], int] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SourceManifestError(
                    f"{manifest_path}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(record, Mapping):
                raise SourceManifestError(
                    f"{manifest_path}:{line_number}: source record must be a JSON object"
                )
            source = _source_from_record(record, manifest_path, line_number)
            key = (source.benchmark, source.split)
            previous = datasets.get(key)
            if previous is not None:
                raise SourceManifestError(
                    f"{manifest_path}:{line_number}: duplicate source {key[0]}/{key[1]} "
                    f"(first seen at line {previous})"
                )
            datasets[key] = line_number
            sources.append(source)
    if not sources:
        raise SourceManifestError(f"{manifest_path}: source manifest is empty")
    return sources


load_evaluation_source_manifest = load_source_manifest
load_evaluation_sources = load_source_manifest
EvaluationSourceSpec = EvaluationSource
SourceManifestRecord = EvaluationSourceRecord


__all__ = [
    "EvaluationSource",
    "EvaluationSourceRecord",
    "EvaluationSourceSpec",
    "GENERIC_ADAPTERS",
    "SOURCE_ADAPTERS",
    "SourceManifestError",
    "SourceManifestRecord",
    "load_evaluation_source_manifest",
    "load_evaluation_sources",
    "load_source_manifest",
]
