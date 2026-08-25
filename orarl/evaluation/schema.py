"""Canonical JSONL contract for OraRL evaluation samples."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict, Union

from orarl.data.schema import (
    CANONICAL_FIELDS,
    MEDIA_PATH_KEYS,
)
from orarl.data.schema import validation_errors as core_validation_errors

from .layout import (
    LayoutError,
    artifact_directory,
    is_dataset_id,
    validate_artifact_path,
    validate_dataset_id,
    validate_media_path,
    validate_repository_assets,
    validate_repository_path,
)

EVALUATION_SCHEMA_VERSION = 1
EVALUATION_CORE_FIELDS = CANONICAL_FIELDS
EVALUATION_REQUIRED_FIELDS = (
    "schema_version",
    "eval_task",
    "sample_id",
    "benchmark",
    "split",
    *EVALUATION_CORE_FIELDS,
)
EVALUATION_FIELDS = (
    *EVALUATION_REQUIRED_FIELDS,
    "family",
    "choices",
    "subtitles",
    "preprocessed",
    "task_payload",
    "metadata",
    "evaluation",
)

_PATH_KEY_RE = re.compile(
    r"(?:^|_)(?:artifact|artifacts|dir|directory|file|files|image|images|"
    r"media|path|paths|root|subtitle|subtitles|tensor|tensors|video|videos)$"
)


class _RequiredEvaluationRow(TypedDict):
    schema_version: int
    eval_task: str
    sample_id: str
    benchmark: str
    split: str
    problem: str
    answer: Any
    images: list[Any]
    videos: list[Any]
    problem_type: str
    source: str


class EvaluationRow(_RequiredEvaluationRow, total=False):
    """Typed canonical evaluation row.

    ``task_payload`` retains benchmark-specific coordinates, masks, temporal
    labels, and grouping information without flattening incompatible schemas.
    """

    family: str
    choices: list[str]
    subtitles: list[Any]
    preprocessed: dict[str, Any]
    task_payload: dict[str, Any]
    metadata: dict[str, Any]
    evaluation: dict[str, Any]


class EvaluationSchemaError(ValueError):
    """Raised when canonical evaluation rows are invalid or ambiguous."""


def _append_layout_error(errors: list[str], callback: Any) -> None:
    try:
        callback()
    except LayoutError as error:
        errors.append(str(error))


def _media_entry_paths(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, Mapping):
        return []
    return [
        candidate
        for key in MEDIA_PATH_KEYS
        if isinstance((candidate := value.get(key)), str)
    ]


def _is_path_key(key: object) -> bool:
    return isinstance(key, str) and bool(_PATH_KEY_RE.search(key.casefold()))


def _path_values(value: Any, context: str) -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield context, value
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            yield from _path_values(item, f"{context}[{index}]")
        return
    if isinstance(value, Mapping):
        matched = False
        for key, item in value.items():
            if _is_path_key(key):
                matched = True
                yield from _path_values(item, f"{context}.{key}")
        if matched:
            return
    raise LayoutError(f"{context} must contain a path string or list of path strings")


def _declared_paths(value: Any, context: str) -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child_context = f"{context}.{key}"
            if _is_path_key(key):
                yield from _path_values(item, child_context)
            elif isinstance(item, (Mapping, list, tuple)):
                yield from _declared_paths(item, child_context)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            yield from _declared_paths(item, f"{context}[{index}]")


def declared_repository_paths(
    value: Any,
    *,
    context: str = "payload",
) -> tuple[tuple[str, str], ...]:
    """Return paths held by explicitly path-named keys in a JSON value."""

    return tuple(_declared_paths(value, context))


def _evaluation_paths(record: Mapping[str, Any]) -> Iterable[tuple[str, str, str | None]]:
    benchmark = record.get("benchmark")
    benchmark_id = benchmark if isinstance(benchmark, str) else ""

    for field in ("images", "videos"):
        values = record.get(field)
        if not isinstance(values, list):
            continue
        for index, entry in enumerate(values):
            for path_index, path in enumerate(_media_entry_paths(entry)):
                suffix = "" if path_index == 0 else f".path[{path_index}]"
                yield f"{field}[{index}]{suffix}", path, field

    subtitles = record.get("subtitles")
    if isinstance(subtitles, list):
        for index, entry in enumerate(subtitles):
            for path_index, path in enumerate(_media_entry_paths(entry)):
                suffix = "" if path_index == 0 else f".path[{path_index}]"
                yield f"subtitles[{index}]{suffix}", path, "subtitles"

    preprocessed = record.get("preprocessed")
    if isinstance(preprocessed, Mapping):
        for context, path in _declared_paths(preprocessed, "preprocessed"):
            yield context, path, "artifacts"

    for field in ("task_payload", "metadata", "evaluation"):
        payload = record.get(field)
        if isinstance(payload, Mapping):
            for context, path in _declared_paths(payload, field):
                kind: str | None = None
                if benchmark_id and path.startswith(
                    f"{artifact_directory(benchmark_id)}/"
                ):
                    kind = "artifacts"
                yield context, path, kind


def evaluation_asset_paths(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every first-class or explicitly named asset path in a row."""

    paths: list[str] = []
    seen: set[str] = set()
    for _context, path, _kind in _evaluation_paths(record):
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return tuple(paths)


def evaluation_row_errors(record: Any) -> list[str]:
    """Collect all deterministic schema and layout errors for one eval row."""

    if not isinstance(record, Mapping):
        return ["record must be a JSON object"]

    errors = list(core_validation_errors(record))
    version = record.get("schema_version")
    if isinstance(version, bool) or version != EVALUATION_SCHEMA_VERSION:
        errors.append(f"schema_version must be {EVALUATION_SCHEMA_VERSION}")

    for field in ("eval_task", "benchmark", "split"):
        value = record.get(field)
        _append_layout_error(
            errors,
            lambda value=value, field=field: validate_dataset_id(value, context=field),
        )

    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        errors.append("sample_id must be a nonempty string")
    elif sample_id != sample_id.strip():
        errors.append("sample_id must not contain surrounding whitespace")

    family = record.get("family")
    if family is not None:
        _append_layout_error(
            errors,
            lambda: validate_dataset_id(family, context="family"),
        )

    choices = record.get("choices")
    if choices is not None:
        if not isinstance(choices, list):
            errors.append("choices must be a list when provided")
        else:
            for index, choice in enumerate(choices):
                if not isinstance(choice, str) or not choice.strip():
                    errors.append(f"choices[{index}] must be a nonempty string")

    subtitles = record.get("subtitles")
    if subtitles is not None:
        if not isinstance(subtitles, list):
            errors.append("subtitles must be a list when provided")
        else:
            for index, entry in enumerate(subtitles):
                if not _media_entry_paths(entry):
                    errors.append(
                        f"subtitles[{index}] must contain a nonempty repository path"
                    )

    for field in ("preprocessed", "task_payload", "metadata", "evaluation"):
        value = record.get(field)
        if value is not None and not isinstance(value, Mapping):
            errors.append(f"{field} must be a JSON object when provided")

    try:
        declared_paths = list(_evaluation_paths(record))
    except LayoutError as error:
        errors.append(str(error))
        declared_paths = []

    benchmark = record.get("benchmark")
    if is_dataset_id(benchmark):
        for context, path, kind in declared_paths:
            if kind in {"images", "videos", "subtitles"}:
                _append_layout_error(
                    errors,
                    lambda path=path, context=context, kind=kind: validate_media_path(
                        path,
                        benchmark,
                        kind=kind,
                        context=context,
                    ),
                )
            elif kind == "artifacts":
                _append_layout_error(
                    errors,
                    lambda path=path, context=context: validate_artifact_path(
                        path,
                        benchmark,
                        context=context,
                    ),
                )
            else:
                _append_layout_error(
                    errors,
                    lambda path=path, context=context: validate_repository_path(
                        path,
                        context=context,
                    ),
                )

    try:
        json.dumps(record, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        errors.append(f"record must contain only finite JSON values: {error}")
    return errors


def validate_evaluation_row(
    record: Any,
    *,
    context: str = "record",
    repository_root: Union[str, Path, None] = None,
    checksums: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
    """Validate one eval row and optionally verify all referenced assets."""

    errors = evaluation_row_errors(record)
    if errors:
        raise EvaluationSchemaError(f"{context}: " + "; ".join(errors))

    if checksums is not None and repository_root is None:
        raise EvaluationSchemaError(f"{context}: repository_root is required for checksums")
    if repository_root is not None:
        try:
            assets = evaluation_asset_paths(record)
            validate_repository_assets(
                assets,
                repository_root,
                checksums=checksums,
                file_paths=assets,
                context=f"{context} assets",
            )
        except LayoutError as error:
            raise EvaluationSchemaError(str(error)) from error
    return record


def validate_evaluation_rows(
    records: Iterable[Any],
    *,
    benchmark: str | None = None,
    split: str | None = None,
    eval_task: str | None = None,
    repository_root: Union[str, Path, None] = None,
    checksums: Mapping[str, str] | None = None,
    context: str = "evaluation rows",
) -> list[Mapping[str, Any]]:
    """Validate eval rows, dataset membership, IDs, path spelling, and assets."""

    expected = {
        "benchmark": benchmark,
        "split": split,
        "eval_task": eval_task,
    }
    for field, value in expected.items():
        if value is not None:
            try:
                validate_dataset_id(value, context=field)
            except LayoutError as error:
                raise EvaluationSchemaError(f"{context}: {error}") from error

    validated: list[Mapping[str, Any]] = []
    row_keys: dict[tuple[str, str, str], int] = {}
    path_spellings: dict[str, str] = {}
    asset_paths: list[str] = []
    dataset_identity: tuple[str, str, str] | None = None

    for index, record in enumerate(records, start=1):
        row_context = f"{context}:{index}"
        validated_record = validate_evaluation_row(record, context=row_context)
        for field, value in expected.items():
            if value is not None and validated_record.get(field) != value:
                raise EvaluationSchemaError(
                    f"{row_context}: {field} must be {value!r}, "
                    f"got {validated_record.get(field)!r}"
                )

        key = (
            str(validated_record["benchmark"]),
            str(validated_record["split"]),
            str(validated_record["sample_id"]),
        )
        identity = (
            str(validated_record["benchmark"]),
            str(validated_record["split"]),
            str(validated_record["eval_task"]),
        )
        if dataset_identity is None:
            dataset_identity = identity
        elif identity != dataset_identity:
            raise EvaluationSchemaError(
                f"{row_context}: rows must belong to one benchmark/split/task; "
                f"expected {dataset_identity[0]}/{dataset_identity[1]}/"
                f"{dataset_identity[2]}, got {identity[0]}/{identity[1]}/{identity[2]}"
            )
        previous = row_keys.get(key)
        if previous is not None:
            raise EvaluationSchemaError(
                f"{row_context}: duplicate sample_id {key[2]!r} for "
                f"{key[0]}/{key[1]} (first seen at row {previous})"
            )
        row_keys[key] = index

        for path in evaluation_asset_paths(validated_record):
            folded = path.casefold()
            previous_path = path_spellings.get(folded)
            if previous_path is not None and previous_path != path:
                raise EvaluationSchemaError(
                    f"{row_context}: asset path case collision: "
                    f"{previous_path!r} and {path!r}"
                )
            path_spellings[folded] = path
            asset_paths.append(path)
        validated.append(validated_record)

    if checksums is not None and repository_root is None:
        raise EvaluationSchemaError(f"{context}: repository_root is required for checksums")
    if repository_root is not None:
        try:
            validate_repository_assets(
                asset_paths,
                repository_root,
                checksums=checksums,
                file_paths=asset_paths,
                context=f"{context} assets",
            )
        except LayoutError as error:
            raise EvaluationSchemaError(str(error)) from error
    return validated


def load_evaluation_jsonl(
    path: Union[str, Path],
    *,
    benchmark: str | None = None,
    split: str | None = None,
    eval_task: str | None = None,
    repository_root: Union[str, Path, None] = None,
    checksums: Mapping[str, str] | None = None,
) -> list[Mapping[str, Any]]:
    """Load and validate canonical evaluation rows from a JSONL file."""

    input_path = Path(path)
    records: list[Any] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvaluationSchemaError(
                    f"{input_path}:{line_number}: invalid JSON: {error}"
                ) from error
            records.append(record)

    return validate_evaluation_rows(
        records,
        benchmark=benchmark,
        split=split,
        eval_task=eval_task,
        repository_root=repository_root,
        checksums=checksums,
        context=str(input_path),
    )
