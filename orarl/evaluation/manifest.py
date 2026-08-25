"""Validation and loading for the canonical ``datasets.jsonl`` manifest."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TypedDict, Union
from urllib.parse import urlsplit

from .layout import (
    MANIFEST_FILENAME,
    LayoutError,
    artifact_directory,
    media_directory,
    path_is_within,
    validate_annotation_path,
    validate_artifact_path,
    validate_dataset_id,
    validate_media_path,
    validate_repository_assets,
    validate_repository_path,
    validate_sha256,
    validate_unique_paths,
)
from .schema import (
    EVALUATION_SCHEMA_VERSION,
    EvaluationSchemaError,
    declared_repository_paths,
    evaluation_asset_paths,
    load_evaluation_jsonl,
)

_ENVIRONMENT_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_PATH_ENV_SUFFIXES = (
    "_FILE",
    "_DIR",
    "_ROOT",
    "_PATH",
    "_CKPT",
    "_CFG",
    "_BASE",
    "_PREFIX",
)


class _RequiredDatasetManifestRecord(TypedDict):
    schema_version: int
    benchmark: str
    split: str
    task: str
    annotation_path: str
    media_paths: list[str]
    artifact_paths: list[str]
    expected_count: int
    license: str
    source_url: str
    redistribution_authorized: bool
    evaluation: dict[str, Any]


class DatasetManifestRecord(_RequiredDatasetManifestRecord, total=False):
    """One canonical manifest record for a benchmark/split pair."""

    family: str
    preprocessing: dict[str, Any]
    legacy_environment: dict[str, Any]
    checksums: dict[str, str]
    metadata: dict[str, Any]


class ManifestError(ValueError):
    """Raised when ``datasets.jsonl`` or its referenced repository is invalid."""


def _append_layout_error(errors: list[str], callback: Any) -> None:
    try:
        callback()
    except LayoutError as error:
        errors.append(str(error))


def _valid_source_url(value: object) -> bool:
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    if any(character.isspace() for character in value):
        return False
    parsed = urlsplit(value)
    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.netloc)
        and parsed.username is None
        and parsed.password is None
    )


def _string_paths(
    value: Any,
    field: str,
    errors: list[str],
) -> list[str]:
    if not isinstance(value, list):
        errors.append(f"{field} must be a list")
        return []
    paths: list[str] = []
    for index, path in enumerate(value):
        if not isinstance(path, str):
            errors.append(f"{field}[{index}] must be a repository path string")
        else:
            paths.append(path)
    return paths


def _manifest_checksums(record: Mapping[str, Any], errors: list[str]) -> dict[str, str]:
    raw_checksums = record.get("checksums")
    if raw_checksums is None:
        return {}
    if not isinstance(raw_checksums, Mapping):
        errors.append("checksums must be a JSON object when provided")
        return {}

    checksums: dict[str, str] = {}
    for raw_path, raw_digest in raw_checksums.items():
        if not isinstance(raw_path, str):
            errors.append("checksum keys must be repository path strings")
            continue
        try:
            path = validate_repository_path(raw_path, context="checksum path")
            digest = validate_sha256(raw_digest, context=f"checksum for {path}")
        except LayoutError as error:
            errors.append(str(error))
            continue
        checksums[path] = digest
    _append_layout_error(
        errors,
        lambda: validate_unique_paths(checksums, context="checksum paths"),
    )
    return checksums


def dataset_manifest_record_errors(record: Any) -> list[str]:
    """Collect schema and canonical-layout errors for one manifest record."""

    if not isinstance(record, Mapping):
        return ["record must be a JSON object"]

    errors: list[str] = []
    version = record.get("schema_version")
    if isinstance(version, bool) or version != EVALUATION_SCHEMA_VERSION:
        errors.append(f"schema_version must be {EVALUATION_SCHEMA_VERSION}")

    for field in ("benchmark", "split", "task"):
        value = record.get(field)
        _append_layout_error(
            errors,
            lambda value=value, field=field: validate_dataset_id(value, context=field),
        )

    family = record.get("family")
    if family is not None:
        _append_layout_error(
            errors,
            lambda: validate_dataset_id(family, context="family"),
        )

    benchmark = record.get("benchmark")
    split = record.get("split")
    if isinstance(benchmark, str) and isinstance(split, str):
        _append_layout_error(
            errors,
            lambda: validate_annotation_path(
                record.get("annotation_path"),
                benchmark,
                split,
            ),
        )
    elif "annotation_path" not in record:
        errors.append("annotation_path is required")

    media_paths = _string_paths(record.get("media_paths"), "media_paths", errors)
    artifact_paths = _string_paths(record.get("artifact_paths"), "artifact_paths", errors)
    if isinstance(benchmark, str):
        for index, path in enumerate(media_paths):
            _append_layout_error(
                errors,
                lambda path=path, index=index: validate_media_path(
                    path,
                    benchmark,
                    allow_directory=True,
                    context=f"media_paths[{index}]",
                ),
            )
        for index, path in enumerate(artifact_paths):
            _append_layout_error(
                errors,
                lambda path=path, index=index: validate_artifact_path(
                    path,
                    benchmark,
                    allow_directory=True,
                    context=f"artifact_paths[{index}]",
                ),
            )
    _append_layout_error(
        errors,
        lambda: validate_unique_paths(media_paths, context="media_paths"),
    )
    _append_layout_error(
        errors,
        lambda: validate_unique_paths(artifact_paths, context="artifact_paths"),
    )

    expected_count = record.get("expected_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 0
    ):
        errors.append("expected_count must be a nonnegative integer")

    license_name = record.get("license")
    if not isinstance(license_name, str) or not license_name.strip():
        errors.append("license must be a nonempty string")
    if not _valid_source_url(record.get("source_url")):
        errors.append("source_url must be an HTTP(S) URL without embedded credentials")
    if not isinstance(record.get("redistribution_authorized"), bool):
        errors.append("redistribution_authorized must be an explicit boolean")

    evaluation = record.get("evaluation")
    if not isinstance(evaluation, Mapping) or not evaluation:
        errors.append("evaluation must be a nonempty JSON object")

    for field in ("preprocessing", "metadata"):
        value = record.get(field)
        if value is not None and not isinstance(value, Mapping):
            errors.append(f"{field} must be a JSON object when provided")

    legacy = record.get("legacy_environment")
    legacy_paths: list[str] = []
    if legacy is not None:
        if not isinstance(legacy, Mapping):
            errors.append("legacy_environment must be a JSON object when provided")
        else:
            for name, value in legacy.items():
                if not isinstance(name, str) or not _ENVIRONMENT_NAME_RE.fullmatch(name):
                    errors.append(
                        "legacy_environment names must use uppercase letters, digits, "
                        "and underscores"
                    )
                if isinstance(value, (Mapping, tuple, set)) or value is None:
                    errors.append(
                        f"legacy_environment[{name!r}] must be a JSON scalar or list"
                    )
                    continue
                if isinstance(value, list) and not all(
                    isinstance(item, (str, int, float, bool)) for item in value
                ):
                    errors.append(
                        f"legacy_environment[{name!r}] lists must contain only JSON scalars"
                    )
                    continue
                if isinstance(name, str) and name.endswith(_PATH_ENV_SUFFIXES):
                    if not isinstance(value, str):
                        errors.append(
                            f"legacy_environment[{name!r}] must be a path string"
                        )
                        continue
                    legacy_paths.append(value)
                    _append_layout_error(
                        errors,
                        lambda value=value, name=name: validate_repository_path(
                            value,
                            context=f"legacy_environment[{name!r}]",
                        ),
                    )

    profile_paths: list[str] = []
    for field in ("preprocessing", "evaluation", "metadata"):
        value = record.get(field)
        if isinstance(value, Mapping):
            try:
                declared = declared_repository_paths(value, context=field)
            except LayoutError as error:
                errors.append(str(error))
                continue
            for context, path in declared:
                profile_paths.append(path)
                _append_layout_error(
                    errors,
                    lambda path=path, context=context: validate_repository_path(
                        path,
                        context=context,
                    ),
                )

    checksums = _manifest_checksums(record, errors)
    declared_paths = []
    annotation = record.get("annotation_path")
    if isinstance(annotation, str):
        declared_paths.append(annotation)
    declared_paths.extend(media_paths)
    declared_paths.extend(artifact_paths)
    declared_paths.extend(legacy_paths)
    declared_paths.extend(profile_paths)
    declared_paths.extend(checksums)
    _append_layout_error(
        errors,
        lambda: validate_unique_paths(
            declared_paths,
            context="manifest paths",
            allow_exact_duplicates=True,
        ),
    )

    try:
        json.dumps(record, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        errors.append(f"record must contain only finite JSON values: {error}")
    return errors


def validate_dataset_manifest_record(
    record: Any,
    *,
    context: str = "manifest record",
) -> Mapping[str, Any]:
    """Validate one ``datasets.jsonl`` record and return it unchanged."""

    errors = dataset_manifest_record_errors(record)
    if errors:
        raise ManifestError(f"{context}: " + "; ".join(errors))
    return record


def _record_paths(record: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    paths = [str(record["annotation_path"])]
    paths.extend(str(path) for path in record["media_paths"])
    paths.extend(str(path) for path in record["artifact_paths"])
    legacy = record.get("legacy_environment")
    if isinstance(legacy, Mapping):
        paths.extend(
            str(value)
            for name, value in legacy.items()
            if isinstance(name, str)
            and name.endswith(_PATH_ENV_SUFFIXES)
            and isinstance(value, str)
        )
    for field in ("preprocessing", "evaluation", "metadata"):
        payload = record.get(field)
        if isinstance(payload, Mapping):
            paths.extend(path for _context, path in declared_repository_paths(payload))

    files = [str(record["annotation_path"])]
    checksums = record.get("checksums")
    if isinstance(checksums, Mapping):
        files.extend(str(path) for path in checksums)
    return paths, files


def validate_dataset_manifest(
    records: Iterable[Any],
    *,
    repository_root: Union[str, Path, None] = None,
    checksums: Mapping[str, str] | None = None,
    context: str = MANIFEST_FILENAME,
) -> list[Mapping[str, Any]]:
    """Validate manifest records, dataset uniqueness, paths, and optional assets."""

    validated: list[Mapping[str, Any]] = []
    datasets: dict[tuple[str, str], int] = {}
    path_spellings: dict[str, str] = {}
    asset_paths: list[str] = []
    file_paths: list[str] = []
    merged_checksums: dict[str, str] = {}

    for index, record in enumerate(records, start=1):
        row_context = f"{context}:{index}"
        validated_record = validate_dataset_manifest_record(record, context=row_context)
        key = (
            str(validated_record["benchmark"]),
            str(validated_record["split"]),
        )
        previous = datasets.get(key)
        if previous is not None:
            raise ManifestError(
                f"{row_context}: duplicate dataset {key[0]}/{key[1]} "
                f"(first seen at row {previous})"
            )
        datasets[key] = index

        paths, files = _record_paths(validated_record)
        for path in paths:
            folded = path.casefold()
            previous_path = path_spellings.get(folded)
            if previous_path is not None and previous_path != path:
                raise ManifestError(
                    f"{row_context}: manifest path case collision: "
                    f"{previous_path!r} and {path!r}"
                )
            path_spellings[folded] = path
        asset_paths.extend(paths)
        file_paths.extend(files)

        record_checksums = validated_record.get("checksums")
        if isinstance(record_checksums, Mapping):
            for path, digest in record_checksums.items():
                previous_digest = merged_checksums.get(str(path))
                if previous_digest is not None and previous_digest != digest:
                    raise ManifestError(f"{row_context}: conflicting checksums for {path}")
                merged_checksums[str(path)] = str(digest)
        validated.append(validated_record)

    for raw_path, raw_digest in (checksums or {}).items():
        try:
            path = validate_repository_path(raw_path, context="checksum path")
            digest = validate_sha256(raw_digest, context=f"checksum for {path}")
        except LayoutError as error:
            raise ManifestError(f"{context}: {error}") from error
        previous_digest = merged_checksums.get(path)
        if previous_digest is not None and previous_digest != digest:
            raise ManifestError(f"{context}: conflicting checksums for {path}")
        merged_checksums[path] = digest

    if checksums is not None and repository_root is None:
        raise ManifestError(f"{context}: repository_root is required for checksums")
    if repository_root is not None:
        try:
            validate_repository_assets(
                asset_paths,
                repository_root,
                checksums=merged_checksums,
                file_paths=file_paths,
                context=f"{context} assets",
            )
        except LayoutError as error:
            raise ManifestError(str(error)) from error
    return validated


def _manifest_path(path: Union[str, Path]) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / MANIFEST_FILENAME
    if candidate.name != MANIFEST_FILENAME:
        raise ManifestError(f"evaluation manifest must be named {MANIFEST_FILENAME}")
    return candidate


def load_dataset_manifest(
    path: Union[str, Path],
    *,
    repository_root: Union[str, Path, None] = None,
    checksums: Mapping[str, str] | None = None,
) -> list[Mapping[str, Any]]:
    """Load and validate ``datasets.jsonl`` from a file or repository root."""

    input_path = _manifest_path(path)
    records: list[Any] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ManifestError(
                    f"{input_path}:{line_number}: invalid JSON: {error}"
                ) from error
            records.append(record)
    return validate_dataset_manifest(
        records,
        repository_root=repository_root,
        checksums=checksums,
        context=str(input_path),
    )


def _path_is_declared(path: str, declarations: Iterable[str]) -> bool:
    return any(path_is_within(path, declaration) for declaration in declarations)


def validate_evaluation_repository(
    repository_root: Union[str, Path],
    *,
    checksums: Mapping[str, str] | None = None,
    require_redistribution_authorized: bool = True,
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    """Validate a staged repository, its manifests, row counts, and all assets."""

    root = Path(repository_root)
    manifest = load_dataset_manifest(
        root,
        repository_root=root,
        checksums=checksums,
    )
    datasets: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    path_spellings: dict[str, str] = {}

    for index, dataset in enumerate(manifest, start=1):
        benchmark = str(dataset["benchmark"])
        split = str(dataset["split"])
        task = str(dataset["task"])
        if require_redistribution_authorized and not dataset["redistribution_authorized"]:
            raise ManifestError(
                f"{MANIFEST_FILENAME}:{index}: redistribution is not authorized for "
                f"{benchmark}/{split}"
            )

        annotation = root.joinpath(str(dataset["annotation_path"]))
        try:
            rows = load_evaluation_jsonl(
                annotation,
                benchmark=benchmark,
                split=split,
                eval_task=task,
                repository_root=root,
            )
        except EvaluationSchemaError as error:
            raise ManifestError(str(error)) from error

        expected_count = int(dataset["expected_count"])
        if len(rows) != expected_count:
            raise ManifestError(
                f"{benchmark}/{split}: expected {expected_count} rows, found {len(rows)}"
            )

        media_paths = [str(path) for path in dataset["media_paths"]]
        artifact_paths = [str(path) for path in dataset["artifact_paths"]]
        media_root = f"{media_directory(benchmark)}/"
        artifact_root = f"{artifact_directory(benchmark)}/"
        for row_number, row in enumerate(rows, start=1):
            for path in evaluation_asset_paths(row):
                if path.startswith(media_root):
                    declarations = media_paths
                elif path.startswith(artifact_root):
                    declarations = artifact_paths
                else:
                    declarations = []
                if declarations and not _path_is_declared(path, declarations):
                    raise ManifestError(
                        f"{benchmark}/{split}:{row_number}: asset {path!r} is not covered "
                        "by the dataset manifest"
                    )
                if not declarations and path.startswith(("media/", "artifacts/")):
                    raise ManifestError(
                        f"{benchmark}/{split}:{row_number}: asset {path!r} has no declared root"
                    )

                folded = path.casefold()
                previous = path_spellings.get(folded)
                if previous is not None and previous != path:
                    raise ManifestError(
                        f"{benchmark}/{split}:{row_number}: asset path case collision: "
                        f"{previous!r} and {path!r}"
                    )
                path_spellings[folded] = path
        datasets[(benchmark, split)] = rows
    return datasets
