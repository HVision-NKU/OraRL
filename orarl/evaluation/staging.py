"""Read-only inventory and atomic staging for canonical evaluation data."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, TypedDict

from .card import (
    DATASET_CARD_FILENAME,
    DEFAULT_DATASET_REPO_ID,
    DEFAULT_INDEX_REPO_ID,
    GIT_ATTRIBUTES_FILENAME,
    DatasetCardError,
    render_index_card,
    validate_huggingface_metadata,
    write_huggingface_metadata,
)
from .converters import ConvertedSource, PlannedAsset, convert_sources
from .layout import (
    ANNOTATIONS_DIRECTORY,
    ARTIFACTS_DIRECTORY,
    ASSET_MANIFEST_FILENAME,
    MANIFEST_FILENAME,
    MEDIA_DIRECTORY,
    LayoutError,
    annotation_path,
    artifact_directory,
    benchmark_directory,
    media_directory,
    sha256_file,
    validate_artifact_path,
    validate_dataset_id,
    validate_media_path,
    validate_repository_path,
    validate_sha256,
)
from .manifest import (
    ManifestError,
    load_dataset_manifest,
    validate_dataset_manifest,
    validate_evaluation_repository,
)
from .schema import (
    EVALUATION_SCHEMA_VERSION,
    declared_repository_paths,
    evaluation_asset_paths,
)
from .sources import EvaluationSource, load_source_manifest

_PATH_KEY_RE = re.compile(
    r"(?:^|_)(?:artifact|artifacts|base|cfg|checkpoint|ckpt|dir|directory|file|"
    r"files|image|images|media|path|paths|prefix|root|subtitle|"
    r"subtitles|tensor|tensors|video|videos)$"
)
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_ASSET_KINDS = frozenset({"annotations", "artifacts", "images", "subtitles", "videos"})
_STAGED_DIRECTORIES = (ANNOTATIONS_DIRECTORY, MEDIA_DIRECTORY, ARTIFACTS_DIRECTORY)
_DROP = object()


class _RequiredAssetManifestRecord(TypedDict):
    path: str
    bytes: int
    kind: str
    benchmark: str
    license: str


class AssetManifestRecord(_RequiredAssetManifestRecord, total=False):
    """One deterministic path-and-size record in ``assets.jsonl``."""

    sha256: str


class StagingError(ValueError):
    """Raised when inventory, build, or staged validation fails."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for record in records:
                handle.write(_json_bytes(record))
                handle.write(b"\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _load_raw_jsonl(path: Path, *, context: str) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise StagingError(
                        f"{context}:{line_number}: invalid JSON: {error}"
                    ) from error
                if not isinstance(record, Mapping):
                    raise StagingError(
                        f"{context}:{line_number}: record must be a JSON object"
                    )
                records.append(record)
    except OSError as error:
        raise StagingError(f"cannot read {context}: {error}") from error
    return records


def _is_path_key(key: object) -> bool:
    return isinstance(key, str) and bool(_PATH_KEY_RE.search(key.casefold()))


def _canonical_profile_path(value: str) -> bool:
    return value.startswith(
        (
            f"{ANNOTATIONS_DIRECTORY}/",
            f"{MEDIA_DIRECTORY}/",
            f"{ARTIFACTS_DIRECTORY}/",
        )
    )


def _path_rewrites(
    converted: ConvertedSource,
    canonical_annotation: str,
) -> tuple[dict[str, str], set[str]]:
    source = converted.source
    rewrites: dict[str, str] = {
        os.path.normcase(str(source.annotation)): canonical_annotation,
        os.path.normcase(str(source.annotation.resolve())): canonical_annotation,
    }
    directories: set[str] = set()
    for name, root in source.media_roots.items():
        destination = (
            media_directory(source.benchmark)
            if name == "default"
            else media_directory(source.benchmark, name)
        )
        rewrites[os.path.normcase(str(root))] = destination
        try:
            rewrites[os.path.normcase(str(root.resolve()))] = destination
        except OSError:
            pass
        directories.add(destination)
    for root in source.preprocessed_roots.values():
        destination = artifact_directory(source.benchmark)
        rewrites[os.path.normcase(str(root))] = destination
        try:
            rewrites[os.path.normcase(str(root.resolve()))] = destination
        except OSError:
            pass
        directories.add(destination)
    for asset in converted.assets:
        if asset.source_path is None:
            continue
        rewrites[os.path.normcase(str(asset.source_path))] = asset.repository_path
        try:
            rewrites[os.path.normcase(str(asset.source_path.resolve()))] = asset.repository_path
        except OSError:
            pass
    return rewrites, directories


def _rewrite_profile_path(
    value: Any,
    *,
    source: EvaluationSource,
    rewrites: Mapping[str, str],
    key: str,
) -> Any:
    if isinstance(value, str):
        if _canonical_profile_path(value):
            try:
                return validate_repository_path(value)
            except LayoutError:
                return _DROP
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value):
            return _DROP
        raw_path = Path(os.path.expanduser(value))
        if raw_path.is_absolute():
            candidates = [raw_path]
        else:
            candidates = [
                source.annotation.parent / raw_path,
                source.preprocessed_root(key) / raw_path,
                *(root / raw_path for root in source.media_roots.values()),
            ]
        for candidate in candidates:
            local = Path(os.path.abspath(os.path.normpath(str(candidate))))
            rewritten = rewrites.get(os.path.normcase(str(local)))
            if rewritten is not None:
                return rewritten
            try:
                rewritten = rewrites.get(os.path.normcase(str(local.resolve())))
            except OSError:
                rewritten = None
            if rewritten is not None:
                return rewritten
        return _DROP
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str) or _private_source_string(raw_key, source):
                continue
            rewritten = _rewrite_profile_path(
                item,
                source=source,
                rewrites=rewrites,
                key=raw_key,
            )
            if rewritten is not _DROP:
                converted[raw_key] = rewritten
        return converted
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        converted_items = []
        for item in value:
            rewritten = _rewrite_profile_path(
                item,
                source=source,
                rewrites=rewrites,
                key=key,
            )
            if rewritten is not _DROP:
                converted_items.append(rewritten)
        return converted_items
    return _DROP


def _private_source_string(value: str, source: EvaluationSource) -> bool:
    if value.startswith(("/", "~/", "file://")) or _WINDOWS_ABSOLUTE_RE.match(value):
        return True
    roots = {
        str(source.annotation),
        str(source.annotation.parent),
        *(str(path) for path in source.media_roots.values()),
        *(str(path) for path in source.preprocessed_roots.values()),
    }
    return any(root and root in value for root in roots)


def _portable_profile(
    value: Any,
    *,
    source: EvaluationSource,
    rewrites: Mapping[str, str],
    path_context: str | None = None,
) -> Any:
    if path_context:
        return _rewrite_profile_path(
            value,
            source=source,
            rewrites=rewrites,
            key=path_context,
        )
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _DROP
    if isinstance(value, str):
        return _DROP if _private_source_string(value, source) else value
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str) or _private_source_string(raw_key, source):
                continue
            portable = _portable_profile(
                item,
                source=source,
                rewrites=rewrites,
                path_context=raw_key if _is_path_key(raw_key) else None,
            )
            if portable is not _DROP:
                converted[raw_key] = portable
        return converted
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        converted_items = []
        for item in value:
            portable = _portable_profile(
                item,
                source=source,
                rewrites=rewrites,
            )
            if portable is not _DROP:
                converted_items.append(portable)
        return converted_items
    return _DROP


def _converted_inventory(converted: ConvertedSource) -> dict[str, Any]:
    assets = {asset.repository_path: asset for asset in converted.assets}
    missing = [asset for asset in assets.values() if asset.missing]
    existing = [asset for asset in assets.values() if not asset.missing]
    expected = converted.source.expected_count
    return {
        "adapter": converted.source.adapter,
        "benchmark": converted.source.benchmark,
        "split": converted.source.split,
        "eval_task": converted.source.eval_task,
        "source_rows": converted.source_rows,
        "rows": len(converted.rows),
        "expected_count": expected,
        "count_matches": None if expected is None else len(converted.rows) == expected,
        "referenced_assets": len(assets),
        "existing_assets": len(existing),
        "missing_assets": len(missing),
        "bytes": sum(asset.bytes for asset in existing),
        "redistribution_authorized": converted.source.redistribution_authorized,
    }


def _inventory_from_converted(
    converted: Sequence[ConvertedSource],
    *,
    checksums: bool = False,
) -> dict[str, Any]:
    source_inventory = [_converted_inventory(item) for item in converted]

    unique_assets: dict[str, PlannedAsset] = {}
    for item in converted:
        for asset in item.assets:
            unique_assets.setdefault(asset.repository_path, asset)
    unique_existing = [asset for asset in unique_assets.values() if not asset.missing]
    unique_missing = [asset for asset in unique_assets.values() if asset.missing]
    expected_counts = [
        item.source.expected_count for item in converted if item.source.expected_count is not None
    ]
    unlocked_sources = sum(item.source.expected_count is None for item in converted)
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "asset_checksums": checksums,
        "sources": source_inventory,
        "totals": {
            "sources": len(converted),
            "source_rows": sum(item.source_rows for item in converted),
            "rows": sum(len(item.rows) for item in converted),
            "expected_count": None if unlocked_sources else sum(expected_counts),
            "locked_expected_count": sum(expected_counts),
            "unlocked_sources": unlocked_sources,
            "referenced_assets": len(unique_assets),
            "existing_assets": len(unique_existing),
            "missing_assets": len(unique_missing),
            "bytes": sum(asset.bytes for asset in unique_existing),
            "count_mismatches": sum(
                item.source.expected_count is not None
                and len(item.rows) != item.source.expected_count
                for item in converted
            ),
            "unauthorized_sources": sum(
                not item.source.redistribution_authorized for item in converted
            ),
        },
    }


def _select_source_tasks(
    sources: Sequence[EvaluationSource],
    tasks: Iterable[str] | None,
) -> list[EvaluationSource]:
    requested = {
        str(task).strip().casefold()
        for task in (tasks or ())
        if str(task).strip()
    }
    if not requested:
        return list(sources)
    selected = [
        source
        for source in sources
        if source.eval_task.casefold() in requested
    ]
    covered = {source.eval_task.casefold() for source in selected}
    missing = sorted(requested - covered)
    if missing:
        raise StagingError(
            "source manifest has no records for task(s): " + ", ".join(missing)
        )
    return selected


def inventory_evaluation_sources(
    source_manifest: str | os.PathLike[str],
    *,
    tasks: Iterable[str] | None = None,
    workers: int = 1,
    checksums: bool = False,
) -> dict[str, Any]:
    """Read source rows and assets without creating or modifying output paths."""

    sources = _select_source_tasks(load_source_manifest(source_manifest), tasks)
    return _inventory_from_converted(
        convert_sources(
            sources,
            workers=workers,
            hash_assets=checksums,
        ),
        checksums=checksums,
    )


def _locked_source_record(converted: ConvertedSource) -> dict[str, Any]:
    source = converted.source
    record: dict[str, Any] = {
        "benchmark": source.benchmark,
        "eval_task": source.eval_task,
        "split": source.split,
        "family": source.family,
        "adapter": source.adapter,
        "annotation_input": str(source.annotation.resolve()),
        "expected_count": len(converted.rows),
        "license": source.license,
        "source_url": source.source_url,
        "redistribution_authorized": source.redistribution_authorized,
        "evaluation": dict(source.evaluation),
    }
    if source.media_roots:
        record["media_roots"] = {
            name: str(path.resolve()) for name, path in sorted(source.media_roots.items())
        }
    if source.preprocessed_roots:
        record["preprocessed_roots"] = {
            name: str(path.resolve()) for name, path in sorted(source.preprocessed_roots.items())
        }
    for field, value in (
        ("preprocessing", source.preprocessing),
        ("legacy_environment", source.legacy_environment),
        ("metadata", source.metadata),
    ):
        if value:
            record[field] = dict(value)
    return record


def write_locked_source_manifest(
    source_manifest: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    tasks: Iterable[str] | None = None,
    workers: int = 1,
    checksums: bool = False,
) -> dict[str, Any]:
    """Write a deterministic private manifest with actual converted row counts."""

    manifest_path = Path(source_manifest).expanduser().resolve()
    output = Path(output_path).expanduser().resolve(strict=False)
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite locked source manifest: {output}")
    sources = _select_source_tasks(load_source_manifest(manifest_path), tasks)
    converted = convert_sources(
        sources,
        workers=workers,
        hash_assets=checksums,
    )
    records = [_locked_source_record(item) for item in converted]
    records.sort(key=lambda item: (str(item["benchmark"]), str(item["split"])))
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_jsonl(output, records)
    return {
        "path": str(output),
        "sources": len(records),
        "rows": sum(int(record["expected_count"]) for record in records),
    }


def inventory_and_write_locked_manifest(
    source_manifest: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    tasks: Iterable[str] | None = None,
    workers: int = 1,
    checksums: bool = False,
) -> dict[str, Any]:
    """Inventory once and write the same conversion's actual counts."""

    manifest_path = Path(source_manifest).expanduser().resolve()
    output = Path(output_path).expanduser().resolve(strict=False)
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite locked source manifest: {output}")
    sources = _select_source_tasks(load_source_manifest(manifest_path), tasks)
    converted = convert_sources(
        sources,
        workers=workers,
        hash_assets=checksums,
    )
    inventory = _inventory_from_converted(
        converted,
        checksums=checksums,
    )
    records = [_locked_source_record(item) for item in converted]
    records.sort(key=lambda item: (str(item["benchmark"]), str(item["split"])))
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_jsonl(output, records)
    return {
        **inventory,
        "locked_manifest": {
            "path": str(output),
            "sources": len(records),
            "rows": sum(int(record["expected_count"]) for record in records),
        },
    }


def _merge_assets(converted: Sequence[ConvertedSource]) -> dict[str, PlannedAsset]:
    merged: dict[str, PlannedAsset] = {}
    for item in converted:
        for asset in item.assets:
            previous = merged.get(asset.repository_path)
            if previous is not None:
                if (
                    previous.sha256 != asset.sha256
                    or previous.bytes != asset.bytes
                    or previous.kind != asset.kind
                ):
                    raise StagingError(
                        f"conflicting staged asset definitions: {asset.repository_path}"
                    )
                if previous.license != asset.license:
                    raise StagingError(
                        f"conflicting licenses for shared asset: {asset.repository_path}"
                    )
                continue
            merged[asset.repository_path] = asset
    return merged


def _materialize_asset(
    root: Path,
    asset: PlannedAsset,
    *,
    copy_mode: str,
) -> str:
    destination = root.joinpath(*PurePosixPath(asset.repository_path).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise StagingError(f"duplicate staged output path: {asset.repository_path}")
    if asset.missing:
        source_label = asset.source_path.name if asset.source_path is not None else "asset"
        raise StagingError(
            f"referenced source asset is missing for {asset.benchmark}: {source_label}"
        )
    if asset.content is not None:
        destination.write_bytes(asset.content)
        return "copied"
    if asset.source_path is None:
        raise StagingError(f"asset has no materialization source: {asset.repository_path}")

    source_path = asset.source_path.resolve(strict=True)
    if copy_mode == "hardlink":
        try:
            os.link(source_path, destination, follow_symlinks=True)
            return "hardlinked"
        except OSError:
            pass
    shutil.copyfile(source_path, destination, follow_symlinks=True)
    return "copied"


def _asset_record(asset: PlannedAsset) -> AssetManifestRecord:
    record: AssetManifestRecord = {
        "path": asset.repository_path,
        "bytes": asset.bytes,
        "kind": asset.kind,
        "benchmark": asset.benchmark,
        "license": asset.license,
    }
    if asset.sha256 is not None:
        record["sha256"] = asset.sha256
    return record


def _annotation_asset(
    path: Path,
    *,
    repository_path: str,
    source: EvaluationSource,
    checksums: bool = False,
) -> AssetManifestRecord:
    record: AssetManifestRecord = {
        "path": repository_path,
        "bytes": path.stat().st_size,
        "kind": "annotations",
        "benchmark": source.benchmark,
        "license": source.license,
    }
    if checksums:
        record["sha256"] = sha256_file(path)
    return record


def _manifest_record(
    converted: ConvertedSource,
    annotation_record: AssetManifestRecord,
) -> tuple[dict[str, Any], set[str]]:
    source = converted.source
    if source.expected_count is None:
        raise StagingError(
            f"{source.benchmark}/{source.split}: expected_count must be locked before build"
        )
    canonical_annotation = str(annotation_record["path"])
    rewrites, profile_directories = _path_rewrites(converted, canonical_annotation)

    evaluation = _portable_profile(
        source.evaluation,
        source=source,
        rewrites=rewrites,
    )
    if not isinstance(evaluation, Mapping) or not evaluation:
        raise StagingError(f"{source.benchmark}/{source.split}: evaluation profile became empty")
    preprocessing = _portable_profile(
        source.preprocessing,
        source=source,
        rewrites=rewrites,
    )
    legacy_environment = _portable_profile(
        source.legacy_environment,
        source=source,
        rewrites=rewrites,
    )
    metadata = _portable_profile(
        source.metadata,
        source=source,
        rewrites=rewrites,
    )

    existing_assets = [asset for asset in converted.assets if not asset.missing]
    media_paths = {
        str(PurePosixPath(asset.repository_path).parent)
        for asset in existing_assets
        if asset.kind in {"images", "subtitles", "videos"}
    }
    artifact_paths = {
        artifact_directory(source.benchmark)
        for asset in existing_assets
        if asset.kind == "artifacts"
    }
    for directory in profile_directories:
        if directory.startswith(f"{MEDIA_DIRECTORY}/"):
            media_paths.add(directory)
        elif directory.startswith(f"{ARTIFACTS_DIRECTORY}/"):
            artifact_paths.add(directory)
    record: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "benchmark": source.benchmark,
        "split": source.split,
        "task": source.eval_task,
        "family": source.family,
        "annotation_path": canonical_annotation,
        "media_paths": sorted(media_paths),
        "artifact_paths": sorted(artifact_paths),
        "expected_count": source.expected_count,
        "license": source.license,
        "source_url": source.source_url,
        "redistribution_authorized": source.redistribution_authorized,
        "evaluation": dict(evaluation),
    }
    annotation_checksum = annotation_record.get("sha256")
    if isinstance(annotation_checksum, str):
        record["checksums"] = {canonical_annotation: annotation_checksum}
    if isinstance(preprocessing, Mapping) and preprocessing:
        record["preprocessing"] = dict(preprocessing)
    if isinstance(legacy_environment, Mapping) and legacy_environment:
        record["legacy_environment"] = dict(legacy_environment)
    if isinstance(metadata, Mapping) and metadata:
        record["metadata"] = dict(metadata)
    return record, profile_directories


def _overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _protect_sources(
    target: Path,
    manifest_path: Path,
    sources: Sequence[EvaluationSource],
) -> None:
    protected = [manifest_path]
    for source in sources:
        protected.append(source.annotation)
        protected.extend(source.media_roots.values())
        protected.extend(source.preprocessed_roots.values())
    for path in protected:
        normalized = path.resolve(strict=False)
        if _overlaps(target, normalized):
            raise StagingError(f"output target must not overlap a source path: {normalized}")


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _publish_directory(staging: Path, target: Path, *, overwrite: bool) -> None:
    if target.exists() or target.is_symlink():
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite existing evaluation repository: {target}")
        if target.is_symlink():
            raise StagingError(f"refusing to overwrite a symbolic-link target: {target}")
        backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
        os.replace(target, backup)
        try:
            os.replace(staging, target)
        except Exception:
            os.replace(backup, target)
            raise
        _remove_path(backup)
        return
    os.replace(staging, target)


def build_evaluation_repository(
    source_manifest: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    copy_mode: str = "copy",
    overwrite: bool = False,
    workers: int = 1,
    checksums: bool = False,
) -> dict[str, Any]:
    """Atomically build a deterministic canonical evaluation repository."""

    if copy_mode not in {"copy", "hardlink"}:
        raise StagingError("copy_mode must be 'copy' or 'hardlink'")
    target = Path(output_root).expanduser().resolve(strict=False)
    if target.exists() or target.is_symlink():
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite existing evaluation repository: {target}")
        if target.is_symlink():
            raise StagingError(f"refusing to overwrite a symbolic-link target: {target}")

    manifest_path = Path(source_manifest).expanduser().resolve()
    sources = load_source_manifest(manifest_path)
    _protect_sources(target, manifest_path, sources)
    converted = convert_sources(
        sources,
        workers=workers,
        hash_assets=checksums,
    )
    for item in converted:
        actual = len(item.rows)
        expected = item.source.expected_count
        if expected is None:
            raise StagingError(
                f"{item.source.benchmark}/{item.source.split}: expected_count is not locked; "
                "run inventory --write-locked-manifest first"
            )
        if actual != expected:
            raise StagingError(
                f"{item.source.benchmark}/{item.source.split}: expected "
                f"{expected} rows, converted {actual}"
            )
        if not item.source.redistribution_authorized:
            raise StagingError(
                f"{item.source.benchmark}/{item.source.split}: redistribution is not authorized"
            )
        if item.missing_assets:
            names = sorted(
                asset.source_path.name
                for asset in item.missing_assets
                if asset.source_path is not None
            )
            detail = ", ".join(names[:5]) or "unknown asset"
            raise StagingError(
                f"{item.source.benchmark}/{item.source.split}: "
                f"{len(item.missing_assets)} referenced asset(s) are missing: {detail}"
            )

    merged_assets = _merge_assets(converted)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.staging-",
            dir=str(target.parent),
        )
    )
    materialization_counts = {"copied": 0, "hardlinked": 0}
    try:
        ordered_assets = sorted(
            merged_assets.values(),
            key=lambda item: item.repository_path,
        )

        def _stage_asset(asset: PlannedAsset) -> str:
            return _materialize_asset(staging, asset, copy_mode=copy_mode)

        if workers == 1:
            actions = map(_stage_asset, ordered_assets)
        else:
            executor = ThreadPoolExecutor(max_workers=workers)
            actions = executor.map(_stage_asset, ordered_assets)
        try:
            for action in actions:
                materialization_counts[action] += 1
        finally:
            if workers != 1:
                executor.shutdown(wait=True, cancel_futures=True)

        annotation_records: dict[tuple[str, str], AssetManifestRecord] = {}
        asset_records: list[AssetManifestRecord] = [
            _asset_record(asset)
            for asset in sorted(merged_assets.values(), key=lambda item: item.repository_path)
        ]
        for item in converted:
            relative = annotation_path(item.source.benchmark, item.source.split)
            annotation = staging.joinpath(*PurePosixPath(relative).parts)
            _atomic_write_jsonl(annotation, item.rows)
            annotation_record = _annotation_asset(
                annotation,
                repository_path=relative,
                source=item.source,
                checksums=checksums,
            )
            annotation_records[(item.source.benchmark, item.source.split)] = annotation_record
            asset_records.append(annotation_record)

        manifest_records: list[dict[str, Any]] = []
        declared_directories: set[str] = set()
        for item in converted:
            key = (item.source.benchmark, item.source.split)
            record, directories = _manifest_record(item, annotation_records[key])
            manifest_records.append(record)
            declared_directories.update(directories)
        manifest_records.sort(key=lambda item: (str(item["benchmark"]), str(item["split"])))
        asset_records.sort(key=lambda item: str(item["path"]))

        for relative in sorted(declared_directories):
            staging.joinpath(*PurePosixPath(relative).parts).mkdir(
                parents=True,
                exist_ok=True,
            )
        _atomic_write_jsonl(staging / MANIFEST_FILENAME, manifest_records)
        _atomic_write_jsonl(staging / ASSET_MANIFEST_FILENAME, asset_records)
        write_huggingface_metadata(staging)
        if checksums:
            summary = validate_staged_repository(staging, checksums=True)
        else:
            summary = {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "datasets": len(manifest_records),
                "rows": sum(len(item.rows) for item in converted),
                "assets": len(asset_records),
                "bytes": sum(int(record["bytes"]) for record in asset_records),
            }
        _publish_directory(staging, target, overwrite=overwrite)
    except Exception:
        if staging.exists() or staging.is_symlink():
            _remove_path(staging)
        raise

    return {
        **summary,
        "copy_mode": copy_mode,
        "asset_checksums": checksums,
        **materialization_counts,
    }


def _reconstruct_asset_manifest(
    root: Path,
    datasets: Sequence[Mapping[str, Any]],
) -> list[AssetManifestRecord]:
    """Reconstruct a deleted asset manifest without rehashing addressed payloads."""

    licenses: dict[str, str] = {}
    for dataset in datasets:
        benchmark = str(dataset["benchmark"])
        license_name = str(dataset["license"])
        previous = licenses.get(benchmark)
        if previous is not None and previous != license_name:
            raise StagingError(
                f"cannot reconstruct assets with conflicting licenses for {benchmark}"
            )
        licenses[benchmark] = license_name

    records: list[AssetManifestRecord] = []
    for directory in _STAGED_DIRECTORIES:
        asset_root = root / directory
        if not asset_root.exists():
            continue
        for path in sorted(asset_root.rglob("*")):
            if path.is_symlink():
                raise StagingError(f"staged asset must not be a symlink: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            parts = PurePosixPath(relative).parts
            if len(parts) < 3:
                raise StagingError(f"cannot infer asset metadata from path: {relative}")
            candidates = [
                benchmark
                for benchmark in licenses
                if relative.startswith(
                    f"{directory}/{benchmark_directory(benchmark)}/"
                )
            ]
            if len(candidates) != 1:
                raise StagingError(
                    f"asset has no dataset license declaration: {relative}"
                )
            benchmark = candidates[0]
            license_name = licenses[benchmark]
            if directory == ANNOTATIONS_DIRECTORY:
                kind = "annotations"
            elif directory == ARTIFACTS_DIRECTORY:
                kind = "artifacts"
            else:
                media_parts = PurePosixPath(media_directory(benchmark)).parts
                if (
                    len(parts) <= len(media_parts)
                    or parts[len(media_parts)] not in {"images", "subtitles", "videos"}
                ):
                    raise StagingError(f"cannot infer media kind from path: {relative}")
                kind = parts[len(media_parts)]
            records.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "kind": kind,
                    "benchmark": benchmark,
                    "license": license_name,
                }
            )
    records.sort(key=lambda item: str(item["path"]))
    return validate_asset_manifest(records)


def _require_manifested_asset(
    root: Path,
    record: Mapping[str, Any],
    *,
    context: str,
) -> Path:
    relative = str(record["path"])
    path = root.joinpath(*PurePosixPath(relative).parts)
    if path.is_symlink() or not path.is_file():
        raise StagingError(f"{context} asset is missing or a symlink: {relative}")
    actual_bytes = path.stat().st_size
    if actual_bytes != int(record["bytes"]):
        raise StagingError(
            f"{context} asset byte-size mismatch for {relative}: "
            f"expected {record['bytes']}, got {actual_bytes}"
        )
    return path


def merge_evaluation_repository(
    source_root: str | os.PathLike[str],
    target_root: str | os.PathLike[str],
    *,
    copy_mode: str = "hardlink",
    repair_missing_asset_manifest: bool = False,
    workers: int = 1,
    replace_benchmarks: Iterable[str] = (),
) -> dict[str, Any]:
    """Incrementally add or explicitly replace benchmarks in an existing release."""

    if copy_mode not in {"copy", "hardlink"}:
        raise StagingError("copy_mode must be 'copy' or 'hardlink'")
    if workers <= 0:
        raise StagingError("workers must be positive")
    source = Path(source_root).expanduser().resolve()
    target = Path(target_root).expanduser().resolve()
    if source == target or source in target.parents or target in source.parents:
        raise StagingError("source and target evaluation repositories must be separate")
    if not source.is_dir():
        raise StagingError(f"source evaluation repository does not exist: {source}")
    if not target.is_dir():
        raise StagingError(f"target evaluation repository does not exist: {target}")

    replacements = {
        validate_dataset_id(value, context="replacement benchmark")
        for value in replace_benchmarks
    }
    source_datasets = load_dataset_manifest(source)
    source_assets = load_asset_manifest(source)
    if replacements:
        raw_target_datasets = _load_raw_jsonl(
            target / MANIFEST_FILENAME,
            context=str(target / MANIFEST_FILENAME),
        )
        legacy_datasets: list[Mapping[str, Any]] = []
        current_datasets: list[Mapping[str, Any]] = []
        for index, record in enumerate(raw_target_datasets, start=1):
            benchmark = record.get("benchmark")
            if benchmark in replacements:
                validate_dataset_id(
                    benchmark,
                    context=f"legacy target dataset {index} benchmark",
                )
                validate_dataset_id(
                    record.get("split"),
                    context=f"legacy target dataset {index} split",
                )
                legacy_datasets.append(record)
            else:
                current_datasets.append(record)
        target_datasets = [
            *validate_dataset_manifest(
                current_datasets,
                context=str(target / MANIFEST_FILENAME),
            ),
            *legacy_datasets,
        ]
    else:
        target_datasets = load_dataset_manifest(target)

    target_asset_path = target / ASSET_MANIFEST_FILENAME
    repaired = False
    if target_asset_path.is_file():
        if replacements:
            raw_target_assets = _load_raw_jsonl(
                target_asset_path,
                context=str(target_asset_path),
            )
            legacy_assets: list[Mapping[str, Any]] = []
            current_assets: list[Mapping[str, Any]] = []
            for index, record in enumerate(raw_target_assets, start=1):
                benchmark = record.get("benchmark")
                if benchmark in replacements:
                    validate_dataset_id(
                        benchmark,
                        context=f"legacy target asset {index} benchmark",
                    )
                    validate_repository_path(
                        record.get("path"),
                        context=f"legacy target asset {index} path",
                    )
                    byte_count = record.get("bytes")
                    if (
                        isinstance(byte_count, bool)
                        or not isinstance(byte_count, int)
                        or byte_count < 0
                    ):
                        raise StagingError(
                            f"legacy target asset {index} bytes must be nonnegative"
                        )
                    if record.get("kind") not in _ASSET_KINDS:
                        raise StagingError(
                            f"legacy target asset {index} has unsupported kind"
                        )
                    legacy_assets.append(record)
                else:
                    current_assets.append(record)
            target_assets = [
                *validate_asset_manifest(
                    current_assets,
                    context=str(target_asset_path),
                ),
                *legacy_assets,
            ]
        else:
            target_assets = load_asset_manifest(target_asset_path)
    elif repair_missing_asset_manifest:
        if replacements:
            raise StagingError(
                "cannot reconstruct a legacy asset manifest during benchmark replacement"
            )
        target_assets = _reconstruct_asset_manifest(target, target_datasets)
        repaired = True
    else:
        raise StagingError(
            f"target asset manifest is missing: {target_asset_path}; "
            "pass repair_missing_asset_manifest=True to reconstruct it"
        )

    source_benchmarks = {str(record["benchmark"]) for record in source_datasets}
    unavailable = sorted(replacements - source_benchmarks)
    if unavailable:
        raise StagingError(
            "replacement benchmark is absent from source repository: "
            + ", ".join(unavailable)
        )
    for benchmark in sorted(replacements):
        source_splits = {
            str(record["split"])
            for record in source_datasets
            if str(record["benchmark"]) == benchmark
        }
        target_splits = {
            str(record["split"])
            for record in target_datasets
            if str(record["benchmark"]) == benchmark
        }
        missing_splits = sorted(target_splits - source_splits)
        if missing_splits:
            raise StagingError(
                f"replacement for {benchmark} must include every existing split; "
                f"missing: {', '.join(missing_splits)}"
            )

    datasets = {
        (str(record["benchmark"]), str(record["split"])): record
        for record in target_datasets
        if str(record["benchmark"]) not in replacements
    }
    for record in source_datasets:
        key = (str(record["benchmark"]), str(record["split"]))
        previous = datasets.get(key)
        if previous is not None and previous != record:
            raise StagingError(f"conflicting dataset definition: {key[0]}/{key[1]}")
        datasets[key] = record

    replaced_target_assets = [
        record
        for record in target_assets
        if str(record["benchmark"]) in replacements
    ]
    replaced_target_assets_by_path = {
        str(record["path"]): record for record in replaced_target_assets
    }
    for record in replaced_target_assets:
        _require_manifested_asset(target, record, context="replaced target")
    assets = {
        str(record["path"]): record
        for record in target_assets
        if str(record["benchmark"]) not in replacements
    }
    for record in source_assets:
        relative = str(record["path"])
        previous = assets.get(relative)
        if previous is not None and previous != record:
            raise StagingError(f"conflicting asset definition: {relative}")
        assets[relative] = record

    def _source_entry(record: Mapping[str, Any]) -> tuple[str, Path]:
        relative = str(record["path"])
        return (
            relative,
            _require_manifested_asset(source, record, context="source"),
        )

    if workers == 1:
        source_entries = map(_source_entry, source_assets)
    else:
        source_executor = ThreadPoolExecutor(max_workers=workers)
        source_entries = source_executor.map(_source_entry, source_assets)
    try:
        source_paths = dict(source_entries)
    finally:
        if workers != 1:
            source_executor.shutdown(wait=True, cancel_futures=True)

    def _merge_asset(record: Mapping[str, Any]) -> str:
        relative = str(record["path"])
        destination = target.joinpath(*PurePosixPath(relative).parts)
        if destination.exists() or destination.is_symlink():
            replaced_record = replaced_target_assets_by_path.get(relative)
            if replaced_record is not None and replaced_record != record:
                _require_manifested_asset(
                    target,
                    replaced_record,
                    context="replaced target",
                )
                temporary = destination.with_name(
                    f".{destination.name}.merge-{uuid.uuid4().hex}"
                )
                try:
                    if copy_mode == "hardlink":
                        try:
                            os.link(
                                source_paths[relative],
                                temporary,
                                follow_symlinks=True,
                            )
                            action = "hardlinked"
                        except OSError:
                            shutil.copyfile(
                                source_paths[relative],
                                temporary,
                                follow_symlinks=True,
                            )
                            action = "copied"
                    else:
                        shutil.copyfile(
                            source_paths[relative],
                            temporary,
                            follow_symlinks=True,
                        )
                        action = "copied"
                    os.replace(temporary, destination)
                    return action
                finally:
                    if temporary.exists() or temporary.is_symlink():
                        temporary.unlink()
            _require_manifested_asset(target, record, context="existing target")
            return "existing"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if copy_mode == "hardlink":
            try:
                os.link(source_paths[relative], destination, follow_symlinks=True)
                return "hardlinked"
            except OSError:
                pass
        shutil.copyfile(source_paths[relative], destination, follow_symlinks=True)
        return "copied"

    counts = {"copied": 0, "hardlinked": 0, "existing": 0}
    if workers == 1:
        actions = map(_merge_asset, source_assets)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        actions = executor.map(_merge_asset, source_assets)
    try:
        for action in actions:
            counts[action] += 1
    finally:
        if workers != 1:
            executor.shutdown(wait=True, cancel_futures=True)

    merged_datasets = sorted(
        datasets.values(),
        key=lambda item: (str(item["benchmark"]), str(item["split"])),
    )
    merged_assets = sorted(assets.values(), key=lambda item: str(item["path"]))
    _atomic_write_jsonl(target / MANIFEST_FILENAME, merged_datasets)
    _atomic_write_jsonl(target_asset_path, merged_assets)
    write_huggingface_metadata(target)

    retained_paths = {str(record["path"]) for record in merged_assets}
    removed = 0
    for record in replaced_target_assets:
        relative = str(record["path"])
        if relative in retained_paths:
            continue
        stale = target.joinpath(*PurePosixPath(relative).parts)
        if stale.is_symlink() or not stale.is_file():
            raise StagingError(f"replaced target asset disappeared: {relative}")
        stale.unlink()
        removed += 1
        parent = stale.parent
        while parent != target:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "datasets": len(merged_datasets),
        "rows": sum(int(record["expected_count"]) for record in merged_datasets),
        "assets": len(merged_assets),
        "bytes": sum(int(record["bytes"]) for record in merged_assets),
        "copy_mode": copy_mode,
        "workers": workers,
        "repaired_asset_manifest": repaired,
        "replaced_benchmarks": sorted(replacements),
        "removed": removed,
        **counts,
    }


def _without_artifact_references(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(
        (f"{ARTIFACTS_DIRECTORY}/", f"./{ARTIFACTS_DIRECTORY}/")
    ):
        return _DROP
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            converted = _without_artifact_references(item)
            if converted is not _DROP:
                cleaned[str(key)] = converted
        return cleaned
    if isinstance(value, list):
        return [
            converted
            for item in value
            if (converted := _without_artifact_references(item)) is not _DROP
        ]
    return value


def _public_dataset_record(record: Mapping[str, Any]) -> dict[str, Any]:
    public = dict(record)
    public["artifact_paths"] = []
    if public.get("task") == "segmentation":
        legacy = dict(public.get("legacy_environment", {}))
        legacy["SEGMENTATION_VIDEO_READER"] = "decord"
        setting = str(legacy.get("SEGMENTATION_SETTING", ""))
        if setting and "readerdecord" not in setting:
            marker = setting.rfind("-new")
            legacy["SEGMENTATION_SETTING"] = (
                f"{setting[:marker]}-readerdecord{setting[marker:]}"
                if marker >= 0
                else f"{setting}-readerdecord"
            )
        public["legacy_environment"] = legacy
        preprocessing = dict(public.get("preprocessing", {}))
        preprocessing["video_reader"] = "decord"
        public["preprocessing"] = preprocessing
    return public


def export_public_evaluation_repository(
    repository_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    copy_mode: str = "hardlink",
    overwrite: bool = False,
    workers: int = 8,
    checksums: bool = False,
) -> dict[str, Any]:
    """Export raw media and annotations while excluding processed artifacts."""

    if copy_mode not in {"copy", "hardlink"}:
        raise StagingError("copy_mode must be 'copy' or 'hardlink'")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise StagingError("workers must be a positive integer")

    source = Path(repository_root).expanduser().resolve()
    target = Path(output_root).expanduser().resolve(strict=False)
    if source == target or source in target.parents or target in source.parents:
        raise StagingError("public export and complete repository must be separate")
    if target.exists() or target.is_symlink():
        if not overwrite:
            raise FileExistsError(
                f"refusing to overwrite public evaluation repository: {target}"
            )
        if target.is_symlink():
            raise StagingError("refusing to overwrite a symbolic-link target")

    try:
        validate_evaluation_repository(
            source,
            require_redistribution_authorized=True,
        )
        source_datasets = load_dataset_manifest(source)
        source_assets = load_asset_manifest(
            source,
            repository_root=source,
            checksums=checksums,
        )
        validate_huggingface_metadata(source)
    except (DatasetCardError, ManifestError) as error:
        raise StagingError(str(error)) from error
    public_datasets = [
        _public_dataset_record(record)
        for record in source_datasets
    ]
    retained_assets = [
        dict(record)
        for record in source_assets
        if record["kind"] not in {"annotations", "artifacts"}
    ]
    excluded_artifacts = [
        record for record in source_assets if record["kind"] == "artifacts"
    ]

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.staging-",
            dir=str(target.parent),
        )
    )
    counts = {"copied": 0, "hardlinked": 0}
    try:
        annotation_assets: list[dict[str, Any]] = []
        for dataset in public_datasets:
            relative = str(dataset["annotation_path"])
            source_annotation = source.joinpath(*PurePosixPath(relative).parts)
            rows = _load_raw_jsonl(
                source_annotation,
                context=f"public annotation {relative}",
            )
            cleaned_rows: list[Mapping[str, Any]] = []
            for row in rows:
                cleaned = _without_artifact_references(row)
                if not isinstance(cleaned, Mapping):
                    raise StagingError(f"{relative}: cleaned row must remain an object")
                if not cleaned.get("preprocessed"):
                    cleaned = {
                        key: value
                        for key, value in cleaned.items()
                        if key != "preprocessed"
                    }
                cleaned_rows.append(cleaned)
            destination = staging.joinpath(*PurePosixPath(relative).parts)
            _atomic_write_jsonl(destination, cleaned_rows)

            original = next(
                (
                    record
                    for record in source_assets
                    if str(record["path"]) == relative
                ),
                None,
            )
            if original is None:
                raise StagingError(f"annotation is not manifested: {relative}")
            annotation_asset = dict(original)
            annotation_asset["bytes"] = destination.stat().st_size
            if "sha256" in annotation_asset:
                annotation_asset["sha256"] = sha256_file(destination)
            annotation_assets.append(annotation_asset)

        def materialize(record: Mapping[str, Any]) -> str:
            relative = str(record["path"])
            source_path = _require_manifested_asset(
                source,
                record,
                context="public source",
            )
            destination = staging.joinpath(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if copy_mode == "hardlink":
                try:
                    os.link(source_path, destination, follow_symlinks=True)
                    return "hardlinked"
                except OSError:
                    pass
            shutil.copyfile(source_path, destination, follow_symlinks=True)
            return "copied"

        executor = (
            ThreadPoolExecutor(max_workers=workers)
            if workers > 1
            else None
        )
        try:
            actions = (
                executor.map(materialize, retained_assets)
                if executor is not None
                else map(materialize, retained_assets)
            )
            for action in actions:
                counts[action] += 1
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

        public_assets = sorted(
            [*annotation_assets, *retained_assets],
            key=lambda item: str(item["path"]),
        )
        _atomic_write_jsonl(staging / MANIFEST_FILENAME, public_datasets)
        _atomic_write_jsonl(staging / ASSET_MANIFEST_FILENAME, public_assets)
        write_huggingface_metadata(
            staging,
            repo_id=DEFAULT_INDEX_REPO_ID,
        )
        summary = validate_staged_repository(
            staging,
            checksums=checksums,
            repo_id=DEFAULT_INDEX_REPO_ID,
        )
        _publish_directory(staging, target, overwrite=overwrite)
    except Exception:
        if staging.exists() or staging.is_symlink():
            _remove_path(staging)
        raise

    return {
        **summary,
        **counts,
        "excluded_artifacts": len(excluded_artifacts),
        "excluded_artifact_bytes": sum(
            int(record["bytes"]) for record in excluded_artifacts
        ),
    }


def export_evaluation_index(
    repository_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    checksums: bool = False,
) -> dict[str, Any]:
    """Atomically export JSONL metadata while keeping large assets external."""

    source = Path(repository_root).expanduser().resolve()
    target = Path(output_root).expanduser().resolve(strict=False)
    if source == target or source in target.parents or target in source.parents:
        raise StagingError("evaluation index and complete repository must be separate")
    if target.exists() or target.is_symlink():
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite evaluation index: {target}")
        if target.is_symlink():
            raise StagingError(f"refusing to overwrite a symbolic-link target: {target}")

    if checksums:
        summary = validate_staged_repository(source, checksums=True)
    else:
        source_records = load_dataset_manifest(source)
        source_assets = load_asset_manifest(source)
        summary = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "datasets": len(source_records),
            "rows": sum(int(record["expected_count"]) for record in source_records),
            "assets": len(source_assets),
            "bytes": sum(int(asset["bytes"]) for asset in source_assets),
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.staging-",
            dir=str(target.parent),
        )
    )
    try:
        shutil.copy2(source / MANIFEST_FILENAME, staging / MANIFEST_FILENAME)
        shutil.copytree(
            source / ANNOTATIONS_DIRECTORY,
            staging / ANNOTATIONS_DIRECTORY,
        )
        records = load_dataset_manifest(staging / MANIFEST_FILENAME)
        assets = load_asset_manifest(source / ASSET_MANIFEST_FILENAME)
        annotation_assets = [
            asset for asset in assets if str(asset["kind"]) == "annotations"
        ]
        if checksums:
            validate_asset_manifest(
                annotation_assets,
                repository_root=staging,
                context=str(staging / ASSET_MANIFEST_FILENAME),
            )
        else:
            validate_asset_manifest(
                annotation_assets,
                context=str(staging / ASSET_MANIFEST_FILENAME),
            )
            for asset in annotation_assets:
                _require_manifested_asset(
                    staging,
                    asset,
                    context="exported annotation",
                )
        declared_annotations = {
            str(record["annotation_path"]) for record in records
        }
        manifested_annotations = {
            str(asset["path"]) for asset in annotation_assets
        }
        if declared_annotations != manifested_annotations:
            raise StagingError(
                "exported annotation files do not match datasets.jsonl"
            )
        (staging / DATASET_CARD_FILENAME).write_text(
            render_index_card(records, annotation_assets),
            encoding="utf-8",
        )
        parent_stat = target.parent.stat()
        for path in (staging, *staging.rglob("*")):
            if os.geteuid() == 0:
                os.chown(path, parent_stat.st_uid, parent_stat.st_gid)
            path.chmod(0o755 if path.is_dir() else 0o644)
        _publish_directory(staging, target, overwrite=overwrite)
    except Exception:
        if staging.exists() or staging.is_symlink():
            _remove_path(staging)
        raise

    return {
        **summary,
        "output": str(target),
        "annotation_assets": len(annotation_assets),
        "external_assets": len(assets) - len(annotation_assets),
        "asset_checksums": checksums,
    }


def asset_manifest_record_errors(record: Any) -> list[str]:
    """Collect schema and canonical-layout errors for one asset record."""

    if not isinstance(record, Mapping):
        return ["record must be a JSON object"]
    errors: list[str] = []
    benchmark = record.get("benchmark")
    try:
        benchmark_id = validate_dataset_id(benchmark, context="benchmark")
    except LayoutError as error:
        errors.append(str(error))
        benchmark_id = None

    kind = record.get("kind")
    if kind not in _ASSET_KINDS:
        choices = ", ".join(sorted(_ASSET_KINDS))
        errors.append(f"kind must be one of {{{choices}}}")
    path = record.get("path")
    try:
        canonical_path = validate_repository_path(path, context="path")
    except LayoutError as error:
        errors.append(str(error))
        canonical_path = None
    if canonical_path is not None and benchmark_id is not None and kind in _ASSET_KINDS:
        try:
            if kind == "annotations":
                expected = (
                    f"{PurePosixPath(annotation_path(benchmark_id, 'split')).parent}/"
                )
                if not canonical_path.startswith(expected):
                    raise LayoutError(f"path must be within {expected!r}")
            elif kind == "artifacts":
                validate_artifact_path(canonical_path, benchmark_id, context="path")
            else:
                validate_media_path(
                    canonical_path,
                    benchmark_id,
                    kind=str(kind),
                    context="path",
                )
        except LayoutError as error:
            errors.append(str(error))

    raw_digest = record.get("sha256")
    digest = None
    if raw_digest is not None:
        try:
            digest = validate_sha256(raw_digest)
        except LayoutError as error:
            errors.append(str(error))
    if canonical_path is not None and digest is not None and kind != "annotations":
        filename = PurePosixPath(canonical_path).name
        if filename.split(".", 1)[0] != digest:
            errors.append("content-addressed asset filename must begin with its sha256")
    byte_count = record.get("bytes")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        errors.append("bytes must be a nonnegative integer")
    license_name = record.get("license")
    if not isinstance(license_name, str) or not license_name.strip():
        errors.append("license must be a nonempty string")
    try:
        json.dumps(record, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        errors.append(f"record must contain only finite JSON values: {error}")
    return errors


def validate_asset_manifest_record(
    record: Any,
    *,
    context: str = "asset record",
) -> Mapping[str, Any]:
    """Validate one ``assets.jsonl`` record."""

    errors = asset_manifest_record_errors(record)
    if errors:
        raise StagingError(f"{context}: " + "; ".join(errors))
    return record


def validate_asset_manifest(
    records: Iterable[Any],
    *,
    repository_root: str | os.PathLike[str] | None = None,
    context: str = ASSET_MANIFEST_FILENAME,
    checksums: bool = False,
) -> list[Mapping[str, Any]]:
    """Validate asset records, paths, sizes, and optionally file checksums."""

    validated: list[Mapping[str, Any]] = []
    seen: dict[str, str] = {}
    previous_path: str | None = None
    root = Path(repository_root).resolve() if repository_root is not None else None
    for index, record in enumerate(records, start=1):
        item = validate_asset_manifest_record(record, context=f"{context}:{index}")
        path = str(item["path"])
        folded = path.casefold()
        prior = seen.get(folded)
        if prior is not None:
            collision = "duplicate" if prior == path else "case-colliding"
            raise StagingError(f"{context}:{index}: {collision} asset path {path!r}")
        if previous_path is not None and path < previous_path:
            raise StagingError(f"{context}:{index}: asset records must be sorted by path")
        seen[folded] = path
        previous_path = path

        if root is not None:
            candidate = root.joinpath(*PurePosixPath(path).parts)
            if candidate.is_symlink():
                raise StagingError(f"{context}:{index}: staged asset must not be a symlink: {path}")
            if not candidate.is_file():
                raise StagingError(f"{context}:{index}: staged asset does not exist: {path}")
            actual_bytes = candidate.stat().st_size
            if actual_bytes != item["bytes"]:
                raise StagingError(
                    f"{context}:{index}: byte-size mismatch for {path}: "
                    f"expected {item['bytes']}, got {actual_bytes}"
                )
            expected_digest = item.get("sha256")
            if checksums and isinstance(expected_digest, str):
                actual_digest = sha256_file(candidate)
                if actual_digest != expected_digest:
                    raise StagingError(
                        f"{context}:{index}: checksum mismatch for {path}: "
                        f"expected {expected_digest}, got {actual_digest}"
                    )
        validated.append(item)
    return validated


def load_asset_manifest(
    path: str | os.PathLike[str],
    *,
    repository_root: str | os.PathLike[str] | None = None,
    checksums: bool = False,
) -> list[Mapping[str, Any]]:
    """Load and validate ``assets.jsonl`` from a file or staged root."""

    input_path = Path(path)
    if input_path.is_dir():
        input_path = input_path / ASSET_MANIFEST_FILENAME
    if input_path.name != ASSET_MANIFEST_FILENAME:
        raise StagingError(f"asset manifest must be named {ASSET_MANIFEST_FILENAME}")
    records: list[Any] = []
    try:
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise StagingError(
                        f"{input_path}:{line_number}: invalid JSON: {error}"
                    ) from error
    except OSError as error:
        raise StagingError(f"cannot read asset manifest {input_path}: {error}") from error
    return validate_asset_manifest(
        records,
        repository_root=repository_root,
        context=str(input_path),
        checksums=checksums,
    )


def _staged_files(root: Path) -> set[str]:
    files: set[str] = set()
    for directory_name in _STAGED_DIRECTORIES:
        directory = root / directory_name
        if not directory.exists():
            continue
        if directory.is_symlink():
            raise StagingError(f"staged directory must not be a symlink: {directory_name}")
        for current, directories, filenames in os.walk(directory, followlinks=False):
            current_path = Path(current)
            for name in directories:
                candidate = current_path / name
                if candidate.is_symlink():
                    relative = candidate.relative_to(root).as_posix()
                    raise StagingError(f"staged directory must not be a symlink: {relative}")
            for name in filenames:
                candidate = current_path / name
                relative = candidate.relative_to(root).as_posix()
                if candidate.is_symlink():
                    raise StagingError(f"staged asset must not be a symlink: {relative}")
                if not candidate.is_file():
                    raise StagingError(f"staged asset must be a regular file: {relative}")
                files.add(relative)
    return files


def _manifest_references(
    root: Path,
    manifest: Sequence[Mapping[str, Any]],
    datasets: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
) -> set[str]:
    references: set[str] = set()
    for record in manifest:
        key = (str(record["benchmark"]), str(record["split"]))
        references.add(str(record["annotation_path"]))
        for field in ("evaluation", "preprocessing", "metadata"):
            value = record.get(field)
            if isinstance(value, Mapping):
                references.update(
                    path for _context, path in declared_repository_paths(value, context=field)
                )
        legacy = record.get("legacy_environment")
        if isinstance(legacy, Mapping):
            for name, value in legacy.items():
                if _is_path_key(name) and isinstance(value, str):
                    references.add(value)
        for row in datasets.get(key, ()):
            references.update(evaluation_asset_paths(row))

    return {path for path in references if root.joinpath(*PurePosixPath(path).parts).is_file()}


def validate_staged_repository(
    repository_root: str | os.PathLike[str],
    *,
    require_redistribution_authorized: bool = True,
    checksums: bool = False,
    repo_id: str = DEFAULT_DATASET_REPO_ID,
) -> dict[str, Any]:
    """Validate contracts, sizes, symlink safety, references, and optional checksums."""

    raw_root = Path(repository_root).expanduser()
    if raw_root.is_symlink():
        raise StagingError("evaluation repository root must not be a symlink")
    root = raw_root.resolve()
    if not root.is_dir():
        raise StagingError(f"evaluation repository does not exist: {root}")
    for manifest_name in (
        MANIFEST_FILENAME,
        ASSET_MANIFEST_FILENAME,
        DATASET_CARD_FILENAME,
        GIT_ATTRIBUTES_FILENAME,
    ):
        manifest_path = root / manifest_name
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise StagingError(f"required staged file is missing or a symlink: {manifest_name}")
    try:
        datasets = validate_evaluation_repository(
            root,
            require_redistribution_authorized=require_redistribution_authorized,
        )
        manifest = load_dataset_manifest(root)
    except ManifestError as error:
        raise StagingError(str(error)) from error
    assets = load_asset_manifest(
        root,
        repository_root=root,
        checksums=checksums,
    )
    try:
        validate_huggingface_metadata(root, repo_id=repo_id)
    except DatasetCardError as error:
        raise StagingError(str(error)) from error

    staged_files = _staged_files(root)
    manifested_files = {str(asset["path"]) for asset in assets}
    if staged_files != manifested_files:
        unmanifested = sorted(staged_files - manifested_files)
        missing = sorted(manifested_files - staged_files)
        details = []
        if unmanifested:
            details.append("unmanifested: " + ", ".join(unmanifested[:5]))
        if missing:
            details.append("missing: " + ", ".join(missing[:5]))
        raise StagingError("unreferenced or unmanifested staged assets; " + "; ".join(details))

    references = _manifest_references(root, manifest, datasets)
    unreferenced = sorted(manifested_files - references)
    if unreferenced:
        raise StagingError("unreferenced staged assets: " + ", ".join(unreferenced[:10]))
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "datasets": len(manifest),
        "rows": sum(len(rows) for rows in datasets.values()),
        "assets": len(assets),
        "bytes": sum(int(asset["bytes"]) for asset in assets),
    }


inventory_evaluation_data = inventory_evaluation_sources
inventory_evaluation_repository = inventory_evaluation_sources
inventory_repository = inventory_evaluation_sources
build_evaluation_data = build_evaluation_repository
build_repository = build_evaluation_repository
load_assets_manifest = load_asset_manifest
validate_assets_manifest = validate_asset_manifest
validate_evaluation_data_repository = validate_staged_repository
validate_repository = validate_staged_repository


__all__ = [
    "AssetManifestRecord",
    "StagingError",
    "asset_manifest_record_errors",
    "build_evaluation_data",
    "build_evaluation_repository",
    "build_repository",
    "export_evaluation_index",
    "inventory_evaluation_data",
    "inventory_and_write_locked_manifest",
    "inventory_evaluation_repository",
    "inventory_evaluation_sources",
    "inventory_repository",
    "load_asset_manifest",
    "load_assets_manifest",
    "merge_evaluation_repository",
    "validate_asset_manifest",
    "validate_asset_manifest_record",
    "validate_assets_manifest",
    "validate_evaluation_data_repository",
    "validate_repository",
    "validate_staged_repository",
    "write_locked_source_manifest",
]
