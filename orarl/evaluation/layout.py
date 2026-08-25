"""Canonical repository layout and asset checks for OraRL evaluation data."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Union

MANIFEST_FILENAME = "datasets.jsonl"
ASSET_MANIFEST_FILENAME = "assets.jsonl"
ANNOTATIONS_DIRECTORY = "annotations"
MEDIA_DIRECTORY = "media"
ARTIFACTS_DIRECTORY = "artifacts"
MEDIA_KINDS = frozenset({"images", "videos", "subtitles"})
BENCHMARK_GROUPS = {
    "videomme": "video_qa",
    "videommev2": "video_qa",
    "mvbench": "video_qa",
    "mmvu": "video_qa",
    "videoholmes": "video_qa",
    "longvideobench": "video_qa",
    "mlvu": "video_qa",
    "vsi": "spatial_intelligence",
    "mmsi": "spatial_intelligence",
    "mindcube": "spatial_intelligence",
    "revsi": "spatial_intelligence",
}

_DATASET_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class LayoutError(ValueError):
    """Raised when an evaluation data path or identifier is not canonical."""


def is_dataset_id(value: object) -> bool:
    """Return whether ``value`` is a lowercase snake_case dataset identifier."""

    return isinstance(value, str) and bool(_DATASET_ID_RE.fullmatch(value))


def validate_dataset_id(value: object, *, context: str = "dataset id") -> str:
    """Validate and return one lowercase snake_case dataset identifier."""

    if not is_dataset_id(value):
        raise LayoutError(f"{context} must be a lowercase snake_case identifier")
    return value


def validate_repository_path(value: object, *, context: str = "path") -> str:
    """Validate and return a normalized repository-relative POSIX path.

    Canonical paths are already normalized when stored. In particular, this
    rejects URL/URI references, absolute paths, Windows separators, empty or
    dot components, and parent traversal.
    """

    if not isinstance(value, str) or not value:
        raise LayoutError(f"{context} must be a nonempty repository-relative POSIX path")
    if value != value.strip() or any(ord(character) < 32 for character in value):
        raise LayoutError(f"{context} must not contain surrounding whitespace or controls")
    if "\\" in value:
        raise LayoutError(f"{context} must use POSIX '/' separators, not backslashes")
    if _URI_SCHEME_RE.match(value):
        raise LayoutError(f"{context} must not be an absolute path or URL: {value}")
    if value.startswith("/") or PurePosixPath(value).is_absolute():
        raise LayoutError(f"{context} must be repository-relative, not absolute: {value}")
    if value == "~" or value.startswith("~/"):
        raise LayoutError(f"{context} must be repository-relative: {value}")

    parts = value.split("/")
    if ".." in parts:
        raise LayoutError(f"{context} must not contain '..': {value}")
    if "." in parts or "" in parts:
        raise LayoutError(f"{context} must be normalized without '.' or empty components")
    return value


def benchmark_directory(benchmark: str) -> str:
    """Return the canonical directory namespace for one benchmark."""

    benchmark_id = validate_dataset_id(benchmark, context="benchmark")
    group = BENCHMARK_GROUPS.get(benchmark_id)
    if group is None:
        return benchmark_id
    group_id = validate_dataset_id(group, context="benchmark group")
    return f"{group_id}/{benchmark_id}"


def annotation_path(benchmark: str, split: str) -> str:
    """Return the canonical annotation JSONL path for a benchmark split."""

    directory = benchmark_directory(benchmark)
    split_id = validate_dataset_id(split, context="split")
    return f"{ANNOTATIONS_DIRECTORY}/{directory}/{split_id}.jsonl"


def media_directory(benchmark: str, kind: str | None = None) -> str:
    """Return the canonical media directory for a benchmark and optional kind."""

    base = f"{MEDIA_DIRECTORY}/{benchmark_directory(benchmark)}"
    if kind is None:
        return base
    kind_id = validate_dataset_id(kind, context="media kind")
    if kind_id not in MEDIA_KINDS:
        choices = ", ".join(sorted(MEDIA_KINDS))
        raise LayoutError(f"media kind must be one of {{{choices}}}")
    return f"{base}/{kind_id}"


def artifact_directory(benchmark: str) -> str:
    """Return the canonical artifact directory for a benchmark."""

    return f"{ARTIFACTS_DIRECTORY}/{benchmark_directory(benchmark)}"


def path_is_within(path: str, directory: str) -> bool:
    """Return whether a canonical repository path is at or below a directory."""

    path_parts = PurePosixPath(validate_repository_path(path)).parts
    directory_parts = PurePosixPath(validate_repository_path(directory)).parts
    return path_parts[: len(directory_parts)] == directory_parts


def validate_annotation_path(value: object, benchmark: str, split: str) -> str:
    """Require the exact canonical annotation path for a benchmark split."""

    path = validate_repository_path(value, context="annotation_path")
    expected = annotation_path(benchmark, split)
    if path != expected:
        raise LayoutError(f"annotation_path must be {expected!r}, got {path!r}")
    return path


def validate_media_path(
    value: object,
    benchmark: str,
    *,
    kind: str | None = None,
    allow_directory: bool = False,
    context: str = "media path",
) -> str:
    """Require a path below the benchmark's canonical media directory."""

    path = validate_repository_path(value, context=context)
    directory = media_directory(benchmark, kind)
    if not path_is_within(path, directory):
        raise LayoutError(f"{context} must be within {directory!r}, got {path!r}")
    if not allow_directory and path == directory:
        raise LayoutError(f"{context} must name an asset below {directory!r}")
    return path


def validate_artifact_path(
    value: object,
    benchmark: str,
    *,
    allow_directory: bool = False,
    context: str = "artifact path",
) -> str:
    """Require a path below the benchmark's canonical artifact directory."""

    path = validate_repository_path(value, context=context)
    directory = artifact_directory(benchmark)
    if not path_is_within(path, directory):
        raise LayoutError(f"{context} must be within {directory!r}, got {path!r}")
    if not allow_directory and path == directory:
        raise LayoutError(f"{context} must name an asset below {directory!r}")
    return path


def validate_unique_paths(
    paths: Iterable[str],
    *,
    context: str = "paths",
    allow_exact_duplicates: bool = False,
) -> tuple[str, ...]:
    """Validate path spelling and reject duplicates or case collisions."""

    normalized: list[str] = []
    seen: dict[str, str] = {}
    for index, value in enumerate(paths):
        path = validate_repository_path(value, context=f"{context}[{index}]")
        key = path.casefold()
        previous = seen.get(key)
        if previous is not None:
            if previous != path:
                raise LayoutError(
                    f"{context} contain a case-colliding path: {previous!r} and {path!r}"
                )
            if not allow_exact_duplicates:
                raise LayoutError(f"{context} contain duplicate path {path!r}")
        else:
            seen[key] = path
        normalized.append(path)
    return tuple(normalized)


def sha256_file(path: Union[str, Path]) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sha256(value: object, *, context: str = "sha256") -> str:
    """Validate and return a lowercase hexadecimal SHA-256 digest."""

    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise LayoutError(f"{context} must be 64 lowercase hexadecimal characters")
    return value


def validate_repository_assets(
    paths: Iterable[str],
    repository_root: Union[str, Path],
    *,
    checksums: Mapping[str, str] | None = None,
    file_paths: Iterable[str] = (),
    context: str = "assets",
) -> dict[str, Path]:
    """Require local assets and optionally verify their SHA-256 checksums.

    Directories are accepted for declared media and artifact roots. Paths in
    ``file_paths`` and every checksummed path must resolve to regular files.
    Existing symlinks are resolved and may not escape ``repository_root``.
    """

    root = Path(repository_root).expanduser().resolve()
    if not root.is_dir():
        raise LayoutError(f"{context} repository root does not exist: {root}")

    expected_checksums: dict[str, str] = {}
    for raw_path, raw_digest in (checksums or {}).items():
        path = validate_repository_path(raw_path, context=f"{context} checksum path")
        expected_checksums[path] = validate_sha256(
            raw_digest,
            context=f"{context} checksum for {path}",
        )

    required_files = {
        validate_repository_path(path, context=f"{context} file path") for path in file_paths
    }
    all_paths = list(paths)
    all_paths.extend(expected_checksums)
    all_paths.extend(required_files)
    normalized = validate_unique_paths(
        all_paths,
        context=context,
        allow_exact_duplicates=True,
    )

    resolved: dict[str, Path] = {}
    for path in normalized:
        candidate = root.joinpath(*PurePosixPath(path).parts)
        try:
            asset = candidate.resolve(strict=True)
        except FileNotFoundError as error:
            raise LayoutError(f"{context} path does not exist: {path}") from error
        try:
            asset.relative_to(root)
        except ValueError as error:
            raise LayoutError(f"{context} path resolves outside the repository: {path}") from error

        if path in required_files or path in expected_checksums:
            if not asset.is_file():
                raise LayoutError(f"{context} path must be a file: {path}")
        elif not asset.is_file() and not asset.is_dir():
            raise LayoutError(f"{context} path is not a file or directory: {path}")

        expected = expected_checksums.get(path)
        if expected is not None:
            actual = sha256_file(asset)
            if actual != expected:
                raise LayoutError(
                    f"{context} checksum mismatch for {path}: expected {expected}, got {actual}"
                )
        resolved[path] = asset
    return resolved
