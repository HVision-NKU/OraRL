"""Build a deterministic, leakage-filtered OraRL JSONL mixture.

Run from the OraRL directory with::

    cp configs/data_sources.example.yaml ./data_sources.local.yaml
    python -m orarl.data.build --config ./data_sources.local.yaml --output ./prepared/train.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from .identity import media_anchors, media_leakage_tokens, prompt_identity, stable_rank
from .schema import (
    SchemaError,
    canonicalize_record,
    is_remote_reference,
    read_records,
)

PathLike = Union[str, os.PathLike[str]]


class ConfigError(ValueError):
    """Raised for an invalid source manifest."""


class ShortfallError(RuntimeError):
    """Raised when a requested train or canary count cannot be reached."""


@dataclass(frozen=True)
class SourceSpec:
    name: str
    input_path: Path
    task: str
    family: str
    quota: int
    media_root: Path
    preserve_problem_type: bool
    license: Optional[str]
    url: Optional[str]


@dataclass(frozen=True)
class Candidate:
    record: dict[str, Any]
    spec: SourceSpec
    source_index: int
    identity: str
    anchors: tuple[str, ...]
    leakage_tokens: frozenset[str]
    row_digest: str


@dataclass(frozen=True)
class Exclusions:
    identities: frozenset[str]
    media_tokens: frozenset[str]
    rows: int
    paths: tuple[Path, ...]
    checksums: Mapping[str, str]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _record_digest(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_bytes(record)).hexdigest()


def _atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
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


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as error:
        raise ConfigError(
            f"{path}: YAML input requires PyYAML; use a JSON manifest if it is unavailable"
        ) from error
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except yaml.YAMLError as error:
        raise ConfigError(f"{path}: invalid YAML: {error}") from error


def _load_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.casefold()
    if suffix in {".yaml", ".yml"}:
        payload = _load_yaml(path)
    else:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            payload = _load_yaml(path)
    if not isinstance(payload, dict):
        raise ConfigError(f"{path}: source manifest must be a mapping")
    return payload


def _local_path(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a nonempty local path")
    if is_remote_reference(value):
        raise ConfigError(f"{label} must be local; URL fields are metadata only")
    path = Path(os.path.expanduser(value.strip()))
    if not path.is_absolute():
        path = base / path
    return Path(os.path.abspath(os.path.normpath(str(path))))


def _optional_text(value: Any, label: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a nonempty string when supplied")
    return value.strip()


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{label} must be an integer") from error
    if parsed != value and not (isinstance(value, str) and str(parsed) == value.strip()):
        raise ConfigError(f"{label} must be an integer")
    if parsed < minimum:
        raise ConfigError(f"{label} must be at least {minimum}")
    return parsed


def _boolean(value: Any, label: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ConfigError(f"{label} must be true or false")
    return value


def _source_specs(config_path: Path, config: Mapping[str, Any]) -> list[SourceSpec]:
    raw_sources = config.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ConfigError(f"{config_path}: sources must be a nonempty list")

    specs: list[SourceSpec] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_sources):
        label = f"{config_path}: sources[{index}]"
        if not isinstance(raw, Mapping):
            raise ConfigError(f"{label} must be a mapping")

        raw_input = raw.get("input", raw.get("path"))
        input_path = _local_path(config_path.parent, raw_input, f"{label}.input")
        raw_name = raw.get("name", raw.get("source"))
        name = _optional_text(raw_name, f"{label}.name")
        if name is None:
            name = input_path.stem
        if name in names:
            raise ConfigError(f"{label}.name is duplicated: {name}")
        names.add(name)

        task = _optional_text(raw.get("task"), f"{label}.task")
        family = _optional_text(raw.get("family"), f"{label}.family")
        if task is None or family is None:
            raise ConfigError(f"{label} must define task and family")
        quota = _integer(raw.get("quota"), f"{label}.quota")
        media_root = _local_path(
            config_path.parent,
            raw.get("media_root"),
            f"{label}.media_root",
        )
        if not input_path.is_file():
            raise ConfigError(f"{label}.input does not exist: {input_path}")

        specs.append(
            SourceSpec(
                name=name,
                input_path=input_path,
                task=task,
                family=family,
                quota=quota,
                media_root=media_root,
                preserve_problem_type=_boolean(
                    raw.get("preserve_problem_type"),
                    f"{label}.preserve_problem_type",
                    False,
                ),
                license=_optional_text(raw.get("license"), f"{label}.license"),
                url=_optional_text(raw.get("url"), f"{label}.url"),
            )
        )
    return specs


def _benchmark_paths(
    config_path: Path,
    config: Mapping[str, Any],
    command_line_paths: Sequence[PathLike],
) -> list[Path]:
    raw_paths = config.get(
        "benchmark_excludes",
        config.get("benchmark_exclude", []),
    )
    if raw_paths is None:
        raw_paths = []
    if isinstance(raw_paths, (str, Mapping)):
        raw_paths = [raw_paths]
    if not isinstance(raw_paths, list):
        raise ConfigError(f"{config_path}: benchmark_excludes must be a list")

    paths: list[Path] = []
    for index, raw in enumerate(raw_paths):
        value = raw.get("path") if isinstance(raw, Mapping) else raw
        paths.append(
            _local_path(
                config_path.parent,
                value,
                f"{config_path}: benchmark_excludes[{index}]",
            )
        )
    for index, raw in enumerate(command_line_paths):
        paths.append(_local_path(Path.cwd(), str(raw), f"--benchmark-exclude[{index}]"))

    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = os.path.normcase(str(path))
        if key not in seen:
            seen.add(key)
            if not path.is_file():
                raise ConfigError(f"benchmark exclusion file does not exist: {path}")
            unique.append(path)
    return unique


def _loose_benchmark_record(raw: Mapping[str, Any], root: Path) -> tuple[dict[str, Any], bool]:
    has_problem = any(
        isinstance(raw.get(key), str) and bool(str(raw[key]).strip())
        for key in ("problem", "question", "prompt")
    )
    working = dict(raw)
    if not has_problem:
        working["problem"] = "media exclusion"
    working["answer"] = "excluded"
    if not working.get("problem_type"):
        working["problem_type"] = "benchmark"
    if not any(
        working.get(key)
        for key in ("images", "videos", "image", "video", "image_path", "video_path")
    ):
        generic_media = raw.get("media", raw.get("media_path", raw.get("path")))
        if generic_media:
            working["videos"] = generic_media
    return (
        canonicalize_record(
            working,
            source="benchmark",
            media_root=root,
            require_media=False,
            context="benchmark exclusion",
        ),
        has_problem and bool(raw.get("problem_type")),
    )


def _load_exclusions(paths: Sequence[Path]) -> Exclusions:
    identities: set[str] = set()
    tokens: set[str] = set()
    rows = 0
    checksums: dict[str, str] = {}
    for path in paths:
        checksums[str(path)] = _sha256_file(path)
        for raw in read_records(path):
            rows += 1
            explicit_identity = raw.get("prompt_identity", raw.get("identity"))
            if isinstance(explicit_identity, str) and explicit_identity.strip():
                identities.add(explicit_identity.strip())
            try:
                record, has_full_identity = _loose_benchmark_record(raw, path.parent)
            except SchemaError:
                if explicit_identity:
                    continue
                raise
            tokens.update(media_leakage_tokens(record))
            if has_full_identity:
                identities.add(prompt_identity(record))
    return Exclusions(
        identities=frozenset(identities),
        media_tokens=frozenset(tokens),
        rows=rows,
        paths=tuple(paths),
        checksums=checksums,
    )


def _load_candidates(
    specs: Sequence[SourceSpec],
    require_media: bool,
) -> tuple[list[Candidate], dict[str, int], dict[str, str]]:
    candidates: list[Candidate] = []
    input_counts: dict[str, int] = {}
    checksums: dict[str, str] = {}
    for spec in specs:
        raw_records = read_records(spec.input_path)
        input_counts[spec.name] = len(raw_records)
        checksums[spec.name] = _sha256_file(spec.input_path)
        for index, raw in enumerate(raw_records):
            prepared = dict(raw)
            if (
                not spec.preserve_problem_type
                or not str(prepared.get("problem_type") or "").strip()
            ):
                prepared["problem_type"] = spec.task
            context = f"{spec.input_path}: record {index + 1}"
            record = canonicalize_record(
                prepared,
                source=spec.name,
                family=spec.family,
                media_root=spec.media_root,
                require_media=require_media,
                context=context,
            )
            try:
                digest = _record_digest(record)
            except (TypeError, ValueError) as error:
                raise SchemaError(f"{context}: record is not strict JSON: {error}") from error
            candidates.append(
                Candidate(
                    record=record,
                    spec=spec,
                    source_index=index,
                    identity=prompt_identity(record),
                    anchors=media_anchors(record),
                    leakage_tokens=media_leakage_tokens(record),
                    row_digest=digest,
                )
            )
    return candidates, input_counts, checksums


def _candidate_order(candidate: Candidate, seed: int, namespace: str) -> tuple[int, str, int]:
    key = f"{candidate.identity}|{candidate.row_digest}"
    return (
        stable_rank(seed, namespace, key),
        candidate.row_digest,
        candidate.source_index,
    )


def _select_train(
    specs: Sequence[SourceSpec],
    candidates: Sequence[Candidate],
    seed: int,
    max_prompts_per_media: int,
) -> tuple[list[Candidate], dict[str, int], dict[str, int]]:
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.spec.name].append(candidate)

    unique_capacity = {
        spec.name: len({candidate.identity for candidate in grouped.get(spec.name, [])})
        for spec in specs
    }
    spec_order = sorted(
        specs,
        key=lambda spec: (
            unique_capacity[spec.name] - spec.quota,
            spec.name.casefold(),
        ),
    )

    selected: list[Candidate] = []
    selected_identities: set[str] = set()
    anchor_counts: Counter[str] = Counter()
    cap_rejections: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()

    for spec in spec_order:
        ordered = sorted(
            grouped.get(spec.name, []),
            key=lambda candidate: _candidate_order(
                candidate,
                seed,
                f"train:{spec.name}",
            ),
        )
        for candidate in ordered:
            if selected_counts[spec.name] >= spec.quota:
                break
            if candidate.identity in selected_identities:
                continue
            if any(anchor_counts[anchor] >= max_prompts_per_media for anchor in candidate.anchors):
                cap_rejections[spec.name] += 1
                continue
            selected.append(candidate)
            selected_counts[spec.name] += 1
            selected_identities.add(candidate.identity)
            anchor_counts.update(candidate.anchors)

    shortfalls = {
        spec.name: spec.quota - selected_counts[spec.name]
        for spec in specs
        if selected_counts[spec.name] < spec.quota
    }
    return selected, shortfalls, dict(cap_rejections)


def _select_canary(
    candidates: Sequence[Candidate],
    train: Sequence[Candidate],
    canary_size: int,
    seed: int,
    max_prompts_per_media: int,
) -> tuple[list[Candidate], dict[str, int]]:
    train_identities = {candidate.identity for candidate in train}
    train_media_tokens: set[str] = set()
    for candidate in train:
        train_media_tokens.update(candidate.leakage_tokens)

    ordered = sorted(
        candidates,
        key=lambda candidate: _candidate_order(candidate, seed, "canary"),
    )
    canary: list[Candidate] = []
    canary_identities: set[str] = set()
    canary_anchor_counts: Counter[str] = Counter()
    stats: Counter[str] = Counter()
    for candidate in ordered:
        if len(canary) >= canary_size:
            break
        if candidate.identity in train_identities:
            stats["train_identity"] += 1
            continue
        if candidate.identity in canary_identities:
            stats["duplicate_identity"] += 1
            continue
        if candidate.leakage_tokens & train_media_tokens:
            stats["train_media"] += 1
            continue
        if any(
            canary_anchor_counts[anchor] >= max_prompts_per_media for anchor in candidate.anchors
        ):
            stats["media_cap"] += 1
            continue
        canary.append(candidate)
        canary_identities.add(candidate.identity)
        canary_anchor_counts.update(candidate.anchors)
    return canary, dict(stats)


def _sorted_output(candidates: Sequence[Candidate], seed: int, namespace: str) -> list[Candidate]:
    return sorted(
        candidates,
        key=lambda candidate: _candidate_order(candidate, seed, namespace),
    )


def _counts(candidates: Sequence[Candidate]) -> dict[str, Any]:
    by_task = Counter(str(candidate.record["problem_type"]) for candidate in candidates)
    by_source = Counter(str(candidate.record["source"]) for candidate in candidates)
    by_family = Counter(str(candidate.record.get("family") or "") for candidate in candidates)
    return {
        "rows": len(candidates),
        "by_task": dict(sorted(by_task.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_family": dict(sorted(by_family.items())),
    }


def _default_canary_path(output: Path) -> Path:
    suffix = output.suffix or ".jsonl"
    stem = output.name[: -len(output.suffix)] if output.suffix else output.name
    return output.with_name(stem + ".canary" + suffix)


def _default_manifest_path(output: Path) -> Path:
    stem = output.name[: -len(output.suffix)] if output.suffix else output.name
    return output.with_name(stem + ".manifest.json")


def _output_path(value: PathLike) -> Path:
    path = Path(os.path.expanduser(str(value)))
    return Path(os.path.abspath(os.path.normpath(str(path))))


def _check_output_paths(
    paths: Sequence[Path],
    overwrite: bool,
    protected_paths: Sequence[Path],
) -> None:
    normalized = [os.path.normcase(str(path)) for path in paths]
    if len(normalized) != len(set(normalized)):
        raise ConfigError("train, canary, and manifest outputs must be different paths")
    protected = {os.path.normcase(str(path)) for path in protected_paths}
    collisions = [
        str(path)
        for path, normalized_path in zip(paths, normalized)
        if normalized_path in protected
    ]
    if collisions:
        raise ConfigError(
            "output paths cannot replace configs, sources, or exclusions: " + ", ".join(collisions)
        )
    if not overwrite:
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing output(s): "
                + ", ".join(existing)
                + "; pass --overwrite"
            )


def _output_leakage(
    train: Sequence[Candidate],
    canary: Sequence[Candidate],
    exclusions: Exclusions,
) -> dict[str, int]:
    train_ids = {candidate.identity for candidate in train}
    canary_ids = {candidate.identity for candidate in canary}
    train_tokens: set[str] = set()
    canary_tokens: set[str] = set()
    for candidate in train:
        train_tokens.update(candidate.leakage_tokens)
    for candidate in canary:
        canary_tokens.update(candidate.leakage_tokens)
    return {
        "benchmark_identity": len((train_ids | canary_ids) & exclusions.identities),
        "benchmark_media": len((train_tokens | canary_tokens) & exclusions.media_tokens),
        "train_canary_identity": len(train_ids & canary_ids),
        "train_canary_media": len(train_tokens & canary_tokens),
    }


def build_dataset(
    config_path: PathLike,
    output_path: PathLike,
    *,
    canary_output: Optional[PathLike] = None,
    manifest_output: Optional[PathLike] = None,
    seed: Optional[int] = None,
    target: Optional[int] = None,
    canary_size: Optional[int] = None,
    max_prompts_per_media: Optional[int] = None,
    require_media: Optional[bool] = None,
    benchmark_excludes: Sequence[PathLike] = (),
    allow_shortfall: Optional[bool] = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build train, canary, and audit manifest files."""

    config_file = _output_path(config_path)
    if not config_file.is_file():
        raise ConfigError(f"source manifest does not exist: {config_file}")
    config = _load_config(config_file)
    specs = _source_specs(config_file, config)

    effective_seed = _integer(
        seed if seed is not None else config.get("seed", 42),
        "seed",
    )
    effective_canary_size = _integer(
        canary_size if canary_size is not None else config.get("canary_size", 0),
        "canary_size",
    )
    effective_cap = _integer(
        (
            max_prompts_per_media
            if max_prompts_per_media is not None
            else config.get(
                "max_prompts_per_media",
                config.get("max_prompts_per_anchor", 1),
            )
        ),
        "max_prompts_per_media",
        minimum=1,
    )
    effective_require_media = (
        require_media
        if require_media is not None
        else _boolean(config.get("require_media"), "require_media", False)
    )
    effective_allow_shortfall = (
        allow_shortfall
        if allow_shortfall is not None
        else _boolean(config.get("allow_shortfall"), "allow_shortfall", False)
    )

    quota_total = sum(spec.quota for spec in specs)
    effective_target = _integer(
        target if target is not None else config.get("target", quota_total),
        "target",
    )
    if effective_target != quota_total:
        raise ConfigError(
            f"target ({effective_target}) must equal the sum of source quotas ({quota_total})"
        )

    output = _output_path(output_path)
    canary_path = (
        _output_path(canary_output) if canary_output is not None else _default_canary_path(output)
    )
    manifest_path = (
        _output_path(manifest_output)
        if manifest_output is not None
        else _default_manifest_path(output)
    )

    benchmark_paths = _benchmark_paths(config_file, config, benchmark_excludes)
    protected_paths = [
        config_file,
        *(spec.input_path for spec in specs),
        *benchmark_paths,
    ]
    _check_output_paths(
        (output, canary_path, manifest_path),
        overwrite,
        protected_paths,
    )
    exclusions = _load_exclusions(benchmark_paths)
    all_candidates, input_counts, source_checksums = _load_candidates(
        specs,
        require_media=effective_require_media,
    )

    identity_counts = Counter(candidate.identity for candidate in all_candidates)
    duplicate_rows = sum(count - 1 for count in identity_counts.values())
    duplicate_groups = sum(count > 1 for count in identity_counts.values())

    eligible: list[Candidate] = []
    identity_leak_rows = 0
    media_leak_rows = 0
    excluded_rows = 0
    for candidate in all_candidates:
        identity_hit = candidate.identity in exclusions.identities
        media_hit = bool(candidate.leakage_tokens & exclusions.media_tokens)
        identity_leak_rows += int(identity_hit)
        media_leak_rows += int(media_hit)
        if identity_hit or media_hit:
            excluded_rows += 1
        else:
            eligible.append(candidate)

    train, source_shortfalls, cap_rejections = _select_train(
        specs,
        eligible,
        effective_seed,
        effective_cap,
    )
    if source_shortfalls and not effective_allow_shortfall:
        detail = ", ".join(
            f"{name} short by {count}" for name, count in sorted(source_shortfalls.items())
        )
        raise ShortfallError(f"source quota shortfall: {detail}")

    canary, canary_rejections = _select_canary(
        eligible,
        train,
        effective_canary_size,
        effective_seed,
        effective_cap,
    )
    canary_shortfall = effective_canary_size - len(canary)
    if canary_shortfall and not effective_allow_shortfall:
        raise ShortfallError(
            f"canary shortfall: requested {effective_canary_size}, got {len(canary)}"
        )

    train = _sorted_output(train, effective_seed, "train-output")
    canary = _sorted_output(canary, effective_seed, "canary-output")
    output_leakage = _output_leakage(train, canary, exclusions)
    if any(output_leakage.values()):
        raise RuntimeError(f"internal leakage invariant failed: {output_leakage}")

    _atomic_write_jsonl(output, (candidate.record for candidate in train))
    _atomic_write_jsonl(canary_path, (candidate.record for candidate in canary))

    source_details = []
    selected_by_source = Counter(candidate.spec.name for candidate in train)
    for spec in sorted(specs, key=lambda value: value.name.casefold()):
        source_details.append(
            {
                "name": spec.name,
                "input": str(spec.input_path),
                "media_root": str(spec.media_root),
                "task": spec.task,
                "family": spec.family,
                "quota": spec.quota,
                "selected": selected_by_source[spec.name],
                "shortfall": spec.quota - selected_by_source[spec.name],
                "preserve_problem_type": spec.preserve_problem_type,
                "license": spec.license,
                "url": spec.url,
                "input_sha256": source_checksums[spec.name],
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "seed": effective_seed,
        "target": effective_target,
        "actual_train_rows": len(train),
        "canary_target": effective_canary_size,
        "actual_canary_rows": len(canary),
        "allow_shortfall": effective_allow_shortfall,
        "media": {
            "require_existing": effective_require_media,
            "max_prompts_per_anchor": effective_cap,
        },
        "files": {
            "train": str(output),
            "canary": str(canary_path),
            "manifest": str(manifest_path),
        },
        "checksums": {
            "algorithm": "sha256",
            "config": _sha256_file(config_file),
            "train": _sha256_file(output),
            "canary": _sha256_file(canary_path),
            "sources": dict(sorted(source_checksums.items())),
            "benchmark_excludes": dict(sorted(exclusions.checksums.items())),
        },
        "counts": {
            "input_rows": len(all_candidates),
            "input_by_source": dict(sorted(input_counts.items())),
            "eligible_rows": len(eligible),
            "train": _counts(train),
            "canary": _counts(canary),
        },
        "duplicates": {
            "candidate_prompt_identity_groups": duplicate_groups,
            "candidate_prompt_rows": duplicate_rows,
            "train_prompt_rows": len(train) - len({candidate.identity for candidate in train}),
            "canary_prompt_rows": len(canary) - len({candidate.identity for candidate in canary}),
        },
        "leakage": {
            "benchmark_reference_rows": exclusions.rows,
            "candidate_identity_rows_excluded": identity_leak_rows,
            "candidate_media_rows_excluded": media_leak_rows,
            "candidate_rows_excluded": excluded_rows,
            "output": output_leakage,
        },
        "selection": {
            "source_shortfalls": dict(sorted(source_shortfalls.items())),
            "canary_shortfall": canary_shortfall,
            "media_cap_rejections_by_source": dict(sorted(cap_rejections.items())),
            "canary_rejections": dict(sorted(canary_rejections.items())),
        },
        "sources": source_details,
        "benchmark_excludes": [str(path) for path in exclusions.paths],
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML or JSON source manifest")
    parser.add_argument("--output", required=True, help="Training JSONL output")
    parser.add_argument("--canary-output", default=None)
    parser.add_argument("--manifest-output", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--target", type=int, default=None)
    parser.add_argument("--canary-size", type=int, default=None)
    parser.add_argument(
        "--max-prompts-per-media",
        "--max-prompts-per-anchor",
        dest="max_prompts_per_media",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--require-media",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require every normalized media path to exist",
    )
    parser.add_argument(
        "--require-existing-media",
        dest="require_media",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--benchmark-exclude",
        action="append",
        default=[],
        help="JSON or JSONL benchmark identities/media to exclude; repeatable",
    )
    parser.add_argument(
        "--allow-shortfall",
        action="store_true",
        default=None,
        help="Write available rows instead of failing a source or canary shortfall",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = build_dataset(
            args.config,
            args.output,
            canary_output=args.canary_output,
            manifest_output=args.manifest_output,
            seed=args.seed,
            target=args.target,
            canary_size=args.canary_size,
            max_prompts_per_media=args.max_prompts_per_media,
            require_media=args.require_media,
            benchmark_excludes=args.benchmark_exclude,
            allow_shortfall=args.allow_shortfall,
            overwrite=args.overwrite,
        )
    except (ConfigError, FileExistsError, OSError, SchemaError, ShortfallError) as error:
        print(f"[FATAL] {error}", file=sys.stderr)
        return 2
    print(
        "OraRL data prepared: "
        f"train={manifest['actual_train_rows']} "
        f"canary={manifest['actual_canary_rows']} "
        f"manifest={manifest['files']['manifest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
