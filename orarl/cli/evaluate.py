"""Explicit-path evaluation wrapper for the paper benchmark suite."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, distribution
from importlib.util import find_spec
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml

from orarl.evaluation import (
    EvaluationSchemaError,
    ManifestError,
    StagingError,
    evaluation_asset_paths,
    load_asset_manifest,
    load_dataset_manifest,
    load_evaluation_jsonl,
    validate_asset_manifest,
    validate_dataset_manifest,
)

VIDEO_QA_TASKS = (
    "videomme",
    "videommev2",
    "mvbench",
    "mmvu",
    "videoholmes",
    "longvideobench",
    "mlvu",
)
PAPER_TASKS = (
    *VIDEO_QA_TASKS,
    "vsi",
    "mmsi",
    "mindcube",
    "revsi",
    "spatial_grounding",
    "tracking",
    "stvg",
    "temporal_grounding",
    "segmentation",
)
_TASK_PREFIX = {
    **{task: task.upper() for task in VIDEO_QA_TASKS},
    "vsi": "VSI",
    "mmsi": "MMSI",
    "mindcube": "MINDCUBE",
    "revsi": "REVSI",
    "spatial_grounding": "SPATIAL_GROUNDING",
    "tracking": "TRACKING",
    "stvg": "STVG",
    "temporal_grounding": "TIMELENS",
    "segmentation": "SEGMENTATION",
}
_REQUIRED_INPUTS = {
    "videomme": {"VIDEOMME_DATA_FILE", "VIDEOMME_VIDEO_BASE"},
    "videommev2": {"VIDEOMMEV2_DATA_FILE", "VIDEOMMEV2_VIDEO_BASE"},
    "mvbench": {"MVBENCH_DATA_FILE", "MVBENCH_VIDEO_ROOT"},
    "mmvu": {"MMVU_DATA_FILE", "MMVU_VIDEO_ROOT"},
    "videoholmes": {"VIDEOHOLMES_DATA_FILE", "VIDEOHOLMES_VIDEO_ROOT"},
    "longvideobench": {"LONGVIDEOBENCH_DATA_FILE", "LONGVIDEOBENCH_VIDEO_ROOT"},
    "mlvu": {"MLVU_DATA_FILE", "MLVU_VIDEO_ROOT"},
    "vsi": {"VSI_DATA_FILE", "VSI_PREPROCESSED_VIDEO_DIR"},
    "mmsi": {"MMSI_DATA_FILE"},
    "mindcube": {"MINDCUBE_DATA_FILE"},
    "revsi": {"REVSI_DATA_FILE", "REVSI_VIDEO_ROOT"},
    "spatial_grounding": {
        "SPATIAL_GROUNDING_BENCH_DIR",
        "SPATIAL_GROUNDING_IMAGE_ROOT",
    },
    "tracking": {"TRACKING_BENCH_DIR", "TRACKING_BASE_PREFIX"},
    "stvg": {"STVG_BENCH_DIR", "STVG_BASE_PREFIX"},
    "temporal_grounding": {"TIMELENS_BENCH_DIR"},
    "segmentation": {"SEGMENTATION_BENCH_DIR", "SEGMENTATION_DATA_ROOT"},
}
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
_STRUCTURED_TASKS = frozenset(
    {
        "spatial_grounding",
        "tracking",
        "stvg",
        "temporal_grounding",
        "segmentation",
    }
)
_SPATIAL_GROUNDING_SPLIT_NAMES = {
    "refcoco_val": "refcoco-val",
    "refcoco_test_a": "refcoco-testA",
    "refcoco_test_b": "refcoco-testB",
    "refcocop_val": "refcoco+-val",
    "refcocop_test_a": "refcoco+-testA",
    "refcocop_test_b": "refcoco+-testB",
    "refcocog_val": "refcocog-val",
    "refcocog_test": "refcocog-test",
}
_TRACKING_SPLIT_NAMES = {
    "got10k": "eval_got10k",
}
_STVG_SPLIT_NAMES = {
    "stvg": "eval_stvg",
}


def _evaluator_split_name(task: str, split: object) -> str:
    value = str(split)
    if task == "spatial_grounding":
        return _SPATIAL_GROUNDING_SPLIT_NAMES.get(value, value)
    if task == "tracking":
        return _TRACKING_SPLIT_NAMES.get(value, value)
    if task == "stvg":
        return _STVG_SPLIT_NAMES.get(value, value)
    return value


_MANIFEST_FILES = ("datasets.jsonl", "assets.jsonl")
_INDEX_MANIFEST_FILES = ("datasets.jsonl",)
_SENSITIVE_NAME = re.compile(
    r"(?:^|_)(?:api_?key|credential|password|private_?key|secret|token)(?:$|_)",
    flags=re.IGNORECASE,
)
_URI_USERINFO = re.compile(r"://[^/\s:@]+:[^@\s/]+@")


class CliError(ValueError):
    """Raised for invalid evaluation requests."""


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orarl-eval",
        description=(
            "Run the repository evaluator with explicit benchmark inputs and "
            "write one aggregate JSON summary. The default is a dry run."
        ),
    )
    parser.add_argument("--model", required=True, help="Local model or checkpoint path.")
    parser.add_argument(
        "--tasks",
        required=True,
        help=(
            "Comma-separated paper tasks. Use video_qa for the documented Video QA "
            "group or paper for the full suite."
        ),
    )
    parser.add_argument(
        "--dataset",
        help=(
            "Canonical local staging root or Hugging Face dataset repository id. "
            "This is the preferred evaluation-data input."
        ),
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face dataset revision (default: main).",
    )
    parser.add_argument(
        "--splits",
        help="Optional comma-separated canonical split ids to evaluate.",
    )
    parser.add_argument(
        "--asset-root",
        help=(
            "Optional local root containing canonical media/ and artifacts/ "
            "when JSONL metadata is stored separately."
        ),
    )
    parser.add_argument("--cache-dir", help="Optional Hugging Face snapshot cache directory.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Resolve a Hugging Face dataset only from the local cache.",
    )
    parser.add_argument(
        "--data-root",
        help=(
            "Legacy root containing one directory per task. Cannot be combined "
            "with --dataset."
        ),
    )
    parser.add_argument(
        "--task-config",
        action="append",
        default=[],
        metavar="TASK=FILE",
        help="YAML/JSON environment mapping for one task; repeat as needed.",
    )
    parser.add_argument(
        "--evaluator",
        help="Path to eval/task/eval.sh (normally discovered from the source checkout).",
    )
    parser.add_argument("--gpus", default="0", help="Comma-separated local GPU ids.")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Optional per-task batch-size override for this launch.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help=(
            "Optional sample limit for supported task smoke tests."
        ),
    )
    parser.add_argument("--force-merge", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--merged-model")
    parser.add_argument("--base-model")
    parser.add_argument("--segmentation-run-sam2", action="store_true")
    parser.add_argument(
        "--summary",
        default="orarl-eval-summary.json",
        help="Aggregate JSON output path.",
    )
    parser.add_argument(
        "--results-root",
        help="Directory to scan for evaluator summary.json files.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="dry_run", action="store_true")
    mode.add_argument("--run", dest="dry_run", action="store_false")
    parser.set_defaults(dry_run=True)
    return parser


def _existing(value: str, label: str, *, directory: bool | None = None) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise CliError(f"{label} does not exist: {path}")
    if directory is True and not path.is_dir():
        raise CliError(f"{label} must be a directory: {path}")
    if directory is False and not path.is_file():
        raise CliError(f"{label} must be a file: {path}")
    return path


def _parse_tasks(value: str) -> list[str]:
    requested = [item.strip().casefold() for item in value.split(",") if item.strip()]
    if not requested:
        raise CliError("--tasks cannot be empty")
    expanded: list[str] = []
    for task in requested:
        values = (
            PAPER_TASKS if task == "paper" else VIDEO_QA_TASKS if task == "video_qa" else (task,)
        )
        for value_item in values:
            if value_item not in PAPER_TASKS:
                raise CliError(
                    f"unsupported task {value_item!r}; choose from {', '.join(PAPER_TASKS)}"
                )
            if value_item not in expanded:
                expanded.append(value_item)
    return expanded


def _installed_evaluator() -> Path | None:
    try:
        package_distribution = distribution("orarl")
    except PackageNotFoundError:
        return None

    suffix = ("share", "orarl", "eval", "task", "eval.sh")
    for entry in package_distribution.files or ():
        parts = PurePosixPath(str(entry)).parts
        if len(parts) < len(suffix) or tuple(parts[-len(suffix) :]) != suffix:
            continue
        candidate = Path(package_distribution.locate_file(entry)).resolve()
        if candidate.is_file():
            return candidate
    return None


def _discover_evaluator(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidate = Path(explicit).expanduser()
        if not candidate.is_file():
            raise CliError(f"evaluator does not exist or is not a file: {candidate}")
        return candidate.resolve()
    configured = os.environ.get("ORARL_EVALUATOR")
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_file():
            raise CliError(f"ORARL_EVALUATOR does not name an evaluator file: {candidate}")
        return candidate.resolve()
    runtime_root = os.environ.get("ORARL_RUNTIME_ROOT")
    if runtime_root:
        candidates.append(Path(runtime_root).expanduser() / "eval" / "task" / "eval.sh")

    try:
        runtime_spec = find_spec("verl")
    except (ImportError, ModuleNotFoundError, ValueError):
        runtime_spec = None
    if runtime_spec is not None:
        if runtime_spec.origin:
            candidates.append(
                Path(runtime_spec.origin).resolve().parent.parent / "eval" / "task" / "eval.sh"
            )
        for location in runtime_spec.submodule_search_locations or ():
            candidates.append(Path(location).resolve().parent / "eval" / "task" / "eval.sh")
    candidates.extend(
        (
            Path.cwd() / "eval" / "task" / "eval.sh",
            Path(__file__).resolve().parents[2] / "eval" / "task" / "eval.sh",
            Path(__file__).resolve().parents[3] / "eval" / "task" / "eval.sh",
        )
    )
    installed_evaluator = _installed_evaluator()
    if installed_evaluator is not None:
        candidates.append(installed_evaluator)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise CliError(
        "eval/task/eval.sh was not found; pass --evaluator, set "
        "ORARL_EVALUATOR, or set ORARL_RUNTIME_ROOT to a checkout that ships it"
    )


def _root_environment(root: Path, tasks: Sequence[str]) -> dict[str, str]:
    environment: dict[str, str] = {}
    for task in tasks:
        task_root = root / task
        if task == "videomme":
            environment.update(
                {
                    "VIDEOMME_VIDEO_BASE": str(task_root),
                    "VIDEOMME_VIDEO_DIR": str(task_root / "videos"),
                    "VIDEOMME_DATA_FILE": str(task_root / "annotations.jsonl"),
                    "VIDEOMME_PREPROCESSED_VIDEO_DIR": str(task_root / "preprocessed_videos"),
                }
            )
        elif task == "videommev2":
            environment.update(
                {
                    "VIDEOMMEV2_VIDEO_BASE": str(task_root),
                    "VIDEOMMEV2_VIDEO_DIR": str(task_root / "videos"),
                    "VIDEOMMEV2_DATA_FILE": str(task_root / "annotations.jsonl"),
                    "VIDEOMMEV2_PREPROCESSED_VIDEO_DIR": str(task_root / "preprocessed_videos"),
                }
            )
        elif task in {
            "mvbench",
            "mmvu",
            "videoholmes",
            "longvideobench",
            "mlvu",
        }:
            prefix = _TASK_PREFIX[task]
            environment[f"{prefix}_DATA_FILE"] = str(task_root / "annotations.jsonl")
            environment[f"{prefix}_VIDEO_ROOT"] = str(task_root)
        elif task == "vsi":
            environment.update(
                {
                    "VSI_DATA_FILE": str(task_root / "annotations.jsonl"),
                    "VSI_PREPROCESSED_VIDEO_DIR": str(task_root / "preprocessed_videos"),
                }
            )
        elif task == "mmsi":
            environment["MMSI_DATA_FILE"] = str(task_root / "MMSI_bench.tsv")
        elif task == "mindcube":
            environment.update(
                {
                    "MINDCUBE_DATA_FILE": str(task_root / "combined-00000-of-00001.parquet"),
                    "MINDCUBE_EXPECTED_SAMPLES": "1050",
                }
            )
        elif task == "revsi":
            environment.update(
                {
                    "REVSI_DATA_FILE": str(
                        task_root / "all_frame" / "test-00000-of-00001.parquet"
                    ),
                    "REVSI_VIDEO_ROOT": str(task_root / "all_frame"),
                }
            )
        elif task == "spatial_grounding":
            environment.update(
                {
                    "SPATIAL_GROUNDING_BENCH_DIR": str(task_root),
                    "SPATIAL_GROUNDING_IMAGE_ROOT": str(task_root / "images"),
                }
            )
        elif task == "tracking":
            environment.update(
                {
                    "TRACKING_BENCH_DIR": str(task_root),
                    "TRACKING_BASE_PREFIX": str(task_root),
                }
            )
        elif task == "stvg":
            environment.update(
                {
                    "STVG_BENCH_DIR": str(task_root),
                    "STVG_BASE_PREFIX": str(task_root),
                }
            )
        elif task == "temporal_grounding":
            environment["TIMELENS_BENCH_DIR"] = str(task_root)
        elif task == "segmentation":
            environment.update(
                {
                    "SEGMENTATION_BENCH_DIR": str(task_root),
                    "SEGMENTATION_DATA_ROOT": str(task_root),
                }
            )
    return environment


def _stringify_env(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _download_dataset_snapshot(
    repo_id: str,
    namespace: argparse.Namespace,
    allow_patterns: Sequence[str],
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise CliError(
            "resolving a Hugging Face --dataset requires huggingface_hub"
        ) from error

    cache_dir = (
        str(Path(namespace.cache_dir).expanduser())
        if namespace.cache_dir
        else None
    )
    try:
        snapshot = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=namespace.revision,
            cache_dir=cache_dir,
            local_files_only=namespace.local_files_only,
            allow_patterns=list(allow_patterns),
        )
    except Exception as error:
        raise CliError(f"cannot resolve Hugging Face dataset {repo_id!r}: {error}") from error
    return _existing(snapshot, "downloaded dataset root", directory=True)


def _load_canonical_manifests(
    root: Path,
    asset_root: Path | None = None,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    assets_root = asset_root or root
    required = {
        "datasets.jsonl": root / "datasets.jsonl",
        "assets.jsonl": assets_root / "assets.jsonl",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise CliError(
            f"canonical dataset root {root} is missing: {', '.join(missing)}"
        )
    try:
        datasets = load_dataset_manifest(root)
        assets = load_asset_manifest(assets_root)
    except (ManifestError, StagingError, OSError) as error:
        raise CliError(f"invalid canonical dataset root {root}: {error}") from error
    if not datasets:
        raise CliError(f"canonical dataset manifest is empty: {root / 'datasets.jsonl'}")
    return list(datasets), list(assets)


def _select_dataset_records(
    records: Sequence[Mapping[str, Any]],
    tasks: Sequence[str],
    splits: set[str] | None = None,
) -> list[Mapping[str, Any]]:
    if splits is not None:
        available_splits = {
            str(record["split"])
            for record in records
            if str(record["task"]) in tasks
        }
        missing_splits = sorted(splits - available_splits)
        if missing_splits:
            raise CliError(
                "canonical dataset has no requested split(s): "
                + ", ".join(missing_splits)
            )
    selected = [
        record
        for record in records
        if str(record["task"]) in tasks
        and (splits is None or str(record["split"]) in splits)
    ]
    covered = {str(record["task"]) for record in selected}
    missing = sorted(set(tasks) - covered)
    if missing:
        raise CliError(
            "canonical dataset has no configuration record for: "
            + ", ".join(missing)
        )
    return selected


def _selected_asset_records(
    assets: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    annotation_paths = {str(record["annotation_path"]) for record in records}
    declared_roots = {
        str(path).rstrip("/")
        for record in records
        for field in ("media_paths", "artifact_paths")
        for path in record.get(field, ())
    }

    def selected(path: str) -> bool:
        return path in annotation_paths or any(
            path == root or path.startswith(f"{root}/")
            for root in declared_roots
        )

    return [
        asset
        for asset in assets
        if selected(str(asset["path"]))
    ]


def _verify_runtime_assets(
    root: Path,
    records: Sequence[Mapping[str, Any]],
    assets: Sequence[Mapping[str, Any]],
    asset_root: Path | None = None,
) -> None:
    selected_assets = _selected_asset_records(assets, records)
    manifested = {str(asset["path"]) for asset in selected_assets}
    annotations = {str(record["annotation_path"]) for record in records}
    missing_annotations = sorted(annotations - manifested)
    if missing_annotations:
        raise CliError(
            "canonical annotations are missing from assets.jsonl: "
            + ", ".join(missing_annotations)
        )

    try:
        if asset_root is None:
            validate_asset_manifest(
                selected_assets,
                repository_root=root,
                context=str(root / "assets.jsonl"),
            )
            validate_dataset_manifest(
                records,
                repository_root=root,
                context=str(root / "datasets.jsonl"),
            )
        else:
            annotation_assets = [
                asset
                for asset in selected_assets
                if str(asset["path"]).startswith("annotations/")
            ]
            media_assets = [
                asset
                for asset in selected_assets
                if not str(asset["path"]).startswith("annotations/")
            ]
            validate_asset_manifest(
                annotation_assets,
                repository_root=root,
                context=str(root / "assets.jsonl"),
            )
            validate_asset_manifest(
                media_assets,
                repository_root=asset_root,
                context=str(root / "assets.jsonl"),
            )
            validate_dataset_manifest(
                records,
                context=str(root / "datasets.jsonl"),
            )
        for record in records:
            annotation = root / str(record["annotation_path"])
            rows = load_evaluation_jsonl(
                annotation,
                benchmark=str(record["benchmark"]),
                split=str(record["split"]),
                eval_task=str(record["task"]),
                    repository_root=root if asset_root is None else None,
            )
            expected = int(record["expected_count"])
            if len(rows) != expected:
                raise CliError(
                    f"{record['benchmark']}/{record['split']}: expected "
                    f"{expected} rows, found {len(rows)}"
                )
            unmanifested = sorted(
                {
                    path
                    for row in rows
                    for path in evaluation_asset_paths(row)
                    if path not in manifested
                }
            )
            if unmanifested:
                raise CliError(
                    f"{record['benchmark']}/{record['split']} references "
                    "unmanifested assets: "
                    + ", ".join(unmanifested[:5])
                )
    except (EvaluationSchemaError, ManifestError, StagingError, OSError) as error:
        raise CliError(f"canonical dataset validation failed: {error}") from error


def _resolve_canonical_dataset(
    namespace: argparse.Namespace,
    tasks: Sequence[str],
    asset_root: Path | None = None,
) -> tuple[Path, list[Mapping[str, Any]]]:
    raw_dataset = str(namespace.dataset).strip()
    if not raw_dataset:
        raise CliError("--dataset cannot be empty")

    local = Path(raw_dataset).expanduser()
    remote = not local.exists()
    if remote and (
        local.is_absolute()
        or raw_dataset.startswith((".", "~"))
        or raw_dataset.endswith(("/", os.sep))
    ):
        raise CliError(f"local canonical dataset root does not exist: {local}")

    root = (
        _download_dataset_snapshot(
            raw_dataset,
            namespace,
            _INDEX_MANIFEST_FILES if asset_root is not None else _MANIFEST_FILES,
        )
        if remote
        else _existing(raw_dataset, "canonical dataset root", directory=True)
    )
    requested_splits = (
        {
            value.strip().casefold().replace("-", "_")
            for value in str(namespace.splits).split(",")
            if value.strip()
        }
        if namespace.splits
        else None
    )
    if requested_splits == set():
        raise CliError("--splits cannot be empty")
    all_records, all_assets = _load_canonical_manifests(root, asset_root)
    selected = _select_dataset_records(all_records, tasks, requested_splits)

    if remote and not namespace.dry_run:
        selected_assets = _selected_asset_records(all_assets, selected)
        if asset_root is not None:
            selected_assets = [
                asset
                for asset in selected_assets
                if str(asset["path"]).startswith("annotations/")
            ]
        allow_patterns = [
            *(_INDEX_MANIFEST_FILES if asset_root is not None else _MANIFEST_FILES),
            *(str(asset["path"]) for asset in selected_assets),
        ]
        root = _download_dataset_snapshot(
            raw_dataset,
            namespace,
            tuple(dict.fromkeys(allow_patterns)),
        )
        all_records, all_assets = _load_canonical_manifests(root, asset_root)
        selected = _select_dataset_records(all_records, tasks, requested_splits)

    if not namespace.dry_run:
        _verify_runtime_assets(root, selected, all_assets, asset_root)
    return root, selected


def _resolve_profile_path(
    root: Path,
    value: str,
    name: str,
    asset_root: Path | None = None,
) -> str:
    relative = Path(value)
    selected_root = (
        asset_root
        if asset_root is not None
        and relative.parts
        and relative.parts[0].casefold() in {"artifacts", "media"}
        else root
    )
    path = (selected_root / relative).resolve()
    try:
        path.relative_to(selected_root)
    except ValueError as error:
        raise CliError(
            f"canonical profile path escapes dataset root: {name}={value}"
        ) from error
    return str(path)


def _merge_profile_value(
    environment: dict[str, str],
    *,
    task: str,
    name: str,
    value: str,
) -> None:
    previous = environment.get(name)
    if previous is not None and previous != value:
        raise CliError(
            f"conflicting canonical profiles for {task}: "
            f"{name} is both {previous!r} and {value!r}"
        )
    environment[name] = value


def _validate_task_profiles(
    task: str,
    records: Sequence[Mapping[str, Any]],
) -> None:
    splits: set[str] = set()
    baseline: dict[str, Any] | None = None
    for record in records:
        split = str(record["split"])
        if split in splits:
            raise CliError(
                f"duplicate canonical profile split for {task}: {split}"
            )
        splits.add(split)

        profile = {
            "family": record.get("family"),
            "evaluation": record.get("evaluation"),
        }
        if task != "temporal_grounding":
            profile["preprocessing"] = record.get("preprocessing")
        if baseline is None:
            baseline = profile
        elif profile != baseline:
            raise CliError(
                f"conflicting canonical evaluation profiles for task {task}"
            )

    if task not in _STRUCTURED_TASKS and len(records) > 1:
        raise CliError(
            f"duplicate canonical profiles for non-structured task {task}"
        )


def _canonical_task_defaults(
    root: Path,
    task: str,
    records: Sequence[Mapping[str, Any]],
    asset_root: Path | None = None,
) -> dict[str, str]:
    annotations = [
        (root / str(record["annotation_path"])).resolve() for record in records
    ]
    prefix = _TASK_PREFIX[task]
    defaults: dict[str, str] = {}
    runtime_asset_root = asset_root or root

    if task in _STRUCTURED_TASKS:
        parents = {path.parent for path in annotations}
        if len(parents) != 1:
            raise CliError(
                f"{task} canonical annotations must share one benchmark directory; "
                "use benchmark equal to eval_task and splits for sub-benchmarks"
            )
        bench_dir = str(next(iter(parents)))
        defaults[f"{prefix}_BENCH_DIR"] = bench_dir
        defaults[f"{prefix}_DATASETS"] = ",".join(
            _evaluator_split_name(task, record["split"])
            for record in records
        )
        if task == "spatial_grounding":
            defaults["SPATIAL_GROUNDING_IMAGE_ROOT"] = str(runtime_asset_root)
        elif task == "tracking":
            defaults["TRACKING_BASE_PREFIX"] = str(runtime_asset_root)
        elif task == "stvg":
            defaults["STVG_BASE_PREFIX"] = str(runtime_asset_root)
        elif task == "segmentation":
            defaults["SEGMENTATION_DATA_ROOT"] = str(runtime_asset_root)
        return defaults

    annotation = str(annotations[0])
    defaults[f"{prefix}_DATA_FILE"] = annotation
    if task in {"videomme", "videommev2"}:
        defaults[f"{prefix}_VIDEO_BASE"] = str(runtime_asset_root)
        defaults[f"{prefix}_VIDEO_DIR"] = str(runtime_asset_root)
        defaults[f"{prefix}_PREPROCESSED_VIDEO_DIR"] = str(runtime_asset_root)
    elif task in {
        "mvbench",
        "mmvu",
        "videoholmes",
        "longvideobench",
        "mlvu",
    }:
        defaults[f"{prefix}_VIDEO_ROOT"] = str(runtime_asset_root)
        if task == "longvideobench":
            defaults["LONGVIDEOBENCH_SUBTITLE_ROOT"] = str(runtime_asset_root)
    elif task == "vsi":
        defaults["VSI_PREPROCESSED_VIDEO_DIR"] = str(runtime_asset_root)
        defaults["VSI_EXPECTED_SAMPLES"] = str(records[0]["expected_count"])
    elif task == "mmsi":
        defaults["MMSI_EXPECTED_SAMPLES"] = str(records[0]["expected_count"])
    elif task == "mindcube":
        defaults["MINDCUBE_EXPECTED_SAMPLES"] = str(records[0]["expected_count"])
    elif task == "revsi":
        defaults["REVSI_VIDEO_ROOT"] = str(runtime_asset_root)
        defaults["REVSI_EXPECTED_SAMPLES"] = str(records[0]["expected_count"])
    return defaults


def _canonical_environment(
    root: Path,
    records: Sequence[Mapping[str, Any]],
    tasks: Sequence[str],
    asset_root: Path | None = None,
) -> dict[str, str]:
    environment = {
        "ORARL_EVAL_DATA_ROOT": str(root),
        "ORARL_EVAL_DATASETS_JSONL": str(root / "datasets.jsonl"),
    }
    if asset_root is not None:
        environment["ORARL_EVAL_ASSET_ROOT"] = str(asset_root)
    for task in tasks:
        task_records = [record for record in records if record["task"] == task]
        _validate_task_profiles(task, task_records)
        prefix = _TASK_PREFIX[task] + "_"
        for record in task_records:
            values = record.get("legacy_environment", {})
            if not isinstance(values, Mapping):
                continue
            for raw_name, raw_value in values.items():
                name = str(raw_name).strip().upper()
                if not name.startswith(prefix) and name != "EVAL_CONDA_ENV":
                    raise CliError(
                        f"unrelated canonical profile setting for {task}: {name}"
                    )
                if _SENSITIVE_NAME.search(name):
                    raise CliError("canonical profiles cannot contain sensitive settings")
                value = _stringify_env(raw_value)
                if _URI_USERINFO.search(value):
                    raise CliError("canonical profiles cannot contain credential-bearing URLs")
                if name.endswith(_PATH_ENV_SUFFIXES) and value:
                    value = _resolve_profile_path(root, value, name, asset_root)
                _merge_profile_value(
                    environment,
                    task=task,
                    name=name,
                    value=value,
                )

        for name, value in _canonical_task_defaults(
            root,
            task,
            task_records,
            asset_root,
        ).items():
            if name.endswith("_DATASETS"):
                selected = value.split(",")
                previous = environment.get(name, "").split(",")
                ordered = [
                    item
                    for prior in previous
                    for item in selected
                    if prior.strip().casefold().replace("-", "_")
                    == item.casefold().replace("-", "_")
                ]
                environment[name] = ",".join(
                    dict.fromkeys([*ordered, *selected])
                )
            else:
                environment.setdefault(name, value)
        if task == "temporal_grounding":
            for field, name in {
                "fps": "TIMELENS_FPS",
                "min_tokens": "TIMELENS_MIN_TOKENS",
                "max_frames": "TIMELENS_MAX_FRAMES",
                "max_pixels": "TIMELENS_MAX_PIXELS",
                "total_tokens": "TIMELENS_TOTAL_TOKENS",
            }.items():
                values = {
                    str(preprocessing[field])
                    for record in task_records
                    if isinstance(
                        preprocessing := record.get("preprocessing"),
                        Mapping,
                    )
                    and field in preprocessing
                }
                if len(values) == 1:
                    environment[name] = values.pop()
    return environment


def _task_configs(
    specifications: Sequence[str],
    tasks: Sequence[str],
    data_root: Path | None,
) -> tuple[dict[str, str], set[str]]:
    environment: dict[str, str] = {}
    configured_tasks: set[str] = set()
    for specification in specifications:
        task, separator, raw_path = specification.partition("=")
        task = task.strip().casefold()
        if not separator or task not in PAPER_TASKS:
            raise CliError(f"--task-config expects TASK=FILE for a paper task: {specification!r}")
        if task not in tasks:
            raise CliError(f"task config supplied for unrequested task: {task}")
        config_path = _existing(raw_path, f"{task} config", directory=False)
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise CliError(f"cannot load {config_path}: {error}") from error
        if not isinstance(payload, Mapping):
            raise CliError(f"task config must be a mapping: {config_path}")
        declared_task = payload.get("task")
        if declared_task is not None and str(declared_task).casefold() != task:
            raise CliError(f"{config_path} declares task {declared_task!r}, expected {task!r}")
        values = payload.get("environment", payload.get("env", payload))
        if not isinstance(values, Mapping):
            raise CliError(f"{config_path}: environment must be a mapping")
        prefix = _TASK_PREFIX[task] + "_"
        for raw_name, raw_value in values.items():
            name = str(raw_name).strip().upper()
            if name == "TASK":
                continue
            if not name.startswith(prefix) and name != "EVAL_CONDA_ENV":
                raise CliError(f"{config_path}: {name} is not allowlisted for {task}")
            if _SENSITIVE_NAME.search(name):
                raise CliError(f"{config_path}: sensitive environment names are not accepted")
            value = _stringify_env(raw_value)
            if _URI_USERINFO.search(value):
                raise CliError(f"{config_path}: credential-bearing URLs are not accepted")
            if name.endswith(_PATH_ENV_SUFFIXES) and value and not Path(value).is_absolute():
                base = data_root if data_root is not None else config_path.parent
                value = str((base / value).resolve())
            environment[name] = value
        configured_tasks.add(task)
    return environment, configured_tasks


def build_launch(
    namespace: argparse.Namespace,
) -> tuple[list[str], dict[str, str], list[str], Path, Path]:
    model = _existing(namespace.model, "model", directory=True)
    tasks = _parse_tasks(namespace.tasks)
    evaluator = _discover_evaluator(namespace.evaluator)
    if not re.fullmatch(r"\d+(?:,\d+)*", namespace.gpus):
        raise CliError("--gpus must be a comma-separated list of non-negative ids")
    if namespace.tp_size <= 0:
        raise CliError("--tp-size must be positive")
    if namespace.batch_size is not None and namespace.batch_size <= 0:
        raise CliError("--batch-size must be positive")
    if namespace.max_samples is not None and namespace.max_samples <= 0:
        raise CliError("--max-samples must be positive")
    if namespace.force_merge and namespace.skip_merge:
        raise CliError("--force-merge and --skip-merge are mutually exclusive")
    if namespace.dataset and namespace.data_root:
        raise CliError("--dataset and legacy --data-root are mutually exclusive")
    if namespace.asset_root and not namespace.dataset:
        raise CliError("--asset-root requires --dataset")
    if namespace.splits and not namespace.dataset:
        raise CliError("--splits requires --dataset")

    canonical_root: Path | None = None
    canonical_records: list[Mapping[str, Any]] = []
    asset_root = (
        _existing(namespace.asset_root, "asset root", directory=True)
        if namespace.asset_root
        else None
    )
    if namespace.dataset:
        canonical_root, canonical_records = _resolve_canonical_dataset(
            namespace,
            tasks,
            asset_root,
        )
        environment = _canonical_environment(
            canonical_root,
            canonical_records,
            tasks,
            asset_root,
        )
    else:
        environment = {}

    data_root = None
    if namespace.data_root:
        data_root = _existing(namespace.data_root, "data root", directory=True)
        environment.update(_root_environment(data_root, tasks))
    config_root = canonical_root if canonical_root is not None else data_root
    configured_environment, configured_tasks = _task_configs(
        namespace.task_config, tasks, config_root
    )
    environment.update(configured_environment)
    if canonical_root is not None:
        configured_tasks.update(tasks)
    if namespace.batch_size is not None:
        for task in tasks:
            environment[f"{_TASK_PREFIX[task]}_BATCH_SIZE"] = str(namespace.batch_size)
    if namespace.max_samples is not None:
        unsupported = set(tasks) - {
            *VIDEO_QA_TASKS,
            "temporal_grounding",
            "spatial_grounding",
            "tracking",
            "stvg",
            "revsi",
            "segmentation",
        }
        if unsupported:
            raise CliError(
                "--max-samples only supports video QA, temporal_grounding, "
                "spatial_grounding, tracking, stvg, revsi, or segmentation"
            )
        if "temporal_grounding" in tasks:
            environment["TIMELENS_MAX_SAMPLES"] = str(namespace.max_samples)
        if "spatial_grounding" in tasks:
            environment["SPATIAL_GROUNDING_MAX_SAMPLES"] = str(
                namespace.max_samples
            )
        if "tracking" in tasks:
            environment["TRACKING_MAX_SAMPLES"] = str(namespace.max_samples)
        if "stvg" in tasks:
            environment["STVG_MAX_SAMPLES"] = str(namespace.max_samples)
        if "revsi" in tasks:
            environment["REVSI_MAX_SAMPLES"] = str(namespace.max_samples)
        if "segmentation" in tasks:
            environment["SEGMENTATION_MAX_SAMPLES"] = str(namespace.max_samples)
        for task in tasks:
            if task in VIDEO_QA_TASKS:
                environment[f"{_TASK_PREFIX[task]}_MAX_SAMPLES"] = str(
                    namespace.max_samples
                )
    eval_conda_env = environment.pop("EVAL_CONDA_ENV", "").strip()
    if not namespace.dry_run:
        missing_configs = (
            sorted(set(tasks) - configured_tasks)
            if data_root is None and canonical_root is None
            else []
        )
        if missing_configs:
            raise CliError(
                "--run requires --dataset, --data-root, or one --task-config per task; "
                f"missing: {', '.join(missing_configs)}"
            )
        for task in tasks:
            missing_inputs = sorted(_REQUIRED_INPUTS[task] - environment.keys())
            if missing_inputs:
                raise CliError(
                    f"{task} is missing explicit input settings: " + ", ".join(missing_inputs)
                )
            for name in sorted(_REQUIRED_INPUTS[task]):
                path = Path(environment[name])
                expected_file = name.endswith("_FILE")
                if not path.exists():
                    raise CliError(f"{task} input does not exist: {name}={path}")
                if expected_file and not path.is_file():
                    raise CliError(f"{task} input must be a file: {name}={path}")
                if not expected_file and not path.is_dir():
                    raise CliError(f"{task} input must be a directory: {name}={path}")
        if namespace.segmentation_run_sam2:
            postprocessor_inputs = {
                "SEGMENTATION_POSTPROCESSOR_PATH",
                "SEGMENTATION_SAM2_CFG",
                "SEGMENTATION_SAM2_CKPT",
            }
            missing_postprocessor = sorted(postprocessor_inputs - environment.keys())
            if missing_postprocessor:
                raise CliError(
                    "segmentation post-processing requires explicit settings: "
                    + ", ".join(missing_postprocessor)
                )
            for name in sorted(postprocessor_inputs):
                path = Path(environment[name])
                if not path.is_file():
                    raise CliError(
                        f"segmentation post-processing input must be a file: {name}={path}"
                    )

    command = [
        "bash",
        str(evaluator),
        "--model",
        str(model),
        "--tasks",
        ",".join(tasks),
        "--gpus",
        namespace.gpus,
        "--tp-size",
        str(namespace.tp_size),
    ]
    if namespace.force_merge:
        command.append("--force-merge")
    if namespace.skip_merge:
        command.append("--skip-merge")
    if namespace.merged_model:
        command.extend(("--merged-model", str(Path(namespace.merged_model).expanduser().resolve())))
    if namespace.base_model:
        command.extend(("--base-model", str(_existing(namespace.base_model, "base model"))))
    if namespace.segmentation_run_sam2:
        command.append("--segmentation-run-sam2")
    if eval_conda_env:
        command.extend(("--env", eval_conda_env))

    project_root = evaluator.parents[2]
    results_root = (
        Path(namespace.results_root).expanduser().resolve()
        if namespace.results_root
        else project_root / "outputs"
    )
    summary_path = Path(namespace.summary).expanduser().resolve()
    return command, environment, tasks, results_root, summary_path


def _execution_environment(configured: Mapping[str, str]) -> dict[str, str]:
    """Build an evaluator environment without inherited task-path state."""

    environment = os.environ.copy()
    task_prefixes = tuple(prefix + "_" for prefix in set(_TASK_PREFIX.values()))
    for name in tuple(environment):
        if (
            name.startswith(task_prefixes)
            or name == "EVAL_CONDA_ENV"
            or name.startswith("ORARL_EVAL_")
        ):
            environment.pop(name, None)
    environment.update(configured)
    return environment


def _summary_state(root: Path) -> dict[Path, int]:
    if not root.is_dir():
        return {}
    return {path: path.stat().st_mtime_ns for path in root.rglob("summary.json") if path.is_file()}


def _task_for_summary(path: Path, tasks: Sequence[str]) -> str | None:
    lowered_parts = {part.casefold() for part in path.parts}
    for task in tasks:
        if task in lowered_parts:
            return task
    return None


def _write_aggregate(
    destination: Path,
    *,
    model: str,
    tasks: Sequence[str],
    results_root: Path,
    before: Mapping[Path, int],
    returncode: int,
) -> None:
    after = _summary_state(results_root)
    changed = [
        path for path, modified in after.items() if path not in before or before[path] != modified
    ]
    records: list[dict[str, Any]] = []
    observed: set[str] = set()
    for path in sorted(changed):
        task = _task_for_summary(path, tasks)
        if task is not None:
            observed.add(task)
        try:
            metrics: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            metrics = {"error": str(error)}
        try:
            display_path = str(path.relative_to(results_root))
        except ValueError:
            display_path = str(path)
        records.append({"task": task, "summary": display_path, "metrics": metrics})

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "requested_tasks": list(tasks),
        "completed_tasks": sorted(observed),
        "missing_tasks": sorted(set(tasks) - observed),
        "evaluator_returncode": returncode,
        "results": records,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    namespace = parser.parse_args(argv)
    try:
        command, configured_env, tasks, results_root, summary_path = build_launch(namespace)
    except CliError as error:
        parser.error(str(error))

    print(f"mode={'dry-run' if namespace.dry_run else 'run'} tasks={','.join(tasks)}")
    if configured_env:
        print("configured environment: " + ", ".join(sorted(configured_env)))
    print(shlex.join(command))
    if namespace.dry_run:
        print(f"aggregate summary would be written to {summary_path}")
        return 0

    before = _summary_state(results_root)
    environment = _execution_environment(configured_env)
    completed = subprocess.run(command, env=environment, check=False)
    _write_aggregate(
        summary_path,
        model=str(Path(namespace.model).expanduser().resolve()),
        tasks=tasks,
        results_root=results_root,
        before=before,
        returncode=completed.returncode,
    )
    print(f"aggregate summary: {summary_path}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
