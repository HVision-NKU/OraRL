#!/usr/bin/env python3
"""Discover only the OraRL paper evaluation sources and write a private manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

PAPER_TASKS = (
    "videomme",
    "videommev2",
    "mvbench",
    "mmvu",
    "videoholmes",
    "longvideobench",
    "mlvu",
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

_SKIPPED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        "__pycache__",
        "checkpoint",
        "checkpoints",
        "logs",
        "lvbench",
        "onethinker-train-data",
        "output",
        "outputs",
        "train",
        "training",
        "videommmu",
    }
)
_UPSTREAM_TERMS = "See the upstream benchmark terms"
_MINDCUBE_EXPECTED_SAMPLES = 1050
_REVSI_EXPECTED_SAMPLES = 6808
_SEGMENTATION_EXPECTED_SAMPLES = {
    "mevis": 424,
    "reasonvos": 458,
    "refcoco": 3811,
    "refcocog": 2537,
    "refcocop": 3805,
}

_VIDEO_QA_SPECS: tuple[dict[str, Any], ...] = (
    {
        "task": "videomme",
        "folder": "Video-MME",
        "split": "test",
        "annotation_tiers": (
            ("videomme_preprocessed_384f_262k_total0.jsonl",),
            ("videomme.jsonl", "videomme.json"),
        ),
        "source_url": "https://github.com/BradyFU/Video-MME",
        "evaluation": {
            "prompt_profile": "videomme_default",
            "parser_profile": "multiple_choice",
            "metric_profile": "accuracy",
        },
        "preprocessing": {
            "fps": 2,
            "max_frames": 384,
            "video_min_pixels": 4096,
            "video_max_pixels": 262144,
            "video_total_pixels": 0,
        },
        "legacy": {
            "VIDEOMME_SETTING": (
                "all-qwen3_vl-sub0-f384-fps2-min4096-max262144-total0-"
                "videomme_preprocessed_384f_262k_total0"
            ),
            "VIDEOMME_FPS": 2,
            "VIDEOMME_MAX_FRAMES": 384,
            "VIDEOMME_VIDEO_MIN_PIXELS": 4096,
            "VIDEOMME_VIDEO_MAX_PIXELS": 262144,
            "VIDEOMME_VIDEO_TOTAL_PIXELS": 0,
        },
    },
    {
        "task": "videommev2",
        "folder": "Video-MME-v2",
        "split": "test",
        "annotation_tiers": (
            ("videommev2_preprocessed_384f_262k_total0.jsonl",),
            ("videommev2.jsonl", "videommev2.json"),
        ),
        "source_url": "https://github.com/BradyFU/Video-MME",
        "evaluation": {
            "prompt_profile": "videommev2_default",
            "parser_profile": "multiple_choice",
            "metric_profile": "accuracy",
        },
        "preprocessing": {
            "fps": 2,
            "max_frames": 384,
            "video_min_pixels": 4096,
            "video_max_pixels": 262144,
            "video_total_pixels": 0,
        },
        "legacy": {
            "VIDEOMMEV2_PROMPT_MODE": "default",
            "VIDEOMMEV2_ANSWER_FILTER": "",
            "VIDEOMMEV2_FPS": 2,
            "VIDEOMMEV2_MAX_FRAMES": 384,
            "VIDEOMMEV2_VIDEO_MIN_PIXELS": 4096,
            "VIDEOMMEV2_VIDEO_MAX_PIXELS": 262144,
            "VIDEOMMEV2_VIDEO_TOTAL_PIXELS": 0,
        },
    },
    {
        "task": "mvbench",
        "folder": "MVBench",
        "split": "test",
        "annotation_tiers": (("mvbench.json", "mvbench.jsonl"),),
        "source_url": "https://github.com/OpenGVLab/Ask-Anything",
        "evaluation": {
            "prompt_profile": "mvbench_default",
            "parser_profile": "multiple_choice",
            "metric_profile": "accuracy",
        },
    },
    {
        "task": "mmvu",
        "folder": "MMVU",
        "split": "test",
        "annotation_tiers": (
            ("mmvu_mc.jsonl",),
            ("mmvu-mc.json", "mmvu_mc.json", "mmvu.jsonl"),
        ),
        "source_url": "https://github.com/yale-nlp/MMVU",
        "evaluation": {
            "prompt_profile": "mmvu_default",
            "parser_profile": "multiple_choice",
            "metric_profile": "accuracy",
        },
    },
    {
        "task": "videoholmes",
        "folder": "Video-Holmes",
        "split": "test",
        "annotation_tiers": (("videoholmes.jsonl", "videoholmes.json"),),
        "source_url": "https://github.com/TencentARC/Video-Holmes",
        "evaluation": {
            "prompt_profile": "videoholmes_default",
            "parser_profile": "multiple_choice",
            "metric_profile": "accuracy",
        },
    },
    {
        "task": "longvideobench",
        "folder": "LongVideoBench",
        "split": "val",
        "annotation_tiers": (
            ("longvideobench_val.jsonl",),
            ("longvideobench.jsonl", "longvideobench.json"),
        ),
        "source_url": "https://github.com/longvideobench/LongVideoBench",
        "evaluation": {
            "prompt_profile": "longvideobench_default",
            "parser_profile": "multiple_choice",
            "metric_profile": "accuracy",
        },
        "legacy": {"LONGVIDEOBENCH_USE_SUBTITLES": True},
    },
    {
        "task": "mlvu",
        "folder": "MLVU",
        "split": "dev",
        "annotation_tiers": (
            ("mlvu_mc.jsonl",),
            ("mlvu-dev.json", "mlvu_mc.json", "mlvu.json"),
        ),
        "source_url": "https://github.com/JUNJIE99/MLVU",
        "evaluation": {
            "prompt_profile": "mlvu_default",
            "parser_profile": "multiple_choice",
            "metric_profile": "macro_accuracy",
        },
    },
)

_COMMON_VIDEO_SETTINGS = {
    "batch_size": 1,
    "enable_thinking": False,
    "fps": 2,
    "gpu_memory_utilization": 0.9,
    "max_frames": 384,
    "max_model_len": 65536,
    "max_new_tokens": 128,
    "max_num_batched_tokens": 65536,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "prompt_mode": "default",
    "temperature": 0.0,
    "top_k": -1,
    "top_p": 1.0,
    "video_max_pixels": 262144,
    "video_min_pixels": 4096,
    "video_total_pixels": 0,
}


class DiscoveryError(ValueError):
    """Raised when a required paper source is missing or ambiguous."""


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(directory.resolve(strict=False))
    except ValueError:
        return False
    return True


def _required_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise DiscoveryError(f"{label} directory does not exist: {resolved}")
    return resolved


def _walk_named_files(root: Path, names: Iterable[str]) -> list[Path]:
    wanted = set(names)
    matches: list[Path] = []
    for current, directories, filenames in os.walk(root, followlinks=False):
        directories[:] = sorted(
            name for name in directories if name.casefold() not in _SKIPPED_DIRECTORY_NAMES
        )
        current_path = Path(current)
        for filename in sorted(filenames):
            if filename in wanted:
                matches.append((current_path / filename).resolve())
    return sorted(set(matches), key=str)


def _walk_suffix_files(root: Path, suffixes: Iterable[str]) -> list[Path]:
    wanted = {suffix.casefold() for suffix in suffixes}
    matches: list[Path] = []
    for current, directories, filenames in os.walk(root, followlinks=False):
        directories[:] = sorted(
            name for name in directories if name.casefold() not in _SKIPPED_DIRECTORY_NAMES
        )
        current_path = Path(current)
        for filename in sorted(filenames):
            path = current_path / filename
            if path.suffix.casefold() in wanted:
                matches.append(path.resolve())
    return sorted(set(matches), key=str)


def _one_file(label: str, tiers: Sequence[Sequence[Path]]) -> Path:
    attempted: list[str] = []
    for tier in tiers:
        candidates = sorted(
            {path.resolve() for path in tier if path.is_file()},
            key=str,
        )
        attempted.extend(str(path) for path in tier)
        if len(candidates) > 1:
            rendered = ", ".join(str(path) for path in candidates)
            raise DiscoveryError(f"ambiguous {label}; found: {rendered}")
        if candidates:
            return candidates[0]
    raise DiscoveryError(f"required {label} was not found; tried: {', '.join(attempted)}")


def _runtime_annotation(
    runtime_root: Path,
    source_root: Path,
    label: str,
    name_tiers: Sequence[Sequence[str]],
) -> Path:
    valid_data = runtime_root / "eval" / "data" / "valid_data"
    tiers: list[list[Path]] = []
    for names in name_tiers:
        tiers.append([valid_data / name for name in names])
    for names in name_tiers:
        tiers.append(_walk_named_files(source_root, names))
    return _one_file(label, tiers)


def _named_annotation(root: Path, label: str, names: Sequence[str]) -> Path:
    return _one_file(label, [_walk_named_files(root, names)])


def _first_directory(label: str, tiers: Sequence[Sequence[Path]]) -> Path:
    attempted: list[str] = []
    for tier in tiers:
        candidates = sorted(
            {path.resolve() for path in tier if path.is_dir()},
            key=str,
        )
        attempted.extend(str(path) for path in tier)
        if len(candidates) > 1:
            rendered = ", ".join(str(path) for path in candidates)
            raise DiscoveryError(f"ambiguous {label}; found: {rendered}")
        if candidates:
            return candidates[0]
    raise DiscoveryError(f"required {label} directory was not found; tried: {', '.join(attempted)}")


def _cheap_expected_count(path: Path) -> int | None:
    suffix = path.suffix.casefold()
    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            return sum(bool(line.strip()) for line in handle)
    if suffix == ".tsv":
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            csv.field_size_limit(sys.maxsize)
            return sum(1 for _row in csv.DictReader(handle, delimiter="\t"))
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError:
            return None
        try:
            return int(parquet.ParquetFile(path).metadata.num_rows)
        except (OSError, ValueError):
            return None
    return None


def _mindcube_annotation(root: Path) -> Path:
    candidates = _walk_suffix_files(root, (".parquet",))
    if not candidates:
        raise DiscoveryError(
            "official MindCube-Tiny parquet was not found under "
            f"{root}; expected {_MINDCUBE_EXPECTED_SAMPLES} rows"
        )

    counted = [(path, _cheap_expected_count(path)) for path in candidates]
    official = [
        path for path, count in counted if count == _MINDCUBE_EXPECTED_SAMPLES
    ]
    if len(official) == 1:
        return official[0]
    if len(official) > 1:
        rendered = ", ".join(str(path) for path in official)
        raise DiscoveryError(
            "ambiguous official MindCube-Tiny parquet; "
            f"found multiple {_MINDCUBE_EXPECTED_SAMPLES}-row files: {rendered}"
        )

    canonical_unreadable = [
        path
        for path, count in counted
        if count is None and path.name == "combined-00000-of-00001.parquet"
    ]
    if len(canonical_unreadable) == 1:
        return canonical_unreadable[0]

    rendered = ", ".join(
        f"{path} ({'unknown' if count is None else count} rows)"
        for path, count in counted
    )
    raise DiscoveryError(
        f"official MindCube-Tiny requires {_MINDCUBE_EXPECTED_SAMPLES} rows; "
        f"found: {rendered}"
    )


def _revsi_annotation(root: Path) -> Path:
    annotation = root / "all_frame" / "test-00000-of-00001.parquet"
    if not annotation.is_file():
        raise DiscoveryError(
            "official ReVSI all-frame annotation was not found; expected "
            f"{annotation}"
        )
    count = _cheap_expected_count(annotation)
    if count is not None and count != _REVSI_EXPECTED_SAMPLES:
        raise DiscoveryError(
            f"official ReVSI all-frame split requires {_REVSI_EXPECTED_SAMPLES} rows; "
            f"found {count} in {annotation}"
        )
    return annotation


def _source_record(
    *,
    task: str,
    split: str,
    family: str,
    adapter: str,
    annotation: Path,
    source_url: str,
    evaluation: Mapping[str, Any],
    preprocessing: Mapping[str, Any] | None,
    legacy_environment: Mapping[str, Any] | None,
    media_roots: Mapping[str, Path] | None,
    preprocessed_roots: Mapping[str, Path] | None,
    authorized: bool,
    expected_count: int | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "benchmark": task,
        "eval_task": task,
        "split": split,
        "family": family,
        "adapter": adapter,
        "annotation_input": str(annotation.resolve()),
        "expected_count": (
            expected_count
            if expected_count is not None
            else _cheap_expected_count(annotation)
        ),
        "license": _UPSTREAM_TERMS,
        "source_url": source_url,
        "redistribution_authorized": authorized,
        "evaluation": dict(evaluation),
    }
    if preprocessing:
        record["preprocessing"] = dict(preprocessing)
    if legacy_environment:
        record["legacy_environment"] = dict(legacy_environment)
    if media_roots:
        record["media_roots"] = {
            name: str(path.resolve()) for name, path in sorted(media_roots.items())
        }
    if preprocessed_roots:
        record["preprocessed_roots"] = {
            name: str(path.resolve()) for name, path in sorted(preprocessed_roots.items())
        }
    return record


def _video_qa_records(
    data_root: Path,
    runtime_root: Path,
    *,
    authorized: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec in _VIDEO_QA_SPECS:
        task = str(spec["task"])
        source_root = _required_directory(data_root / str(spec["folder"]), str(spec["folder"]))
        annotation = _runtime_annotation(
            runtime_root,
            source_root,
            f"{task} annotation",
            spec["annotation_tiers"],
        )
        if task == "videomme":
            video_root = _first_directory(
                "videomme videos",
                (
                    (source_root / "data",),
                    (source_root / "videos",),
                    (source_root,),
                ),
            )
        elif task == "mvbench":
            # MVBench annotations already include their dataset subdirectory
            # (for example, ``./star/...`` and ``./clevrer/...``).  A separate
            # ``videos/`` directory may exist but is not the common parent of
            # those references.
            video_root = source_root
        else:
            video_root = source_root / "videos"
            if not video_root.is_dir():
                video_root = source_root
        media_roots: dict[str, Path] = {"videos": video_root}
        if task == "longvideobench":
            subtitle_root = source_root / "subtitles"
            media_roots["subtitles"] = subtitle_root if subtitle_root.is_dir() else source_root

        preprocessed_roots: dict[str, Path] = {}
        if task in {"videomme", "videommev2"}:
            global_preprocessed = data_root / "preprocessed_videos"
            prepared_name = (
                "videomme_preprocessed_384f_262k_total0"
                if task == "videomme"
                else "videommev2_preprocessed_384f_262k_total0"
            )
            prepared = _first_directory(
                f"{task} preprocessed videos",
                (
                    (source_root / "preprocessed_videos_384f_262k_total0",),
                    (source_root / "preprocessed_videos",),
                    (global_preprocessed / prepared_name,),
                    (global_preprocessed / task,),
                    (global_preprocessed,),
                ),
            )
            preprocessed_roots["default"] = prepared

        preprocessing = {**_COMMON_VIDEO_SETTINGS, **spec.get("preprocessing", {})}
        prefix = {
            "videomme": "VIDEOMME",
            "videommev2": "VIDEOMMEV2",
            "mvbench": "MVBENCH",
            "mmvu": "MMVU",
            "videoholmes": "VIDEOHOLMES",
            "longvideobench": "LONGVIDEOBENCH",
            "mlvu": "MLVU",
        }[task]
        legacy = {f"{prefix}_{key.upper()}": value for key, value in _COMMON_VIDEO_SETTINGS.items()}
        legacy.update(spec.get("legacy", {}))
        records.append(
            _source_record(
                task=task,
                split=str(spec["split"]),
                family="video_qa",
                adapter="generic",
                annotation=annotation,
                source_url=str(spec["source_url"]),
                evaluation=spec["evaluation"],
                preprocessing=preprocessing,
                legacy_environment=legacy,
                media_roots=media_roots,
                preprocessed_roots=preprocessed_roots,
                authorized=authorized,
            )
        )
    return records


def _spatial_intelligence_records(
    data_root: Path,
    runtime_root: Path,
    *,
    authorized: bool,
) -> list[dict[str, Any]]:
    vsi_root = _required_directory(data_root / "VSI-Bench", "VSI-Bench")
    vsi_annotation = _runtime_annotation(
        runtime_root,
        vsi_root,
        "vsi annotation",
        (
            ("vsibench_preprocessed_128f_16M.jsonl",),
            ("vsibench.jsonl", "vsibench.json"),
        ),
    )
    global_preprocessed = data_root / "preprocessed_videos"
    vsi_preprocessed = _first_directory(
        "vsi preprocessed videos",
        (
            (
                vsi_root / "preprocessed_videos_128f_16M",
                vsi_root / "preprocessed_videos",
            ),
            (
                global_preprocessed / "vsibench_preprocessed_128f_16M",
                global_preprocessed / "vsi",
                global_preprocessed / "VSI-Bench",
            ),
            (global_preprocessed,),
        ),
    )
    records = [
        _source_record(
            task="vsi",
            split="test",
            family="spatial_intelligence",
            adapter="generic",
            annotation=vsi_annotation,
            source_url="https://huggingface.co/datasets/nyu-visionx/VSI-Bench",
            evaluation={
                "prompt_profile": "vsi_default",
                "parser_profile": "vsi_answer",
                "metric_profile": "vsi_official",
            },
            preprocessing={
                "fps": 2,
                "max_frames": 128,
                "video_min_pixels": 65536,
                "video_total_pixels": 16777216,
            },
            legacy_environment={
                "VSI_SETTING": (
                    "video128-16M-video-f128-fps2-min65536-maxnone-total16777216-"
                    "vsibench_preprocessed_128f_16M"
                ),
                "VSI_BATCH_SIZE": 16,
                "VSI_MAX_MODEL_LEN": 32768,
                "VSI_MAX_NEW_TOKENS": 1024,
                "VSI_EXPECTED_SAMPLES": 5130,
            },
            media_roots={"videos": vsi_root},
            preprocessed_roots={"default": vsi_preprocessed},
            authorized=authorized,
        )
    ]

    mmsi_root = _required_directory(data_root / "MMSI-Bench", "MMSI-Bench")
    mmsi_annotation = _named_annotation(
        mmsi_root,
        "mmsi annotation",
        ("MMSI_bench.tsv",),
    )
    records.append(
        _source_record(
            task="mmsi",
            split="test",
            family="spatial_intelligence",
            adapter="mmsi",
            annotation=mmsi_annotation,
            source_url="https://huggingface.co/datasets/RunsenXu/MMSI-Bench",
            evaluation={
                "prompt_profile": "mmsi_default",
                "parser_profile": "multiple_choice",
                "metric_profile": "accuracy",
            },
            preprocessing={"image_min_pixels": 4096, "image_max_pixels": 262144},
            legacy_environment={
                "MMSI_BACKEND": "transformers",
                "MMSI_MAX_IMAGES": 0,
                "MMSI_IMAGE_MIN_PIXELS": 4096,
                "MMSI_IMAGE_MAX_PIXELS": 262144,
                "MMSI_ENABLE_THINKING": False,
            },
            media_roots=None,
            preprocessed_roots=None,
            authorized=authorized,
        )
    )

    mindcube_root = _required_directory(data_root / "MindCube-Tiny", "MindCube-Tiny")
    mindcube_annotation = _mindcube_annotation(mindcube_root)
    records.append(
        _source_record(
            task="mindcube",
            split="test",
            family="spatial_intelligence",
            adapter="mindcube",
            annotation=mindcube_annotation,
            source_url=(
                "https://huggingface.co/datasets/oscarqjh/MindCube_lmmseval/"
                "tree/7dd2725d9bd4149f2aad00a9843f72a3824da003"
            ),
            evaluation={
                "prompt_profile": "mindcube_official",
                "parser_profile": "multiple_choice",
                "metric_profile": "micro_accuracy",
                "aggregation": "micro",
                "expected_group_counts": {
                    "rotation": 200,
                    "among": 600,
                    "around": 250,
                },
            },
            preprocessing={"max_images": 4},
            legacy_environment={
                "MINDCUBE_MAX_IMAGES": 4,
                "MINDCUBE_BATCH_SIZE": 16,
                "MINDCUBE_MAX_MODEL_LEN": 32768,
                "MINDCUBE_MAX_NEW_TOKENS": 1024,
            },
            media_roots={"images": mindcube_root},
            preprocessed_roots=None,
            authorized=authorized,
        )
    )

    revsi_root = _required_directory(data_root / "ReVSI", "ReVSI")
    revsi_annotation = _revsi_annotation(revsi_root)
    revsi_frame_root = revsi_annotation.parent
    records.append(
        _source_record(
            task="revsi",
            split="test",
            family="spatial_intelligence",
            adapter="revsi",
            annotation=revsi_annotation,
            source_url="https://arxiv.org/abs/2605.25979",
            evaluation={
                "prompt_profile": "revsi_official",
                "parser_profile": "revsi_answer",
                "metric_profile": "revsi_macro_average",
                "aggregation": "macro_by_question_type",
                "frame_protocol": "native_all_frame",
            },
            preprocessing={
                "fps": 2,
                "max_frames": 128,
                "exact_nframes": True,
                "video_min_pixels": 65536,
                "video_total_pixels": 16777216,
            },
            legacy_environment={
                "REVSI_SETTING": (
                    "native-all-f128-exacttrue-fps2-min65536-"
                    "maxnone-total16777216"
                ),
                "REVSI_FRAME_BUDGET": "all",
                "REVSI_MAX_FRAMES": 128,
                "REVSI_EXACT_NFRAMES": True,
                "REVSI_FPS": 2,
                "REVSI_VIDEO_MIN_PIXELS": 65536,
                "REVSI_VIDEO_TOTAL_PIXELS": 16777216,
                "REVSI_BATCH_SIZE": 16,
                "REVSI_MAX_MODEL_LEN": 32768,
                "REVSI_MAX_NEW_TOKENS": 64,
                "REVSI_GPU_MEMORY_UTILIZATION": 0.9,
                "REVSI_EXPECTED_SAMPLES": _REVSI_EXPECTED_SAMPLES,
                "REVSI_ENABLE_THINKING": False,
            },
            media_roots={"videos": revsi_frame_root},
            preprocessed_roots=None,
            authorized=authorized,
        )
    )
    return records


def _structured_records(
    data_root: Path,
    runtime_root: Path,
    *,
    authorized: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    one_thinker_root = _required_directory(
        data_root / "OneThinker-eval",
        "OneThinker-eval",
    )
    grounding_root = _required_directory(
        data_root / "Spatial-Grounding",
        "Spatial-Grounding",
    )
    grounding_image_root = _first_directory(
        "Spatial Grounding images",
        (
            (
                one_thinker_root / "Refcoco",
                one_thinker_root / "RefCOCO",
            ),
            (
                grounding_root / "train2014",
                grounding_root / "Refcoco",
                grounding_root / "RefCOCO",
            ),
            (grounding_root,),
        ),
    )
    runtime_grounding_root = (
        runtime_root / "eval" / "task" / "spatial_grounding" / "rec_jsons_processed"
    )
    grounding_splits = (
        ("refcoco_val", "refcoco_val.json", "refcoco-val"),
        ("refcoco_test_a", "refcoco_testA.json", "refcoco-testA"),
        ("refcoco_test_b", "refcoco_testB.json", "refcoco-testB"),
        ("refcocop_val", "refcocop_val.json", "refcoco+-val"),
        ("refcocop_test_a", "refcocop_testA.json", "refcoco+-testA"),
        ("refcocop_test_b", "refcocop_testB.json", "refcoco+-testB"),
        ("refcocog_val", "refcocog_val.json", "refcocog-val"),
        ("refcocog_test", "refcocog_test.json", "refcocog-test"),
    )
    grounding_profile = {
        "prompt_profile": "qwen_native",
        "parser_profile": "norm1000_bbox",
        "metric_profile": "refcoco_iou",
    }
    grounding_legacy = {
        "SPATIAL_GROUNDING_DATASETS": ",".join(
            evaluator_name for _split, _filename, evaluator_name in grounding_splits
        ),
        "SPATIAL_GROUNDING_PROMPT_STYLE": "qwen_native",
        "SPATIAL_GROUNDING_COORD_SYSTEM": "norm1000",
        "SPATIAL_GROUNDING_BBOX_SELECT": "first",
        "SPATIAL_GROUNDING_MIN_TOKENS": 64,
        "SPATIAL_GROUNDING_TOTAL_TOKENS": 1024,
        "SPATIAL_GROUNDING_MAX_NEW_TOKENS": 1024,
    }
    for split, filename, _evaluator_name in grounding_splits:
        jsonl_filename = f"{Path(filename).stem}.jsonl"
        annotation = _one_file(
            f"spatial_grounding/{split} annotation",
            (
                (runtime_grounding_root / filename,),
                (runtime_grounding_root / jsonl_filename,),
                _walk_named_files(grounding_root, (filename,)),
                _walk_named_files(grounding_root, (jsonl_filename,)),
            ),
        )
        records.append(
            _source_record(
                task="spatial_grounding",
                split=split,
                family="spatial_grounding",
                adapter="spatial_grounding",
                annotation=annotation,
                source_url="https://huggingface.co/datasets/OneThink/OneThinker-eval",
                evaluation=grounding_profile,
                preprocessing={"coordinate_system": "norm1000"},
                legacy_environment=grounding_legacy,
                media_roots={"images": grounding_image_root},
                preprocessed_roots=None,
                authorized=authorized,
            )
        )

    one_thinker_specs = (
        (
            "tracking",
            "got10k",
            "eval_got10k.json",
            "tracking",
            {
                "prompt_profile": "tracking_default",
                "parser_profile": "tracking_boxes",
                "metric_profile": "got10k_ao",
            },
            {
                "TRACKING_DATASETS": "eval_got10k",
                "TRACKING_MAX_FRAMES": 32,
                "TRACKING_FPS": 1,
                "TRACKING_VIDEO_MIN_PIXELS": 4096,
                "TRACKING_VIDEO_MAX_PIXELS": 786432,
                "TRACKING_VIDEO_TOTAL_PIXELS": 8388608,
                "TRACKING_MAX_NEW_TOKENS": 8192,
                "TRACKING_PROMPT_MODE": "default",
            },
        ),
        (
            "stvg",
            "stvg",
            "eval_stvg.json",
            "spatial_temporal_grounding",
            {
                "prompt_profile": "train_stvg",
                "parser_profile": "temporal_spatial_boxes",
                "metric_profile": "stvg_official",
            },
            {
                "STVG_DATASETS": "eval_stvg",
                "STVG_MAX_FRAMES": 128,
                "STVG_FPS": 2,
                "STVG_VIDEO_MIN_PIXELS": 65536,
                "STVG_VIDEO_MAX_PIXELS": 393216,
                "STVG_VIDEO_TOTAL_PIXELS": 10485760,
                "STVG_MAX_NEW_TOKENS": 2048,
                "STVG_PROMPT_MODE": "train_stvg",
            },
        ),
    )
    for task, split, filename, family, evaluation, legacy in one_thinker_specs:
        annotation = _named_annotation(
            one_thinker_root,
            f"{task}/{split} annotation",
            (filename, f"{Path(filename).stem}.jsonl"),
        )
        records.append(
            _source_record(
                task=task,
                split=split,
                family=family,
                adapter="one_thinker",
                annotation=annotation,
                source_url="https://huggingface.co/datasets/OneThink/OneThinker-eval",
                evaluation=evaluation,
                preprocessing={
                    key.casefold(): value
                    for key, value in legacy.items()
                    if key.endswith(
                        (
                            "_FPS",
                            "_MAX_FRAMES",
                            "_VIDEO_MIN_PIXELS",
                            "_VIDEO_MAX_PIXELS",
                            "_VIDEO_TOTAL_PIXELS",
                        )
                    )
                },
                legacy_environment=legacy,
                media_roots={"default": one_thinker_root},
                preprocessed_roots=None,
                authorized=authorized,
            )
        )

    segmentation_splits = (
        ("refcoco", "eval_seg_refcoco.json"),
        ("refcocop", "eval_seg_refcocop.json"),
        ("refcocog", "eval_seg_refcocog.json"),
        ("mevis", "eval_seg_mevis.json"),
        ("reasonvos", "eval_seg_reasonvos.json"),
    )
    segmentation_legacy = {
        "SEGMENTATION_DATASETS": ",".join(split for split, _name in segmentation_splits),
        "SEGMENTATION_DATA_TYPE": "all",
        "SEGMENTATION_PROMPT_MODE": "train_seg",
        "SEGMENTATION_ENABLE_THINKING": False,
        "SEGMENTATION_MAX_FRAMES": 128,
        "SEGMENTATION_FPS": 2,
        "SEGMENTATION_VIDEO_READER": "decord",
        "SEGMENTATION_VIDEO_MIN_PIXELS": 4096,
        "SEGMENTATION_VIDEO_MAX_PIXELS": 262144,
        "SEGMENTATION_VIDEO_TOTAL_PIXELS": 16777216,
        "SEGMENTATION_MAX_PIXELS_IMAGE": 1048576,
        "SEGMENTATION_MIN_PIXELS_IMAGE": 4096,
        "SEGMENTATION_BATCH_SIZE": 16,
        "SEGMENTATION_MAX_MODEL_LEN": 32768,
        "SEGMENTATION_MAX_NEW_TOKENS": 1024,
        "SEGMENTATION_GPU_MEM_UTIL": 0.85,
        "SEGMENTATION_SEED": 42,
        "SEGMENTATION_RUN_SAM2": False,
        "SEGMENTATION_SETTING": (
            "segmentation-refcoco_refcocop_refcocog_mevis_reasonvos-"
            "train_seg-f128-fps2-min4096-max262144-total16777216-"
            "readerdecord-new1024"
        ),
    }
    for split, filename in segmentation_splits:
        annotation = _named_annotation(
            one_thinker_root,
            f"segmentation/{split} annotation",
            (filename, f"{Path(filename).stem}.jsonl"),
        )
        records.append(
            _source_record(
                task="segmentation",
                split=split,
                family="segmentation",
                adapter="one_thinker",
                annotation=annotation,
                source_url="https://huggingface.co/datasets/OneThink/OneThinker-eval",
                evaluation={
                    "prompt_profile": "train_seg",
                    "parser_profile": "sam2_prompt",
                    "metric_profile": "segmentation_official",
                    "oracle_profile": "segmentation_rle",
                    "oracle_location": "task_payload.segmentation_output",
                },
                preprocessing={
                    "fps": 2,
                    "max_frames": 128,
                    "video_reader": "decord",
                    "video_min_pixels": 4096,
                    "video_max_pixels": 262144,
                    "video_total_pixels": 16777216,
                },
                legacy_environment=segmentation_legacy,
                media_roots={"default": one_thinker_root},
                preprocessed_roots=None,
                authorized=authorized,
                expected_count=_SEGMENTATION_EXPECTED_SAMPLES[split],
            )
        )

    timelens_root = _required_directory(data_root / "TimeLens-Bench", "TimeLens-Bench")
    timelens_specs = (
        (
            "charades_timelens",
            "charades-timelens",
            4,
            ((timelens_root / "video_shards" / "charades",),),
        ),
        (
            "activitynet_timelens",
            "activitynet-timelens",
            4,
            ((timelens_root / "video_shards" / "activitynet",),),
        ),
        (
            "qvhighlights_timelens",
            "qvhighlights-timelens",
            4,
            (
                (timelens_root / "video_shards" / "qvhighlights",),
                (data_root / "qvhighlights-videos",),
                (timelens_root / "qvhighlights-videos",),
            ),
        ),
    )
    timelens_datasets = ",".join(split for split, _stem, _fps, _roots in timelens_specs)
    common_timelens_environment = {
        "TIMELENS_DATASETS": timelens_datasets,
        "TIMELENS_ENABLE_THINKING": False,
        "TIMELENS_FPS": 4,
        "TIMELENS_MIN_TOKENS": 1,
        "TIMELENS_MAX_FRAMES": 2048,
        "TIMELENS_MAX_PIXELS": 409600,
        "TIMELENS_TOTAL_TOKENS": 128000,
        "TIMELENS_MAX_NEW_TOKENS": 128,
        "TIMELENS_PROMPT_MODE": "same",
        "TIMELENS_STOP_AFTER_ANSWER": True,
        "TIMELENS_NUM_WORKERS": 2,
    }
    for split, stem, fps, video_root_tiers in timelens_specs:
        annotation = _named_annotation(
            timelens_root,
            f"temporal_grounding/{split} annotation",
            (f"{stem}.json", f"{stem}.jsonl"),
        )
        video_root = _first_directory(f"{split} videos", video_root_tiers)
        records.append(
            _source_record(
                task="temporal_grounding",
                split=split,
                family="temporal_grounding",
                adapter="timelens",
                annotation=annotation,
                source_url="https://huggingface.co/datasets/TencentARC/TimeLens-Bench",
                evaluation={
                    "prompt_profile": "timelens_same",
                    "parser_profile": "temporal_spans",
                    "metric_profile": "temporal_iou",
                },
                preprocessing={
                    "fps": fps,
                    "min_tokens": 1,
                    "max_frames": 2048,
                    "max_pixels": 409600,
                    "total_tokens": 128000,
                },
                legacy_environment=common_timelens_environment,
                media_roots={"videos": video_root},
                preprocessed_roots=None,
                authorized=authorized,
            )
        )
    return records


def discover_eval_sources(
    data_root: str | os.PathLike[str],
    runtime_root: str | os.PathLike[str],
    *,
    redistribution_authorized: bool = False,
) -> list[dict[str, Any]]:
    """Discover the fixed paper suite without scanning unrelated data trees."""

    data = _required_directory(Path(data_root), "data root")
    runtime = _required_directory(Path(runtime_root), "runtime root")
    records = [
        *_video_qa_records(data, runtime, authorized=redistribution_authorized),
        *_spatial_intelligence_records(data, runtime, authorized=redistribution_authorized),
        *_structured_records(
            data,
            runtime,
            authorized=redistribution_authorized,
        ),
    ]
    records.sort(key=lambda item: (str(item["benchmark"]), str(item["split"])))
    observed_tasks = {str(record["eval_task"]) for record in records}
    if observed_tasks != set(PAPER_TASKS):
        missing = sorted(set(PAPER_TASKS) - observed_tasks)
        unexpected = sorted(observed_tasks - set(PAPER_TASKS))
        raise DiscoveryError(
            f"paper task discovery mismatch; missing={missing}, unexpected={unexpected}"
        )
    if any(record["benchmark"] != record["eval_task"] for record in records):
        raise DiscoveryError("canonical discovery requires benchmark equal to eval_task")
    return records


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite source manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
                handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Root containing licensed sources.")
    parser.add_argument(
        "--runtime-root",
        required=True,
        help="Compatible runtime checkout containing generated paper annotations.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Private JSONL source manifest outside the OraRL release tree.",
    )
    parser.add_argument(
        "--confirm-redistribution-authorized",
        action="store_true",
        help="Record explicit authorization for every discovered source.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = create_parser().parse_args(argv)
    output = Path(namespace.output).expanduser().resolve(strict=False)
    release_tree = Path(namespace.runtime_root).expanduser().resolve() / "OraRL"
    if _is_within(output, release_tree):
        raise SystemExit("ERROR: --output must be private and outside the OraRL release tree")
    try:
        records = discover_eval_sources(
            namespace.data_root,
            namespace.runtime_root,
            redistribution_authorized=namespace.confirm_redistribution_authorized,
        )
        _write_jsonl(output, records, overwrite=namespace.overwrite)
    except (DiscoveryError, FileExistsError, OSError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    unlocked = sum(record["expected_count"] is None for record in records)
    print(
        json.dumps(
            {
                "output": str(output),
                "records": len(records),
                "tasks": len({record["eval_task"] for record in records}),
                "unlocked_counts": unlocked,
                "redistribution_authorized": bool(namespace.confirm_redistribution_authorized),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
