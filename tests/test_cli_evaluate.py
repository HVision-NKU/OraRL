from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT))

from orarl.cli import evaluate  # noqa: E402
from orarl.evaluation.layout import (  # noqa: E402
    annotation_path as canonical_annotation_path,
)
from orarl.evaluation.layout import media_directory  # noqa: E402


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _canonical_root(
    tmp_path: Path,
    specs: list[dict[str, object]],
) -> Path:
    root = tmp_path / "canonical"
    manifests: list[dict[str, object]] = []
    assets: list[dict[str, object]] = []
    for spec in specs:
        task = str(spec["task"])
        benchmark = str(spec.get("benchmark", task))
        split = str(spec.get("split", "test"))
        video_bytes = f"{benchmark}/{split}".encode()
        video_digest = hashlib.sha256(video_bytes).hexdigest()
        video_path = f"{media_directory(benchmark, 'videos')}/{video_digest}.mp4"
        (root / video_path).parent.mkdir(parents=True, exist_ok=True)
        (root / video_path).write_bytes(video_bytes)

        row = {
            "schema_version": 1,
            "eval_task": task,
            "sample_id": f"{benchmark}-{split}",
            "benchmark": benchmark,
            "split": split,
            "problem": "What happens?",
            "answer": "A",
            "images": [],
            "videos": [video_path],
            "problem_type": task,
            "source": benchmark,
            "task_payload": {},
            "evaluation": {"metric_profile": "fixture"},
        }
        annotation_path = canonical_annotation_path(benchmark, split)
        _write_jsonl(root / annotation_path, [row])
        annotation_bytes = (root / annotation_path).read_bytes()
        annotation_digest = hashlib.sha256(annotation_bytes).hexdigest()

        manifest = {
            "schema_version": 1,
            "benchmark": benchmark,
            "split": split,
            "task": task,
            "family": str(spec.get("family", task)),
            "annotation_path": annotation_path,
            "media_paths": [media_directory(benchmark, "videos")],
            "artifact_paths": [],
            "expected_count": 1,
            "license": "fixture-only",
            "source_url": f"https://example.org/{benchmark}",
            "redistribution_authorized": True,
            "evaluation": spec.get(
                "evaluation",
                {
                    "prompt_profile": "fixture",
                    "parser_profile": "fixture",
                    "metric_profile": "fixture",
                },
            ),
            "checksums": {annotation_path: annotation_digest},
        }
        if "preprocessing" in spec:
            manifest["preprocessing"] = spec["preprocessing"]
        if "legacy_environment" in spec:
            manifest["legacy_environment"] = spec["legacy_environment"]
        manifests.append(manifest)
        assets.extend(
            [
                {
                    "path": annotation_path,
                    "sha256": annotation_digest,
                    "bytes": len(annotation_bytes),
                    "kind": "annotations",
                    "benchmark": benchmark,
                    "license": "fixture-only",
                },
                {
                    "path": video_path,
                    "sha256": video_digest,
                    "bytes": len(video_bytes),
                    "kind": "videos",
                    "benchmark": benchmark,
                    "license": "fixture-only",
                },
            ]
        )

    manifests.sort(key=lambda record: (str(record["benchmark"]), str(record["split"])))
    assets.sort(key=lambda record: str(record["path"]))
    _write_jsonl(root / "datasets.jsonl", manifests)
    _write_jsonl(root / "assets.jsonl", assets)
    return root


def _canonical_arguments(tmp_path: Path, root: str, tasks: str = "vsi") -> list[str]:
    model = tmp_path / "canonical-model"
    model.mkdir(exist_ok=True)
    evaluator = tmp_path / "canonical-eval.sh"
    evaluator.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    return [
        "--model",
        str(model),
        "--tasks",
        tasks,
        "--dataset",
        root,
        "--evaluator",
        str(evaluator),
        "--summary",
        str(tmp_path / "canonical-summary.json"),
    ]


def _base_arguments(tmp_path: Path, tasks: str = "vsi") -> list[str]:
    model = tmp_path / "model"
    model.mkdir()
    evaluator = tmp_path / "eval.sh"
    evaluator.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    data_root = tmp_path / "benchmarks"
    data_root.mkdir()
    return [
        "--model",
        str(model),
        "--tasks",
        tasks,
        "--data-root",
        str(data_root),
        "--evaluator",
        str(evaluator),
        "--summary",
        str(tmp_path / "aggregate.json"),
    ]


def test_released_temporal_profiles_match_the_paper_evaluation() -> None:
    records = [
        json.loads(line)
        for line in (RELEASE_ROOT / "data/eval/datasets.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    records = [
        record for record in records if record["task"] == "temporal_grounding"
    ]

    assert {record["split"] for record in records} == {
        "activitynet_timelens",
        "charades_timelens",
        "qvhighlights_timelens",
    }
    assert all(
        record["preprocessing"]
        == {
            "fps": 4,
            "max_frames": 2048,
            "max_pixels": 409600,
            "min_tokens": 1,
            "total_tokens": 128000,
        }
        for record in records
    )


def test_eval_builds_restricted_parent_command(tmp_path: Path) -> None:
    namespace = evaluate.create_parser().parse_args(_base_arguments(tmp_path, "vsi,mindcube"))
    command, environment, tasks, _, summary = evaluate.build_launch(namespace)

    assert namespace.dry_run is True
    assert tasks == ["vsi", "mindcube"]
    assert command[:2] == ["bash", str((tmp_path / "eval.sh").resolve())]
    assert command[command.index("--tasks") + 1] == "vsi,mindcube"
    assert environment["MINDCUBE_EXPECTED_SAMPLES"] == "1050"
    assert environment["VSI_DATA_FILE"].endswith("vsi/annotations.jsonl")
    assert summary == (tmp_path / "aggregate.json").resolve()


def test_video_qa_alias_expands_to_documented_tasks(tmp_path: Path) -> None:
    namespace = evaluate.create_parser().parse_args(_base_arguments(tmp_path, "video_qa"))
    _, _, tasks, _, _ = evaluate.build_launch(namespace)
    assert tuple(tasks) == (
        "videomme",
        "videommev2",
        "mvbench",
        "mmvu",
        "videoholmes",
        "longvideobench",
        "mlvu",
    )


def test_selected_assets_follow_manifested_video_qa_roots() -> None:
    records = [
        {
            "benchmark": "videomme",
            "annotation_path": "annotations/video_qa/videomme/test.jsonl",
            "media_paths": ["media/video_qa/videomme/videos"],
            "artifact_paths": ["artifacts/video_qa/videomme"],
        }
    ]
    paths = (
        "annotations/video_qa/videomme/test.jsonl",
        "media/video_qa/videomme/videos/test/clip.mp4",
        "artifacts/video_qa/videomme/test/cache.pt",
        "media/video_qa/videommev2/videos/test/other.mp4",
    )
    assets = [{"path": path} for path in paths]

    selected = evaluate._selected_asset_records(assets, records)

    assert [record["path"] for record in selected] == list(paths[:3])


def test_videommev2_accepts_bounded_smoke_test(tmp_path: Path) -> None:
    root = _canonical_root(
        tmp_path,
        [
            {
                "task": "videommev2",
                "split": "test",
                "legacy_environment": {
                    "VIDEOMMEV2_SETTING": "videommev2-fixture",
                },
            }
        ],
    )
    namespace = evaluate.create_parser().parse_args(
        [
            *_canonical_arguments(tmp_path, str(root), tasks="videommev2"),
            "--max-samples",
            "8",
        ]
    )

    _, environment, _, _, _ = evaluate.build_launch(namespace)

    assert environment["VIDEOMMEV2_MAX_SAMPLES"] == "8"
    assert environment["VIDEOMMEV2_DATA_FILE"] == str(
        (root / "annotations/video_qa/videommev2/test.jsonl").resolve()
    )


def test_videomme_accepts_bounded_smoke_test(tmp_path: Path) -> None:
    root = _canonical_root(
        tmp_path,
        [
            {
                "task": "videomme",
                "split": "test",
                "legacy_environment": {
                    "VIDEOMME_SETTING": "videomme-fixture",
                },
            }
        ],
    )
    namespace = evaluate.create_parser().parse_args(
        [
            *_canonical_arguments(tmp_path, str(root), tasks="videomme"),
            "--max-samples",
            "8",
        ]
    )

    _, environment, _, _, _ = evaluate.build_launch(namespace)

    assert environment["VIDEOMME_MAX_SAMPLES"] == "8"
    assert environment["VIDEOMME_DATA_FILE"] == str(
        (root / "annotations/video_qa/videomme/test.jsonl").resolve()
    )


@pytest.mark.parametrize(
    ("task", "prefix"),
    [
        ("mvbench", "MVBENCH"),
        ("mmvu", "MMVU"),
        ("videoholmes", "VIDEOHOLMES"),
        ("longvideobench", "LONGVIDEOBENCH"),
        ("mlvu", "MLVU"),
    ],
)
def test_other_video_qa_tasks_accept_bounded_smoke_tests(
    tmp_path: Path,
    task: str,
    prefix: str,
) -> None:
    root = _canonical_root(
        tmp_path,
        [
            {
                "task": task,
                "split": "test",
                "legacy_environment": {
                    f"{prefix}_SETTING": f"{task}-fixture",
                },
            }
        ],
    )
    namespace = evaluate.create_parser().parse_args(
        [
            *_canonical_arguments(tmp_path, str(root), tasks=task),
            "--max-samples",
            "8",
        ]
    )

    _, environment, _, _, _ = evaluate.build_launch(namespace)

    assert environment[f"{prefix}_MAX_SAMPLES"] == "8"
    assert environment[f"{prefix}_DATA_FILE"] == str(
        (root / f"annotations/video_qa/{task}/test.jsonl").resolve()
    )


def test_eval_rejects_unknown_task(tmp_path: Path) -> None:
    namespace = evaluate.create_parser().parse_args(_base_arguments(tmp_path, "unknown"))
    with pytest.raises(evaluate.CliError, match="unsupported task"):
        evaluate.build_launch(namespace)


def test_eval_rejects_benchmark_outside_paper_suite(tmp_path: Path) -> None:
    namespace = evaluate.create_parser().parse_args(_base_arguments(tmp_path, "videommmu"))
    with pytest.raises(evaluate.CliError, match="unsupported task"):
        evaluate.build_launch(namespace)


def test_evaluator_discovery_uses_installed_runtime_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    package = runtime_root / "verl"
    package.mkdir(parents=True)
    init_file = package / "__init__.py"
    init_file.touch()
    evaluator = runtime_root / "eval" / "task" / "eval.sh"
    evaluator.parent.mkdir(parents=True)
    evaluator.touch()
    monkeypatch.setattr(
        evaluate,
        "find_spec",
        lambda _: SimpleNamespace(
            origin=str(init_file),
            submodule_search_locations=[str(package)],
        ),
    )

    assert evaluate._discover_evaluator(None) == evaluator.resolve()


def test_installed_evaluator_uses_wheel_data_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = tmp_path / "share" / "orarl" / "eval" / "task" / "eval.sh"
    evaluator.parent.mkdir(parents=True)
    evaluator.touch()
    package_distribution = SimpleNamespace(
        files=["../../../share/orarl/eval/task/eval.sh"],
        locate_file=lambda _: evaluator,
    )
    monkeypatch.setattr(evaluate, "distribution", lambda _: package_distribution)

    assert evaluate._installed_evaluator() == evaluator.resolve()


def test_explicit_missing_evaluator_is_not_silently_ignored(
    tmp_path: Path,
) -> None:
    with pytest.raises(evaluate.CliError, match="evaluator does not exist"):
        evaluate._discover_evaluator(str(tmp_path / "missing.sh"))


def test_task_config_is_allowlisted_and_resolves_relative_paths(tmp_path: Path) -> None:
    arguments = _base_arguments(tmp_path)
    config = tmp_path / "vsi.yaml"
    config.write_text(
        "task: vsi\n"
        "environment:\n"
        "  VSI_DATA_FILE: annotations.jsonl\n"
        "  VSI_PREPROCESSED_VIDEO_DIR: media\n"
        "  EVAL_CONDA_ENV: orarl-eval\n",
        encoding="utf-8",
    )
    arguments.extend(("--task-config", f"vsi={config}"))
    namespace = evaluate.create_parser().parse_args(arguments)
    command, environment, _, _, _ = evaluate.build_launch(namespace)

    root = (tmp_path / "benchmarks").resolve()
    assert environment["VSI_DATA_FILE"] == str(root / "annotations.jsonl")
    assert environment["VSI_PREPROCESSED_VIDEO_DIR"] == str(root / "media")
    assert "EVAL_CONDA_ENV" not in environment
    assert command[command.index("--env") + 1] == "orarl-eval"


def test_canonical_local_root_resolves_profiles_and_paths(tmp_path: Path) -> None:
    root = _canonical_root(
        tmp_path,
        [
            {
                "task": "vsi",
                "legacy_environment": {
                    "VSI_DATA_FILE": "annotations/spatial_intelligence/vsi/test.jsonl",
                    "VSI_PREPROCESSED_VIDEO_DIR": "media/spatial_intelligence/vsi/videos",
                    "VSI_TASK_FILTER": ["distance", "direction"],
                },
            }
        ],
    )
    namespace = evaluate.create_parser().parse_args(
        _canonical_arguments(tmp_path, str(root))
    )

    _, environment, tasks, _, _ = evaluate.build_launch(namespace)

    assert tasks == ["vsi"]
    assert environment["ORARL_EVAL_DATA_ROOT"] == str(root.resolve())
    assert environment["ORARL_EVAL_DATASETS_JSONL"] == str(
        (root / "datasets.jsonl").resolve()
    )
    assert environment["VSI_DATA_FILE"] == str(
        (root / "annotations/spatial_intelligence/vsi/test.jsonl").resolve()
    )
    assert environment["VSI_PREPROCESSED_VIDEO_DIR"] == str(
        (root / "media/spatial_intelligence/vsi/videos").resolve()
    )
    assert environment["VSI_EXPECTED_SAMPLES"] == "1"
    assert environment["VSI_TASK_FILTER"] == "distance,direction"


def test_canonical_revsi_uses_all_frame_profile_and_assets(tmp_path: Path) -> None:
    root = _canonical_root(
        tmp_path,
        [
            {
                "task": "revsi",
                "family": "spatial_intelligence",
                "legacy_environment": {
                    "REVSI_FRAME_BUDGET": "all",
                    "REVSI_MAX_FRAMES": 128,
                    "REVSI_EXACT_NFRAMES": True,
                    "REVSI_VIDEO_TOTAL_PIXELS": 16777216,
                },
            }
        ],
    )
    namespace = evaluate.create_parser().parse_args(
        [
            *_canonical_arguments(tmp_path, str(root), tasks="revsi"),
            "--batch-size",
            "4",
            "--max-samples",
            "8",
        ]
    )

    _, environment, tasks, _, _ = evaluate.build_launch(namespace)

    assert tasks == ["revsi"]
    assert environment["REVSI_DATA_FILE"] == str(
        (root / "annotations/spatial_intelligence/revsi/test.jsonl").resolve()
    )
    assert environment["REVSI_VIDEO_ROOT"] == str(root.resolve())
    assert environment["REVSI_EXPECTED_SAMPLES"] == "1"
    assert environment["REVSI_FRAME_BUDGET"] == "all"
    assert environment["REVSI_EXACT_NFRAMES"] == "true"
    assert environment["REVSI_BATCH_SIZE"] == "4"
    assert environment["REVSI_MAX_SAMPLES"] == "8"


def test_canonical_segmentation_supports_bounded_smoke_test(tmp_path: Path) -> None:
    splits = ("mevis", "reasonvos", "refcoco", "refcocog", "refcocop")
    root = _canonical_root(
        tmp_path,
        [
            {
                "task": "segmentation",
                "split": split,
                "family": "segmentation",
                "legacy_environment": {
                    "SEGMENTATION_RUN_SAM2": False,
                    "SEGMENTATION_MAX_FRAMES": 128,
                },
            }
            for split in splits
        ],
    )
    namespace = evaluate.create_parser().parse_args(
        [
            *_canonical_arguments(tmp_path, str(root), tasks="segmentation"),
            "--max-samples",
            "1",
        ]
    )

    _, environment, tasks, _, _ = evaluate.build_launch(namespace)

    assert tasks == ["segmentation"]
    assert environment["SEGMENTATION_DATASETS"] == ",".join(splits)
    assert environment["SEGMENTATION_DATA_ROOT"] == str(root.resolve())
    assert environment["SEGMENTATION_MAX_SAMPLES"] == "1"
    assert environment["SEGMENTATION_RUN_SAM2"] == "false"


def test_canonical_metadata_can_use_a_separate_asset_root(tmp_path: Path) -> None:
    root = _canonical_root(
        tmp_path,
        [
            {
                "task": "vsi",
                "legacy_environment": {
                    "VSI_PREPROCESSED_VIDEO_DIR": "media/spatial_intelligence/vsi/videos",
                },
            }
        ],
    )
    asset_root = tmp_path / "external-assets"
    asset_root.mkdir()
    (root / "assets.jsonl").rename(asset_root / "assets.jsonl")
    namespace = evaluate.create_parser().parse_args(
        [
            *_canonical_arguments(tmp_path, str(root)),
            "--asset-root",
            str(asset_root),
        ]
    )

    _, environment, _, _, _ = evaluate.build_launch(namespace)

    assert environment["ORARL_EVAL_DATA_ROOT"] == str(root.resolve())
    assert environment["ORARL_EVAL_ASSET_ROOT"] == str(asset_root.resolve())
    assert environment["VSI_PREPROCESSED_VIDEO_DIR"] == str(
        (asset_root / "media/spatial_intelligence/vsi/videos").resolve()
    )


def test_temporal_grounding_accepts_per_split_preprocessing_profiles(
    tmp_path: Path,
) -> None:
    datasets = "charades_timelens,activitynet_timelens,qvhighlights_timelens"
    root = _canonical_root(
        tmp_path,
        [
            {
                "task": "temporal_grounding",
                "split": split,
                "preprocessing": {
                    "fps": fps,
                    "min_tokens": 1,
                    "max_frames": 2048,
                    "max_pixels": 409600,
                    "total_tokens": 128000,
                },
                "legacy_environment": {"TIMELENS_DATASETS": datasets},
            }
            for split, fps in (
                ("charades_timelens", 4),
                ("activitynet_timelens", 2),
                ("qvhighlights_timelens", 2),
            )
        ],
    )
    namespace = evaluate.create_parser().parse_args(
        _canonical_arguments(tmp_path, str(root), tasks="temporal_grounding")
    )

    _, environment, _, _, _ = evaluate.build_launch(namespace)

    assert environment["TIMELENS_DATASETS"] == datasets
    assert environment["TIMELENS_BENCH_DIR"] == str(
        (root / "annotations/temporal_grounding").resolve()
    )


def test_temporal_grounding_can_select_sta_for_a_bounded_smoke_test(
    tmp_path: Path,
) -> None:
    datasets = "charades_timelens,activitynet_timelens,qvhighlights_timelens"
    root = _canonical_root(
        tmp_path,
        [
            {
                "task": "temporal_grounding",
                "split": split,
                "preprocessing": {
                    "fps": fps,
                    "min_tokens": 1,
                    "max_frames": 2048,
                    "max_pixels": 409600,
                    "total_tokens": 128000,
                },
                "legacy_environment": {"TIMELENS_DATASETS": datasets},
            }
            for split, fps in (
                ("charades_timelens", 4),
                ("activitynet_timelens", 2),
                ("qvhighlights_timelens", 2),
            )
        ],
    )
    namespace = evaluate.create_parser().parse_args(
        [
            *_canonical_arguments(
                tmp_path,
                str(root),
                tasks="temporal_grounding",
            ),
            "--splits",
            "charades_timelens",
            "--max-samples",
            "8",
        ]
    )

    _, environment, _, _, _ = evaluate.build_launch(namespace)

    assert environment["TIMELENS_DATASETS"] == "charades_timelens"
    assert environment["TIMELENS_MAX_SAMPLES"] == "8"
    assert environment["TIMELENS_FPS"] == "4"
    assert environment["TIMELENS_MIN_TOKENS"] == "1"
    assert environment["TIMELENS_MAX_FRAMES"] == "2048"
    assert environment["TIMELENS_MAX_PIXELS"] == "409600"
    assert environment["TIMELENS_TOTAL_TOKENS"] == "128000"


def test_spatial_grounding_maps_canonical_split_to_evaluator_name(
    tmp_path: Path,
) -> None:
    evaluator_datasets = "refcoco-val,refcoco+-testA"
    root = _canonical_root(
        tmp_path,
        [
            {
                "task": "spatial_grounding",
                "split": split,
                "preprocessing": {"coordinate_system": "norm1000"},
                "legacy_environment": {
                    "SPATIAL_GROUNDING_DATASETS": evaluator_datasets,
                },
            }
            for split in ("refcoco_val", "refcocop_test_a")
        ],
    )
    namespace = evaluate.create_parser().parse_args(
        [
            *_canonical_arguments(
                tmp_path,
                str(root),
                tasks="spatial_grounding",
            ),
            "--splits",
            "refcocop_test_a",
            "--max-samples",
            "16",
        ]
    )

    _, environment, _, _, _ = evaluate.build_launch(namespace)

    assert environment["SPATIAL_GROUNDING_DATASETS"] == "refcoco+-testA"
    assert environment["SPATIAL_GROUNDING_MAX_SAMPLES"] == "16"
    assert environment["SPATIAL_GROUNDING_BENCH_DIR"] == str(
        (root / "annotations/spatial_grounding").resolve()
    )


def test_tracking_maps_canonical_split_to_evaluator_name(tmp_path: Path) -> None:
    root = _canonical_root(
        tmp_path,
        [
            {
                "task": "tracking",
                "split": "got10k",
                "preprocessing": {
                    "tracking_fps": 1,
                    "tracking_max_frames": 32,
                },
                "legacy_environment": {
                    "TRACKING_DATASETS": "eval_got10k",
                },
            }
        ],
    )
    namespace = evaluate.create_parser().parse_args(
        [
            *_canonical_arguments(tmp_path, str(root), tasks="tracking"),
            "--max-samples",
            "8",
        ]
    )

    _, environment, _, _, _ = evaluate.build_launch(namespace)

    assert environment["TRACKING_DATASETS"] == "eval_got10k"
    assert environment["TRACKING_MAX_SAMPLES"] == "8"
    assert environment["TRACKING_BENCH_DIR"] == str(
        (root / "annotations/tracking").resolve()
    )


def test_stvg_maps_canonical_split_and_smoke_limit(tmp_path: Path) -> None:
    root = _canonical_root(
        tmp_path,
        [
            {
                "task": "stvg",
                "split": "stvg",
                "preprocessing": {
                    "stvg_fps": 2,
                    "stvg_max_frames": 128,
                },
                "legacy_environment": {
                    "STVG_DATASETS": "eval_stvg",
                },
            }
        ],
    )
    namespace = evaluate.create_parser().parse_args(
        [
            *_canonical_arguments(tmp_path, str(root), tasks="stvg"),
            "--max-samples",
            "8",
        ]
    )

    _, environment, _, _, _ = evaluate.build_launch(namespace)

    assert environment["STVG_DATASETS"] == "eval_stvg"
    assert environment["STVG_MAX_SAMPLES"] == "8"
    assert environment["STVG_BENCH_DIR"] == str(
        (root / "annotations/stvg").resolve()
    )


def test_canonical_dry_run_needs_only_valid_manifests(tmp_path: Path) -> None:
    root = _canonical_root(tmp_path, [{"task": "vsi"}])
    for path in (root / "annotations").rglob("*"):
        if path.is_file():
            path.unlink()
    for path in (root / "media").rglob("*"):
        if path.is_file():
            path.unlink()
    namespace = evaluate.create_parser().parse_args(
        _canonical_arguments(tmp_path, str(root))
    )

    _, environment, _, _, _ = evaluate.build_launch(namespace)

    assert environment["VSI_DATA_FILE"].endswith(
        "annotations/spatial_intelligence/vsi/test.jsonl"
    )


def test_canonical_hf_dataset_uses_manifest_only_snapshot_for_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _canonical_root(tmp_path, [{"task": "vsi"}])
    calls: list[dict[str, object]] = []
    module = types.ModuleType("huggingface_hub")

    def snapshot_download(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        return str(root)

    module.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    cache_dir = tmp_path / "hf-cache"
    arguments = _canonical_arguments(tmp_path, "org/evaluation-data")
    arguments.extend(
        (
            "--revision",
            "release",
            "--cache-dir",
            str(cache_dir),
            "--local-files-only",
        )
    )
    namespace = evaluate.create_parser().parse_args(arguments)

    evaluate.build_launch(namespace)

    assert calls == [
        {
            "repo_id": "org/evaluation-data",
            "repo_type": "dataset",
            "revision": "release",
            "cache_dir": str(cache_dir),
            "local_files_only": True,
            "allow_patterns": ["datasets.jsonl", "assets.jsonl"],
        }
    ]


def test_canonical_hf_run_downloads_only_selected_task_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _canonical_root(
        tmp_path,
        [{"task": "vsi"}, {"task": "mmsi"}],
    )
    calls: list[dict[str, object]] = []
    module = types.ModuleType("huggingface_hub")

    def snapshot_download(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        return str(root)

    module.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    namespace = evaluate.create_parser().parse_args(
        [
            *_canonical_arguments(tmp_path, "org/evaluation-data"),
            "--run",
        ]
    )

    evaluate.build_launch(namespace)

    assert len(calls) == 2
    selected = calls[1]["allow_patterns"]
    assert isinstance(selected, list)
    assert "annotations/spatial_intelligence/vsi/test.jsonl" in selected
    assert any(
        str(path).startswith("media/spatial_intelligence/vsi/")
        for path in selected
    )
    assert not any(
        str(path).startswith("media/spatial_intelligence/mmsi/")
        for path in selected
    )


def test_canonical_dataset_requires_task_coverage(tmp_path: Path) -> None:
    root = _canonical_root(tmp_path, [{"task": "vsi"}])
    namespace = evaluate.create_parser().parse_args(
        _canonical_arguments(tmp_path, str(root), "vsi,mmsi")
    )

    with pytest.raises(evaluate.CliError, match="no configuration record.*mmsi"):
        evaluate.build_launch(namespace)


def test_canonical_dataset_rejects_conflicting_structured_profiles(
    tmp_path: Path,
) -> None:
    root = _canonical_root(
        tmp_path,
        [
            {
                "task": "tracking",
                "split": "got10k",
                "evaluation": {"metric_profile": "one"},
            },
            {
                "task": "tracking",
                "split": "lasot",
                "evaluation": {"metric_profile": "two"},
            },
        ],
    )
    namespace = evaluate.create_parser().parse_args(
        _canonical_arguments(tmp_path, str(root), "tracking")
    )

    with pytest.raises(evaluate.CliError, match="conflicting canonical evaluation"):
        evaluate.build_launch(namespace)


def test_canonical_structured_records_merge_identical_profiles(tmp_path: Path) -> None:
    root = _canonical_root(
        tmp_path,
        [
            {"task": "tracking", "split": "got10k"},
            {"task": "tracking", "split": "lasot"},
        ],
    )
    namespace = evaluate.create_parser().parse_args(
        _canonical_arguments(tmp_path, str(root), "tracking")
    )

    _, environment, _, _, _ = evaluate.build_launch(namespace)

    assert environment["TRACKING_DATASETS"] == "eval_got10k,lasot"
    assert environment["TRACKING_BENCH_DIR"] == str(
        (root / "annotations/tracking").resolve()
    )
    assert environment["TRACKING_BASE_PREFIX"] == str(root.resolve())


def test_canonical_dataset_rejects_unrelated_profile_settings(tmp_path: Path) -> None:
    root = _canonical_root(
        tmp_path,
        [
            {
                "task": "mmsi",
                "legacy_environment": {
                    "VSI_DATA_FILE": "annotations/spatial_intelligence/mmsi/test.jsonl",
                },
            }
        ],
    )
    namespace = evaluate.create_parser().parse_args(
        _canonical_arguments(tmp_path, str(root), "mmsi")
    )

    with pytest.raises(evaluate.CliError, match="unrelated canonical profile"):
        evaluate.build_launch(namespace)


def test_canonical_run_verifies_asset_sizes_without_rehashing(tmp_path: Path) -> None:
    root = _canonical_root(tmp_path, [{"task": "vsi"}])
    video = next((root / "media/spatial_intelligence/vsi/videos").iterdir())
    video.write_bytes(b"bad")
    namespace = evaluate.create_parser().parse_args(
        [*_canonical_arguments(tmp_path, str(root)), "--run"]
    )

    with pytest.raises(evaluate.CliError, match="byte-size mismatch"):
        evaluate.build_launch(namespace)


def test_canonical_run_validates_separate_asset_root(tmp_path: Path) -> None:
    root = _canonical_root(tmp_path, [{"task": "vsi"}])
    asset_root = tmp_path / "external-assets"
    asset_root.mkdir()
    (root / "assets.jsonl").rename(asset_root / "assets.jsonl")
    (root / "media").rename(asset_root / "media")
    namespace = evaluate.create_parser().parse_args(
        [
            *_canonical_arguments(tmp_path, str(root)),
            "--asset-root",
            str(asset_root),
            "--run",
        ]
    )

    _, environment, _, _, _ = evaluate.build_launch(namespace)

    assert environment["ORARL_EVAL_ASSET_ROOT"] == str(asset_root.resolve())


def test_dataset_and_data_root_are_mutually_exclusive(tmp_path: Path) -> None:
    root = _canonical_root(tmp_path, [{"task": "vsi"}])
    arguments = _canonical_arguments(tmp_path, str(root))
    legacy_root = tmp_path / "legacy-root"
    legacy_root.mkdir()
    arguments.extend(("--data-root", str(legacy_root)))
    namespace = evaluate.create_parser().parse_args(arguments)

    with pytest.raises(evaluate.CliError, match="mutually exclusive"):
        evaluate.build_launch(namespace)


def test_asset_root_requires_canonical_dataset(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    namespace = evaluate.create_parser().parse_args(
        [*_base_arguments(tmp_path), "--asset-root", str(asset_root)]
    )

    with pytest.raises(evaluate.CliError, match="requires --dataset"):
        evaluate.build_launch(namespace)


def test_split_selection_requires_canonical_dataset(tmp_path: Path) -> None:
    namespace = evaluate.create_parser().parse_args(
        [*_base_arguments(tmp_path), "--splits", "test"]
    )

    with pytest.raises(evaluate.CliError, match="requires --dataset"):
        evaluate.build_launch(namespace)


def test_run_rejects_missing_conventional_inputs(tmp_path: Path) -> None:
    namespace = evaluate.create_parser().parse_args([*_base_arguments(tmp_path), "--run"])
    with pytest.raises(evaluate.CliError, match="input does not exist"):
        evaluate.build_launch(namespace)


def test_sam2_run_requires_explicit_postprocessor_files(tmp_path: Path) -> None:
    arguments = _base_arguments(tmp_path, "segmentation")
    data_root = tmp_path / "benchmarks"
    (data_root / "segmentation").mkdir()
    for name in ("sam2.yaml", "sam2.pt", "postprocess.py"):
        (data_root / name).touch()
    config = tmp_path / "segmentation.yaml"
    config.write_text(
        "task: segmentation\n"
        "environment:\n"
        "  SEGMENTATION_SAM2_CFG: sam2.yaml\n"
        "  SEGMENTATION_SAM2_CKPT: sam2.pt\n"
        "  SEGMENTATION_POSTPROCESSOR_PATH: postprocess.py\n",
        encoding="utf-8",
    )
    arguments.extend(
        (
            "--task-config",
            f"segmentation={config}",
            "--segmentation-run-sam2",
            "--run",
        )
    )
    namespace = evaluate.create_parser().parse_args(arguments)

    command, environment, _, _, _ = evaluate.build_launch(namespace)

    assert "--segmentation-run-sam2" in command
    assert environment["SEGMENTATION_POSTPROCESSOR_PATH"] == str(data_root / "postprocess.py")


def test_aggregate_embeds_new_official_summary(tmp_path: Path) -> None:
    results_root = tmp_path / "outputs"
    task_output = (
        results_root
        / "model"
        / "spatial_intelligence"
        / "vsi"
        / "base"
        / "setting"
        / "run"
    )
    task_output.mkdir(parents=True)
    (task_output / "summary.json").write_text(
        json.dumps({"accuracy": 42.0}),
        encoding="utf-8",
    )
    aggregate = tmp_path / "aggregate.json"

    evaluate._write_aggregate(
        aggregate,
        model=str(tmp_path / "model"),
        tasks=["vsi"],
        results_root=results_root,
        before={},
        returncode=0,
    )

    payload = json.loads(aggregate.read_text(encoding="utf-8"))
    assert payload["completed_tasks"] == ["vsi"]
    assert payload["missing_tasks"] == []
    assert payload["results"][0]["metrics"] == {"accuracy": 42.0}
