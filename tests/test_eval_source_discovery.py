from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

RELEASE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = RELEASE_ROOT / "scripts" / "create_eval_source_manifest.py"
SPEC = importlib.util.spec_from_file_location("orarl_eval_source_discovery", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
DISCOVERY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DISCOVERY
SPEC.loader.exec_module(DISCOVERY)


def _write(path: Path, content: str = "[]\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _fixture_roots(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "licensed-data"
    runtime_root = tmp_path / "runtime"
    valid_data = runtime_root / "eval" / "data" / "valid_data"
    valid_data.mkdir(parents=True)

    for folder in (
        "Video-MME",
        "Video-MME-v2",
        "MVBench",
        "MMVU",
        "Video-Holmes",
        "LongVideoBench",
        "MLVU",
        "VSI-Bench",
        "MMSI-Bench",
        "MindCube-Tiny",
        "ReVSI",
        "Spatial-Grounding",
        "OneThinker-eval",
        "TimeLens-Bench",
        "preprocessed_videos",
        "qvhighlights-videos",
    ):
        (data_root / folder).mkdir(parents=True)
    (data_root / "Video-MME" / "data").mkdir()
    (data_root / "Video-MME" / "preprocessed_videos_384f_262k_total0").mkdir()
    (data_root / "Video-MME-v2" / "preprocessed_videos_384f_262k_total0").mkdir()
    (data_root / "MVBench" / "videos").mkdir()
    (data_root / "OneThinker-eval" / "Refcoco").mkdir()
    (data_root / "TimeLens-Bench" / "video_shards" / "charades").mkdir(parents=True)
    (data_root / "TimeLens-Bench" / "video_shards" / "activitynet").mkdir()
    (data_root / "TimeLens-Bench" / "video_shards" / "qvhighlights").mkdir()

    for name in (
        "videomme_preprocessed_384f_262k_total0.jsonl",
        "videommev2_preprocessed_384f_262k_total0.jsonl",
        "mvbench.json",
        "mmvu_mc.jsonl",
        "videoholmes.jsonl",
        "longvideobench_val.jsonl",
        "mlvu_mc.jsonl",
        "vsibench_preprocessed_128f_16M.jsonl",
    ):
        _write(valid_data / name, "{}\n" if name.endswith(".jsonl") else "[]\n")

    _write(data_root / "MMSI-Bench" / "MMSI_bench.tsv", "id\tquestion\tanswer\timage\n")
    _write(data_root / "MindCube-Tiny" / "combined-00000-of-00001.parquet", "")
    _write(data_root / "ReVSI" / "all_frame" / "test-00000-of-00001.parquet", "")
    for name in (
        "refcoco_val.json",
        "refcoco_testA.json",
        "refcoco_testB.json",
        "refcocop_val.json",
        "refcocop_testA.json",
        "refcocop_testB.json",
        "refcocog_val.json",
        "refcocog_test.json",
    ):
        _write(
            runtime_root
            / "eval"
            / "task"
            / "spatial_grounding"
            / "rec_jsons_processed"
            / name
        )
    for name in (
        "eval_got10k.json",
        "eval_stvg.json",
        "eval_seg_refcoco.json",
        "eval_seg_refcocop.json",
        "eval_seg_refcocog.json",
        "eval_seg_mevis.json",
        "eval_seg_reasonvos.json",
    ):
        _write(data_root / "OneThinker-eval" / name)
    for name in (
        "charades-timelens.json",
        "activitynet-timelens.json",
        "qvhighlights-timelens.json",
    ):
        _write(data_root / "TimeLens-Bench" / name, "{}\n")

    for excluded in ("VideoMMMU", "LVBench", "outputs", "training"):
        directory = data_root / excluded
        directory.mkdir()
        _write(directory / "annotations.jsonl", "{}\n")
    return data_root, runtime_root


def test_discovery_covers_only_paper_tasks_and_defaults_to_unauthorized(
    tmp_path: Path,
) -> None:
    data_root, runtime_root = _fixture_roots(tmp_path)

    records = DISCOVERY.discover_eval_sources(data_root, runtime_root)

    assert {record["eval_task"] for record in records} == set(DISCOVERY.PAPER_TASKS)
    assert len({record["eval_task"] for record in records}) == 16
    assert all(record["benchmark"] == record["eval_task"] for record in records)
    assert all(record["redistribution_authorized"] is False for record in records)
    assert all(str(record["split"]) == str(record["split"]).casefold() for record in records)
    serialized = "\n".join(json.dumps(record, sort_keys=True) for record in records)
    for excluded in ("VideoMMMU", "LVBench", "outputs", "training"):
        assert excluded not in serialized
    temporal = [
        record for record in records if record["eval_task"] == "temporal_grounding"
    ]
    assert {record["split"] for record in temporal} == {
        "activitynet_timelens",
        "charades_timelens",
        "qvhighlights_timelens",
    }
    qvhighlights = next(
        record for record in temporal if record["split"] == "qvhighlights_timelens"
    )
    assert qvhighlights["media_roots"]["videos"].endswith(
        "TimeLens-Bench/video_shards/qvhighlights"
    )
    charades = next(record for record in temporal if record["split"] == "charades_timelens")
    assert charades["preprocessing"]["fps"] == 4
    for record in temporal:
        assert record["preprocessing"] == {
            "fps": 4,
            "min_tokens": 1,
            "max_frames": 2048,
            "max_pixels": 409600,
            "total_tokens": 128000,
        }
        assert record["legacy_environment"]["TIMELENS_NUM_WORKERS"] == 2
    grounding = [
        record for record in records if record["eval_task"] == "spatial_grounding"
    ]
    assert all(
        "/eval/task/spatial_grounding/rec_jsons_processed/"
        in record["annotation_input"]
        for record in grounding
    )
    assert {
        record["split"] for record in grounding
    } == {
        "refcoco_val",
        "refcoco_test_a",
        "refcoco_test_b",
        "refcocop_val",
        "refcocop_test_a",
        "refcocop_test_b",
        "refcocog_val",
        "refcocog_test",
    }
    assert {
        record["legacy_environment"]["SPATIAL_GROUNDING_DATASETS"]
        for record in grounding
    } == {
        (
            "refcoco-val,refcoco-testA,refcoco-testB,"
            "refcoco+-val,refcoco+-testA,refcoco+-testB,"
            "refcocog-val,refcocog-test"
        )
    }
    assert all(
        record["media_roots"]["images"].endswith("OneThinker-eval/Refcoco")
        for record in grounding
    )
    videomme = next(record for record in records if record["eval_task"] == "videomme")
    assert videomme["media_roots"]["videos"].endswith("Video-MME/data")
    mvbench = next(record for record in records if record["eval_task"] == "mvbench")
    assert mvbench["media_roots"]["videos"].endswith("licensed-data/MVBench")
    mindcube = next(record for record in records if record["eval_task"] == "mindcube")
    assert mindcube["evaluation"] == {
        "prompt_profile": "mindcube_official",
        "parser_profile": "multiple_choice",
        "metric_profile": "micro_accuracy",
        "aggregation": "micro",
        "expected_group_counts": {
            "rotation": 200,
            "among": 600,
            "around": 250,
        },
    }
    assert mindcube["source_url"].endswith(
        "tree/7dd2725d9bd4149f2aad00a9843f72a3824da003"
    )
    revsi = next(record for record in records if record["eval_task"] == "revsi")
    assert revsi["annotation_input"].endswith(
        "ReVSI/all_frame/test-00000-of-00001.parquet"
    )
    assert revsi["media_roots"]["videos"].endswith("ReVSI/all_frame")
    assert revsi["evaluation"]["frame_protocol"] == "native_all_frame"
    assert revsi["legacy_environment"]["REVSI_SETTING"].startswith("native-all-f128")
    assert revsi["legacy_environment"]["REVSI_EXACT_NFRAMES"] is True
    assert revsi["legacy_environment"]["REVSI_EXPECTED_SAMPLES"] == 6808
    tracking = next(
        record for record in records if record["eval_task"] == "tracking"
    )
    assert tracking["split"] == "got10k"
    assert tracking["legacy_environment"]["TRACKING_DATASETS"] == "eval_got10k"
    assert tracking["media_roots"]["default"].endswith("OneThinker-eval")
    stvg = next(record for record in records if record["eval_task"] == "stvg")
    assert stvg["split"] == "stvg"
    assert stvg["legacy_environment"]["STVG_DATASETS"] == "eval_stvg"
    assert stvg["media_roots"]["default"].endswith("OneThinker-eval")
    segmentation = [
        record for record in records if record["eval_task"] == "segmentation"
    ]
    assert {record["split"] for record in segmentation} == {
        "refcoco",
        "refcocop",
        "refcocog",
        "mevis",
        "reasonvos",
    }
    assert {
        record["split"]: record["expected_count"] for record in segmentation
    } == {
        "mevis": 424,
        "reasonvos": 458,
        "refcoco": 3811,
        "refcocog": 2537,
        "refcocop": 3805,
    }
    assert all(
        record["legacy_environment"]["SEGMENTATION_BATCH_SIZE"] == 16
        and record["legacy_environment"]["SEGMENTATION_MAX_PIXELS_IMAGE"] == 1048576
        and record["legacy_environment"]["SEGMENTATION_VIDEO_READER"] == "decord"
        and record["legacy_environment"]["SEGMENTATION_RUN_SAM2"] is False
        and record["preprocessing"]["video_reader"] == "decord"
        for record in segmentation
    )


def test_authorization_requires_the_explicit_confirmation_flag(tmp_path: Path) -> None:
    data_root, runtime_root = _fixture_roots(tmp_path)
    output = tmp_path / "private" / "sources.jsonl"

    assert (
        DISCOVERY.main(
            [
                "--data-root",
                str(data_root),
                "--runtime-root",
                str(runtime_root),
                "--output",
                str(output),
                "--confirm-redistribution-authorized",
            ]
        )
        == 0
    )
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    assert records
    assert all(record["redistribution_authorized"] is True for record in records)


def test_discovery_fails_on_ambiguous_required_annotation(tmp_path: Path) -> None:
    data_root, runtime_root = _fixture_roots(tmp_path)
    _write(data_root / "MMSI-Bench" / "duplicate" / "MMSI_bench.tsv", "id\n")

    with pytest.raises(DISCOVERY.DiscoveryError, match="ambiguous mmsi annotation"):
        DISCOVERY.discover_eval_sources(data_root, runtime_root)


def test_tsv_count_accepts_large_embedded_image_fields(tmp_path: Path) -> None:
    annotation = _write(
        tmp_path / "mmsi.tsv",
        f"id\timage\nsample-1\t{'a' * 200_000}\n",
    )

    assert DISCOVERY._cheap_expected_count(annotation) == 1


def test_mindcube_discovery_accepts_one_verified_official_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_subset = _write(tmp_path / "data" / "test.parquet", "")
    official = _write(tmp_path / "official.parquet", "")
    counts = {old_subset: 120, official: 1050}
    monkeypatch.setattr(DISCOVERY, "_cheap_expected_count", counts.__getitem__)

    assert DISCOVERY._mindcube_annotation(tmp_path) == official.resolve()


def test_mindcube_discovery_rejects_only_reduced_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reduced = _write(tmp_path / "data" / "test.parquet", "")
    monkeypatch.setattr(DISCOVERY, "_cheap_expected_count", lambda _path: 120)

    with pytest.raises(DISCOVERY.DiscoveryError, match="requires 1050 rows"):
        DISCOVERY._mindcube_annotation(tmp_path)

    assert reduced.is_file()
