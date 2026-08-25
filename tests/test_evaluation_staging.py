from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

import pytest

import orarl.evaluation.converters as evaluation_converters
import orarl.evaluation.layout as evaluation_layout
import orarl.evaluation.staging as evaluation_staging
from orarl.cli import eval_data
from orarl.evaluation import (
    StagingError,
    build_evaluation_repository,
    inventory_evaluation_sources,
    load_asset_manifest,
    load_evaluation_jsonl,
    load_source_manifest,
    merge_evaluation_repository,
    validate_staged_repository,
)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _source_record(
    annotation: Path,
    *,
    benchmark: str,
    eval_task: str,
    family: str,
    adapter: str = "generic",
    expected_count: int = 1,
    **roots: Path,
) -> dict[str, object]:
    record: dict[str, object] = {
        "benchmark": benchmark,
        "eval_task": eval_task,
        "split": "test",
        "family": family,
        "adapter": adapter,
        "annotation_input": str(annotation),
        "expected_count": expected_count,
        "license": "fixture-only",
        "source_url": f"https://example.org/{benchmark}",
        "redistribution_authorized": True,
        "evaluation": {
            "prompt_profile": "fixture",
            "parser_profile": "fixture",
            "metric_profile": "fixture",
        },
        "preprocessing": {"fps": 2, "max_frames": 32},
    }
    record.update({key: str(value) for key, value in roots.items()})
    return record


def _write_manifest(path: Path, records: list[dict[str, object]]) -> Path:
    _write_jsonl(path, records)
    return path


def _generic_fixture(tmp_path: Path) -> tuple[Path, Path, bytes]:
    video_root = tmp_path / "source-videos"
    subtitle_root = tmp_path / "source-subtitles"
    preprocessed_root = tmp_path / "source-artifacts"
    video_root.mkdir()
    subtitle_root.mkdir()
    preprocessed_root.mkdir()
    video_bytes = b"fixture-video"
    (video_root / "clip.mp4").write_bytes(video_bytes)
    (subtitle_root / "clip.srt").write_text("fixture subtitle\n", encoding="utf-8")
    (preprocessed_root / "clip.npz").write_bytes(b"fixture-array")

    annotation = tmp_path / "video-qa.json"
    annotation.write_text(
        json.dumps(
            [
                {
                    "question_id": "q-1",
                    "question": "What happens?",
                    "answer": "B",
                    "options": ["A. Nothing", "B. Motion"],
                    "path": "clip.mp4",
                    "image": "",
                    "video": "",
                    "subtitles": "",
                    "subtitle_path": "clip.srt",
                    "artifact_path": "",
                    "preprocessed_video": "clip.npz",
                    "preprocessed_video_settings": {"frames": 32},
                    "question_category": "motion",
                }
            ]
        ),
        encoding="utf-8",
    )
    source = _source_record(
        annotation,
        benchmark="longvideobench",
        eval_task="longvideobench",
        family="video_qa",
        video_root=video_root,
        subtitle_root=subtitle_root,
        preprocessed_root=preprocessed_root,
    )
    source["legacy_environment"] = {
        "LONGVIDEOBENCH_DATA_FILE": str(annotation),
        "LONGVIDEOBENCH_VIDEO_ROOT": str(video_root),
        "LONGVIDEOBENCH_SUBTITLE_ROOT": str(subtitle_root),
    }
    manifest = _write_manifest(tmp_path / "sources.jsonl", [source])
    return manifest, video_root / "clip.mp4", video_bytes


def test_generic_build_stages_subtitles_artifacts_and_is_deterministic(
    tmp_path: Path,
) -> None:
    manifest, source_video, video_bytes = _generic_fixture(tmp_path)
    copied = tmp_path / "copied"
    linked = tmp_path / "linked"

    copy_summary = build_evaluation_repository(
        manifest,
        copied,
        copy_mode="copy",
        checksums=True,
    )
    link_summary = build_evaluation_repository(
        manifest,
        linked,
        copy_mode="hardlink",
        checksums=True,
    )

    for relative in (
        "datasets.jsonl",
        "assets.jsonl",
        "annotations/video_qa/longvideobench/test.jsonl",
    ):
        assert (copied / relative).read_bytes() == (linked / relative).read_bytes()
    assert copy_summary["datasets"] == 1
    assert copy_summary["rows"] == 1
    assert link_summary["hardlinked"] == 3

    digest = hashlib.sha256(video_bytes).hexdigest()
    video_path = f"media/video_qa/longvideobench/videos/{digest}.mp4"
    assert (copied / video_path).read_bytes() == video_bytes
    assert os.stat(source_video).st_ino == os.stat(linked / video_path).st_ino

    rows = load_evaluation_jsonl(
        copied / "annotations/video_qa/longvideobench/test.jsonl"
    )
    row = rows[0]
    assert row["videos"] == [video_path]
    assert row["subtitles"][0].startswith(
        "media/video_qa/longvideobench/subtitles/"
    )
    assert row["preprocessed"]["preprocessed_video"].startswith(
        "artifacts/video_qa/longvideobench/"
    )
    assert row["metadata"]["preprocessed_video_settings"] == {"frames": 32}
    assert row["choices"] == ["A. Nothing", "B. Motion"]
    assert row["metadata"]["question_category"] == "motion"
    assert row["task_payload"] == {}

    serialized = (copied / "datasets.jsonl").read_text(encoding="utf-8")
    serialized += (
        copied / "annotations/video_qa/longvideobench/test.jsonl"
    ).read_text(
        encoding="utf-8"
    )
    assert str(tmp_path) not in serialized
    assert not any(path.is_symlink() for path in copied.rglob("*"))
    assert validate_staged_repository(copied) == {
        key: copy_summary[key]
        for key in ("schema_version", "datasets", "rows", "assets", "bytes")
    }


def test_mvbench_prefers_materialized_path_over_upstream_video_name(
    tmp_path: Path,
) -> None:
    video_root = tmp_path / "MVBench"
    materialized = video_root / "ssv2_video" / "166583.mp4"
    materialized.parent.mkdir(parents=True)
    materialized.write_bytes(b"mvbench-video")
    annotation = tmp_path / "mvbench.json"
    annotation.write_text(
        json.dumps(
            [
                {
                    "problem_id": 1,
                    "question": "What happens?",
                    "options": ["A. Nothing", "B. Motion"],
                    "answer": "Motion",
                    "solution": "<answer>B</answer>",
                    "data_type": "video",
                    "video": "166583.webm",
                    "path": "./ssv2_video/166583.mp4",
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest = _write_manifest(
        tmp_path / "sources.jsonl",
        [
            _source_record(
                annotation,
                benchmark="mvbench",
                eval_task="mvbench",
                family="video_qa",
                video_root=video_root,
            )
        ],
    )
    output = tmp_path / "staged"

    report = inventory_evaluation_sources(manifest, checksums=False)
    summary = build_evaluation_repository(
        manifest,
        output,
        copy_mode="hardlink",
        checksums=False,
    )

    assert report["totals"]["missing_assets"] == 0
    assert report["totals"]["existing_assets"] == 1
    assert summary["hardlinked"] == 1
    rows = load_evaluation_jsonl(
        output / "annotations/video_qa/mvbench/test.jsonl"
    )
    assert rows[0]["videos"] == [
        "media/video_qa/mvbench/videos/test/ssv2_video__166583.mp4"
    ]
    assert rows[0]["answer"] == "B"


def test_hardlink_falls_back_to_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _, _ = _generic_fixture(tmp_path)

    def _fail_link(*args: object, **kwargs: object) -> None:
        raise OSError("fixture cross-device link")

    monkeypatch.setattr(os, "link", _fail_link)
    summary = build_evaluation_repository(
        manifest,
        tmp_path / "fallback",
        copy_mode="hardlink",
    )
    assert summary["hardlinked"] == 0
    assert summary["copied"] == 3


def test_checksum_free_build_never_reads_asset_contents_for_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _, _ = _generic_fixture(tmp_path)
    output = tmp_path / "checksum-free"

    def _fail_hash(_path: Path) -> str:
        pytest.fail("checksum-free build must not calculate SHA-256")

    monkeypatch.setattr(evaluation_converters, "sha256_file", _fail_hash)
    monkeypatch.setattr(evaluation_staging, "sha256_file", _fail_hash)

    summary = build_evaluation_repository(
        manifest,
        output,
        copy_mode="hardlink",
        checksums=False,
    )

    assert summary["asset_checksums"] is False
    assert (
        output / "media/video_qa/longvideobench/videos/test/clip.mp4"
    ).read_bytes() == b"fixture-video"
    assert all("sha256" not in record for record in load_asset_manifest(output))
    assert validate_staged_repository(output)["datasets"] == 1


def test_checksum_free_build_shortens_long_asset_filenames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_root = tmp_path / "videos"
    video_root.mkdir()
    long_name = f"{'long-video-title-' * 14}clip.mp4"
    source_video = video_root / long_name
    source_video.write_bytes(b"long-name-video")
    annotation = tmp_path / "longvideobench.json"
    annotation.write_text(
        json.dumps(
            [
                {
                    "question_id": "q-1",
                    "question": "What happens?",
                    "options": ["A. Nothing", "B. Motion"],
                    "answer": "B",
                    "data_type": "video",
                    "path": long_name,
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest = _write_manifest(
        tmp_path / "sources.jsonl",
        [
            _source_record(
                annotation,
                benchmark="longvideobench",
                eval_task="longvideobench",
                family="video_qa",
                video_root=video_root,
            )
        ],
    )
    output = tmp_path / "staged"

    def _fail_hash(_path: Path) -> str:
        pytest.fail("checksum-free build must not calculate SHA-256")

    monkeypatch.setattr(evaluation_converters, "sha256_file", _fail_hash)
    monkeypatch.setattr(evaluation_staging, "sha256_file", _fail_hash)

    build_evaluation_repository(
        manifest,
        output,
        copy_mode="hardlink",
        checksums=False,
    )

    staged = next(
        (output / "media/video_qa/longvideobench/videos/test").iterdir()
    )
    assert len(staged.name.encode("utf-8")) <= 220
    assert "--" in staged.name
    assert staged.suffix == ".mp4"
    assert staged.read_bytes() == b"long-name-video"


def test_incremental_merge_repairs_missing_target_asset_manifest(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_manifest, _, _ = _generic_fixture(first_root)
    second_manifest, _, _ = _generic_fixture(second_root)
    second_record = json.loads(second_manifest.read_text(encoding="utf-8"))
    second_record.update(
        {
            "benchmark": "mvbench",
            "eval_task": "mvbench",
            "source_url": "https://example.org/mvbench",
        }
    )
    _write_jsonl(second_manifest, [second_record])

    target = tmp_path / "cumulative"
    source = tmp_path / "next-task"
    build_evaluation_repository(first_manifest, target, copy_mode="hardlink")
    build_evaluation_repository(second_manifest, source, copy_mode="hardlink")
    (target / "assets.jsonl").unlink()

    summary = merge_evaluation_repository(
        source,
        target,
        repair_missing_asset_manifest=True,
        workers=2,
    )

    assert summary["datasets"] == 2
    assert summary["rows"] == 2
    assert summary["assets"] == 8
    assert summary["hardlinked"] == 4
    assert summary["repaired_asset_manifest"] is True
    assert summary["workers"] == 2
    assert (target / "annotations/video_qa/mvbench/test.jsonl").is_file()
    assert validate_staged_repository(target)["datasets"] == 2


def test_incremental_merge_can_replace_legacy_benchmark_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _, _ = _generic_fixture(tmp_path)
    target = tmp_path / "legacy-target"
    source = tmp_path / "grouped-source"

    with monkeypatch.context() as legacy_layout:
        legacy_layout.delitem(
            evaluation_layout.BENCHMARK_GROUPS,
            "longvideobench",
        )
        build_evaluation_repository(manifest, target, copy_mode="hardlink")

    assert (target / "annotations/longvideobench/test.jsonl").is_file()
    build_evaluation_repository(manifest, source, copy_mode="hardlink")

    summary = merge_evaluation_repository(
        source,
        target,
        replace_benchmarks=["longvideobench"],
        workers=2,
    )

    assert summary["replaced_benchmarks"] == ["longvideobench"]
    assert summary["removed"] == 4
    assert not (target / "annotations/longvideobench").exists()
    assert not (target / "media/longvideobench").exists()
    assert not (target / "artifacts/longvideobench").exists()
    assert (
        target / "annotations/video_qa/longvideobench/test.jsonl"
    ).is_file()
    assert validate_staged_repository(target)["datasets"] == 1


def test_incremental_merge_replaces_changed_asset_at_the_same_path(
    tmp_path: Path,
) -> None:
    target_input = tmp_path / "target-input"
    source_input = tmp_path / "source-input"
    target_input.mkdir()
    source_input.mkdir()
    target_manifest, _, _ = _generic_fixture(target_input)
    source_manifest, _, _ = _generic_fixture(source_input)
    source_annotation = source_input / "video-qa.json"
    source_rows = json.loads(source_annotation.read_text(encoding="utf-8"))
    source_rows[0]["answer"] = "updated-answer-with-a-different-size"
    source_annotation.write_text(json.dumps(source_rows), encoding="utf-8")

    target = tmp_path / "target"
    source = tmp_path / "source"
    build_evaluation_repository(target_manifest, target, copy_mode="hardlink")
    build_evaluation_repository(source_manifest, source, copy_mode="hardlink")

    summary = merge_evaluation_repository(
        source,
        target,
        replace_benchmarks=["longvideobench"],
        workers=2,
    )

    rows = load_evaluation_jsonl(
        target / "annotations/video_qa/longvideobench/test.jsonl"
    )
    assert rows[0]["answer"] == "updated-answer-with-a-different-size"
    assert summary["hardlinked"] == 1
    assert summary["existing"] == 3
    assert summary["removed"] == 0
    assert validate_staged_repository(target)["datasets"] == 1


def test_inventory_is_read_only_and_reports_missing_assets(tmp_path: Path) -> None:
    annotation = tmp_path / "missing.json"
    annotation.write_text(
        json.dumps(
            [
                {
                    "id": "missing-1",
                    "question": "Where is the clip?",
                    "answer": "A",
                    "video": "not-there.mp4",
                }
            ]
        ),
        encoding="utf-8",
    )
    video_root = tmp_path / "video-root"
    video_root.mkdir()
    manifest = _write_manifest(
        tmp_path / "missing-sources.jsonl",
        [
            _source_record(
                annotation,
                benchmark="missing_bench",
                eval_task="video_qa",
                family="video_qa",
                video_root=video_root,
            )
        ],
    )
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    inventory = inventory_evaluation_sources(manifest, workers=2)

    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert before == after
    assert inventory["totals"]["source_rows"] == 1
    assert inventory["totals"]["rows"] == 1
    assert inventory["totals"]["referenced_assets"] == 1
    assert inventory["totals"]["missing_assets"] == 1
    target = tmp_path / "must-not-exist"
    with pytest.raises(StagingError, match="referenced asset"):
        build_evaluation_repository(manifest, target)
    assert not target.exists()


def test_repeated_asset_references_are_resolved_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _, _ = _generic_fixture(tmp_path)
    source = load_source_manifest(manifest)[0]
    planner = evaluation_converters._AssetPlanner(source, hash_assets=False)
    root = source.media_root("videos")
    original_is_file = Path.is_file
    checks = 0

    def _counted_is_file(path: Path) -> bool:
        nonlocal checks
        if path.name == "clip.mp4":
            checks += 1
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", _counted_is_file)

    for _ in range(20):
        assert planner.resolve("clip.mp4", root).name == "clip.mp4"

    assert checks == 1


def test_parallel_inventory_matches_single_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_root = tmp_path / "videos"
    video_root.mkdir()
    (video_root / "clip.mp4").write_bytes(b"video")
    records = []
    for name in ("first", "second"):
        annotation = tmp_path / f"{name}.json"
        annotation.write_text(
            json.dumps(
                [
                    {
                        "id": name,
                        "question": f"Question {name}?",
                        "answer": "A",
                        "video": "clip.mp4",
                    }
                ]
            ),
            encoding="utf-8",
        )
        records.append(
            _source_record(
                annotation,
                benchmark=name,
                eval_task="video_qa",
                family="video_qa",
                video_root=video_root,
            )
        )
    manifest = _write_manifest(tmp_path / "parallel.jsonl", records)

    sequential = inventory_evaluation_sources(manifest, workers=1)
    parallel = inventory_evaluation_sources(manifest, workers=4)

    assert parallel == sequential
    monkeypatch.setattr(
        evaluation_converters,
        "sha256_file",
        lambda _path: pytest.fail("fast inventory must not hash media"),
    )
    fast = inventory_evaluation_sources(
        manifest,
        workers=4,
        checksums=False,
    )
    assert fast["asset_checksums"] is False
    assert fast["totals"] == sequential["totals"]
    with pytest.raises(ValueError, match="workers"):
        inventory_evaluation_sources(manifest, workers=0)


def test_mmsi_embedded_images_are_decoded_to_content_addressed_files(
    tmp_path: Path,
) -> None:
    image_bytes = b"\xff\xd8\xff\xe0fixture-jpeg"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    annotation = tmp_path / "mmsi.tsv"
    annotation.write_text(
        "id\tquestion\tanswer\timage\tgroup\n"
        f"m-1\tWhich view? A. left B. right\tA\t{[encoded]!r}\trotation\n",
        encoding="utf-8",
    )
    manifest = _write_manifest(
        tmp_path / "mmsi-sources.jsonl",
        [
            _source_record(
                annotation,
                benchmark="mmsi",
                eval_task="mmsi",
                family="spatial_intelligence",
                adapter="mmsi",
            )
        ],
    )
    output = tmp_path / "mmsi-output"

    build_evaluation_repository(manifest, output, checksums=True)

    row = load_evaluation_jsonl(
        output / "annotations/spatial_intelligence/mmsi/test.jsonl"
    )[0]
    image_path = row["images"][0]
    assert image_path == (
        "media/spatial_intelligence/mmsi/images/"
        f"{hashlib.sha256(image_bytes).hexdigest()}.jpg"
    )
    assert (output / image_path).read_bytes() == image_bytes
    assert "image" not in row.get("metadata", {})


def test_timelens_nested_annotations_flatten_per_query_and_span(tmp_path: Path) -> None:
    video_root = tmp_path / "timelens-videos"
    video_root.mkdir()
    (video_root / "clip.mp4").write_bytes(b"timelens video")
    annotation = tmp_path / "timelens.json"
    annotation.write_text(
        json.dumps(
            {
                "clip": {
                    "duration": 12.5,
                    "queries": ["opens door.", "sits down"],
                    "spans": [[1.0, 2.5], [[7.0, 8.0], [9.0, 10.0]]],
                    "group": "charades",
                }
            }
        ),
        encoding="utf-8",
    )
    manifest = _write_manifest(
        tmp_path / "timelens-sources.jsonl",
        [
            _source_record(
                annotation,
                benchmark="charades_timelens",
                eval_task="temporal_grounding",
                family="temporal_grounding",
                adapter="timelens",
                expected_count=2,
                video_root=video_root,
            )
        ],
    )
    output = tmp_path / "timelens-output"

    build_evaluation_repository(manifest, output)

    rows = load_evaluation_jsonl(
        output / "annotations/charades_timelens/test.jsonl"
    )
    assert [row["sample_id"] for row in rows] == ["clip:000000", "clip:000001"]
    assert [row["problem"] for row in rows] == ["opens door", "sits down"]
    assert rows[0]["task_payload"] == {"duration": 12.5, "span": [1.0, 2.5]}
    assert rows[1]["answer"] == [[7.0, 8.0], [9.0, 10.0]]
    assert rows[0]["metadata"]["group"] == "charades"
    media_assets = [
        asset
        for asset in load_asset_manifest(output)
        if asset["kind"] == "videos"
    ]
    assert len(media_assets) == 1


def test_generic_spatial_grounding_preserves_payload_and_metadata(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "scene.jpg").write_bytes(b"fixture image")
    annotation = tmp_path / "grounding.jsonl"
    _write_jsonl(
        annotation,
        [
            {
                "id": "g-1",
                "normal_caption": "the red box",
                "problem": (
                    "Please provide the bounding box coordinate of the region "
                    "this sentence describes: the red box."
                ),
                "solution": [10.0, 20.0, 300.0, 400.0],
                "normalized_solution": [16, 47, 469, 935],
                "image": "scene.jpg",
                "dataset_group": "refcoco_val",
            }
        ],
    )
    manifest = _write_manifest(
        tmp_path / "grounding-sources.jsonl",
        [
            _source_record(
                annotation,
                benchmark="refcoco",
                eval_task="spatial_grounding",
                family="spatial_grounding",
                adapter="spatial_grounding",
                image_root=image_root,
            )
        ],
    )
    output = tmp_path / "grounding-output"

    build_evaluation_repository(manifest, output)

    row = load_evaluation_jsonl(output / "annotations/refcoco/test.jsonl")[0]
    assert row["problem"] == "the red box"
    assert row["answer"] == [16, 47, 469, 935]
    assert row["task_payload"]["normalized_solution"] == [16, 47, 469, 935]
    assert row["metadata"]["dataset_group"] == "refcoco_val"
    assert row["images"][0].startswith("media/refcoco/images/")


def test_onethinker_generic_adapter_extracts_structured_ground_truth(
    tmp_path: Path,
) -> None:
    video_root = tmp_path / "tracking-videos"
    video_root.mkdir()
    (video_root / "track.mp4").write_bytes(b"tracking video")
    annotation = tmp_path / "tracking.json"
    annotation.write_text(
        json.dumps(
            [
                {
                    "problem_id": 7,
                    "data_type": "video",
                    "problem_type": "tracking",
                    "problem": "<video>Track the target.",
                    "path": "track.mp4",
                    "solution": (
                        '<answer>{"boxes":{"1":[1,2,3,4],'
                        '"2":[2,3,4,5]}}</answer>'
                    ),
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest = _write_manifest(
        tmp_path / "tracking-sources.jsonl",
        [
            _source_record(
                annotation,
                benchmark="got10k",
                eval_task="tracking",
                family="tracking",
                adapter="one_thinker",
                video_root=video_root,
            )
        ],
    )
    output = tmp_path / "tracking-output"

    build_evaluation_repository(manifest, output)

    row = load_evaluation_jsonl(output / "annotations/got10k/test.jsonl")[0]
    assert row["task_payload"]["boxes"]["1"] == [1, 2, 3, 4]
    assert row["metadata"]["data_type"] == "video"


def test_segmentation_stages_media_hints_and_mask_oracles(tmp_path: Path) -> None:
    media_root = tmp_path / "OneThinker-eval"
    image = media_root / "Evaluation" / "Refcoco" / "cat.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"segmentation image")
    annotation = tmp_path / "segmentation.json"
    annotation.write_text(
        json.dumps(
            [
                {
                    "problem_id": 9,
                    "data_type": "image",
                    "problem_type": "segmentation",
                    "problem": "<image>Segment the cat.",
                    "path": "./Evaluation/Refcoco/cat.jpg",
                    "solution": (
                        '<answer>{"boxes":[100,100,800,800],'
                        '"positive_points":[[300,300]],'
                        '"negative_points":[[20,20]]}</answer>'
                    ),
                    "segmentation_output": {
                        "size": [2, 2],
                        "counts": [0, 1, 3],
                    },
                },
                {
                    "problem_id": 10,
                    "data_type": "image",
                    "problem_type": "segmentation",
                    "problem": "<image>Segment the unlabeled benchmark object.",
                    "path": "./Evaluation/Refcoco/cat.jpg",
                    "solution": "",
                    "segmentation_output": {
                        "frames": ["00000"],
                        "segmentation_rle": {
                            "00000": {"size": [2, 2], "counts": "13"}
                        },
                    },
                },
            ]
        ),
        encoding="utf-8",
    )
    manifest = _write_manifest(
        tmp_path / "segmentation-sources.jsonl",
        [
            _source_record(
                annotation,
                benchmark="segmentation",
                eval_task="segmentation",
                family="segmentation",
                adapter="one_thinker",
                expected_count=2,
                media_root=media_root,
            )
        ],
    )
    output = tmp_path / "segmentation-output"

    build_evaluation_repository(manifest, output)

    rows = load_evaluation_jsonl(output / "annotations/segmentation/test.jsonl")
    row = next(item for item in rows if item["sample_id"] == "9")
    assert row["problem_type"] == "segmentation"
    assert row["metadata"]["data_type"] == "image"
    assert row["task_payload"]["boxes"] == [100, 100, 800, 800]
    assert row["task_payload"]["positive_points"] == [[300, 300]]
    assert row["task_payload"]["segmentation_output"]["size"] == [2, 2]
    assert (output / row["images"][0]).read_bytes() == b"segmentation image"
    mask_only = next(item for item in rows if item["sample_id"] == "10")
    assert mask_only["answer"] is None
    assert mask_only["task_payload"]["segmentation_output"]["frames"] == ["00000"]


def test_mindcube_parquet_materializes_embedded_images_lazily(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    image_bytes = b"\x89PNG\r\n\x1a\nfixture"
    annotation = tmp_path / "mindcube.parquet"
    pd.DataFrame(
        [
            {
                "id": "rotation_1",
                "question": "Which direction?",
                "choices": ["left", "right"],
                "_".join(("gt", "answer")): 0,
                "images": [{"bytes": image_bytes, "path": None}],
            }
        ]
    ).to_parquet(annotation)
    manifest = _write_manifest(
        tmp_path / "mindcube-sources.jsonl",
        [
            _source_record(
                annotation,
                benchmark="mindcube",
                eval_task="mindcube",
                family="spatial_intelligence",
                adapter="mindcube",
            )
        ],
    )
    output = tmp_path / "mindcube-output"

    build_evaluation_repository(manifest, output)

    row = load_evaluation_jsonl(
        output / "annotations/spatial_intelligence/mindcube/test.jsonl"
    )[0]
    assert row["answer"] == 0
    assert row["choices"] == ["left", "right"]
    assert (output / row["images"][0]).read_bytes() == image_bytes
    assert "_".join(("gt", "answer")) not in row.get("metadata", {})


def test_mindcube_official_distribution_is_strict(tmp_path: Path) -> None:
    annotation = tmp_path / "mindcube.parquet"
    annotation.write_bytes(b"fixture")
    manifest = _write_manifest(
        tmp_path / "mindcube-sources.jsonl",
        [
            _source_record(
                annotation,
                benchmark="mindcube",
                eval_task="mindcube",
                family="spatial_intelligence",
                adapter="mindcube",
                expected_count=1050,
            )
        ],
    )
    source = load_source_manifest(manifest)[0]
    records = [
        {"id": f"{group}_{index}"}
        for group, count in {"rotation": 200, "among": 600, "around": 250}.items()
        for index in range(count)
    ]

    evaluation_converters._validate_official_mindcube(records, source)

    records[-1] = {"id": "unknown_0"}
    with pytest.raises(
        evaluation_converters.ConversionError,
        match="official MindCube-Tiny must contain",
    ):
        evaluation_converters._validate_official_mindcube(records, source)


def test_revsi_parquet_stages_all_frame_videos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    annotation = tmp_path / "revsi.parquet"
    annotation.write_bytes(b"fixture")
    video_root = tmp_path / "all_frame"
    video_root.mkdir()
    (video_root / "scene-1.mp4").write_bytes(b"revsi-video")
    manifest = _write_manifest(
        tmp_path / "revsi-sources.jsonl",
        [
            _source_record(
                annotation,
                benchmark="revsi",
                eval_task="revsi",
                family="spatial_intelligence",
                adapter="revsi",
                video_root=video_root,
            )
        ],
    )
    monkeypatch.setattr(
        evaluation_converters,
        "_load_parquet",
        lambda _source: [
            {
                "id": "sample-1",
                "scene_id": "scene-1",
                "question": "How many chairs are visible?",
                "ground_truth": 3,
                "question_type": "object_counting_single",
                "num_frames": "all",
            }
        ],
    )
    output = tmp_path / "revsi-output"

    build_evaluation_repository(manifest, output)

    row = load_evaluation_jsonl(
        output / "annotations/spatial_intelligence/revsi/test.jsonl"
    )[0]
    assert row["answer"] == 3
    assert row["problem_type"] == "object_counting_single"
    assert row["videos"] == [
        "media/spatial_intelligence/revsi/videos/test/scene-1.mp4"
    ]
    assert row["metadata"]["scene_id"] == "scene-1"
    assert row["metadata"]["num_frames"] == "all"
    assert (output / row["videos"][0]).read_bytes() == b"revsi-video"


def test_revsi_official_distribution_is_strict(tmp_path: Path) -> None:
    annotation = tmp_path / "revsi.parquet"
    annotation.write_bytes(b"fixture")
    manifest = _write_manifest(
        tmp_path / "revsi-sources.jsonl",
        [
            _source_record(
                annotation,
                benchmark="revsi",
                eval_task="revsi",
                family="spatial_intelligence",
                adapter="revsi",
                expected_count=6808,
            )
        ],
    )
    source = load_source_manifest(manifest)[0]
    expected = evaluation_converters._REVSI_EXPECTED_GROUP_COUNTS
    representative_types = {
        "object_counting": "object_counting_single",
        "object_rel_direction": "object_rel_direction_forward_easy",
        "object_rel_distance": "object_rel_distance_closest",
        "room_size_estimation": "room_size_estimation_single",
    }
    records = [
        {"question_type": representative_types.get(group, group)}
        for group, count in expected.items()
        for _index in range(count)
    ]

    evaluation_converters._validate_official_revsi(records, source)

    records[-1] = {"question_type": "unknown"}
    with pytest.raises(
        evaluation_converters.ConversionError,
        match="official ReVSI all-frame split must contain",
    ):
        evaluation_converters._validate_official_revsi(records, source)


def test_validation_detects_unreferenced_files_and_checksum_changes(
    tmp_path: Path,
) -> None:
    manifest, _, _ = _generic_fixture(tmp_path)
    output = tmp_path / "validated"
    build_evaluation_repository(manifest, output, checksums=True)
    video = next(
        path
        for path in (output / "media/video_qa/longvideobench/videos").rglob("*")
        if path.is_file()
    )
    original = video.read_bytes()

    video.write_bytes(b"x" * len(original))
    with pytest.raises(StagingError, match="mismatch"):
        validate_staged_repository(output, checksums=True)
    video.write_bytes(original)

    extra = output / "media/video_qa/longvideobench/videos/unreferenced.mp4"
    extra.write_bytes(b"extra")
    with pytest.raises(StagingError, match="unmanifested"):
        validate_staged_repository(output, checksums=True)


def test_build_refuses_existing_target_without_overwrite(tmp_path: Path) -> None:
    manifest, _, _ = _generic_fixture(tmp_path)
    output = tmp_path / "existing"
    build_evaluation_repository(manifest, output)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_evaluation_repository(manifest, output)
    build_evaluation_repository(manifest, output, overwrite=True)
    validate_staged_repository(output)


def test_eval_data_command_exposes_inventory_build_and_validate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest, _, _ = _generic_fixture(tmp_path)
    output = tmp_path / "cli-output"
    parser = eval_data.create_parser()
    assert parser.parse_args(["inventory", str(manifest)]).checksums is False
    assert (
        parser.parse_args(["inventory", str(manifest), "--checksums"]).checksums
        is True
    )

    assert (
        eval_data.main(
            [
                "inventory",
                str(manifest),
                "--workers",
                "2",
            ]
        )
        == 0
    )
    inventory = json.loads(capsys.readouterr().out)
    assert inventory["totals"]["rows"] == 1
    assert inventory["asset_checksums"] is False
    assert (
        eval_data.main(
            [
                "build",
                str(manifest),
                str(output),
                "--workers",
                "2",
            ]
        )
        == 0
    )
    build_summary = json.loads(capsys.readouterr().out)
    assert build_summary["datasets"] == 1
    assert build_summary["asset_checksums"] is False
    assert eval_data.main(["validate", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["rows"] == 1
