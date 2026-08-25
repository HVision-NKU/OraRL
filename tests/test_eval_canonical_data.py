from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RELEASE_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = RELEASE_ROOT / "eval" / "task"
sys.path.insert(0, str(TASK_DIR))
sys.path.insert(0, str(TASK_DIR / "spatial_grounding"))
sys.path.insert(0, str(TASK_DIR / "tracking"))
sys.path.insert(0, str(TASK_DIR / "spatial_temporal_grounding"))
sys.path.insert(0, str(TASK_DIR / "segmentation"))

from _grounding_utils import load_annotations as load_grounding_annotations  # noqa: E402
from canonical_data import (  # noqa: E402
    CanonicalDataError,
    adapt_canonical_row,
    canonical_dataset_record,
    load_json_records,
    repository_relative_output_path,
)
from eval_stvg_vllm import load_dataset as load_stvg_dataset  # noqa: E402
from eval_tracking_vllm import load_dataset as load_tracking_dataset  # noqa: E402
from eval_vllm import vsi_prompt, vsi_qtype, vsi_score  # noqa: E402
from post_sam2 import normalize_missing_rle_counts, sam2_config_name  # noqa: E402


def _row() -> dict[str, object]:
    return {
        "schema_version": 1,
        "eval_task": "tracking",
        "sample_id": "sample-1",
        "benchmark": "tracking",
        "split": "got10k",
        "problem": "Track the object.",
        "answer": {"boxes": {"1": [1, 2, 3, 4]}},
        "images": ["media/tracking/images/frame.jpg"],
        "videos": [{"path": "media/tracking/videos/clip.mp4", "fps": 2}],
        "subtitles": ["media/tracking/subtitles/clip.srt"],
        "problem_type": "tracking",
        "source": "tracking",
        "choices": ["left", "right"],
        "preprocessed": {
            "preprocessed_video": "artifacts/tracking/clip.npz",
        },
        "task_payload": {
            "boxes": {"1": [1, 2, 3, 4]},
            "mask_path": "artifacts/tracking/mask.json",
        },
        "metadata": {"category": "fixture"},
    }


def test_adapter_resolves_paths_and_preserves_canonical_fields(tmp_path: Path) -> None:
    adapted = adapt_canonical_row(_row(), tmp_path)

    assert adapted["eval_task"] == "tracking"
    assert adapted["images"] == [
        str(tmp_path / "media/tracking/images/frame.jpg")
    ]
    assert adapted["videos"][0]["path"] == str(
        tmp_path / "media/tracking/videos/clip.mp4"
    )
    assert adapted["subtitles"] == [
        str(tmp_path / "media/tracking/subtitles/clip.srt")
    ]
    assert adapted["task_payload"]["mask_path"] == str(
        tmp_path / "artifacts/tracking/mask.json"
    )
    assert adapted["preprocessed_video"] == str(
        tmp_path / "artifacts/tracking/clip.npz"
    )
    assert adapted["path"] == str(tmp_path / "media/tracking/videos/clip.mp4")
    assert adapted["video"] == adapted["path"]
    assert adapted["image"] == str(tmp_path / "media/tracking/images/frame.jpg")
    assert adapted["options"] == ["left", "right"]
    assert adapted["ground_truth"] == {"boxes": {"1": [1, 2, 3, 4]}}
    assert adapted["boxes"] == {"1": [1, 2, 3, 4]}
    assert adapted["category"] == "fixture"


def test_adapter_resolves_media_and_artifacts_from_separate_asset_root(
    tmp_path: Path,
) -> None:
    metadata_root = tmp_path / "metadata"
    asset_root = tmp_path / "assets"

    adapted = adapt_canonical_row(
        _row(),
        metadata_root,
        asset_root=asset_root,
    )

    assert adapted["image"] == str(
        asset_root / "media/tracking/images/frame.jpg"
    )
    assert adapted["video"] == str(
        asset_root / "media/tracking/videos/clip.mp4"
    )
    assert adapted["preprocessed_video"] == str(
        asset_root / "artifacts/tracking/clip.npz"
    )


def test_loader_infers_repository_root_from_manifests(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    annotation = root / "annotations/tracking/got10k.jsonl"
    annotation.parent.mkdir(parents=True)
    (root / "datasets.jsonl").write_text("{}\n", encoding="utf-8")
    annotation.write_text(json.dumps(_row()) + "\n", encoding="utf-8")

    loaded = load_json_records(annotation)

    assert loaded[0]["video"] == str(
        root / "media/tracking/videos/clip.mp4"
    )


def test_adapter_leaves_legacy_rows_unchanged(tmp_path: Path) -> None:
    legacy = {"question": "Legacy?", "path": "relative.mp4"}

    assert adapt_canonical_row(legacy, tmp_path) == legacy


def test_repository_relative_output_path_supports_sam2_postprocessing(
    tmp_path: Path,
) -> None:
    media = tmp_path / "media/segmentation/videos/mevis/clip.mp4"

    assert repository_relative_output_path(str(media), tmp_path) == (
        "./media/segmentation/videos/mevis/clip.mp4"
    )
    outside = tmp_path.parent / "outside.mp4"
    assert repository_relative_output_path(str(outside), tmp_path) == str(outside)


def test_sam2_absolute_config_becomes_hydra_package_name(tmp_path: Path) -> None:
    package = tmp_path / "sam2"
    config = package / "configs/sam2.1/sam2.1_hiera_l.yaml"

    assert sam2_config_name(config.as_posix(), package) == (
        "configs/sam2.1/sam2.1_hiera_l.yaml"
    )
    assert sam2_config_name("configs/sam2.1/sam2.1_hiera_l.yaml", package) == (
        "configs/sam2.1/sam2.1_hiera_l.yaml"
    )


def test_sam2_normalizes_polygon_and_empty_video_rles() -> None:
    class CocoMask:
        @staticmethod
        def frPyObjects(polygons, height, width):
            if isinstance(polygons, dict):
                assert polygons == {"size": [height, width], "counts": [6]}
                return {"size": [height, width], "counts": b"empty"}
            assert polygons == [[0, 0, 2, 0, 2, 2]]
            return [{"size": [height, width], "counts": b"encoded"}]

        @staticmethod
        def merge(rles):
            return rles[0]

    payload = {
        "results": [
            {
                "data_type": "image",
                "segmentation_output": {
                    "segmentation_polygon": [[0, 0, 2, 0, 2, 2]],
                    "segmentation_rle": {"size": [2, 3]},
                },
            },
            {
                "data_type": "video",
                "task_payload": {
                    "segmentation_output": {
                        "segmentation_rle": {
                            "00000": {"size": [2, 3]},
                            "00001": {"size": [2, 3], "counts": [6]},
                            "00002": {"size": [2, 3], "counts": "valid"},
                        }
                    }
                },
            },
        ]
    }

    assert normalize_missing_rle_counts(payload, CocoMask) == 3
    assert payload["results"][0]["segmentation_output"]["segmentation_rle"] == {
        "size": [2, 3],
        "counts": "encoded",
    }
    assert payload["results"][1]["segmentation_output"]["segmentation_rle"] == {
        "00000": {"size": [2, 3], "counts": "empty"},
        "00001": {"size": [2, 3], "counts": "empty"},
        "00002": {"size": [2, 3], "counts": "valid"},
    }


def test_vsi_prefers_the_benchmark_subtask_over_generic_answer_type() -> None:
    record = {
        "problem": "How many chairs are in this room?",
        "problem_type": "regression",
        "question_type": "regression",
        "original_question_type": "object_counting",
    }

    assert vsi_qtype(record) == "object_counting"
    assert "Answer with an integer within <answer>" in vsi_prompt(record)
    assert vsi_score(vsi_qtype(record), "<answer>3</answer>", "<answer>3</answer>") == (
        "MRA",
        "3.0",
        1.0,
    )


def test_loader_preserves_whitespace_prefixed_legacy_json(tmp_path: Path) -> None:
    legacy = [{"question": "Legacy?", "path": "relative.mp4"}]
    data_file = tmp_path / "legacy.json"
    data_file.write_text(
        "\n  " + json.dumps(legacy, indent=2) + "\n",
        encoding="utf-8",
    )

    assert load_json_records(data_file) == legacy


def test_adapter_rejects_repository_escape(tmp_path: Path) -> None:
    row = _row()
    row["videos"] = ["../outside.mp4"]

    with pytest.raises(CanonicalDataError, match="outside"):
        adapt_canonical_row(row, tmp_path)


def test_adapter_rejects_absolute_path_outside_repository(tmp_path: Path) -> None:
    row = _row()
    row["videos"] = [str(tmp_path.parent / "outside.mp4")]

    with pytest.raises(CanonicalDataError, match="outside"):
        adapt_canonical_row(row, tmp_path)


def test_dataset_profile_resolves_snake_case_split_alias(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "datasets.jsonl").write_text(
        json.dumps(
            {
                "task": "temporal_grounding",
                "split": "charades_timelens",
                "preprocessing": {"fps": 4},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    record = canonical_dataset_record(
        "temporal_grounding",
        "charades-timelens",
        root,
    )

    assert record is not None
    assert record["preprocessing"]["fps"] == 4


def test_spatial_grounding_loads_canonical_split_with_legacy_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "metadata"
    asset_root = tmp_path / "assets"
    annotation = root / "annotations/spatial_grounding/refcoco_val.jsonl"
    image = asset_root / "media/spatial_grounding/images/image.jpg"
    annotation.parent.mkdir(parents=True)
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fixture")
    row = {
        "schema_version": 1,
        "eval_task": "spatial_grounding",
        "sample_id": "refcoco-val-1",
        "benchmark": "spatial_grounding",
        "split": "refcoco_val",
        "problem": "the red box",
        "answer": [16, 47, 469, 935],
        "images": ["media/spatial_grounding/images/image.jpg"],
        "videos": [],
        "problem_type": "spatial_grounding",
        "source": "refcoco",
        "family": "spatial_grounding",
        "task_payload": {"normalized_solution": [16, 47, 469, 935]},
        "metadata": {"height": 428, "width": 640},
        "evaluation": {},
    }
    annotation.write_text(json.dumps(row) + "\n", encoding="utf-8")
    (root / "datasets.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("ORARL_EVAL_DATA_ROOT", str(root))
    monkeypatch.setenv("ORARL_EVAL_ASSET_ROOT", str(asset_root))

    records = load_grounding_annotations(
        str(annotation.parent),
        "refcoco-val",
    )

    assert records == [
        {
            "problem_id": "refcoco-val-1",
            "image_path": str(image),
            "expression": "the red box",
            "bbox": [16.0, 47.0, 469.0, 935.0],
            "width": 640,
            "height": 428,
        }
    ]


def test_tracking_loads_canonical_split_with_legacy_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "metadata"
    asset_root = tmp_path / "assets"
    annotation = root / "annotations/tracking/got10k.jsonl"
    video = asset_root / "media/tracking/videos/clip.mp4"
    annotation.parent.mkdir(parents=True)
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fixture")
    answer = '<answer>{"boxes":{"1":[1,2,3,4]}}</answer>'
    row = {
        "schema_version": 1,
        "eval_task": "tracking",
        "sample_id": "got10k-1",
        "benchmark": "tracking",
        "split": "got10k",
        "problem": "Track the object.",
        "answer": answer,
        "images": [],
        "videos": ["media/tracking/videos/clip.mp4"],
        "problem_type": "tracking",
        "source": "got10k",
        "family": "tracking",
        "task_payload": {"boxes": {"1": [1, 2, 3, 4]}},
        "evaluation": {},
    }
    annotation.write_text(json.dumps(row) + "\n", encoding="utf-8")
    (root / "datasets.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("ORARL_EVAL_DATA_ROOT", str(root))
    monkeypatch.setenv("ORARL_EVAL_ASSET_ROOT", str(asset_root))

    records = load_tracking_dataset(str(annotation.parent), "eval_got10k")

    assert records[0]["problem"] == "Track the object."
    assert records[0]["solution"] == answer
    assert records[0]["path"] == str(video)


def test_stvg_loads_canonical_split_with_legacy_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "metadata"
    asset_root = tmp_path / "assets"
    annotation = root / "annotations/stvg/stvg.jsonl"
    video = asset_root / "media/stvg/videos/stvg/clip.mp4"
    annotation.parent.mkdir(parents=True)
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fixture")
    answer = (
        '<answer>{"time":[1,2],"boxes":{"1":[1,2,3,4],'
        '"2":[2,3,4,5]}}</answer>'
    )
    row = {
        "schema_version": 1,
        "eval_task": "stvg",
        "sample_id": "stvg-1",
        "benchmark": "stvg",
        "split": "stvg",
        "problem": "When and where is the person moving?",
        "answer": answer,
        "images": [],
        "videos": ["media/stvg/videos/stvg/clip.mp4"],
        "problem_type": "spatial-temporal grounding",
        "source": "stvg",
        "family": "spatial_temporal_grounding",
        "task_payload": {
            "time": [1, 2],
            "boxes": {"1": [1, 2, 3, 4], "2": [2, 3, 4, 5]},
        },
        "evaluation": {},
    }
    annotation.write_text(json.dumps(row) + "\n", encoding="utf-8")
    (root / "datasets.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("ORARL_EVAL_DATA_ROOT", str(root))
    monkeypatch.setenv("ORARL_EVAL_ASSET_ROOT", str(asset_root))

    records = load_stvg_dataset(str(annotation.parent), "eval_stvg")

    assert records[0]["solution"] == answer
    assert records[0]["path"] == str(video)
