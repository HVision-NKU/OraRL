from __future__ import annotations

import importlib
import importlib.util
from typing import Any

import numpy as np
import pytest

from orarl.rewards import RewardRouter, RouterSettings, TaskFamily

SEGMENTATION_GT = {
    "boxes": [100, 100, 400, 400],
    "positive_points": [[150, 150], [250, 250], [350, 350]],
    "negative_points": [[0, 0], [700, 700], [900, 100]],
}

TASK_CASES = [
    (
        {"problem_type": "temporal grounding", "ground_truth": {"time": [1, 3]}},
        "<answer>8 to 9</answer>",
    ),
    (
        {
            "problem_type": "tracking",
            "ground_truth": {
                "boxes": {
                    "1": [0, 0, 10, 10],
                    "2": [10, 10, 20, 20],
                }
            },
        },
        '{"boxes":{"1":[0,0,10,10],"2":[10,10,20,20]}}',
    ),
    (
        {
            "problem_type": "segmentation",
            "data_type": "image",
            "ground_truth": SEGMENTATION_GT,
        },
        '<answer>{"boxes":[100,100,400,400]}</answer>',
    ),
    (
        {
            "problem_type": "spatial grounding",
            "ground_truth": {"bbox_2d": [10, 20, 40, 60]},
        },
        "[10,20,40,60]",
    ),
    (
        {
            "problem_type": "spatial-temporal grounding",
            "ground_truth": {
                "time": [1, 2],
                "boxes": {
                    "1": [0, 0, 10, 10],
                    "2": [10, 10, 20, 20],
                },
            },
        },
        '<answer>{"time":[1,2],"boxes":{"1":[50,50,60,60]}}</answer>',
    ),
    (
        {"problem_type": "object_counting", "ground_truth": "5"},
        "<answer>12</answer>",
    ),
    (
        {"problem_type": "video_qa_mc", "ground_truth": "<answer>H</answer>"},
        "<answer>A</answer>",
    ),
]


@pytest.mark.parametrize(("sample", "wrong_response"), TASK_CASES)
def test_default_router_oracles_maximize_all_families(sample, wrong_response):
    router = RewardRouter()
    oracle_response = router.build_oracle_response(sample)

    oracle_score = router.compute_reward(sample, oracle_response)
    wrong_score = router.compute_reward(sample, wrong_response)

    assert oracle_score["overall"] == pytest.approx(1.0)
    assert oracle_score["format"] == 1.0
    assert wrong_score["overall"] < oracle_score["overall"]


def test_default_paths_are_packaged_and_importable():
    paths = RouterSettings().module_paths

    assert set(paths) == set(TaskFamily)
    assert all(path.startswith("orarl.") for path in paths.values())
    assert all(importlib.util.find_spec(path) is not None for path in paths.values())


def test_default_router_never_requests_an_external_example_module():
    imported: list[str] = []

    def guarded_import(module_path: str):
        assert not module_path.startswith("examples.")
        imported.append(module_path)
        return importlib.import_module(module_path)

    router = RewardRouter(importer=guarded_import)
    for sample, _ in TASK_CASES:
        response = router.build_oracle_response(sample)
        assert router.compute_reward(sample, response)["overall"] == pytest.approx(1.0)

    assert len(imported) == len(TaskFamily)
    assert all(path.startswith("orarl.") for path in imported)


def test_tracking_missing_frame_contributes_zero():
    router = RewardRouter()
    sample = {
        "problem_type": "tracking",
        "ground_truth": {
            "boxes": {
                "1": [0, 0, 10, 10],
                "2": [10, 10, 20, 20],
            }
        },
    }
    response = '<answer>{"boxes":{"1":[0,0,10,10]}}</answer>'

    score = router.compute_reward(sample, response)

    assert score["miou"] == pytest.approx(0.5)
    assert score["overall"] == pytest.approx(0.5)


def test_stvg_combines_temporal_iou_with_strict_spatial_iou():
    router = RewardRouter()
    sample = {
        "problem_type": "stvg",
        "ground_truth": {
            "time": [1, 3],
            "boxes": {
                "1": [0, 0, 10, 10],
                "2": [10, 10, 20, 20],
            },
        },
    }
    response = '<answer>{"time":[1,3],"boxes":{"1":[0,0,10,10]}}</answer>'

    score = router.compute_reward(sample, response)

    assert score["tiou"] == 1.0
    assert score["siou_strict"] == pytest.approx(0.5)
    assert score["overall"] == pytest.approx(0.2 * 1.0 + 0.8 * 0.5)


def _rle_counts(mask: np.ndarray) -> list[int]:
    flattened = mask.T.reshape(-1)
    counts: list[int] = []
    previous = 0
    run = 0
    for value in flattened:
        current = int(value)
        if current == previous:
            run += 1
        else:
            counts.append(run)
            run = 1
            previous = current
    counts.append(run)
    return counts


def _compressed_counts(counts: list[int]) -> str:
    encoded: list[str] = []
    for index, count in enumerate(counts):
        value = count - counts[index - 2] if index > 2 else count
        more = True
        while more:
            code = value & 0x1F
            value >>= 5
            more = value != (-1 if code & 0x10 else 0)
            if more:
                code |= 0x20
            encoded.append(chr(code + 48))
    return "".join(encoded)


def _mask_ground_truth() -> dict[str, Any]:
    return {
        "boxes": [200, 200, 600, 600],
        "positive_points": [[300, 300], [400, 400], [500, 500]],
        "negative_points": [[50, 50], [800, 800], [950, 500]],
    }


def test_segmentation_uses_image_mask_and_proxy_fallback_deterministically():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 20:60] = 1
    sample = {
        "problem_type": "segmentation",
        "data_type": "image",
        "ground_truth": _mask_ground_truth(),
        "segmentation_output": {
            "size": [100, 100],
            "counts": _rle_counts(mask),
        },
    }
    router = RewardRouter()
    response = router.build_oracle_response(sample)

    mask_score = router.compute_reward(sample, response)
    proxy_sample = {key: value for key, value in sample.items() if key != "segmentation_output"}
    first_proxy = router.compute_reward(proxy_sample, response)
    second_proxy = router.compute_reward(proxy_sample, response)

    assert mask_score["overall"] == pytest.approx(1.0)
    assert mask_score["mask_aware_used"] == 1.0
    assert first_proxy["overall"] == pytest.approx(1.0)
    assert first_proxy == second_proxy


def test_segmentation_decodes_compressed_video_rle_using_media_metadata():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 20:60] = 1
    empty = np.zeros_like(mask)
    target = {**_mask_ground_truth(), "time": 1.0}
    sample = {
        "problem_type": "segmentation",
        "data_type": "video",
        "ground_truth": target,
        "videos": [{"video": "clip.mp4", "fps": 10}],
        "segmentation_output": {
            "frames": ["0", "10"],
            "segmentation_rle": {
                "0": {
                    "size": [100, 100],
                    "counts": _compressed_counts(_rle_counts(empty)),
                },
                "10": {
                    "size": [100, 100],
                    "counts": _compressed_counts(_rle_counts(mask)),
                },
            },
        },
    }
    router = RewardRouter()
    response = router.build_oracle_response(sample)

    score = router.compute_reward(sample, response)

    assert score["overall"] == pytest.approx(1.0)
    assert score["mask_aware_used"] == 1.0
    assert score["mask_box_iou"] == pytest.approx(1.0)
