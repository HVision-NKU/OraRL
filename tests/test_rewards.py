from __future__ import annotations

from typing import Any

import pytest

import orarl.rewards.router as router_module
from orarl.rewards import (
    RewardContractError,
    RewardRouter,
    RouterSettings,
    TaskFamily,
    UnknownTaskError,
    build_oracle_response_from_ground_truth,
    compute_reward,
)


class FakeAdapter:
    def __init__(self, family: TaskFamily, value: float) -> None:
        self.family = family
        self.value = value
        self.scored_batches: list[list[dict[str, Any]]] = []
        self.score_options: list[dict[str, Any]] = []
        self.oracle_calls: list[tuple[Any, Any]] = []

    def compute_score(self, batch, **kwargs):
        self.scored_batches.append(batch)
        self.score_options.append(kwargs)
        return [
            {
                "overall": self.value,
                "format": 1.0,
                "response_length_seen": float(len(item.get("response", ""))),
            }
            for item in batch
        ]

    def build_oracle_response_from_ground_truth(
        self,
        ground_truth,
        extra=None,
    ):
        self.oracle_calls.append((ground_truth, extra))
        return f"{self.family.value}:{ground_truth}"


def fake_adapters() -> dict[TaskFamily, FakeAdapter]:
    return {
        family: FakeAdapter(family, (index + 1) / 10.0) for index, family in enumerate(TaskFamily)
    }


@pytest.mark.parametrize(
    ("problem_type", "expected"),
    [
        ("Temporal Grounding", TaskFamily.TEMPORAL_GROUNDING),
        ("video_tracking", TaskFamily.TRACKING),
        ("image-segmentation", TaskFamily.SEGMENTATION),
        ("RefCOCO", TaskFamily.SPATIAL_GROUNDING),
        ("spatial-temporal grounding", TaskFamily.SPATIAL_TEMPORAL_GROUNDING),
        ("object_rel_direction_hard", TaskFamily.SPATIAL_INTELLIGENCE),
        ("video_qa_mc", TaskFamily.VIDEO_QA),
    ],
)
def test_routes_all_seven_families(problem_type, expected):
    adapters = fake_adapters()
    router = RewardRouter(RouterSettings(adapters=adapters))

    score = router.compute_reward(
        {"problem_type": problem_type, "ground_truth": "unused"},
        "response",
    )

    assert score["overall"] == adapters[expected].value
    assert len(adapters[expected].scored_batches) == 1


def test_mixed_batch_groups_adapters_and_restores_order():
    adapters = fake_adapters()
    router = RewardRouter(RouterSettings(adapters=adapters))
    batch = [
        {
            "problem_type": "tracking",
            "ground_truth": "one",
            "response": "first",
        },
        {
            "problem_type": "video_qa_mc",
            "ground_truth": "A",
            "response": "second",
        },
        {
            "problem_type": "object_counting",
            "ground_truth": "2",
            "response": "third",
        },
        {
            "problem_type": "object tracking",
            "ground_truth": "four",
            "response": "fourth",
        },
    ]

    scores = router.compute_score(batch)

    assert [score["overall"] for score in scores] == [
        adapters[TaskFamily.TRACKING].value,
        adapters[TaskFamily.VIDEO_QA].value,
        adapters[TaskFamily.SPATIAL_INTELLIGENCE].value,
        adapters[TaskFamily.TRACKING].value,
    ]
    tracking_batches = adapters[TaskFamily.TRACKING].scored_batches
    assert len(tracking_batches) == 1
    assert [item["response"] for item in tracking_batches[0]] == [
        "first",
        "fourth",
    ]


def test_unknown_task_raises_instead_of_returning_zero():
    router = RewardRouter(RouterSettings(adapters=fake_adapters()))

    with pytest.raises(UnknownTaskError, match="Could not route reward sample"):
        router.compute_score(
            [
                {
                    "problem_type": "unsupported-new-task",
                    "ground_truth": "x",
                    "response": "x",
                }
            ]
        )


def test_builtin_module_import_is_lazy_and_monkeypatchable(monkeypatch):
    adapter = FakeAdapter(TaskFamily.TRACKING, 0.75)
    imported: list[str] = []

    def fake_import(module_path):
        imported.append(module_path)
        return adapter

    monkeypatch.setattr(router_module.importlib, "import_module", fake_import)
    router = RewardRouter()
    assert imported == []

    score = router.compute_reward(
        {"problem_type": "tracking", "ground_truth": "gt"},
        "prediction",
    )

    assert score["overall"] == 0.75
    assert imported == ["orarl.rewards.adapters.tracking"]


def test_video_qa_module_path_override_is_honored():
    adapter = FakeAdapter(TaskFamily.VIDEO_QA, 0.65)
    imported: list[str] = []

    def fake_import(module_path):
        imported.append(module_path)
        return adapter

    router = RewardRouter(
        RouterSettings(module_paths={TaskFamily.VIDEO_QA: "custom.video_reward"}),
        importer=fake_import,
    )

    score = router.compute_reward(
        {"problem_type": "video_qa_mc", "ground_truth": "X"},
        "custom response",
    )

    assert score["overall"] == 0.65
    assert imported == ["custom.video_reward"]


def test_video_qa_settings_remain_configurable():
    router = RewardRouter(
        RouterSettings(
            video_qa_options="XY",
            video_qa_require_answer_tags=False,
        )
    )

    score = router.compute_reward(
        {"problem_type": "video_qa_mc", "ground_truth": "X"},
        "x",
    )

    assert score == {"overall": 1.0, "accuracy": 1.0, "format": 1.0}


def test_oracle_builder_receives_ground_truth_and_sample_extra():
    adapters = fake_adapters()
    router = RewardRouter(RouterSettings(adapters=adapters))
    sample = {
        "problem_type": "tracking",
        "ground_truth": "tracked object",
        "sample_id": 17,
    }

    response = router.build_oracle_response(sample)

    assert response == "tracking:tracked object"
    ground_truth, extra = adapters[TaskFamily.TRACKING].oracle_calls[0]
    assert ground_truth == "tracked object"
    assert extra["sample_id"] == 17


def test_trainer_builder_uses_explicit_task_metadata():
    adapters = fake_adapters()
    router = RewardRouter(RouterSettings(adapters=adapters))

    response = router.build_oracle_response_from_ground_truth(
        "7 to 11",
        {"problem_type": "temporal grounding"},
    )

    assert response == "temporal_grounding:7 to 11"


@pytest.mark.parametrize(
    ("ground_truth", "expected"),
    [
        (
            '{"positive_points": [[1, 1]], "boxes": [0, 0, 2, 2]}',
            TaskFamily.SEGMENTATION,
        ),
        (
            '{"time": [1, 2], "boxes": {"1": [0, 0, 2, 2]}}',
            TaskFamily.SPATIAL_TEMPORAL_GROUNDING,
        ),
        ('{"boxes": {"1": [0, 0, 2, 2]}}', TaskFamily.TRACKING),
        ('{"bbox_2d": [0, 0, 2, 2]}', TaskFamily.SPATIAL_GROUNDING),
        ("<answer>1.5 to 2.5</answer>", TaskFamily.TEMPORAL_GROUNDING),
        ("12", TaskFamily.SPATIAL_INTELLIGENCE),
        ("<answer>H</answer>", TaskFamily.VIDEO_QA),
    ],
)
def test_trainer_builder_infers_supported_ground_truth_shapes(
    ground_truth,
    expected,
):
    adapters = fake_adapters()
    router = RewardRouter(RouterSettings(adapters=adapters))

    response = router.build_oracle_response_from_ground_truth(ground_truth)

    assert response == f"{expected.value}:{ground_truth}"


@pytest.mark.parametrize("option", list("ABCDEFGH"))
def test_video_qa_exact_option_matching_supports_a_through_h(option):
    sample = {
        "problem_type": "video_qa_mc",
        "ground_truth": f"<answer>{option}</answer>",
    }

    score = compute_reward(sample, f"<answer>{option.lower()}</answer>")

    assert score == {"overall": 1.0, "accuracy": 1.0, "format": 1.0}


def test_video_qa_rejects_non_exact_option_content_and_raw_response():
    sample = {
        "problem_type": "multiple choice",
        "ground_truth": "<answer>H</answer>",
    }

    verbose = compute_reward(sample, "<answer>H because it is visible</answer>")
    raw = compute_reward(sample, "H")
    wrong = compute_reward(sample, "<answer>A</answer>")

    assert verbose["overall"] == 0.0
    assert verbose["format"] == 0.0
    assert raw["format"] == 0.0
    assert wrong == {"overall": 0.0, "accuracy": 0.0, "format": 1.0}


def test_video_qa_preserves_tagged_answer_after_reasoning():
    sample = {
        "problem_type": "multiple choice",
        "ground_truth": "<answer>G</answer>",
    }

    score = compute_reward(
        sample,
        "<think>I compare the listed events.</think><answer>G</answer>",
    )

    assert score == {"overall": 1.0, "accuracy": 1.0, "format": 1.0}


def test_video_qa_oracle_builder_normalizes_tags():
    assert (
        build_oracle_response_from_ground_truth(
            "<answer>h</answer>",
            {"problem_type": "video qa"},
        )
        == "<answer>H</answer>"
    )


def test_adapter_contract_requires_an_overall_metric():
    class InvalidAdapter:
        @staticmethod
        def compute_score(batch, **kwargs):
            del batch, kwargs
            return [{"format": 1.0}]

    router = RewardRouter(RouterSettings(adapters={TaskFamily.TRACKING: InvalidAdapter()}))

    with pytest.raises(RewardContractError, match="missing 'overall'"):
        router.compute_reward(
            {"problem_type": "tracking", "ground_truth": "gt"},
            "response",
        )


def test_explicit_per_task_score_settings_are_forwarded():
    adapters = fake_adapters()
    router = RewardRouter(
        RouterSettings(
            adapters=adapters,
            score_kwargs={TaskFamily.TRACKING: {"format_weight": 0.2}},
        )
    )

    router.compute_score(
        [
            {
                "problem_type": "tracking",
                "ground_truth": "gt",
                "response": "response",
            }
        ],
        debug=False,
    )

    assert adapters[TaskFamily.TRACKING].score_options == [{"format_weight": 0.2, "debug": False}]
