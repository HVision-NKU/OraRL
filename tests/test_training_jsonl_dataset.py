from __future__ import annotations

import json
import pickle

import pytest

pytest.importorskip("torch")
pytest.importorskip("qwen_vl_utils")

from verl.utils.dataset import (  # noqa: E402
    LocalJsonlDataset,
    _align_media_placeholders,
)


def test_local_jsonl_preserves_task_specific_columns(tmp_path) -> None:
    path = tmp_path / "mixed.jsonl"
    rows = [
        {
            "problem": "temporal prompt",
            "answer": "<answer>1 to 2</answer>",
            "problem_type": "temporal grounding",
            "videos": ["/media/a.mp4"],
            "pred_span": [1.0, 2.0],
        },
        {
            "problem": "segmentation prompt",
            "answer": "<answer>{}</answer>",
            "problem_type": "segmentation",
            "videos": ["/media/b.mp4"],
            "segmentation_output": {"object_id": 1},
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    dataset = LocalJsonlDataset(str(path))

    assert len(dataset) == 2
    assert dataset[0]["pred_span"] == [1.0, 2.0]
    assert dataset[1]["segmentation_output"] == {"object_id": 1}
    assert "segmentation_output" not in dataset[0]


def test_local_jsonl_filter_and_pickle_keep_random_access(tmp_path) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"problem_type": "tracking", "value": 1}),
                "",
                json.dumps({"problem_type": "video_qa_mc", "value": 2}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = LocalJsonlDataset(str(path))
    filtered = dataset.filter(lambda row: row["value"] == 2)
    restored = pickle.loads(pickle.dumps(filtered))

    assert len(dataset) == 2
    assert len(restored) == 1
    assert restored[0] == {"problem_type": "video_qa_mc", "value": 2}


def test_surplus_media_placeholders_are_removed() -> None:
    prompt = "<image> <image> <image> Question with <image> literal tail"

    aligned = _align_media_placeholders(prompt, "<image>", media_count=2)

    assert aligned.count("<image>") == 2
    assert "Question with" in aligned
    assert "literal tail" in aligned


def test_missing_media_placeholders_are_rejected() -> None:
    with pytest.raises(ValueError, match="1 <video> placeholder"):
        _align_media_placeholders("<video> Question", "<video>", media_count=2)
