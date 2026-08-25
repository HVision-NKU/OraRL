from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT))

from orarl.cli import evaluate  # noqa: E402

EVAL_TASK_ROOT = RELEASE_ROOT / "eval" / "task"

# One entry point per paper task family, so a release cannot drop a family.
FAMILY_ENTRY_POINTS = {
    "video_qa": "eval_vllm.py",
    "spatial_intelligence": "revsi/eval_revsi_vllm.py",
    "temporal_grounding": "temporal_grounding/eval_timelens_hf.py",
    "spatial_grounding": "spatial_grounding/eval_refcoco_vllm.py",
    "tracking": "tracking/eval_tracking_vllm.py",
    "spatial_temporal_grounding": "spatial_temporal_grounding/eval_stvg_vllm.py",
    "segmentation": "segmentation/eval_seg_vllm.py",
}


def test_every_task_family_ships_an_entry_point() -> None:
    missing = sorted(
        family
        for family, relative in FAMILY_ENTRY_POINTS.items()
        if not (EVAL_TASK_ROOT / relative).is_file()
    )

    assert missing == []
    assert (EVAL_TASK_ROOT / "eval.sh").is_file()
    assert (EVAL_TASK_ROOT / "mmsi" / "eval_mmsi_transformers.py").is_file()
    assert (EVAL_TASK_ROOT / "mindcube" / "data_utils.py").is_file()


def test_evaluator_is_discovered_without_external_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("ORARL_EVALUATOR", raising=False)
    monkeypatch.delenv("ORARL_RUNTIME_ROOT", raising=False)
    monkeypatch.setattr(evaluate, "find_spec", lambda _: None)
    monkeypatch.chdir(tmp_path)

    assert evaluate._discover_evaluator(None) == (EVAL_TASK_ROOT / "eval.sh").resolve()


@pytest.mark.parametrize(
    "relative",
    sorted(path.relative_to(EVAL_TASK_ROOT).as_posix() for path in EVAL_TASK_ROOT.rglob("*.sh")),
)
def test_shipped_shell_evaluators_parse(relative: str) -> None:
    result = subprocess.run(
        ["bash", "-n", str(EVAL_TASK_ROOT / relative)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_segmentation_loads_the_shipped_decord_patch() -> None:
    patch = EVAL_TASK_ROOT / "qwenvl_decord_patch.py"
    evaluator = (EVAL_TASK_ROOT / "segmentation" / "eval_seg_vllm.py").read_text(encoding="utf-8")

    assert patch.is_file()
    assert "import qwenvl_decord_patch" in evaluator
    assert "Path(__file__).resolve().parents[1]," in evaluator
