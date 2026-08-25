from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = RELEASE_ROOT / "scripts" / "check_release.py"
SPEC = importlib.util.spec_from_file_location("orarl_release_check", CHECKER_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECKER
SPEC.loader.exec_module(CHECKER)


def _reasons(root: Path, max_bytes: int = CHECKER.DEFAULT_MAX_BYTES) -> list[str]:
    return [finding.reason for finding in CHECKER.check_release(root, max_bytes)]


def test_clean_text_tree_passes(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("portable source\n", encoding="utf-8")
    assert CHECKER.check_release(tmp_path) == []


def test_portable_evaluation_jsonl_is_release_source(tmp_path: Path) -> None:
    annotation = tmp_path / "data" / "eval" / "annotations" / "task.jsonl"
    annotation.parent.mkdir(parents=True)
    natural_text = "".join(("fl", "uffy"))
    annotation.write_text(
        f'{{"problem":"{natural_text}","videos":["media/task/videos/clip.mp4"]}}\n',
        encoding="utf-8",
    )

    assert CHECKER.check_release(tmp_path) == []


def test_canonical_evaluation_jsonl_can_exceed_the_generic_source_limit(
    tmp_path: Path,
) -> None:
    annotation = tmp_path / "data" / "eval" / "annotations" / "task.jsonl"
    annotation.parent.mkdir(parents=True)
    annotation.write_text('{"problem":"' + ("x" * 64) + '"}\n', encoding="utf-8")

    assert CHECKER.check_release(tmp_path, max_bytes=16) == []


def test_evaluation_jsonl_still_rejects_private_paths(tmp_path: Path) -> None:
    annotation = tmp_path / "data" / "eval" / "annotations" / "task.jsonl"
    annotation.parent.mkdir(parents=True)
    private_path = "/" + "mnt" + "/person/video.mp4"
    annotation.write_text(
        f'{{"videos":["{private_path}"]}}\n',
        encoding="utf-8",
    )

    assert "private absolute path at line 1" in _reasons(tmp_path)


def test_non_evaluation_data_tree_is_rejected(tmp_path: Path) -> None:
    generated = tmp_path / "data" / "training"
    generated.mkdir(parents=True)
    (generated / "rows.txt").write_text("private input\n", encoding="utf-8")

    assert "generated root data directory" in _reasons(tmp_path)


def test_excluded_method_terms_are_detected_without_literals_in_test_source(
    tmp_path: Path,
) -> None:
    term_one = "".join(("c", "p", "p", "o"))
    term_two = "".join(("lu", "ff", "y"))
    (tmp_path / "source.py").write_text(
        f"first = '{term_one}'\nsecond = '{term_two}'\n",
        encoding="utf-8",
    )
    reasons = _reasons(tmp_path)
    assert sum("excluded method term" in reason for reason in reasons) == 2


def test_legacy_oracle_names_are_detected(tmp_path: Path) -> None:
    legacy_name = "".join(("build_", "g", "t", "_response"))
    (tmp_path / "source.py").write_text(
        f"def {legacy_name}():\n    pass\n",
        encoding="utf-8",
    )
    assert any("excluded method term" in reason for reason in _reasons(tmp_path))


def test_evaluation_runtime_may_name_benchmark_ground_truth(tmp_path: Path) -> None:
    evaluator = tmp_path / "eval" / "task" / "tracking" / "eval_tracking.py"
    evaluator.parent.mkdir(parents=True)
    annotation_field = "".join(("g", "t", "_", "bbox"))
    evaluator.write_text(
        f'box = record["{annotation_field}"]\n',
        encoding="utf-8",
    )

    assert CHECKER.check_release(tmp_path) == []


def test_evaluation_runtime_still_rejects_excluded_method_terms(tmp_path: Path) -> None:
    evaluator = tmp_path / "eval" / "task" / "eval_vllm.py"
    evaluator.parent.mkdir(parents=True)
    legacy_name = "".join(("build_", "g", "t", "_response"))
    evaluator.write_text(f"def {legacy_name}():\n    pass\n", encoding="utf-8")

    assert any("excluded method term" in reason for reason in _reasons(tmp_path))


def test_generated_test_caches_fail_the_source_gate(tmp_path: Path) -> None:
    cache = tmp_path / ".pytest_cache"
    cache.mkdir()
    (cache / "state").write_text("generated\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("portable source\n", encoding="utf-8")
    findings = CHECKER.check_release(tmp_path)
    assert len(findings) == 1
    assert findings[0].path == Path(".pytest_cache")
    assert findings[0].reason == "generated directory"


def test_gitignored_generated_state_is_not_part_of_release(tmp_path: Path) -> None:
    cache = tmp_path / ".pytest_cache"
    cache.mkdir()
    (cache / "state").write_text("generated\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text(".pytest_cache/\n", encoding="utf-8")

    assert CHECKER.check_release(tmp_path) == []


def test_gitignored_training_runs_are_not_part_of_release(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "smoke-training" / "grpo"
    run.mkdir(parents=True)
    (run / "smoke.log").write_text("/private/training/path\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("/runs/\n", encoding="utf-8")

    assert CHECKER.check_release(tmp_path) == []


def test_local_checkpoint_tree_is_outside_release_scan(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint" / "Video-ORA-9B"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"local weights")
    (tmp_path / "README.md").write_text("portable source\n", encoding="utf-8")
    assert CHECKER.check_release(tmp_path) == []


def test_documented_release_media_is_allowlisted(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "orarl-data-scaling.gif").write_bytes(b"documented data animation")
    (assets / "orarl-hero.gif").write_bytes(b"documented preview")
    (assets / "orarl-method.gif").write_bytes(b"documented method animation")
    (assets / "orarl-model-scaling.gif").write_bytes(b"documented model animation")
    (assets / "paper-results.png").write_bytes(b"documented figure")
    (tmp_path / "orarl.pdf").write_bytes(b"documented paper")

    assert CHECKER.check_release(tmp_path) == []


def test_private_paths_and_secret_values_are_detected(tmp_path: Path) -> None:
    private_path = "/" + "mnt" + "/person/project/model"
    access_value = "AKIA" + ("A" * 16)
    (tmp_path / "settings.txt").write_text(
        f"model={private_path}\ncloud={access_value}\n",
        encoding="utf-8",
    )
    reasons = _reasons(tmp_path)
    assert any("private absolute path" in reason for reason in reasons)
    assert any("credential-like value" in reason for reason in reasons)


def test_generated_large_and_broken_entries_are_detected(tmp_path: Path) -> None:
    (tmp_path / "weights.pt").write_bytes(b"payload")
    (tmp_path / "large.txt").write_text("x" * 17, encoding="utf-8")
    (tmp_path / "missing-link").symlink_to(tmp_path / "not-there")
    reasons = _reasons(tmp_path, max_bytes=16)
    assert "generated or binary artifact" in reasons
    assert any("file is too large" in reason for reason in reasons)
    assert "broken symbolic link" in reasons
