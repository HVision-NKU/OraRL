from __future__ import annotations

from pathlib import Path

import yaml

RELEASE_ROOT = Path(__file__).resolve().parents[1]


def test_h20_environment_uses_the_pinned_cuda_stack() -> None:
    environment = yaml.safe_load((RELEASE_ROOT / "environment.yml").read_text(encoding="utf-8"))
    dependencies = set(environment["dependencies"])

    assert environment["name"] == "orarl"
    assert "python=3.11" in dependencies
    assert "cuda-toolkit=12.9" in dependencies
    assert "ffmpeg=7" in dependencies
    assert "conda-forge" in environment["channels"]

    installer = (RELEASE_ROOT / "scripts" / "create_conda_env.sh").read_text(encoding="utf-8")
    assert "https://download.pytorch.org/whl/cu129" in installer
    assert "torch==2.10.0+cu129" in installer
    assert "vllm==0.19.1" in installer
    assert "flash-attn==2.8.3" in installer
    assert "install_conda_runtime_hook.sh" in installer
    assert "--evaluation-only" in installer
    # A single checkout installs the trainer; no external runtime root remains.
    assert "--runtime-root" not in installer
    assert "ORARL_RUNTIME_ROOT" not in installer

    validator = (RELEASE_ROOT / "scripts" / "check_environment.py").read_text(
        encoding="utf-8"
    )
    assert '"verl.trainer.main"' in validator
    assert "--evaluation-only" in validator

    runtime_hook = (
        RELEASE_ROOT / "scripts" / "install_conda_runtime_hook.sh"
    ).read_text(encoding="utf-8")
    assert "activate.d/orarl-runtime.sh" in runtime_hook
    assert 'Path(sysconfig.get_path("purelib")) / "nvidia"' in runtime_hook


def test_runtime_requirements_match_the_released_qwen35_stack() -> None:
    requirements = {
        line
        for line in (RELEASE_ROOT / "requirements-cu129.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line and not line.startswith("#")
    }

    assert "transformers==5.5.4" in requirements
    assert "qwen-vl-utils[decord]==0.0.14" in requirements
    assert "torchcodec==0.10.0" in requirements
    assert "ray[default]==2.54.0" in requirements
    assert "flash-linear-attention==0.4.2" in requirements

    evaluator = (
        RELEASE_ROOT / "eval" / "task" / "temporal_grounding" / "eval_timelens_hf.py"
    ).read_text(encoding="utf-8")
    assert 'setdefault("FORCE_QWENVL_VIDEO_READER", "decord")' in evaluator

    segmentation_evaluator = (
        RELEASE_ROOT / "eval" / "task" / "segmentation" / "eval_seg_vllm.py"
    ).read_text(encoding="utf-8")
    assert 'setdefault("FORCE_QWENVL_VIDEO_READER", "decord")' in segmentation_evaluator
