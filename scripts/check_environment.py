#!/usr/bin/env python3
"""Validate the pinned OraRL runtime and an optional local checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Sequence

EXPECTED_DISTRIBUTIONS = {
    "flash-attn": "2.8.3",
    "flash-linear-attention": "0.4.2",
    "orarl": "0.1.0",
    "qwen-vl-utils": "0.0.14",
    "ray": "2.54.0",
    "torchcodec": "0.10.0",
    "transformers": "5.5.4",
    "vllm": "0.19.1",
}
TRAINING_MODULES = (
    "verl.trainer.main",
    "verl.trainer.ray_trainer",
    "verl.workers.fsdp_workers",
)
REQUIRED_MODEL_FILES = {
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
}


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="fail when CUDA is unavailable (use this inside a GPU allocation)",
    )
    parser.add_argument(
        "--require-h20",
        action="store_true",
        help="fail unless at least one visible GPU is an NVIDIA H20",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="optionally validate checkpoint metadata and every indexed weight shard",
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="skip the checks that only training runs need",
    )
    return parser.parse_args(argv)


def _check_versions(errors: list[str]) -> None:
    if sys.version_info < (3, 10):
        errors.append(f"Python 3.10+ is required, found {sys.version.split()[0]}")

    for distribution, expected in EXPECTED_DISTRIBUTIONS.items():
        try:
            actual = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            errors.append(f"missing package: {distribution}=={expected}")
            continue
        print(f"{distribution}=={actual}")
        if actual != expected:
            errors.append(f"{distribution}: expected {expected}, found {actual}")

    qwen_module = "transformers.models.qwen3_5.modeling_qwen3_5"
    try:
        qwen_spec = importlib.util.find_spec(qwen_module)
    except (ImportError, ModuleNotFoundError):
        qwen_spec = None
    if qwen_spec is None:
        errors.append(f"Transformers does not expose {qwen_module}")


def _check_training_runtime(errors: list[str]) -> None:
    """Confirm the bundled trainer resolves from the installed environment."""

    for module in TRAINING_MODULES:
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ModuleNotFoundError) as exc:
            errors.append(f"cannot resolve training module {module}: {exc}")
            continue
        if spec is None or spec.origin is None:
            errors.append(f"missing training module: {module}")
            continue
        print(f"{module}: {spec.origin}")


def _check_torch(require_gpu: bool, require_h20: bool, errors: list[str]) -> None:
    try:
        import torch
    except Exception as exc:
        errors.append(f"cannot import torch: {exc}")
        return

    print(f"torch=={torch.__version__}")
    print(f"torch CUDA build: {torch.version.cuda}")
    if not str(torch.__version__).startswith("2.10.0+cu129"):
        errors.append(f"expected torch 2.10.0+cu129, found {torch.__version__}")
    if torch.version.cuda != "12.9":
        errors.append(f"expected a CUDA 12.9 PyTorch build, found {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("CUDA device: unavailable")
        if require_gpu or require_h20:
            errors.append("CUDA is unavailable inside this process")
        return

    names: list[str] = []
    for index in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(index)
        capability = torch.cuda.get_device_capability(index)
        names.append(name)
        print(f"CUDA device {index}: {name} (compute capability {capability[0]}.{capability[1]})")

    if require_h20 and not any("H20" in name.upper() for name in names):
        errors.append(f"expected an NVIDIA H20, found: {', '.join(names)}")


def _check_video_runtime(errors: list[str]) -> None:
    executable = shutil.which("ffmpeg")
    if executable is None:
        errors.append("missing FFmpeg executable; install ffmpeg=7 from conda-forge")
    else:
        result = subprocess.run(
            [executable, "-version"],
            check=False,
            capture_output=True,
            text=True,
        )
        version_line = result.stdout.splitlines()[0] if result.stdout else ""
        print(version_line or f"ffmpeg: {executable}")
        if result.returncode != 0:
            errors.append(f"FFmpeg failed with exit code {result.returncode}")

    try:
        import torchcodec
    except Exception as exc:
        errors.append(f"cannot import torchcodec with FFmpeg runtime: {exc}")
    else:
        print(f"torchcodec runtime=={torchcodec.__version__}")


def _check_model(model: Path, errors: list[str]) -> None:
    model = model.expanduser().resolve()
    if not model.is_dir():
        errors.append(f"model directory does not exist: {model}")
        return

    present = {path.name for path in model.iterdir() if path.is_file()}
    missing = sorted(REQUIRED_MODEL_FILES - present)
    if missing:
        errors.append(f"model is missing required files: {', '.join(missing)}")
        return

    index_path = model / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
        shards = {str(name) for name in weight_map.values()}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"invalid model shard index: {exc}")
        return

    missing_shards = sorted(shard for shard in shards if not (model / shard).is_file())
    if missing_shards:
        errors.append(f"model is missing indexed weight shards: {', '.join(missing_shards)}")
        return

    try:
        from transformers import AutoConfig, AutoProcessor

        config = AutoConfig.from_pretrained(model, trust_remote_code=False)
        AutoProcessor.from_pretrained(model, trust_remote_code=False)
    except Exception as exc:
        errors.append(f"Transformers cannot load checkpoint metadata: {exc}")
        return

    print(f"model: {model}")
    print(f"model type: {config.model_type}")
    print(f"indexed weight shards: {len(shards)}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    errors: list[str] = []

    _check_versions(errors)
    if not args.evaluation_only:
        _check_training_runtime(errors)
    _check_torch(args.require_gpu, args.require_h20, errors)
    _check_video_runtime(errors)
    if args.model is not None:
        _check_model(args.model, errors)

    if errors:
        print("\nEnvironment validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("\nOraRL environment validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
