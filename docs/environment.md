<p align="right"><a href="environment_zh.md">简体中文</a></p>

# Reproduce the OraRL Environment

OraRL uses one reproducible software stack for policy optimization and
task-native evaluation. The released configuration was validated on NVIDIA
H20/Hopper GPUs with CUDA 12.9 and includes both runtimes in this repository.

## Requirements

- Linux with Conda available in `PATH`
- NVIDIA GPU with a recent driver
- CUDA 12.9-compatible hardware for the paper-matched environment

CUDA 12.9 GA officially requires NVIDIA Linux driver 575.51.03 or newer.
Other NVIDIA GPUs may work, but the published setup was validated on H20.

## Install the pinned stack

```bash
bash scripts/create_conda_env.sh
conda activate orarl

python scripts/check_environment.py \
  --require-gpu \
  --model /path/to/local/model
```

The installer pins Python 3.11, PyTorch 2.10.0+cu129, Transformers 5.5.4,
vLLM 0.19.1, FlashAttention 2.8.3, and the remaining packages in
`requirements-cu129.txt`. The final editable install exposes the `orarl-*`
commands and the bundled `verl` trainer package from this checkout; do not
install an unrelated `verl` release from PyPI over it.

## Validate an evaluation-only node

If a node will only run evaluation, skip the training-side checks:

```bash
bash scripts/create_conda_env.sh --evaluation-only
conda activate orarl

python scripts/check_environment.py --evaluation-only
```

## Verify the release checkout

Run the release checks before allocating a long job:

```bash
python scripts/check_release.py
python -m pytest -q
ruff check .
```

On a paper-matched H20 node, add `--require-gpu --require-h20` to
`scripts/check_environment.py`. Use the same source revision and environment on
every node of a distributed run.

Continue with [Training](training.md) or [Evaluation](evaluation.md).
