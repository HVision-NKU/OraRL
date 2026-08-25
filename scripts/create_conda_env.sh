#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORARL_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="${ORARL_CONDA_ENV:-orarl}"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu129"
EVALUATION_ONLY=0

usage() {
  cat <<'EOF'
Usage: bash scripts/create_conda_env.sh [options]

Create the pinned OraRL CUDA 12.9 environment. This checkout ships both the
training runtime and the evaluators, so no external runtime is required. Use
--evaluation-only to skip the training-side environment validation.

Options:
  --name ENV_NAME       Conda environment name (default: orarl).
  --evaluation-only     Validate the environment for evaluation only.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      if [[ $# -lt 2 || -z "$2" ]]; then
        echo "ERROR: --name requires a non-empty environment name." >&2
        exit 2
      fi
      ENV_NAME="$2"
      shift 2
      ;;
    --evaluation-only)
      EVALUATION_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available in PATH." >&2
  exit 1
fi

if [[ ! -d "${ORARL_ROOT}/verl" ]]; then
  echo "ERROR: the bundled training runtime is missing at ${ORARL_ROOT}/verl." >&2
  echo "Clone the full repository instead of copying individual directories." >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "Detected cluster GPU(s):"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
  echo "CUDA 12.9 GA officially requires Linux driver 575.51.03 or newer."
else
  echo "WARNING: nvidia-smi is unavailable; run the GPU check on an allocated H20 node." >&2
fi

if conda run -n "${ENV_NAME}" python -c "pass" >/dev/null 2>&1; then
  echo "ERROR: conda environment '${ENV_NAME}' already exists." >&2
  echo "Choose another name with --name or remove/update it explicitly." >&2
  exit 1
fi

conda env create \
  --name "${ENV_NAME}" \
  --file "${ORARL_ROOT}/environment.yml"

run_in_env() {
  conda run --no-capture-output -n "${ENV_NAME}" "$@"
}

run_in_env python -m pip install --upgrade \
  "pip==26.0.1" \
  "setuptools==82.0.1" \
  "wheel==0.46.3"

# Install the CUDA build explicitly. Generic PyPI resolves PyTorch 2.10 to the
# CUDA 12.8 wheel, which is not the stack validated with vLLM 0.19.1 here.
run_in_env python -m pip install \
  "torch==2.10.0+cu129" \
  "torchvision==0.25.0+cu129" \
  "torchaudio==2.10.0+cu129" \
  --index-url "${PYTORCH_INDEX}"

# PyTorch must be importable before building FlashAttention.
run_in_env python -m pip install \
  "flash-attn==2.8.3" \
  --no-build-isolation

run_in_env python -m pip install \
  "vllm==0.19.1" \
  --extra-index-url "${PYTORCH_INDEX}"

run_in_env python -m pip install \
  --requirement "${ORARL_ROOT}/requirements-cu129.txt"

# One editable install covers the CLIs, the trainer runtime, and the evaluators.
# --no-deps keeps it from replacing the GPU stack pinned above.
run_in_env python -m pip install --no-deps --editable "${ORARL_ROOT}"
run_in_env python -m pip install "pytest" "ruff"

run_in_env bash "${ORARL_ROOT}/scripts/install_conda_runtime_hook.sh"
CHECK_ARGUMENTS=()
if [[ "${EVALUATION_ONLY}" -eq 1 ]]; then
  CHECK_ARGUMENTS+=(--evaluation-only)
fi
run_in_env bash -c \
  'source "${CONDA_PREFIX}/etc/conda/activate.d/orarl-runtime.sh"; shift; exec python "$@"' \
  _ "${ORARL_ROOT}/scripts/check_environment.py" "${CHECK_ARGUMENTS[@]}"

cat <<EOF

OraRL environment created successfully.

  conda activate ${ENV_NAME}
  python ${ORARL_ROOT}/scripts/check_environment.py --require-gpu
EOF
