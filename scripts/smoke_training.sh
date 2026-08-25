#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORARL_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL=""
TRAIN_DATA=""
VAL_DATA=""
MEDIA_ROOT=""
MODEL_SIZE="${MODEL_SIZE:-4b}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
STEPS="${STEPS:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
RESULTS_ROOT=""
RUN_GRPO=1
RUN_ORARL=1
EXECUTE=1

usage() {
  cat <<'EOF'
Usage:
  bash scripts/smoke_training.sh \
    --model PATH \
    --train-data PATH \
    --val-data PATH \
    [options]

Runs two bounded updates with the trainer bundled in this checkout:
  1. GRPO baseline  (grpo_<size>.yaml)
  2. OraRL          (orarl_<size>.yaml)

Each run saves a checkpoint so resume is exercised too.

Required:
  --model PATH             Local base model or checkpoint directory.
  --train-data PATH        Prepared training JSONL.
  --val-data PATH          Prepared canary JSONL.

Options:
  --size 4b|9b             Released recipe scale (default: 4b).
  --media-root PATH        Media root exported as ORARL_MEDIA_ROOT.
  --nodes N                Training nodes (default: 1).
  --gpus-per-node N        GPUs per node (default: 8).
  --steps N                Updates per run (default: 1).
  --rollout-batch-size N   Prompts per rollout batch (default: 8).
  --global-batch-size N    Actor update batch size (default: 8).
  --results-root PATH      Log and checkpoint directory.
  --skip-grpo              Do not run the GRPO baseline.
  --skip-orarl             Do not run OraRL.
  --dry-run                Resolve and print commands without training.
  -h, --help               Show this message.

Environment:
  PYTHON_BIN               Python from the installed OraRL environment.
EOF
}

require_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "${value}" ]]; then
    echo "ERROR: ${option} requires a value." >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      require_value "$1" "${2:-}"
      MODEL="$2"
      shift 2
      ;;
    --train-data)
      require_value "$1" "${2:-}"
      TRAIN_DATA="$2"
      shift 2
      ;;
    --val-data)
      require_value "$1" "${2:-}"
      VAL_DATA="$2"
      shift 2
      ;;
    --media-root)
      require_value "$1" "${2:-}"
      MEDIA_ROOT="$2"
      shift 2
      ;;
    --size)
      require_value "$1" "${2:-}"
      MODEL_SIZE="${2,,}"
      shift 2
      ;;
    --nodes)
      require_value "$1" "${2:-}"
      NODES="$2"
      shift 2
      ;;
    --gpus-per-node)
      require_value "$1" "${2:-}"
      GPUS_PER_NODE="$2"
      shift 2
      ;;
    --steps)
      require_value "$1" "${2:-}"
      STEPS="$2"
      shift 2
      ;;
    --rollout-batch-size)
      require_value "$1" "${2:-}"
      ROLLOUT_BATCH_SIZE="$2"
      shift 2
      ;;
    --global-batch-size)
      require_value "$1" "${2:-}"
      GLOBAL_BATCH_SIZE="$2"
      shift 2
      ;;
    --results-root)
      require_value "$1" "${2:-}"
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --skip-grpo)
      RUN_GRPO=0
      shift
      ;;
    --skip-orarl)
      RUN_ORARL=0
      shift
      ;;
    --dry-run)
      EXECUTE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${MODEL}" || -z "${TRAIN_DATA}" || -z "${VAL_DATA}" ]]; then
  echo "ERROR: --model, --train-data, and --val-data are required." >&2
  usage >&2
  exit 2
fi
if [[ ! -d "${MODEL}" ]]; then
  echo "ERROR: model directory does not exist: ${MODEL}" >&2
  exit 2
fi
for value in "${TRAIN_DATA}" "${VAL_DATA}"; do
  if [[ ! -f "${value}" ]]; then
    echo "ERROR: data file does not exist: ${value}" >&2
    exit 2
  fi
done
if [[ -n "${MEDIA_ROOT}" && ! -d "${MEDIA_ROOT}" ]]; then
  echo "ERROR: media root does not exist: ${MEDIA_ROOT}" >&2
  exit 2
fi
case "${MODEL_SIZE}" in
  4b|9b) ;;
  *)
    echo "ERROR: --size must be 4b or 9b; got ${MODEL_SIZE}" >&2
    exit 2
    ;;
esac
if [[ ! -d "${ORARL_ROOT}/verl" ]]; then
  echo "ERROR: the bundled training runtime is missing: ${ORARL_ROOT}/verl" >&2
  exit 2
fi
if [[ "${RUN_GRPO}" -eq 0 && "${RUN_ORARL}" -eq 0 ]]; then
  echo "ERROR: both smoke runs were disabled." >&2
  exit 2
fi
for value in \
  "${NODES}" \
  "${GPUS_PER_NODE}" \
  "${STEPS}" \
  "${ROLLOUT_BATCH_SIZE}" \
  "${GLOBAL_BATCH_SIZE}"; do
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: numeric settings must be positive integers; got ${value}" >&2
    exit 2
  fi
done

MODEL="$(readlink -f "${MODEL}")"
TRAIN_DATA="$(readlink -f "${TRAIN_DATA}")"
VAL_DATA="$(readlink -f "${VAL_DATA}")"
if [[ -n "${MEDIA_ROOT}" ]]; then
  MEDIA_ROOT="$(readlink -f "${MEDIA_ROOT}")"
  export ORARL_MEDIA_ROOT="${MEDIA_ROOT}"
fi

if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
  echo "ERROR: PYTHON_BIN must be Python 3.10 or newer: ${PYTHON_BIN}" >&2
  exit 2
fi

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_ROOT="${RESULTS_ROOT:-${ORARL_ROOT}/outputs/smoke-training/${RUN_STAMP}}"
mkdir -p "${RESULTS_ROOT}"
LOG="${RESULTS_ROOT}/smoke.log"
exec > >(tee -a "${LOG}") 2>&1

export PYTHONPATH="${ORARL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export FORCE_QWENVL_VIDEO_READER=decord
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"

echo "OraRL root:       ${ORARL_ROOT}"
echo "Python:           $("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
echo "Model:            ${MODEL}"
echo "Train data:       ${TRAIN_DATA}"
echo "Val data:         ${VAL_DATA}"
echo "Recipe scale:     ${MODEL_SIZE}"
echo "World size:       ${NODES} x ${GPUS_PER_NODE}"
echo "Steps per run:    ${STEPS}"
echo "Results:          ${RESULTS_ROOT}"
echo "Execute:          ${EXECUTE}"

if [[ "${EXECUTE}" -eq 1 ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi is unavailable; run this script on a GPU node." >&2
    exit 1
  fi
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

  # The GPU stack is only needed for a real update, so --dry-run stays usable
  # from a login node.
  "${PYTHON_BIN}" - <<'PY'
import importlib

required = (
    "numpy",
    "torch",
    "transformers",
    "vllm",
    "ray",
    "tensordict",
    "codetiming",
    "omegaconf",
    "verl.trainer.main",
    "orarl.rewards",
)
for name in required:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", None)
    print(f"dependency OK: {name} {version or module.__file__}")
PY
fi

RUN_ARGUMENT=()
if [[ "${EXECUTE}" -eq 1 ]]; then
  RUN_ARGUMENT=(--run)
fi

assert_checkpoint() {
  local output="$1"
  local method="$2"
  "${PYTHON_BIN}" - "${output}" "${method}" <<'PY'
import sys
from pathlib import Path

output = Path(sys.argv[1])
method = sys.argv[2]
steps = sorted(output.glob("global_step_*"))
if not steps:
    raise SystemExit(f"{method} wrote no checkpoint under {output}")
print(f"checkpoint OK: {method} -> {steps[-1]}")
PY
}

run_recipe() {
  local method="$1"
  local output="${RESULTS_ROOT}/${method}"
  echo
  echo "=== ${method} smoke (${STEPS} update(s)) ==="
  "${PYTHON_BIN}" -m orarl.cli.train \
    --config "${ORARL_ROOT}/configs/${method}_${MODEL_SIZE}.yaml" \
    --model "${MODEL}" \
    --train-data "${TRAIN_DATA}" \
    --val-data "${VAL_DATA}" \
    --output "${output}" \
    --nodes "${NODES}" \
    --gpus-per-node "${GPUS_PER_NODE}" \
    --set "trainer.max_steps=${STEPS}" \
    --set "trainer.save_freq=${STEPS}" \
    --set "trainer.val_before_train=false" \
    --set "trainer.experiment_name=smoke-${method}-${MODEL_SIZE}" \
    --set "data.rollout_batch_size=${ROLLOUT_BATCH_SIZE}" \
    --set "worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE}" \
    "${RUN_ARGUMENT[@]}"
  if [[ "${EXECUTE}" -eq 1 ]]; then
    assert_checkpoint "${output}" "${method}"
  fi
}

[[ "${RUN_GRPO}" -eq 0 ]] || run_recipe grpo
[[ "${RUN_ORARL}" -eq 0 ]] || run_recipe orarl

echo
echo "Smoke training finished."
echo "Log: ${LOG}"
[[ "${RUN_GRPO}" -eq 0 ]] || echo "GRPO output: ${RESULTS_ROOT}/grpo"
[[ "${RUN_ORARL}" -eq 0 ]] || echo "OraRL output: ${RESULTS_ROOT}/orarl"
