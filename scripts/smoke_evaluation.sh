#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORARL_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EVALUATOR="${ORARL_ROOT}/eval/task/eval.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL=""
DATASET=""
SAM2_CKPT=""
SAM2_CFG=""
POSTPROCESSOR=""
GPUS="${GPUS:-0}"
TP_SIZE="${TP_SIZE:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
VIDEO_SAMPLES="${VIDEO_SAMPLES:-8}"
SEGMENTATION_SAMPLES="${SEGMENTATION_SAMPLES:-2}"
SAM2_WORKERS_PER_GPU="${SAM2_WORKERS_PER_GPU:-1}"
RESULTS_ROOT=""
RUN_VIDEO=1
RUN_SEGMENTATION=1
EXECUTE=1

usage() {
  cat <<'EOF'
Usage:
  bash scripts/smoke_evaluation.sh \
    --model PATH \
    --dataset PATH_OR_HF_REPO \
    --sam2-ckpt PATH \
    --sam2-cfg PATH \
    --postprocessor PATH \
    [options]

Runs two bounded checks using this checkout's eval/task/eval.sh:
  1. VideoMME inference (8 samples by default)
  2. MeViS inference + SAM2 mask metrics (2 samples by default)

Required:
  --model PATH              Exported HF model or veRL actor checkpoint.
  --dataset PATH_OR_REPO    Canonical OraRL evaluation dataset.

Required unless --skip-segmentation:
  --sam2-ckpt PATH          SAM2 checkpoint.
  --sam2-cfg PATH           SAM2 Hydra YAML file.
  --postprocessor PATH      Official OneThinker seg_post_sam2.py.

Options:
  --gpus LIST               GPU IDs (default: 0).
  --tp-size N               Tensor parallel size (default: 1).
  --batch-size N            Inference batch size (default: 1).
  --video-samples N         VideoMME sample count (default: 8).
  --segmentation-samples N  MeViS sample count (default: 2).
  --sam2-workers-per-gpu N  SAM2 workers per GPU (default: 1).
  --results-root PATH       Smoke log and aggregate-summary directory.
  --skip-video              Do not run VideoMME.
  --skip-segmentation       Do not run MeViS + SAM2.
  --dry-run                 Resolve and print commands without GPU inference.
  -h, --help                Show this message.

Environment:
  PYTHON_BIN                 Python from the installed OraRL environment.
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
    --dataset)
      require_value "$1" "${2:-}"
      DATASET="$2"
      shift 2
      ;;
    --sam2-ckpt)
      require_value "$1" "${2:-}"
      SAM2_CKPT="$2"
      shift 2
      ;;
    --sam2-cfg)
      require_value "$1" "${2:-}"
      SAM2_CFG="$2"
      shift 2
      ;;
    --postprocessor)
      require_value "$1" "${2:-}"
      POSTPROCESSOR="$2"
      shift 2
      ;;
    --gpus)
      require_value "$1" "${2:-}"
      GPUS="$2"
      shift 2
      ;;
    --tp-size)
      require_value "$1" "${2:-}"
      TP_SIZE="$2"
      shift 2
      ;;
    --batch-size)
      require_value "$1" "${2:-}"
      BATCH_SIZE="$2"
      shift 2
      ;;
    --video-samples)
      require_value "$1" "${2:-}"
      VIDEO_SAMPLES="$2"
      shift 2
      ;;
    --segmentation-samples)
      require_value "$1" "${2:-}"
      SEGMENTATION_SAMPLES="$2"
      shift 2
      ;;
    --sam2-workers-per-gpu)
      require_value "$1" "${2:-}"
      SAM2_WORKERS_PER_GPU="$2"
      shift 2
      ;;
    --results-root)
      require_value "$1" "${2:-}"
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --skip-video)
      RUN_VIDEO=0
      shift
      ;;
    --skip-segmentation)
      RUN_SEGMENTATION=0
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

if [[ -z "${MODEL}" || -z "${DATASET}" ]]; then
  echo "ERROR: --model and --dataset are required." >&2
  usage >&2
  exit 2
fi
if [[ ! -d "${MODEL}" ]]; then
  echo "ERROR: model directory does not exist: ${MODEL}" >&2
  exit 2
fi
if [[ ! -f "${EVALUATOR}" ]]; then
  echo "ERROR: in-repo evaluator is missing: ${EVALUATOR}" >&2
  exit 2
fi
if [[ -d "${DATASET}" ]]; then
  for manifest in datasets.jsonl assets.jsonl; do
    if [[ ! -f "${DATASET}/${manifest}" ]]; then
      echo "ERROR: canonical dataset is missing ${manifest}: ${DATASET}" >&2
      exit 2
    fi
  done
fi
if [[ "${RUN_SEGMENTATION}" -eq 1 ]]; then
  for value in "${SAM2_CKPT}" "${SAM2_CFG}" "${POSTPROCESSOR}"; do
    if [[ -z "${value}" || ! -f "${value}" ]]; then
      echo "ERROR: segmentation input is missing or not a file: ${value:-<empty>}" >&2
      exit 2
    fi
  done
fi
if [[ "${RUN_VIDEO}" -eq 0 && "${RUN_SEGMENTATION}" -eq 0 ]]; then
  echo "ERROR: both smoke tests were disabled." >&2
  exit 2
fi

MODEL="$(readlink -f "${MODEL}")"
if [[ -d "${DATASET}" ]]; then
  DATASET="$(readlink -f "${DATASET}")"
fi
if [[ "${RUN_SEGMENTATION}" -eq 1 ]]; then
  SAM2_CKPT="$(readlink -f "${SAM2_CKPT}")"
  SAM2_CFG="$(readlink -f "${SAM2_CFG}")"
  POSTPROCESSOR="$(readlink -f "${POSTPROCESSOR}")"
fi

if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
  echo "ERROR: PYTHON_BIN must be Python 3.10 or newer: ${PYTHON_BIN}" >&2
  exit 2
fi

for value in \
  "${TP_SIZE}" \
  "${BATCH_SIZE}" \
  "${VIDEO_SAMPLES}" \
  "${SEGMENTATION_SAMPLES}" \
  "${SAM2_WORKERS_PER_GPU}"; do
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: numeric settings must be positive integers; got ${value}" >&2
    exit 2
  fi
done

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "ERROR: --gpus must contain at least one GPU ID." >&2
  exit 2
fi

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_ROOT="${RESULTS_ROOT:-${ORARL_ROOT}/outputs/smoke-evaluation/${RUN_STAMP}}"
mkdir -p "${RESULTS_ROOT}"
LOG="${RESULTS_ROOT}/smoke.log"
exec > >(tee -a "${LOG}") 2>&1

export PYTHONPATH="${ORARL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export FORCE_QWENVL_VIDEO_READER=decord
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"

echo "OraRL root:       ${ORARL_ROOT}"
echo "Evaluator:        ${EVALUATOR}"
echo "Python:           $("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
echo "Model:            ${MODEL}"
echo "Dataset:          ${DATASET}"
echo "GPUs:             ${GPUS}"
echo "Results:          ${RESULTS_ROOT}"
echo "Execute:          ${EXECUTE}"

if [[ "${EXECUTE}" -eq 1 ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi is unavailable; run this script on a GPU node." >&2
    exit 1
  fi
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
fi

"${PYTHON_BIN}" - <<'PY'
import importlib

required = ("numpy", "torch", "transformers", "vllm", "qwen_vl_utils", "decord")
for name in required:
    module = importlib.import_module(name)
    print(f"dependency OK: {name} {getattr(module, '__version__', '<unknown>')}")
PY

if [[ "${RUN_SEGMENTATION}" -eq 1 ]]; then
  PYTHONPATH="${ORARL_ROOT}/eval/task:${PYTHONPATH}" "${PYTHON_BIN}" - <<'PY'
import os

os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"
import qwenvl_decord_patch
from qwen_vl_utils import vision_process

if vision_process.fetch_video is not qwenvl_decord_patch._decord_fetch_video_new_api:
    raise RuntimeError("the in-repo decord patch did not replace fetch_video")
print(f"decord patch OK: {qwenvl_decord_patch.__file__}")
PY
  "${PYTHON_BIN}" -c 'import sam2; print(f"SAM2 package OK: {sam2.__file__}")'
fi

RUN_ARGUMENT=()
if [[ "${EXECUTE}" -eq 1 ]]; then
  RUN_ARGUMENT=(--run)
fi

COMMON=(
  --model "${MODEL}"
  --dataset "${DATASET}"
  --evaluator "${EVALUATOR}"
  --gpus "${GPUS}"
  --tp-size "${TP_SIZE}"
  --batch-size "${BATCH_SIZE}"
)

assert_aggregate_summary() {
  local summary="$1"
  local task="$2"
  "${PYTHON_BIN}" - "${summary}" "${task}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
task = sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("evaluator_returncode") != 0:
    raise SystemExit(f"{task} evaluator failed: {path}")
if task not in payload.get("completed_tasks", []):
    raise SystemExit(
        f"{task} produced no discoverable task summary: {path}; "
        f"missing={payload.get('missing_tasks', [])}"
    )
print(f"aggregate summary OK: {task} -> {path}")
PY
}

if [[ "${RUN_VIDEO}" -eq 1 ]]; then
  echo
  echo "=== VideoMME smoke (${VIDEO_SAMPLES} samples) ==="
  "${PYTHON_BIN}" -m orarl.cli.evaluate \
    "${COMMON[@]}" \
    --tasks videomme \
    --max-samples "${VIDEO_SAMPLES}" \
    --summary "${RESULTS_ROOT}/videomme-summary.json" \
    "${RUN_ARGUMENT[@]}"
  if [[ "${EXECUTE}" -eq 1 ]]; then
    assert_aggregate_summary "${RESULTS_ROOT}/videomme-summary.json" "videomme"
  fi
fi

if [[ "${RUN_SEGMENTATION}" -eq 1 ]]; then
  TASK_CONFIG="${RESULTS_ROOT}/segmentation-smoke.json"
  "${PYTHON_BIN}" - \
    "${TASK_CONFIG}" \
    "${SAM2_CKPT}" \
    "${SAM2_CFG}" \
    "${POSTPROCESSOR}" \
    "${#GPU_IDS[@]}" \
    "${SAM2_WORKERS_PER_GPU}" <<'PY'
import json
import sys

output, checkpoint, config, postprocessor, gpu_count, workers = sys.argv[1:]
payload = {
    "task": "segmentation",
    "environment": {
        "SEGMENTATION_DATASETS": "mevis",
        "SEGMENTATION_DATA_TYPE": "video",
        "SEGMENTATION_VIDEO_READER": "decord",
        "SEGMENTATION_SETTING": "smoke-mevis-video-decord",
        "SEGMENTATION_SAM2_CKPT": checkpoint,
        "SEGMENTATION_SAM2_CFG": config,
        "SEGMENTATION_POSTPROCESSOR_PATH": postprocessor,
        "SEGMENTATION_SAM2_NUM_GPUS": int(gpu_count),
        "SEGMENTATION_SAM2_WORKERS_PER_GPU": int(workers),
    },
}
with open(output, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY

  echo
  echo "=== MeViS + SAM2 smoke (${SEGMENTATION_SAMPLES} samples) ==="
  "${PYTHON_BIN}" -m orarl.cli.evaluate \
    "${COMMON[@]}" \
    --tasks segmentation \
    --task-config "segmentation=${TASK_CONFIG}" \
    --max-samples "${SEGMENTATION_SAMPLES}" \
    --segmentation-run-sam2 \
    --summary "${RESULTS_ROOT}/segmentation-summary.json" \
    "${RUN_ARGUMENT[@]}"
  if [[ "${EXECUTE}" -eq 1 ]]; then
    assert_aggregate_summary "${RESULTS_ROOT}/segmentation-summary.json" "segmentation"
  fi
fi

echo
echo "Smoke evaluation finished."
echo "Log: ${LOG}"
[[ "${RUN_VIDEO}" -eq 0 ]] || echo "VideoMME summary: ${RESULTS_ROOT}/videomme-summary.json"
[[ "${RUN_SEGMENTATION}" -eq 0 ]] \
  || echo "Segmentation summary: ${RESULTS_ROOT}/segmentation-summary.json"
