#!/usr/bin/env bash
# ReVSI evaluation using the same vLLM multi-node sharding path as VSI-Bench.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
RUNNER="${SCRIPT_DIR}/revsi/run_eval_vllm.sh"
MERGER="${SCRIPT_DIR}/revsi/merge_multinode_shards.py"
PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
HOSTFILE=${HOSTFILE:-}
NUM_NODES=${NUM_NODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

MODEL=${MODEL:-}
MODEL=${MODEL%/}
REVSI_ROOT=${REVSI_ROOT:-}
FRAME_BUDGET=${REVSI_FRAME_BUDGET:-all}
FRAME_DIR="${REVSI_ROOT:+${REVSI_ROOT}/${FRAME_BUDGET}_frame}"
QA_FILE=${REVSI_QA_FILE:-"${FRAME_DIR:+${FRAME_DIR}/test-00000-of-00001.parquet}"}
VIDEO_ROOT=${REVSI_VIDEO_ROOT:-${REVSI_ROOT}}
TASK_FILTER=${REVSI_TASK_FILTER:-}
MAX_SAMPLES=${REVSI_MAX_SAMPLES:-0}
REMOTE_SETUP=${REMOTE_SETUP:-}
SSH_STRICT_HOST_KEY_CHECKING=${SSH_STRICT_HOST_KEY_CHECKING:-yes}

case "${FRAME_BUDGET}" in
  16|32|64|all) ;;
  *)
    echo "[FATAL] REVSI_FRAME_BUDGET must be 16, 32, 64, or all." >&2
    exit 2
    ;;
esac
[[ -n "${MODEL}" ]] || {
  echo "[FATAL] Set MODEL to the model directory." >&2
  exit 2
}
[[ -n "${QA_FILE}" ]] || {
  echo "[FATAL] Set REVSI_QA_FILE or REVSI_ROOT." >&2
  exit 2
}
[[ -n "${HOSTFILE}" ]] || {
  echo "[FATAL] Set HOSTFILE to one host per line." >&2
  exit 2
}
for path in \
  "${RUNNER}" "${MERGER}" "${HOSTFILE}" "${MODEL}/config.json" "${QA_FILE}"; do
  [[ -f "${path}" ]] || {
    echo "[FATAL] Required input is unavailable: ${path}" >&2
    exit 1
  }
done
if [[ "${QA_FILE}" == *.parquet ]]; then
  [[ -n "${FRAME_DIR}" ]] || {
    echo "[FATAL] Parquet input requires REVSI_ROOT." >&2
    exit 2
  }
  compgen -G "${FRAME_DIR}/*.mp4" >/dev/null || {
    echo "[FATAL] No ReVSI videos found under ${FRAME_DIR}." >&2
    exit 1
  }
fi

if [[ -n "${REVSI_EXPECTED_SAMPLES:-}" ]]; then
  EXPECTED_SAMPLES=${REVSI_EXPECTED_SAMPLES}
elif [[ -z "${TASK_FILTER}" && "${MAX_SAMPLES}" == "0" ]]; then
  EXPECTED_SAMPLES=6808
else
  EXPECTED_SAMPLES=0
fi

mapfile -t AVAILABLE_HOSTS < <(awk 'NF && !seen[$1]++ {print $1}' "${HOSTFILE}")
[[ "${NUM_NODES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "[FATAL] NUM_NODES must be a positive integer." >&2
  exit 2
}
((NUM_NODES <= ${#AVAILABLE_HOSTS[@]})) || {
  echo "[FATAL] Requested ${NUM_NODES} nodes, only ${#AVAILABLE_HOSTS[@]} available." >&2
  exit 2
}
HOSTS=("${AVAILABLE_HOSTS[@]:0:NUM_NODES}")
GLOBAL_SHARDS=$((NUM_NODES * GPUS_PER_NODE))

checkpoint_dir=$(dirname "$(dirname "${MODEL%/}")")
checkpoint_tag=$(basename "${checkpoint_dir}")
experiment_tag=$(basename "$(dirname "${checkpoint_dir}")")
timestamp=$(date +%Y%m%d_%H%M%S)
EVAL_MAX_FRAMES=${REVSI_MAX_FRAMES:-${FRAME_BUDGET}}
EVAL_EXACT_NFRAMES=${REVSI_EXACT_NFRAMES:-1}
if [[ "${FRAME_BUDGET}" == "all" ]]; then
  EVAL_MAX_FRAMES=${REVSI_MAX_FRAMES:-128}
fi
setting_tag="revsi-${FRAME_BUDGET}frame-multinode${NUM_NODES}x${GPUS_PER_NODE}-f${EVAL_MAX_FRAMES}-exact${EVAL_EXACT_NFRAMES}-fps${REVSI_FPS:-2}-total${REVSI_VIDEO_TOTAL_PIXELS:-16777216}"
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_DIR}/outputs/${experiment_tag}/spatial_intelligence/revsi/${checkpoint_tag}/${setting_tag}/${timestamp}"}
LOG_DIR=${LOG_DIR:-"${PROJECT_DIR}/logs/revsi_multinode/${timestamp}"}
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

SSH_OPTS=(-o "StrictHostKeyChecking=${SSH_STRICT_HOST_KEY_CHECKING}" -o BatchMode=yes -o ServerAliveInterval=60)
WORKER_PATTERN='eval/task/revsi/eval_revsi_vllm.py'

run_on_host() {
  local node=$1
  shift
  if ((node == 0)); then
    bash -s <<<"$*"
  else
    ssh "${SSH_OPTS[@]}" "${HOSTS[node]}" bash -s <<<"$*"
  fi
}

kill_stale_workers() {
  local node reap_pids=() command
  command="pkill -f $(printf %q "${WORKER_PATTERN}") >/dev/null 2>&1 || true; sleep 5; pkill -9 -f $(printf %q "${WORKER_PATTERN}") >/dev/null 2>&1 || true; sleep 3"
  for ((node = 0; node < NUM_NODES; node++)); do
    run_on_host "${node}" "${command}" >/dev/null 2>&1 &
    reap_pids+=("$!")
  done
  wait "${reap_pids[@]}" 2>/dev/null || true
}

require_free_gpus() {
  local node free bad=0
  local need_gib=${REVSI_MIN_FREE_GIB:-75}
  for ((node = 0; node < NUM_NODES; node++)); do
    free=$(run_on_host "${node}" \
      "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | awk 'NR==1 {print}'" 2>/dev/null |
      tr -d '\r' | awk '/^[0-9]+$/ {value=$0} END {print value}')
    if [[ -z "${free}" ]]; then
      echo "[FATAL] node=${node} host=${HOSTS[node]}: cannot query GPU memory." >&2
      bad=1
    elif ((free / 1024 < need_gib)); then
      echo "[FATAL] node=${node} host=${HOSTS[node]}: only $((free / 1024)) GiB free (need ${need_gib} GiB)." >&2
      bad=1
    fi
  done
  ((bad == 0)) || exit 1
}

pids=()
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "${pids[@]:-}"; do
    [[ -n "${pid}" ]] && kill "${pid}" >/dev/null 2>&1 || true
  done
  ((status == 0)) || kill_stale_workers
  exit "${status}"
}
trap cleanup EXIT INT TERM

kill_stale_workers
require_free_gpus

dispatch() {
  local node=$1
  local offset=$((node * GPUS_PER_NODE))
  local gpu_ids
  gpu_ids=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
  local env_values=(
    "CUDA_VISIBLE_DEVICES=${gpu_ids}"
    "REVSI_ROOT=${REVSI_ROOT}"
    "FRAME_BUDGET=${FRAME_BUDGET}"
    "QA_FILE=${QA_FILE}"
    "VIDEO_ROOT=${VIDEO_ROOT}"
    "OUTPUT_DIR=${OUTPUT_DIR}"
    "GLOBAL_SHARD_COUNT=${GLOBAL_SHARDS}"
    "GLOBAL_SHARD_OFFSET=${offset}"
    "MERGE_SHARDS=0"
    "EXPECTED_SAMPLES=0"
    "TASK_FILTER=${TASK_FILTER}"
    "MAX_SAMPLES=${MAX_SAMPLES}"
    "MEDIA_MODE=video"
    "TP_SIZE=1"
    "BATCH_SIZE=${REVSI_BATCH_SIZE:-16}"
    "MAX_FRAMES=${EVAL_MAX_FRAMES}"
    "EXACT_NFRAMES=${EVAL_EXACT_NFRAMES}"
    "FPS=${REVSI_FPS:-2}"
    "VIDEO_MIN_PIXELS=${REVSI_VIDEO_MIN_PIXELS:-65536}"
    "VIDEO_MAX_PIXELS=${REVSI_VIDEO_MAX_PIXELS:-}"
    "VIDEO_TOTAL_PIXELS=${REVSI_VIDEO_TOTAL_PIXELS:-16777216}"
    "MAX_MODEL_LEN=${REVSI_MAX_MODEL_LEN:-32768}"
    "MAX_NEW_TOKENS=${REVSI_MAX_NEW_TOKENS:-64}"
    "GPU_MEM_UTIL=${REVSI_GPU_MEMORY_UTILIZATION:-0.90}"
    "STRICT_NUMERIC_PROMPT=${REVSI_STRICT_NUMERIC_PROMPT:-0}"
    "ENABLE_THINKING=${REVSI_ENABLE_THINKING:-false}"
    "SCORE_LOG_INTERVAL=${REVSI_SCORE_LOG_INTERVAL:-50}"
    "LAUNCH_DELAY=2"
    "VLLM_BASE_PORT=49000"
  )
  local prefix="" value
  for value in "${env_values[@]}"; do
    prefix+="$(printf %q "${value}") "
  done
  local setup_prefix=""
  if [[ -n "${REMOTE_SETUP}" ]]; then
    setup_prefix="${REMOTE_SETUP} && "
  fi
  local command="${setup_prefix}cd $(printf %q "${PROJECT_DIR}") && ${prefix}bash $(printf %q "${RUNNER}") $(printf %q "${MODEL}")"
  local log="${LOG_DIR}/node${node}.log"
  echo "[revsi] node=${node}/${NUM_NODES} host=${HOSTS[node]} shards=${offset}-$((offset + GPUS_PER_NODE - 1)) log=${log}"
  if ((node == 0)); then
    bash -lc "${command}" 2>&1 | tee "${log}"
  else
    ssh "${SSH_OPTS[@]}" "${HOSTS[node]}" \
      "bash -lc $(printf %q "${command}")" >"${log}" 2>&1 &
    pids+=("$!")
  fi
}

for ((node = 1; node < NUM_NODES; node++)); do
  dispatch "${node}"
done
dispatch 0

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
pids=()
((status == 0)) || {
  echo "[FATAL] One or more ReVSI nodes failed; inspect ${LOG_DIR}." >&2
  exit "${status}"
}

"${PYTHON_BIN}" "${MERGER}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-shards "${GLOBAL_SHARDS}" \
  --expected-samples "${EXPECTED_SAMPLES}"
chmod -R a+rX "${OUTPUT_DIR}" 2>/dev/null || true
echo "[revsi] Multi-node evaluation complete: ${OUTPUT_DIR}/summary.json"
