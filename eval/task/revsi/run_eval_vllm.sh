#!/bin/bash
# =============================================================================
# ReVSI evaluation launcher — vLLM, data-parallel only.
#
# Usage:
#   bash eval/task/revsi/run_eval_vllm.sh [MODEL_PATH]
#
# Env overrides:
#   REVSI_ROOT     root containing 16_frame/32_frame/64_frame/all_frame
#   FRAME_BUDGET   16, 32, 64, or all (default: all)
#   QA_FILE        defaults to FRAME_BUDGET/test-00000-of-00001.parquet
#   OUTPUT_ROOT    default: outputs/revsi
#   OUTPUT_DIR     optional existing run dir; set this to resume an interrupted eval
#   TASK_FILTER    comma-separated question_type list, optional
#   MAX_SAMPLES    optional quick debug cap
#   TP_SIZE        default 1
#   MAX_MODEL_LEN  default 32768
#   MAX_NEW_TOKENS default 64
#   BATCH_SIZE     default 16
#   GPU_MEM_UTIL   default 0.90
#   MEDIA_MODE     image (default) or video
#   VIDEO_ROOT     root for mp4s in video mode, e.g. VSI-590K root
#   MAX_FRAMES/FPS/VIDEO_TOTAL_PIXELS  video sampling controls
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODEL_PATH="${1:-${MODEL_PATH:-}}"
REVSI_ROOT="${REVSI_ROOT:-}"
FRAME_BUDGET="${FRAME_BUDGET:-all}"
FRAME_DIR="${REVSI_ROOT:+${REVSI_ROOT}/${FRAME_BUDGET}_frame}"
QA_FILE="${QA_FILE:-${FRAME_DIR:+${FRAME_DIR}/test-00000-of-00001.parquet}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/revsi}"
TASK_FILTER="${TASK_FILTER:-}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
STRICT_NUMERIC_PROMPT="${STRICT_NUMERIC_PROMPT:-0}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
MEDIA_MODE="${MEDIA_MODE:-video}"
VIDEO_ROOT="${VIDEO_ROOT:-${REVSI_ROOT}}"
MAX_FRAMES="${MAX_FRAMES:-${FRAME_BUDGET/all/128}}"
EXACT_NFRAMES="${EXACT_NFRAMES:-1}"
FPS="${FPS:-2}"
VIDEO_TOTAL_PIXELS="${VIDEO_TOTAL_PIXELS:-16777216}"
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-65536}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-}"

TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
SCORE_LOG_INTERVAL="${SCORE_LOG_INTERVAL:-200}"
GLOBAL_SHARD_COUNT="${GLOBAL_SHARD_COUNT:-}"
GLOBAL_SHARD_OFFSET="${GLOBAL_SHARD_OFFSET:-0}"
MERGE_SHARDS="${MERGE_SHARDS:-1}"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-0}"
VLLM_BASE_PORT="${VLLM_BASE_PORT:-}"
LAUNCH_DELAY="${LAUNCH_DELAY:-2}"

case "${FRAME_BUDGET}" in
    16|32|64|all) ;;
    *)
        echo "ERROR: FRAME_BUDGET must be 16, 32, 64, or all." >&2
        exit 2
        ;;
esac
[[ -n "${MODEL_PATH}" ]] || {
    echo "ERROR: pass MODEL_PATH as the first argument or environment variable." >&2
    exit 2
}
[[ -n "${QA_FILE}" ]] || {
    echo "ERROR: set QA_FILE or REVSI_ROOT." >&2
    exit 2
}
for path in "${MODEL_PATH}/config.json" "${QA_FILE}"; do
    [[ -f "${path}" ]] || {
        echo "ERROR: required input is unavailable: ${path}" >&2
        exit 1
    }
done
if [[ "${QA_FILE}" == *.parquet ]]; then
    [[ -n "${FRAME_DIR}" ]] || {
        echo "ERROR: parquet input requires REVSI_ROOT to locate frame videos." >&2
        exit 2
    }
    compgen -G "${FRAME_DIR}/*.mp4" >/dev/null || {
        echo "ERROR: no ReVSI videos found under ${FRAME_DIR}; extract video.zip first." >&2
        exit 1
    }
fi

# Prefer the CUDA runtime libraries installed alongside PyTorch. In particular,
# pip/conda CUDA 12.9 builds need their matching nvJitLink ahead of an older
# system CUDA toolkit that may already be present in LD_LIBRARY_PATH.
NVJITLINK_LIB="$(
    python - <<'PY'
import site
from pathlib import Path

roots = [*site.getsitepackages(), site.getusersitepackages()]
for root in roots:
    candidate = Path(root) / "nvidia" / "nvjitlink" / "lib"
    if candidate.is_dir():
        print(candidate)
        break
PY
)"
if [[ -n "${NVJITLINK_LIB}" ]]; then
    export LD_LIBRARY_PATH="${NVJITLINK_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if ! python -c "import torch; print(f'PyTorch preflight: {torch.__version__} CUDA {torch.version.cuda}')"; then
    echo "ERROR: PyTorch CUDA libraries cannot be loaded in the active environment." >&2
    exit 1
fi

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $(($(nvidia-smi -L | wc -l)-1)))}"
NUM_GPUS=${#GPULIST[@]}
# Each vLLM subprocess must inherit only its worker-specific GPU mask.
unset CUDA_VISIBLE_DEVICES
if (( NUM_GPUS % TP_SIZE != 0 )); then
    echo "ERROR: NUM_GPUS=$NUM_GPUS must be divisible by TP_SIZE=$TP_SIZE"
    exit 1
fi
DP_SIZE=$(( NUM_GPUS / TP_SIZE ))
if [[ -z "${VLLM_BASE_PORT}" ]]; then
    VLLM_BASE_PORT="$(
        python - "$DP_SIZE" <<'PY'
import socket
import sys

count = int(sys.argv[1])
spacing = 16
for base in range(48000, 64000 - spacing * count, 128):
    sockets = []
    try:
        for index in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", base + index * spacing))
            sockets.append(sock)
    except OSError:
        for sock in sockets:
            sock.close()
        continue
    for sock in sockets:
        sock.close()
    print(base)
    break
else:
    raise SystemExit("no free ReVSI vLLM port block found")
PY
    )"
fi

MODEL_TAG=$(basename "${MODEL_PATH%/}")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_TAG="${RUN_TAG:-eval_revsi_${FRAME_BUDGET}frame_vllm-${MODEL_TAG}-${TIMESTAMP}}"
# If OUTPUT_DIR points to an existing partial run, eval_revsi_vllm.py will skip
# IDs already present in results_shard*.jsonl and continue the remaining samples.
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_TAG}}"
mkdir -p "$OUTPUT_DIR"
export MODEL_PATH QA_FILE VIDEO_ROOT FRAME_BUDGET MAX_FRAMES EXACT_NFRAMES FPS
export VIDEO_TOTAL_PIXELS VIDEO_MIN_PIXELS VIDEO_MAX_PIXELS
export MAX_MODEL_LEN MAX_NEW_TOKENS BATCH_SIZE TP_SIZE MAX_SAMPLES
export EXPECTED_SAMPLES ENABLE_THINKING VLLM_BASE_PORT
python - "$OUTPUT_DIR/run_config.json" <<'PY'
import json
import os
import sys

keys = (
    "MODEL_PATH",
    "QA_FILE",
    "VIDEO_ROOT",
    "FRAME_BUDGET",
    "MAX_FRAMES",
    "EXACT_NFRAMES",
    "FPS",
    "VIDEO_TOTAL_PIXELS",
    "VIDEO_MIN_PIXELS",
    "VIDEO_MAX_PIXELS",
    "MAX_MODEL_LEN",
    "MAX_NEW_TOKENS",
    "BATCH_SIZE",
    "TP_SIZE",
    "MAX_SAMPLES",
    "EXPECTED_SAMPLES",
    "ENABLE_THINKING",
    "VLLM_BASE_PORT",
)
payload = {key.lower(): os.environ.get(key, "") for key in keys}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

cat <<EOF
==============================================
ReVSI Evaluation (vLLM, data-parallel)
==============================================
Model:        $MODEL_PATH
QA file:      $QA_FILE
Output:       $OUTPUT_DIR
GPUs:         ${GPULIST[*]} (${NUM_GPUS} total, TP=${TP_SIZE}, DP=${DP_SIZE})
Max tokens:   new=$MAX_NEW_TOKENS model_len=$MAX_MODEL_LEN
Batch:        $BATCH_SIZE
Task filter:  ${TASK_FILTER:-<none>}
Max samples:  ${MAX_SAMPLES}
Strict num:   ${STRICT_NUMERIC_PROMPT}
Thinking:     ${ENABLE_THINKING}
Media mode:   ${MEDIA_MODE}
Video root:   ${VIDEO_ROOT:-<none>}
Base port:    ${VLLM_BASE_PORT}
Frame budget: ${FRAME_BUDGET} (exact_nframes=${EXACT_NFRAMES})
Video:        max_frames=${MAX_FRAMES} fps=${FPS} total_pixels=${VIDEO_TOTAL_PIXELS}
==============================================
EOF

PIDS=()
cleanup() {
    echo ""; echo "Caught interrupt, killing workers ..."
    for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
    exit 1
}
trap cleanup INT TERM

EFFECTIVE_SHARD_COUNT="${GLOBAL_SHARD_COUNT:-$DP_SIZE}"
for IDX in $(seq 0 $((DP_SIZE - 1))); do
    START=$(( IDX * TP_SIZE ))
    GLOBAL_IDX=$((GLOBAL_SHARD_OFFSET + IDX))
    SHARD_PORT=$((VLLM_BASE_PORT + IDX * 16))
    SHARD_GPUS=""
    for j in $(seq 0 $((TP_SIZE - 1))); do
        g=${GPULIST[$((START + j))]}
        SHARD_GPUS="${SHARD_GPUS}${SHARD_GPUS:+,}${g}"
    done

    OUT_JSONL="${OUTPUT_DIR}/results_shard${GLOBAL_IDX}.jsonl"
    STRICT_FLAG=""
    if [ "$STRICT_NUMERIC_PROMPT" = "1" ] || [ "$STRICT_NUMERIC_PROMPT" = "true" ]; then
        STRICT_FLAG="--strict_numeric_prompt"
    fi
    VIDEO_MAX_PIXELS_ARGS=()
    if [ -n "$VIDEO_MAX_PIXELS" ]; then
        VIDEO_MAX_PIXELS_ARGS=(--video_max_pixels "$VIDEO_MAX_PIXELS")
    fi
    EXACT_NFRAMES_ARGS=()
    if [ "$EXACT_NFRAMES" = "1" ] || [ "$EXACT_NFRAMES" = "true" ]; then
        EXACT_NFRAMES_ARGS=(--exact_nframes)
    fi
    CUDA_VISIBLE_DEVICES="$SHARD_GPUS" \
    VLLM_PORT="$SHARD_PORT" \
    VLLM_HOST_IP=127.0.0.1 \
    MASTER_PORT="$SHARD_PORT" \
    MASTER_ADDR=127.0.0.1 \
    PYTHONUNBUFFERED=1 \
    python "${SCRIPT_DIR}/eval_revsi_vllm.py" \
        --output_json_path "$OUT_JSONL" \
        --model_path "$MODEL_PATH" \
        --qa_file "$QA_FILE" \
        --rank "$GLOBAL_IDX" \
        --world_size "$EFFECTIVE_SHARD_COUNT" \
        --tensor_parallel_size "$TP_SIZE" \
        --max_model_len "$MAX_MODEL_LEN" \
        --gpu_memory_utilization "$GPU_MEM_UTIL" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --batch_size "$BATCH_SIZE" \
        --score_log_interval "$SCORE_LOG_INTERVAL" \
        --task_filter "$TASK_FILTER" \
        --max_samples "$MAX_SAMPLES" \
        --media_mode "$MEDIA_MODE" \
        --video_root "$VIDEO_ROOT" \
        --max_frames "$MAX_FRAMES" \
        --fps "$FPS" \
        --video_total_pixels "$VIDEO_TOTAL_PIXELS" \
        --video_min_pixels "$VIDEO_MIN_PIXELS" \
        "${VIDEO_MAX_PIXELS_ARGS[@]}" \
        "${EXACT_NFRAMES_ARGS[@]}" \
        --enable_thinking "$ENABLE_THINKING" \
        $STRICT_FLAG \
        > "$OUTPUT_DIR/worker_${GLOBAL_IDX}.log" 2>&1 &
    PIDS+=($!)
    echo "Launched shard $GLOBAL_IDX/$EFFECTIVE_SHARD_COUNT on GPU $SHARD_GPUS (PID ${PIDS[-1]})"
    if [ "$IDX" -lt $((DP_SIZE - 1)) ] && [ "$LAUNCH_DELAY" -gt 0 ]; then
        sleep "$LAUNCH_DELAY"
    fi
done

echo "Waiting for ${DP_SIZE} workers ..."
FAILED=0
for i in "${!PIDS[@]}"; do
    RC=0
    wait "${PIDS[$i]}" || RC=$?
    if [ $RC -ne 0 ]; then
        echo "[FAIL] shard $i (PID ${PIDS[$i]}) exit=$RC"
        FAILED=1
    else
        echo "[DONE] shard $i (PID ${PIDS[$i]})"
    fi
done

if [ $FAILED -ne 0 ]; then
    echo "ERROR: some workers failed; not merging incomplete shards." >&2
    echo "Logs: $OUTPUT_DIR/worker_*.log" >&2
    exit 1
fi

if [ "$MERGE_SHARDS" != "1" ]; then
    echo "Local ReVSI shard range completed; centralized merge deferred."
    exit 0
fi

python "${SCRIPT_DIR}/merge_multinode_shards.py" \
    --output-dir "$OUTPUT_DIR" \
    --num-shards "$EFFECTIVE_SHARD_COUNT" \
    --expected-samples "$EXPECTED_SAMPLES"
echo "Done: $OUTPUT_DIR"
exit $FAILED
