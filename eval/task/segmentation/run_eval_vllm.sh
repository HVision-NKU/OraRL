#!/bin/bash
# =============================================================================
# Segmentation evaluation — vLLM inference + optional SAM2 post-processing.
#
# Usage:
#   bash eval/task/segmentation/run_eval_vllm.sh [MODEL_PATH] [OUTPUT_DIR]
#
# Common overrides:
#   BENCH_DIR=/path/to/OneThinker-eval DATASETS=reasonseg-val DATA_ROOT=/path/to/OneThinker-eval \
#   RUN_SAM2=true SAM2_CKPT=/path/to/sam2.1_hiera_large.pt SAM2_CFG=/path/to/sam2.1_hiera_l.yaml \
#   bash eval/task/segmentation/run_eval_vllm.sh /path/to/ckpt
# =============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ---------- paths ----------
MODEL_PATH="${1:-}"
PROCESSOR_PATH="${PROCESSOR_PATH:-$MODEL_PATH}"
BENCH_DIR="${BENCH_DIR:-${PROJECT_DIR}/data/eval/segmentation}"
DATA_ROOT="${DATA_ROOT:-$BENCH_DIR}"
DATASETS="${DATASETS:-eval_seg_refcoco,eval_seg_refcocop,eval_seg_refcocog,eval_seg_mevis,eval_seg_reasonvos}"

if [[ -z "$MODEL_PATH" || ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: MODEL_PATH must name an existing model directory." >&2
    exit 2
fi
if [[ ! -d "$BENCH_DIR" || ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: BENCH_DIR and DATA_ROOT must name existing directories." >&2
    exit 2
fi

# ---------- eval settings ----------
DATA_TYPE="${DATA_TYPE:-all}"              # all | image | video
PROMPT_MODE="${PROMPT_MODE:-train_seg}"   # think | no_think | bare | onethink_system | train_seg
ENABLE_THINKING="${ENABLE_THINKING:-false}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_PIXELS_IMAGE="${MAX_PIXELS_IMAGE:-1048576}"
MIN_PIXELS_IMAGE="${MIN_PIXELS_IMAGE:-4096}"
# Accept both the tracking-style VIDEO_* names and older MAX_PIXELS_VIDEO names.
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-${MAX_PIXELS_VIDEO:-262144}}"
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-${MIN_PIXELS_VIDEO:-4096}}"
VIDEO_TOTAL_PIXELS="${VIDEO_TOTAL_PIXELS:-${TOTAL_PIXELS_VIDEO:-16777216}}"
MAX_FRAMES="${MAX_FRAMES:-128}"
FPS="${FPS:-2}"
PATCH_SIZE="${PATCH_SIZE:-}"
VIDEO_READER="${VIDEO_READER:-decord}"

case "$VIDEO_READER" in
    decord|torchcodec|torchvision) ;;
    *)
        echo "ERROR: VIDEO_READER must be decord, torchcodec, or torchvision; got '$VIDEO_READER'" >&2
        exit 1
        ;;
esac
export FORCE_QWENVL_VIDEO_READER="$VIDEO_READER"

require_int() {
    local name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "ERROR: $name must be a non-empty integer, got '$value'"
        exit 1
    fi
}

require_int MAX_PIXELS_IMAGE "$MAX_PIXELS_IMAGE"
require_int MIN_PIXELS_IMAGE "$MIN_PIXELS_IMAGE"
require_int VIDEO_MAX_PIXELS "$VIDEO_MAX_PIXELS"
require_int VIDEO_MIN_PIXELS "$VIDEO_MIN_PIXELS"
require_int VIDEO_TOTAL_PIXELS "$VIDEO_TOTAL_PIXELS"
require_int MAX_FRAMES "$MAX_FRAMES"
require_int FPS "$FPS"
require_int BATCH_SIZE "$BATCH_SIZE"
require_int MAX_NEW_TOKENS "$MAX_NEW_TOKENS"

# ---------- vLLM ----------
TP_SIZE="${TP_SIZE:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
SEED="${SEED:-42}"
VLLM_BASE_PORT="${VLLM_BASE_PORT:-}"
RESUME_SHARDS="${RESUME_SHARDS:-false}"
RETRY_FAILED_SHARDS="${RETRY_FAILED_SHARDS:-true}"

# ---------- SAM2 ----------
RUN_SAM2="${RUN_SAM2:-false}"
SAM2_CKPT="${SAM2_CKPT:-}"
SAM2_CFG="${SAM2_CFG:-}"
ONETHINKER_SEG_POST="${ONETHINKER_SEG_POST:-${PROJECT_DIR}/third_party/OneThinker/Evaluation/Eval/seg_post_sam2.py}"
SAM2_NUM_GPUS="${SAM2_NUM_GPUS:-}"
SAM2_WORKERS_PER_GPU="${SAM2_WORKERS_PER_GPU:-}"
PRE_EXTRACT_THREADS="${PRE_EXTRACT_THREADS:-4}"
# Each SAM2 epoch spawns `world_size` worker processes that EACH reload the SAM2
# model and then handle only their slice of the epoch. Total model loads =
# world_size * num_epochs. A small epoch with many workers means tiny slices and
# constant model reloading (the real bottleneck). Keep the epoch large so there
# is effectively one epoch and each worker amortizes its model load over a big
# contiguous slice.
SAM2_EPOCH_SIZE="${SAM2_EPOCH_SIZE:-100000}"
VIZ_RATIO="${VIZ_RATIO:-0.0}"

# ---------- output ----------
MODEL_TAG=$(basename "${MODEL_PATH%/}")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${2:-${PROJECT_DIR}/outputs/segmentation/eval_seg_vllm-${MODEL_TAG}-${TIMESTAMP}}"

# ---------- hardware ----------
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

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS="," read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
else
    IFS="," read -ra GPULIST <<< "$(seq -s, 0 $(($(nvidia-smi -L | wc -l)-1)))"
fi
NUM_GPUS=${#GPULIST[@]}
if (( NUM_GPUS % TP_SIZE != 0 )); then
    echo "ERROR: NUM_GPUS ($NUM_GPUS) must be divisible by TP_SIZE ($TP_SIZE)"
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
for base in range(45000, 64000 - spacing * count, 128):
    sockets = []
    try:
        for index in range(count):
            for offset in (0, 1):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(("127.0.0.1", base + index * spacing + offset))
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
    raise SystemExit("no free segmentation vLLM port block found")
PY
    )"
fi

echo "=============================================="
echo "Segmentation Evaluation (vLLM)"
echo "=============================================="
echo "Model:       $MODEL_PATH"
echo "Processor:   $PROCESSOR_PATH"
echo "Bench dir:   $BENCH_DIR"
echo "Data root:   $DATA_ROOT"
echo "Datasets:    $DATASETS"
echo "Data type:   $DATA_TYPE"
echo "Prompt:      $PROMPT_MODE (enable_thinking=$ENABLE_THINKING)"
echo "Video pix:   min=$VIDEO_MIN_PIXELS max=$VIDEO_MAX_PIXELS total=$VIDEO_TOTAL_PIXELS frames=$MAX_FRAMES fps=$FPS"
echo "Video reader:$VIDEO_READER"
echo "GPUs:        ${GPULIST[*]} (${NUM_GPUS} total, TP=${TP_SIZE}, DP=${DP_SIZE})"
echo "Base port:   $VLLM_BASE_PORT"
echo "Output:      $OUTPUT_DIR"
echo "SAM2:        $RUN_SAM2"
echo "=============================================="

mkdir -p "$OUTPUT_DIR"

PY_ARGS=(
    --model_path "$MODEL_PATH"
    --processor_path "$PROCESSOR_PATH"
    --bench_dir "$BENCH_DIR"
    --datasets "$DATASETS"
    --output_dir "$OUTPUT_DIR"
    --base_prefix "$DATA_ROOT"
    --data_type "$DATA_TYPE"
    --prompt_mode "$PROMPT_MODE"
    --batch_size "$BATCH_SIZE"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --tensor_parallel_size "$TP_SIZE"
    --gpu_memory_utilization "$GPU_MEM_UTIL"
    --max_model_len "$MAX_MODEL_LEN"
    --seed "$SEED"
    --max_pixels_image "$MAX_PIXELS_IMAGE"
    --min_pixels_image "$MIN_PIXELS_IMAGE"
    --max_pixels_video "$VIDEO_MAX_PIXELS"
    --min_pixels_video "$VIDEO_MIN_PIXELS"
    --total_pixels_video "$VIDEO_TOTAL_PIXELS"
    --max_frames "$MAX_FRAMES"
    --fps "$FPS"
    --skip_missing_media
)

if [ -n "$MAX_SAMPLES" ]; then
    PY_ARGS+=(--max_samples "$MAX_SAMPLES")
fi
if [ -n "$PATCH_SIZE" ]; then
    PY_ARGS+=(--patch_size "$PATCH_SIZE")
fi
if [ "$ENABLE_THINKING" = "true" ]; then
    PY_ARGS+=(--enable_thinking)
fi

PIDS=()
PID_SHARDS=()
cleanup() {
    echo ""
    echo "Caught interrupt, killing workers ..."
    for pid in "${PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    exit 1
}
trap cleanup INT TERM

dataset_key() {
    python -c \
        "from pathlib import Path; import sys; p=sys.argv[1]; print(Path(p).stem if Path(p).suffix else p)" \
        "$1"
}

shard_is_complete() {
    local shard_id="$1"
    local dataset
    local key
    local datasets=()
    IFS=',' read -ra datasets <<< "$DATASETS"
    for dataset in "${datasets[@]}"; do
        key="$(dataset_key "$dataset")"
        if [[ ! -s "$OUTPUT_DIR/results_${key}_shard${shard_id}.json" ]]; then
            return 1
        fi
    done
    return 0
}

if [ "$DP_SIZE" -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPULIST[*]}") \
    VLLM_PORT="$VLLM_BASE_PORT" \
    VLLM_HOST_IP=127.0.0.1 \
    MASTER_PORT="$((VLLM_BASE_PORT + 1))" \
    MASTER_ADDR=127.0.0.1 \
    PYTHONUNBUFFERED=1 python "$SCRIPT_DIR/eval_seg_vllm.py" "${PY_ARGS[@]}" \
        2>&1 | tee "$OUTPUT_DIR/run.log"
else
    echo ""
    echo ">>> Launching $DP_SIZE vLLM workers (TP=$TP_SIZE each) ..."
    for IDX in $(seq 0 $((DP_SIZE - 1))); do
        if [[ "$RESUME_SHARDS" == "true" ]] && shard_is_complete "$IDX"; then
            echo "  Reusing completed shard $IDX"
            continue
        fi
        START=$(( IDX * TP_SIZE ))
        SHARD_PORT=$((VLLM_BASE_PORT + IDX * 16))
        SHARD_MASTER_PORT=$((SHARD_PORT + 1))
        SHARD_GPUS=""
        for j in $(seq 0 $((TP_SIZE - 1))); do
            g=${GPULIST[$((START + j))]}
            SHARD_GPUS="${SHARD_GPUS}${SHARD_GPUS:+,}${g}"
        done

        CUDA_VISIBLE_DEVICES="$SHARD_GPUS" \
        VLLM_PORT="$SHARD_PORT" \
        VLLM_HOST_IP=127.0.0.1 \
        MASTER_PORT="$SHARD_MASTER_PORT" \
        MASTER_ADDR=127.0.0.1 \
        PYTHONUNBUFFERED=1 \
        python "$SCRIPT_DIR/eval_seg_vllm.py" "${PY_ARGS[@]}" \
            --chunk "$DP_SIZE" --index "$IDX" \
            > "$OUTPUT_DIR/worker_shard${IDX}.log" 2>&1 &
        PIDS+=($!)
        PID_SHARDS+=("$IDX")
        echo "  Launched shard $IDX on GPU $SHARD_GPUS (PID ${PIDS[-1]})"
    done

    FAILED_SHARDS=()
    for i in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$i]}"; then
            FAILED_SHARDS+=("${PID_SHARDS[$i]}")
        fi
    done

    if (( ${#FAILED_SHARDS[@]} > 0 )) && [[ "$RETRY_FAILED_SHARDS" == "true" ]]; then
        echo ""
        echo ">>> Retrying failed shards sequentially with fresh ports ..."
        RETRY_FAILURES=()
        for IDX in "${FAILED_SHARDS[@]}"; do
            START=$(( IDX * TP_SIZE ))
            SHARD_GPUS=""
            for j in $(seq 0 $((TP_SIZE - 1))); do
                g=${GPULIST[$((START + j))]}
                SHARD_GPUS="${SHARD_GPUS}${SHARD_GPUS:+,}${g}"
            done
            RETRY_PORT="$(
                python - <<'PY'
import socket

for base in range(52000, 64000, 8):
    sockets = []
    try:
        for offset in (0, 1):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", base + offset))
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
    raise SystemExit("no free vLLM retry ports found")
PY
            )"
            echo "  Retrying shard $IDX on GPU $SHARD_GPUS (ports $RETRY_PORT/$((RETRY_PORT + 1)))"
            if ! CUDA_VISIBLE_DEVICES="$SHARD_GPUS" \
                VLLM_PORT="$RETRY_PORT" \
                VLLM_HOST_IP=127.0.0.1 \
                MASTER_PORT="$((RETRY_PORT + 1))" \
                MASTER_ADDR=127.0.0.1 \
                PYTHONUNBUFFERED=1 \
                python "$SCRIPT_DIR/eval_seg_vllm.py" "${PY_ARGS[@]}" \
                    --chunk "$DP_SIZE" --index "$IDX" \
                    > "$OUTPUT_DIR/worker_shard${IDX}_retry.log" 2>&1; then
                RETRY_FAILURES+=("$IDX")
            fi
        done
        FAILED_SHARDS=("${RETRY_FAILURES[@]}")
    fi

    if (( ${#FAILED_SHARDS[@]} > 0 )); then
        echo "ERROR: Failed shards: ${FAILED_SHARDS[*]}; not merging or running SAM2."
        exit 1
    fi

    echo ""
    echo ">>> Merging shard results ..."
    IFS=',' read -ra DATASET_LIST <<< "$DATASETS"
    for DATASET in "${DATASET_LIST[@]}"; do
        DATASET_KEY=$(python -c "from pathlib import Path; import sys; p=sys.argv[1]; print(Path(p).stem if Path(p).suffix else p)" "$DATASET")
        python - "$OUTPUT_DIR" "$DATASET_KEY" "$DP_SIZE" <<'PY'
import json
import os
import sys

out_dir, dataset, num_shards = sys.argv[1], sys.argv[2], int(sys.argv[3])
all_samples = []
for sid in range(num_shards):
    path = os.path.join(out_dir, f"results_{dataset}_shard{sid}.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        all_samples.extend(payload.get("results", []))

n = len(all_samples)
if n == 0:
    print(f"  No results for {dataset}")
    sys.exit(0)

parse_ok = sum(1 for row in all_samples if row.get("parse_ok"))
summary = {
    "num_samples": n,
    "parse_ok": parse_ok,
    "parse_rate": round(parse_ok / n * 100.0, 2),
    "by_data_type": {},
}
for data_type in ("image", "video"):
    part = [r for r in all_samples if r.get("data_type") == data_type]
    if part:
        ok = sum(1 for r in part if r.get("parse_ok"))
        summary["by_data_type"][data_type] = {
            "num_samples": len(part),
            "parse_rate": round(ok / len(part) * 100.0, 2),
        }

with open(os.path.join(out_dir, f"results_{dataset}.json"), "w", encoding="utf-8") as f:
    json.dump({"results": all_samples, "metrics": summary}, f, ensure_ascii=False, indent=2)

summary_path = os.path.join(out_dir, "summary.json")
full_summary = json.load(open(summary_path, encoding="utf-8")) if os.path.isfile(summary_path) else {}
full_summary[dataset] = summary
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(full_summary, f, ensure_ascii=False, indent=2)

print(f"  {dataset}: n={n} parse={summary['parse_rate']:.2f}%")
PY
    done
fi

if [ "$RUN_SAM2" = "true" ]; then
    if [ -z "$SAM2_CKPT" ] || [ -z "$SAM2_CFG" ]; then
        echo "ERROR: RUN_SAM2=true requires SAM2_CKPT and SAM2_CFG."
        exit 1
    fi
    echo ""
    echo ">>> Running SAM2 post-processing ..."
    # SAM2 needs visible GPUs. The DP>1 inference path sets CUDA_VISIBLE_DEVICES
    # only inside per-worker subshells, so the parent env may be empty here; an
    # empty CUDA_VISIBLE_DEVICES makes torch.cuda.is_available() False and SAM2
    # falls back to slow CPU serial. Re-derive a non-empty device list.
    export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPULIST[*]}")"
    # Default SAM2 GPU count to all visible GPUs unless caller overrides.
    if [ -z "$SAM2_NUM_GPUS" ]; then
        SAM2_NUM_GPUS="$NUM_GPUS"
    fi
    echo "    SAM2 GPUs: $CUDA_VISIBLE_DEVICES (num_gpus=$SAM2_NUM_GPUS)"
    IFS=',' read -ra DATASET_LIST <<< "$DATASETS"
    for DATASET in "${DATASET_LIST[@]}"; do
        DATASET_KEY=$(python -c "from pathlib import Path; import sys; p=sys.argv[1]; print(Path(p).stem if Path(p).suffix else p)" "$DATASET")
        RESULT_JSON="$OUTPUT_DIR/results_${DATASET_KEY}.json"
        if [ ! -f "$RESULT_JSON" ]; then
            echo "  Skip $DATASET_KEY: missing $RESULT_JSON"
            continue
        fi
        SAM2_ARGS=(
            --input_json "$RESULT_JSON"
            --data_root "$DATA_ROOT"
            --sam2_ckpt "$SAM2_CKPT"
            --sam2_cfg "$SAM2_CFG"
            --onethinker_script "$ONETHINKER_SEG_POST"
            --pre_extract_threads "$PRE_EXTRACT_THREADS"
            --epoch_size "$SAM2_EPOCH_SIZE"
            --viz_ratio "$VIZ_RATIO"
        )
        if [ -n "$SAM2_NUM_GPUS" ]; then
            SAM2_ARGS+=(--num_gpus "$SAM2_NUM_GPUS")
        fi
        if [ -n "$SAM2_WORKERS_PER_GPU" ]; then
            SAM2_ARGS+=(--workers_per_gpu "$SAM2_WORKERS_PER_GPU")
        fi
        PYTHONUNBUFFERED=1 python "$SCRIPT_DIR/post_sam2.py" "${SAM2_ARGS[@]}" \
            2>&1 | tee "$OUTPUT_DIR/sam2_${DATASET_KEY}.log"

        # Merge SAM2 metrics (cIoU / gIoU / J&F) back into summary.json.
        SAM2_JSON="$OUTPUT_DIR/results_${DATASET_KEY}_sam2.json"
        if [ -f "$SAM2_JSON" ]; then
            python - "$OUTPUT_DIR" "$DATASET_KEY" "$SAM2_JSON" <<'PY'
import json
import os
import sys

out_dir, dataset, sam2_json = sys.argv[1], sys.argv[2], sys.argv[3]
with open(sam2_json, encoding="utf-8") as f:
    payload = json.load(f)

metrics = payload.get("metrics", {})
avg_rewards = payload.get("avg_rewards", {})

summary_path = os.path.join(out_dir, "summary.json")
full_summary = json.load(open(summary_path, encoding="utf-8")) if os.path.isfile(summary_path) else {}
entry = full_summary.get(dataset, {})
if isinstance(metrics, dict):
    for k in ("num_samples", "parse_ok", "parse_rate"):
        if k in metrics:
            entry[k] = metrics[k]
entry["avg_rewards"] = avg_rewards
full_summary[dataset] = entry
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(full_summary, f, ensure_ascii=False, indent=2)

parts = ", ".join(f"{k}={v:.4f}" for k, v in avg_rewards.items())
print(f"  {dataset}: {parts}")
PY
        fi
    done
fi

echo ""
echo "=============================================="
echo "Done. Summary: $OUTPUT_DIR/summary.json"
echo "=============================================="
