#!/bin/bash
# =============================================================================
# TimeLens-Bench Evaluation — HuggingFace inference (multi-GPU data parallel)
#
# Usage:
#   bash eval/run_eval.sh [MODEL_PATH] [ENABLE_THINKING] [OUTPUT_DIR]
#
# Examples:
#   # Base model evaluation
#   bash eval/run_eval.sh /path/to/Qwen3.5-9B
#
#   # Trained checkpoint evaluation
#   bash eval/run_eval.sh /path/to/checkpoint/huggingface
#
#   # With thinking mode
#   bash eval/run_eval.sh /path/to/model true
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# eval/task/temporal_grounding/run_eval.sh  →  repository root  (up 3 levels)
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ---------- paths ----------
MODEL_PATH="${1:-}"
ENABLE_THINKING="${2:-false}"
BENCH_DIR="${TIMELENS_BENCH_DIR:-${PROJECT_DIR}/data/eval/temporal_grounding}"

if [[ -z "$MODEL_PATH" || ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: MODEL_PATH must name an existing model directory." >&2
    exit 2
fi
if [[ ! -d "$BENCH_DIR" ]]; then
    echo "ERROR: TIMELENS_BENCH_DIR does not exist: $BENCH_DIR" >&2
    exit 2
fi

# ---------- eval settings ----------
DATASETS="${DATASETS:-charades-timelens}"
SETTING="${SETTING:-timelens-fps${FPS:-4}-min${MIN_TOKENS:-1}-total${TOTAL_TOKENS:-128000}-new${MAX_NEW_TOKENS:-128}-${DATASETS//,/_}}"
MIN_TOKENS="${MIN_TOKENS:-1}"
TOTAL_TOKENS="${TOTAL_TOKENS:-128000}"
MAX_PIXELS="${MAX_PIXELS:-409600}"
MAX_FRAMES="${MAX_FRAMES:-2048}"
FPS="${FPS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
STOP_AFTER_ANSWER="${STOP_AFTER_ANSWER:-true}"
PROMPT_MODE="${PROMPT_MODE:-same}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_SAMPLES="${TIMELENS_MAX_SAMPLES:-0}"
RESUME="${TIMELENS_RESUME:-1}"
MERGE_SHARDS="${TIMELENS_MERGE_SHARDS:-1}"
if ! [[ "$MAX_SAMPLES" =~ ^[0-9]+$ ]]; then
    echo "ERROR: TIMELENS_MAX_SAMPLES must be a non-negative integer." >&2
    exit 1
fi

# ---------- output dir layout ----------
# Default layout:
#   <OUTPUT_ROOT>/<DATASET>/<RUN_TAG>/
#
# If TIMELENS_USE_OUTPUT_ROOT_DIRECT=1, OUTPUT_ROOT is assumed to already be
# the full run directory and each dataset is written under:
#   <OUTPUT_ROOT>/<DATASET>/
MODEL_TAG=$(basename "${MODEL_PATH%/}")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT="${3:-${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/temporal_grounding}}"
RUN_TAG="eval_hf-${MODEL_TAG}-${TIMESTAMP}"
if [ "$ENABLE_THINKING" = "true" ]; then
    RUN_TAG="${RUN_TAG}-think"
fi
if [ "${NO_ANSWER_WRAP:-0}" = "1" ]; then
    RUN_TAG="${RUN_TAG}-naw"
fi

# ---------- hardware ----------
# PyTorch wheels bundle CUDA user-space libraries. Prefer those libraries so a
# cluster-wide CUDA toolkit cannot override them with an older nvJitLink ABI.
if [ "${TIMELENS_USE_SYSTEM_CUDA:-0}" = "1" ]; then
    if [ -z "${CUDA_HOME:-}" ] || [ ! -d "$CUDA_HOME" ]; then
        echo "ERROR: TIMELENS_USE_SYSTEM_CUDA=1 requires a valid CUDA_HOME." >&2
        exit 1
    fi
    export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
else
    PYTHON_NVIDIA_LIBS="$(python - <<'PY'
from pathlib import Path
import sysconfig

root = Path(sysconfig.get_path("purelib")) / "nvidia"
print(":".join(str(path) for path in sorted(root.glob("*/lib")) if path.is_dir()))
PY
)"
    PYTHON_RUNTIME_LIBS="$PYTHON_NVIDIA_LIBS"
    if [ -n "${CONDA_PREFIX:-}" ] && [ -d "$CONDA_PREFIX/lib" ]; then
        PYTHON_RUNTIME_LIBS="${PYTHON_RUNTIME_LIBS:+$PYTHON_RUNTIME_LIBS:}$CONDA_PREFIX/lib"
    fi
    if [ -n "$PYTHON_RUNTIME_LIBS" ]; then
        export LD_LIBRARY_PATH="$PYTHON_RUNTIME_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
fi

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $(($(nvidia-smi -L | wc -l)-1)))}"
NUM_GPUS=${#GPULIST[@]}
NUM_SHARDS="${TIMELENS_GLOBAL_SHARD_COUNT:-${TIMELENS_NUM_SHARDS:-$NUM_GPUS}}"
LOCAL_SHARDS="${TIMELENS_LOCAL_SHARD_COUNT:-$NUM_SHARDS}"
SHARD_OFFSET="${TIMELENS_GLOBAL_SHARD_OFFSET:-0}"
if ! [[ "$NUM_SHARDS" =~ ^[0-9]+$ ]] || [ "$NUM_SHARDS" -lt 1 ]; then
    echo "ERROR: TIMELENS_NUM_SHARDS must be a positive integer, got '$NUM_SHARDS'" >&2
    exit 1
fi
if ! [[ "$LOCAL_SHARDS" =~ ^[0-9]+$ ]] || [ "$LOCAL_SHARDS" -lt 1 ]; then
    echo "ERROR: TIMELENS_LOCAL_SHARD_COUNT must be positive, got '$LOCAL_SHARDS'" >&2
    exit 1
fi
if ! [[ "$SHARD_OFFSET" =~ ^[0-9]+$ ]] || \
   [ $((SHARD_OFFSET + LOCAL_SHARDS)) -gt "$NUM_SHARDS" ]; then
    echo "ERROR: invalid shard range offset=$SHARD_OFFSET count=$LOCAL_SHARDS global=$NUM_SHARDS" >&2
    exit 1
fi
for GPU_ID in "${GPULIST[@]}"; do
    if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
        echo "ERROR: CUDA_VISIBLE_DEVICES must contain comma-separated GPU ids, got '${CUDA_VISIBLE_DEVICES:-<unset>}'" >&2
        exit 1
    fi
done

echo "=============================================="
echo "TimeLens-Bench Evaluation (HuggingFace)"
echo "=============================================="
echo "Model:      $MODEL_PATH"
echo "Thinking:   $ENABLE_THINKING"
echo "Prompt mode:$PROMPT_MODE"
echo "NoAnswerWrap: ${NO_ANSWER_WRAP:-0}"
echo "Datasets:   $DATASETS"
echo "GPUs:       ${GPULIST[*]} (${NUM_GPUS} total)"
echo "Shards:     global=$NUM_SHARDS local=$LOCAL_SHARDS offset=$SHARD_OFFSET"
echo "Tokens:     min=$MIN_TOKENS, total=$TOTAL_TOKENS"
echo "Video:      max_pixels=$MAX_PIXELS, max_frames=$MAX_FRAMES"
echo "FPS:        $FPS"
echo "Decode:     max_new=$MAX_NEW_TOKENS, repetition_penalty=$REPETITION_PENALTY, stop_after_answer=$STOP_AFTER_ANSWER"
echo "Max samples:$MAX_SAMPLES per shard (0 means all)"
echo "Output root:$OUTPUT_ROOT"
echo "Run tag:    $RUN_TAG"
echo "=============================================="

mkdir -p "$OUTPUT_ROOT"

# ---------- cleanup handler ----------
PIDS=()
cleanup() {
    echo ""
    echo "Caught interrupt, killing all workers ..."
    for pid in "${PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo "All workers killed."
    exit 1
}
trap cleanup INT TERM

# ---------- run evaluation for each dataset ----------
IFS=',' read -ra DATASET_LIST <<< "$DATASETS"

for DATASET in "${DATASET_LIST[@]}"; do
    if [ "${TIMELENS_USE_OUTPUT_ROOT_DIRECT:-0}" = "1" ]; then
        OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}"
    else
        OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}/${RUN_TAG}"
    fi
    mkdir -p "$OUTPUT_DIR"
    if [ "$SHARD_OFFSET" -eq 0 ]; then
        SETTING="$SETTING" DATASET="$DATASET" OUTPUT_DIR="$OUTPUT_DIR" MODEL_PATH="$MODEL_PATH" \
        PROJECT_DIR="$PROJECT_DIR" \
        ENABLE_THINKING="$ENABLE_THINKING" BENCH_DIR="$BENCH_DIR" GPUS="${GPULIST[*]}" \
        NUM_GPUS="$NUM_GPUS" NUM_SHARDS="$NUM_SHARDS" LOCAL_SHARDS="$LOCAL_SHARDS" \
        SHARD_OFFSET="$SHARD_OFFSET" MIN_TOKENS="$MIN_TOKENS" TOTAL_TOKENS="$TOTAL_TOKENS" \
        MAX_PIXELS="$MAX_PIXELS" MAX_FRAMES="$MAX_FRAMES" FPS="$FPS" \
        MAX_NEW_TOKENS="$MAX_NEW_TOKENS" REPETITION_PENALTY="$REPETITION_PENALTY" \
        STOP_AFTER_ANSWER="$STOP_AFTER_ANSWER" PROMPT_MODE="$PROMPT_MODE" \
        NUM_WORKERS="$NUM_WORKERS" MAX_SAMPLES="$MAX_SAMPLES" \
        python - <<'PY'
import json
import os
import sys

project_dir = os.environ["PROJECT_DIR"]
sys.path.insert(0, os.path.join(project_dir, "eval", "task"))
try:
    from eval_prompt import PROMPT_WO_THINK, TIMELENS_OFFICIAL_PROMPT
except Exception:
    PROMPT_WO_THINK = None
    TIMELENS_OFFICIAL_PROMPT = None

prompt_mode = os.environ["PROMPT_MODE"]
prompt_template = (
    TIMELENS_OFFICIAL_PROMPT
    if prompt_mode == "timelens_official"
    else PROMPT_WO_THINK
)

config = {
    "task": "temporal_grounding",
    "setting": os.environ["SETTING"],
    "model_path": os.environ["MODEL_PATH"],
    "dataset": os.environ["DATASET"],
    "bench_dir": os.environ["BENCH_DIR"],
    "output_dir": os.environ["OUTPUT_DIR"],
    "enable_thinking": os.environ["ENABLE_THINKING"],
    "gpus": os.environ["GPUS"].split(),
    "num_gpus": int(os.environ["NUM_GPUS"]),
    "video": {
        "min_tokens": int(os.environ["MIN_TOKENS"]),
        "total_tokens": int(os.environ["TOTAL_TOKENS"]),
        "max_pixels": int(os.environ["MAX_PIXELS"]),
        "max_frames": int(os.environ["MAX_FRAMES"]),
        "fps": float(os.environ["FPS"]),
    },
    "decode": {
        "max_new_tokens": int(os.environ["MAX_NEW_TOKENS"]),
        "repetition_penalty": float(os.environ["REPETITION_PENALTY"]),
        "stop_after_answer": os.environ["STOP_AFTER_ANSWER"],
    },
    "runtime": {
        "num_shards": int(os.environ["NUM_SHARDS"]),
        "local_shards": int(os.environ["LOCAL_SHARDS"]),
        "shard_offset": int(os.environ["SHARD_OFFSET"]),
        "num_workers": int(os.environ["NUM_WORKERS"]),
        "max_samples_per_shard": int(os.environ["MAX_SAMPLES"]),
    },
    "prompt": {
        "mode": prompt_mode,
        "source": (
            "eval/task/eval_prompt.py:TIMELENS_OFFICIAL_PROMPT"
            if prompt_mode == "timelens_official"
            else "eval/task/eval_prompt.py:PROMPT_WO_THINK"
        ),
        "template": prompt_template,
    },
}
with open(os.path.join(os.environ["OUTPUT_DIR"], "run_config.json"), "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
PY
    fi
    echo ""
    echo ">>> Evaluating: $DATASET"
    echo "    Output:    $OUTPUT_DIR"
    PIDS=()
    SHARD_IDS=()

    for LOCAL_IDX in $(seq 0 $((LOCAL_SHARDS - 1))); do
        IDX=$((SHARD_OFFSET + LOCAL_IDX))
        RESULT_PATH="$OUTPUT_DIR/results_${DATASET}_shard${IDX}.json"
        SUMMARY_PATH="$OUTPUT_DIR/summary_shard${IDX}.json"
        if [ "$RESUME" = "1" ] && [ -s "$RESULT_PATH" ] && [ -s "$SUMMARY_PATH" ]; then
            echo "  [resume] Skipping completed shard $IDX/$NUM_SHARDS"
            continue
        fi
        GPU_ID="${GPULIST[$((LOCAL_IDX % NUM_GPUS))]}"
        (
            export CUDA_VISIBLE_DEVICES=$GPU_ID
            export PYTHONUNBUFFERED=1
            python - <<'PY'
import os
import torch

print(f"Worker CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
print(f"Worker torch: {torch.__version__}", flush=True)
print(f"Worker torch cuda: {torch.version.cuda}", flush=True)
print(f"Worker cuda available: {torch.cuda.is_available()}", flush=True)
print(f"Worker device count: {torch.cuda.device_count()}", flush=True)
PY
            python "${SCRIPT_DIR}/eval_timelens_hf.py" \
                --model_path "$MODEL_PATH" \
                --bench_dir "$BENCH_DIR" \
                --dataset "$DATASET" \
                --output_dir "$OUTPUT_DIR" \
                --enable_thinking "$ENABLE_THINKING" \
                --prompt_mode "$PROMPT_MODE" \
                --min_tokens $MIN_TOKENS \
                --total_tokens $TOTAL_TOKENS \
                --max_pixels $MAX_PIXELS \
                --max_frames $MAX_FRAMES \
                --fps $FPS \
                --max_new_tokens $MAX_NEW_TOKENS \
                --repetition_penalty $REPETITION_PENALTY \
                --stop_after_answer "$STOP_AFTER_ANSWER" \
                --num_workers $NUM_WORKERS \
                --max_samples $MAX_SAMPLES \
                --chunk $NUM_SHARDS \
                --index $IDX
        ) > "$OUTPUT_DIR/worker_${DATASET}_${IDX}.log" 2>&1 &
        PIDS+=($!)
        SHARD_IDS+=($IDX)
        echo "  Launched shard $IDX on GPU $GPU_ID (PID ${PIDS[-1]})"
    done

    echo "  Waiting for ${#PIDS[@]} local workers ..."
    FAILED=0
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            echo "  [DONE] shard ${SHARD_IDS[$i]} (PID ${PIDS[$i]})"
        else
            RC=$?
            echo "  [FAIL] shard ${SHARD_IDS[$i]} (PID ${PIDS[$i]}) exited with code $RC"
            FAILED=1
        fi
    done

    if [ $FAILED -ne 0 ]; then
        echo "  Some workers failed. Check logs: $OUTPUT_DIR/worker_${DATASET}_*.log"
        exit 1
    fi
    if [ "$MERGE_SHARDS" != "1" ]; then
        echo "  Local shard range completed; centralized merge disabled."
        continue
    fi

    # ---------- merge shard results ----------
    echo "  Merging results ..."
    python -c "
import json, os, sys

output_dir = sys.argv[1]
dataset = sys.argv[2]
num_shards = int(sys.argv[3])

all_samples = []
for sid in range(num_shards):
    path = os.path.join(output_dir, f'results_{dataset}_shard{sid}.json')
    if os.path.isfile(path):
        with open(path) as f:
            all_samples.extend(json.load(f))

n = len(all_samples)
if n == 0:
    print(f'  No results for {dataset}')
    sys.exit(0)

ious = [s['iou'] for s in all_samples]
thresholds = [0.3, 0.5, 0.7]
metrics = {
    'num_samples': n,
    'mIoU': round(sum(ious) / n * 100, 2),
}
for t in thresholds:
    metrics[f'R@{t}'] = round(sum(1 for x in ious if x >= t) / n * 100, 2)
n_parsed = sum(1 for s in all_samples if s.get('pred_span') is not None)
metrics['parse_rate'] = round(n_parsed / n * 100, 2)

print(f'  {dataset}: n={n}  mIoU={metrics[\"mIoU\"]:.2f}%  R@0.3={metrics[\"R@0.3\"]:.2f}%  R@0.5={metrics[\"R@0.5\"]:.2f}%  R@0.7={metrics[\"R@0.7\"]:.2f}%  Parse={metrics[\"parse_rate\"]:.2f}%')

# Save merged results
merged_path = os.path.join(output_dir, f'results_{dataset}.json')
with open(merged_path, 'w') as f:
    json.dump(all_samples, f, ensure_ascii=False, indent=2)

# Save/update summary
summary_path = os.path.join(output_dir, 'summary.json')
summary = {}
if os.path.isfile(summary_path):
    with open(summary_path) as f:
        summary = json.load(f)
summary[dataset] = metrics
with open(summary_path, 'w') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# Cleanup shard files. Keep worker logs so per-sample generation traces
# survive the merge; only the redundant shard JSONs (already folded into
# results_{dataset}.json) are removed. Set TIMELENS_KEEP_SHARDS=1 to keep all.
keep_shards = os.environ.get('TIMELENS_KEEP_SHARDS', '0') == '1'
if not keep_shards:
    for sid in range(num_shards):
        for pattern in [f'results_{dataset}_shard{sid}.json',
                        f'summary_shard{sid}.json']:
            p = os.path.join(output_dir, pattern)
            if os.path.isfile(p):
                os.remove(p)
" "$OUTPUT_DIR" "$DATASET" "$NUM_SHARDS"

done

echo ""
echo "=============================================="
echo "All evaluations complete!"
echo "Output root: $OUTPUT_ROOT"
echo "Run tag:     $RUN_TAG"
echo "=============================================="

# Print each dataset's summary, in order.
for DATASET in "${DATASET_LIST[@]}"; do
    if [ "${TIMELENS_USE_OUTPUT_ROOT_DIRECT:-0}" = "1" ]; then
        SUMMARY="${OUTPUT_ROOT}/${DATASET}/summary.json"
    else
        SUMMARY="${OUTPUT_ROOT}/${DATASET}/${RUN_TAG}/summary.json"
    fi
    if [ -f "$SUMMARY" ]; then
        echo ""
        echo "[${DATASET}]  -> $SUMMARY"
        cat "$SUMMARY"
    fi
done
echo ""
