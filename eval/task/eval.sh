#!/bin/bash
# Thin shell entry for evaluation.
#
# Supported TASKS: videomme,videommev2,videommmu,mmvu,mvbench,videoholmes,longvideobench,lvbench,mlvu,vsi,mmsi,mindcube,revsi,spatial_grounding,tracking,stvg,temporal_grounding,segmentation,all
# Common env overrides:
#   MODEL_PATH=/path/to/model_or_verl_actor
#   TASKS=videomme,videommev2,vsi
#   GPUS=0,1,2,3,4,5,6,7
#   TP_SIZE=1
# Prefer explicit CLI args for new runs, which avoids stale env variables:
#   bash eval/task/eval.sh --model /path/to/model --tasks videomme,videommev2,mvbench --gpus 0,1,2,3,4,5,6,7 --force-merge
#
# MODEL must be provided explicitly with --model.

# # —— Video QA / MC ——
# bash eval/task/eval.sh --model $MODEL --tasks mvbench        --gpus $GPUS --force-merge
# bash eval/task/eval.sh --model $MODEL --tasks mmvu            --gpus $GPUS --force-merge
# bash eval/task/eval.sh --model $MODEL --tasks videomme        --gpus $GPUS --force-merge
# bash eval/task/eval.sh --model $MODEL --tasks videoholmes     --gpus $GPUS --force-merge
# bash eval/task/eval.sh --model $MODEL --tasks longvideobench  --gpus $GPUS --force-merge
# bash eval/task/eval.sh --model $MODEL --tasks mlvu            --gpus $GPUS --force-merge

# # —— Spatial intelligence (MC) ——
# bash eval/task/eval.sh --model $MODEL --tasks vsi             --gpus $GPUS --force-merge
# bash eval/task/eval.sh --model $MODEL --tasks mmsi            --gpus $GPUS --force-merge
# bash eval/task/eval.sh --model $MODEL --tasks mindcube        --gpus $GPUS --force-merge
# bash eval/task/eval.sh --model $MODEL --tasks revsi           --gpus $GPUS --force-merge

# # —— Grounding / tracking / stvg / temporal ——
# bash eval/task/eval.sh --model $MODEL --tasks spatial_grounding   --gpus $GPUS --force-merge
# bash eval/task/eval.sh --model $MODEL --tasks tracking            --gpus $GPUS --force-merge
# bash eval/task/eval.sh --model $MODEL --tasks stvg                --gpus $GPUS --force-merge
# charades（默认）
# TIMELENS_DATASETS=charades-timelens TIMELENS_FPS=4 bash eval/task/eval.sh --model $MODEL --tasks temporal_grounding --gpus $GPUS --force-merge
# TIMELENS_DATASETS=activitynet-timelens TIMELENS_FPS=2 bash eval/task/eval.sh --model $MODEL --tasks temporal_grounding --gpus $GPUS --force-merge
# TIMELENS_DATASETS=qvhighlights-timelens TIMELENS_FPS=2 bash eval/task/eval.sh --model $MODEL --tasks temporal_grounding --gpus $GPUS --force-merge



# # —— Segmentation（需要 SAM2 + OraRL 评测环境）——
# EVAL_CONDA_ENV=orarl SEGMENTATION_RUN_SAM2=true \
#   bash eval/task/eval.sh --model $MODEL --tasks segmentation --gpus $GPUS --force-merge

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_GPUS="0,1,2,3,4,5,6,7"

usage() {
  cat <<'EOF'
Usage:
  bash eval/task/eval.sh [options]

Options:
  --model PATH           Model or checkpoint path.
  --tasks LIST           Comma-separated task list, e.g. videomme,videommev2,mvbench,mlvu.
  --gpus LIST            Comma-separated GPU ids, e.g. 0,1,2,3,4,5,6,7.
  --tp-size N            Tensor parallel size.
  --shard-mode MODE      Sharding mode passed to vLLM task runner.
  --video-cache-size N   Video cache size.
  --videomme-max-new-tokens N
                         Max new tokens for VideoMME generation.
  --force-merge          Force merging FSDP actor checkpoints.
  --skip-merge           Reuse existing merged HF model for actor checkpoints.
  --merged-model PATH    Explicit merged HF output/input path for actor checkpoints.
  --base-model PATH      Base model path used when merging actor checkpoints.
  --env NAME             Conda env to activate. Default: keep current env.
  --segmentation-run-sam2
                         Enable SAM2 post-processing for segmentation.
  --help                 Show this help.

Core launch settings intentionally ignore inherited environment variables to
avoid stale MODEL_PATH/TASKS/GPUS/etc. Use explicit CLI options instead.
EOF
}

# Avoid stale shell environment contaminating evaluation identity. Task-specific
# tuning envs below are still supported, but core launch identity is CLI/defaults.
unset MODEL_PATH TASKS GPUS TP_SIZE SHARD_MODE VIDEO_CACHE_SIZE
unset FORCE_MERGE SKIP_MERGE MERGED_MODEL_PATH BASE_MODEL_PATH EVAL_CONDA_ENV

while (($# > 0)); do
  case "$1" in
    --model|--model-path)
      [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value" >&2; exit 1; }
      MODEL_PATH="$2"
      shift 2
      ;;
    --tasks)
      [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value" >&2; exit 1; }
      TASKS="$2"
      shift 2
      ;;
    --gpus)
      if [[ $# -ge 2 && "$2" != --* ]]; then
        GPUS="$2"
        shift 2
      else
        GPUS="$DEFAULT_GPUS"
        shift
      fi
      ;;
    --tp-size)
      [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value" >&2; exit 1; }
      TP_SIZE="$2"
      shift 2
      ;;
    --shard-mode)
      [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value" >&2; exit 1; }
      SHARD_MODE="$2"
      shift 2
      ;;
    --video-cache-size)
      [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value" >&2; exit 1; }
      VIDEO_CACHE_SIZE="$2"
      shift 2
      ;;
    --videomme-max-new-tokens)
      [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value" >&2; exit 1; }
      VIDEOMME_MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --force-merge)
      FORCE_MERGE=1
      shift
      ;;
    --skip-merge)
      SKIP_MERGE=1
      shift
      ;;
    --merged-model|--merged-model-path)
      [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value" >&2; exit 1; }
      MERGED_MODEL_PATH="$2"
      shift 2
      ;;
    --base-model|--base-model-path)
      [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value" >&2; exit 1; }
      BASE_MODEL_PATH="$2"
      shift 2
      ;;
    --env|--conda-env)
      [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value" >&2; exit 1; }
      EVAL_CONDA_ENV="$2"
      shift 2
      ;;
    --segmentation-run-sam2)
      SEGMENTATION_RUN_SAM2=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "ERROR: unknown option '$1'" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# Require an explicit model so stale machine-local defaults cannot contaminate
# benchmark results.
if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "ERROR: --model PATH is required." >&2
  usage >&2
  exit 2
fi

MODEL_PATH="$(printf '%s' "$MODEL_PATH" | xargs)"

# Common launch defaults.
: "${TASKS:=all}"
: "${GPUS:=$DEFAULT_GPUS}"
: "${TP_SIZE:=1}"
: "${SHARD_MODE:=contiguous}"
: "${VIDEO_CACHE_SIZE:=2}"
: "${PREFETCH_BATCHES:=1}"
: "${FORCE_MERGE:=0}"
: "${SKIP_MERGE:=0}"
: "${MERGE_LOG_DIR:=${PROJECT_DIR}/logs/qwen3.5/eval/merge}"

if [[ ! "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "ERROR: --gpus must be a comma-separated list of GPU ids, got '${GPUS}'" >&2
  exit 1
fi

# VideoMME defaults.
: "${VIDEOMME_VIDEO_BASE:=${PROJECT_DIR}/data/eval/videomme}"
: "${VIDEOMME_VIDEO_DIR:=${VIDEOMME_VIDEO_BASE}/videos}"
: "${VIDEOMME_VIDEO_MIN_PIXELS:=4096}"
: "${VIDEOMME_VIDEO_MAX_PIXELS:=262144}"
: "${VIDEOMME_VIDEO_TOTAL_PIXELS:=0}"
: "${VIDEOMME_MAX_FRAMES:=384}"
: "${VIDEOMME_FPS:=2}"
: "${VIDEOMME_SETTING:=all-qwen3_vl-sub0-f${VIDEOMME_MAX_FRAMES}-fps${VIDEOMME_FPS}-min${VIDEOMME_VIDEO_MIN_PIXELS}-max${VIDEOMME_VIDEO_MAX_PIXELS}-total${VIDEOMME_VIDEO_TOTAL_PIXELS}-videomme_preprocessed_384f_262k_total0}"
: "${VIDEOMME_DATA_FILE:=${PROJECT_DIR}/eval/data/valid_data/videomme_preprocessed_384f_262k_total0.jsonl}"
: "${VIDEOMME_PREPROCESSED_VIDEO_DIR:=${VIDEOMME_VIDEO_BASE}/preprocessed_videos_384f_262k_total0}"
: "${VIDEOMME_BATCH_SIZE:=1}"
: "${VIDEOMME_MAX_MODEL_LEN:=65536}"
: "${VIDEOMME_MAX_NEW_TOKENS:=128}"
: "${VIDEOMME_MAX_NUM_BATCHED_TOKENS:=65536}"
: "${VIDEOMME_GPU_MEMORY_UTILIZATION:=0.90}"
: "${VIDEOMME_MAX_SAMPLES:=0}"

# VideoMME-V2 defaults. Uses the same local schema and evaluator as VideoMME.
: "${VIDEOMMEV2_VIDEO_BASE:=${PROJECT_DIR}/data/eval/videommev2}"
: "${VIDEOMMEV2_VIDEO_DIR:=${VIDEOMMEV2_VIDEO_BASE}/videos}"
: "${VIDEOMMEV2_VIDEO_MIN_PIXELS:=4096}"
: "${VIDEOMMEV2_VIDEO_MAX_PIXELS:=262144}"
: "${VIDEOMMEV2_VIDEO_TOTAL_PIXELS:=0}"
: "${VIDEOMMEV2_MAX_FRAMES:=384}"
: "${VIDEOMMEV2_FPS:=2}"
: "${VIDEOMMEV2_SETTING:=all-qwen3_vl-sub0-f${VIDEOMMEV2_MAX_FRAMES}-fps${VIDEOMMEV2_FPS}-min${VIDEOMMEV2_VIDEO_MIN_PIXELS}-max${VIDEOMMEV2_VIDEO_MAX_PIXELS}-total${VIDEOMMEV2_VIDEO_TOTAL_PIXELS}-videommev2_preprocessed_384f_262k_total0}"
: "${VIDEOMMEV2_DATA_FILE:=${PROJECT_DIR}/eval/data/valid_data/videommev2_preprocessed_384f_262k_total0.jsonl}"
: "${VIDEOMMEV2_PREPROCESSED_VIDEO_DIR:=${VIDEOMMEV2_VIDEO_BASE}/preprocessed_videos_384f_262k_total0}"
: "${VIDEOMMEV2_BATCH_SIZE:=1}"
: "${VIDEOMMEV2_MAX_MODEL_LEN:=65536}"
: "${VIDEOMMEV2_MAX_NEW_TOKENS:=128}"
: "${VIDEOMMEV2_MAX_NUM_BATCHED_TOKENS:=65536}"
: "${VIDEOMMEV2_GPU_MEMORY_UTILIZATION:=0.90}"
: "${VIDEOMMEV2_PROMPT_MODE:=default}"
: "${VIDEOMMEV2_ANSWER_FILTER:=}"
: "${VIDEOMMEV2_MAX_SAMPLES:=0}"

# VideoMMMU defaults.
: "${VIDEOMMMU_SETTING:=videommmu-f128-fps2-min4096-max262144-total0-with-image}"
: "${VIDEOMMMU_DATA_FILE:=${PROJECT_DIR}/eval/data/valid_data/videommmu_with_image.json}"
: "${VIDEOMMMU_VIDEO_ROOT:=${PROJECT_DIR}/data/eval/videommmu}"
: "${VIDEOMMMU_BATCH_SIZE:=4}"
: "${VIDEOMMMU_MAX_MODEL_LEN:=65536}"
: "${VIDEOMMMU_MAX_NEW_TOKENS:=128}"
: "${VIDEOMMMU_MAX_NUM_BATCHED_TOKENS:=65536}"
: "${VIDEOMMMU_GPU_MEMORY_UTILIZATION:=0.90}"
: "${VIDEOMMMU_VIDEO_MIN_PIXELS:=4096}"
: "${VIDEOMMMU_VIDEO_MAX_PIXELS:=262144}"
: "${VIDEOMMMU_VIDEO_TOTAL_PIXELS:=0}"
: "${VIDEOMMMU_MAX_FRAMES:=128}"
: "${VIDEOMMMU_FPS:=2}"
# OneThinker-style Adaptation uses subject videos + PNGs extracted from
# Adaptation/test-00000-of-00001.parquet. Run once before eval:
#   python3 eval/data/extract_videommmu_images.py
# prompt_mode=onethink reproduces OneThinker's CoT prompt; pair it with a large
# VIDEOMMMU_MAX_NEW_TOKENS (OneThinker uses 8192).
: "${VIDEOMMMU_PROMPT_MODE:=default}"
# Separate Adaptation image resolution cap (OneThinker uses 1024*32*32=1048576).
: "${VIDEOMMMU_IMAGE_MAX_PIXELS:=1048576}"
# Qwen3.5 native benchmarking: thinking on + recommended sampling + long output.
# To reproduce the official VideoMMMU number, run with:
#   VIDEOMMMU_PROMPT_MODE=qwen VIDEOMMMU_ENABLE_THINKING=true
#   VIDEOMMMU_MAX_NEW_TOKENS=32768 VIDEOMMMU_MAX_MODEL_LEN=131072
#   VIDEOMMMU_TEMPERATURE=1.0 VIDEOMMMU_TOP_P=0.95 VIDEOMMMU_TOP_K=20 VIDEOMMMU_PRESENCE_PENALTY=1.5
: "${VIDEOMMMU_ENABLE_THINKING:=false}"
: "${VIDEOMMMU_TEMPERATURE:=0.0}"
: "${VIDEOMMMU_TOP_P:=1.0}"
: "${VIDEOMMMU_TOP_K:=-1}"
: "${VIDEOMMMU_PRESENCE_PENALTY:=0.0}"
: "${VIDEOMMMU_MIN_P:=0.0}"

# MMVU defaults (multiple-choice subset, raw video).
# Build the MC JSONL once: python scripts/build_mmvu_mc.py
: "${MMVU_DATA_FILE:=${PROJECT_DIR}/eval/data/valid_data/mmvu_mc.jsonl}"
: "${MMVU_VIDEO_ROOT:=${PROJECT_DIR}/data/eval/mmvu}"
: "${MMVU_BATCH_SIZE:=1}"
: "${MMVU_MAX_MODEL_LEN:=65536}"
: "${MMVU_MAX_NEW_TOKENS:=128}"
: "${MMVU_MAX_NUM_BATCHED_TOKENS:=65536}"
: "${MMVU_GPU_MEMORY_UTILIZATION:=0.90}"
: "${MMVU_VIDEO_MIN_PIXELS:=4096}"
: "${MMVU_VIDEO_MAX_PIXELS:=262144}"
: "${MMVU_VIDEO_TOTAL_PIXELS:=0}"
: "${MMVU_MAX_FRAMES:=384}"
: "${MMVU_FPS:=2}"
: "${MMVU_PROMPT_MODE:=default}"
# Qwen3.5 native: MMVU_PROMPT_MODE=qwen MMVU_ENABLE_THINKING=true with large
# MMVU_MAX_NEW_TOKENS / sampling, mirroring VideoMMMU.
: "${MMVU_ENABLE_THINKING:=false}"
: "${MMVU_TEMPERATURE:=0.0}"
: "${MMVU_TOP_P:=1.0}"
: "${MMVU_TOP_K:=-1}"
: "${MMVU_PRESENCE_PENALTY:=0.0}"
: "${MMVU_MIN_P:=0.0}"
: "${MMVU_MAX_SAMPLES:=0}"
: "${MMVU_SETTING:=mmvu-mc-f${MMVU_MAX_FRAMES}-fps${MMVU_FPS}-min${MMVU_VIDEO_MIN_PIXELS}-max${MMVU_VIDEO_MAX_PIXELS}-total${MMVU_VIDEO_TOTAL_PIXELS}-${MMVU_PROMPT_MODE}}"

# MVBench defaults (multi-task short-video multiple-choice).
: "${MVBENCH_DATA_FILE:=${PROJECT_DIR}/eval/data/valid_data/mvbench.json}"
: "${MVBENCH_VIDEO_ROOT:=${PROJECT_DIR}/data/eval/mvbench}"
: "${MVBENCH_BATCH_SIZE:=1}"
: "${MVBENCH_MAX_MODEL_LEN:=65536}"
: "${MVBENCH_MAX_NEW_TOKENS:=128}"
: "${MVBENCH_MAX_NUM_BATCHED_TOKENS:=65536}"
: "${MVBENCH_GPU_MEMORY_UTILIZATION:=0.90}"
: "${MVBENCH_VIDEO_MIN_PIXELS:=4096}"
: "${MVBENCH_VIDEO_MAX_PIXELS:=262144}"
: "${MVBENCH_VIDEO_TOTAL_PIXELS:=0}"
: "${MVBENCH_MAX_FRAMES:=384}"
: "${MVBENCH_FPS:=2}"
: "${MVBENCH_PROMPT_MODE:=default}"
: "${MVBENCH_ENABLE_THINKING:=false}"
: "${MVBENCH_TEMPERATURE:=0.0}"
: "${MVBENCH_TOP_P:=1.0}"
: "${MVBENCH_TOP_K:=-1}"
: "${MVBENCH_PRESENCE_PENALTY:=0.0}"
: "${MVBENCH_MIN_P:=0.0}"
: "${MVBENCH_MAX_SAMPLES:=0}"
: "${MVBENCH_SETTING:=mvbench-f${MVBENCH_MAX_FRAMES}-fps${MVBENCH_FPS}-min${MVBENCH_VIDEO_MIN_PIXELS}-max${MVBENCH_VIDEO_MAX_PIXELS}-total${MVBENCH_VIDEO_TOTAL_PIXELS}-${MVBENCH_PROMPT_MODE}}"

# Video-Holmes defaults (all multiple-choice, raw cropped videos).
# Download: huggingface-cli download TencentARC/Video-Holmes --repo-type dataset
# Build: python scripts/build_videoholmes.py
: "${VIDEOHOLMES_DATA_FILE:=${PROJECT_DIR}/eval/data/valid_data/videoholmes.jsonl}"
: "${VIDEOHOLMES_VIDEO_ROOT:=${PROJECT_DIR}/data/eval/videoholmes}"
: "${VIDEOHOLMES_BATCH_SIZE:=1}"
: "${VIDEOHOLMES_MAX_MODEL_LEN:=65536}"
: "${VIDEOHOLMES_MAX_NEW_TOKENS:=128}"
: "${VIDEOHOLMES_MAX_NUM_BATCHED_TOKENS:=65536}"
: "${VIDEOHOLMES_GPU_MEMORY_UTILIZATION:=0.90}"
: "${VIDEOHOLMES_VIDEO_MIN_PIXELS:=4096}"
: "${VIDEOHOLMES_VIDEO_MAX_PIXELS:=262144}"
: "${VIDEOHOLMES_VIDEO_TOTAL_PIXELS:=0}"
: "${VIDEOHOLMES_MAX_FRAMES:=384}"
: "${VIDEOHOLMES_FPS:=2}"
: "${VIDEOHOLMES_PROMPT_MODE:=default}"
# Official benchmark prompt is reasoning-style:
#   VIDEOHOLMES_PROMPT_MODE=holmes VIDEOHOLMES_ENABLE_THINKING=true VIDEOHOLMES_MAX_NEW_TOKENS=1024
: "${VIDEOHOLMES_ENABLE_THINKING:=false}"
: "${VIDEOHOLMES_TEMPERATURE:=0.0}"
: "${VIDEOHOLMES_TOP_P:=1.0}"
: "${VIDEOHOLMES_TOP_K:=-1}"
: "${VIDEOHOLMES_PRESENCE_PENALTY:=0.0}"
: "${VIDEOHOLMES_MIN_P:=0.0}"
: "${VIDEOHOLMES_MAX_SAMPLES:=0}"
: "${VIDEOHOLMES_SETTING:=videoholmes-f${VIDEOHOLMES_MAX_FRAMES}-fps${VIDEOHOLMES_FPS}-min${VIDEOHOLMES_VIDEO_MIN_PIXELS}-max${VIDEOHOLMES_VIDEO_MAX_PIXELS}-total${VIDEOHOLMES_VIDEO_TOTAL_PIXELS}-${VIDEOHOLMES_PROMPT_MODE}}"

# LongVideoBench defaults. Main videos/subtitles repo is gated; after getting
# access, download/extract to LONGVIDEOBENCH_VIDEO_ROOT with videos/ + subtitles/.
# Build validation JSONL: python scripts/build_longvideobench.py
: "${LONGVIDEOBENCH_DATA_FILE:=${PROJECT_DIR}/eval/data/valid_data/longvideobench_val.jsonl}"
: "${LONGVIDEOBENCH_VIDEO_ROOT:=${PROJECT_DIR}/data/eval/longvideobench}"
: "${LONGVIDEOBENCH_SUBTITLE_ROOT:=$LONGVIDEOBENCH_VIDEO_ROOT}"
: "${LONGVIDEOBENCH_USE_SUBTITLES:=true}"
: "${LONGVIDEOBENCH_BATCH_SIZE:=1}"
: "${LONGVIDEOBENCH_MAX_MODEL_LEN:=65536}"
: "${LONGVIDEOBENCH_MAX_NEW_TOKENS:=128}"
: "${LONGVIDEOBENCH_MAX_NUM_BATCHED_TOKENS:=65536}"
: "${LONGVIDEOBENCH_GPU_MEMORY_UTILIZATION:=0.90}"
: "${LONGVIDEOBENCH_VIDEO_MIN_PIXELS:=4096}"
: "${LONGVIDEOBENCH_VIDEO_MAX_PIXELS:=262144}"
: "${LONGVIDEOBENCH_VIDEO_TOTAL_PIXELS:=0}"
: "${LONGVIDEOBENCH_MAX_FRAMES:=384}"
: "${LONGVIDEOBENCH_FPS:=2}"
: "${LONGVIDEOBENCH_PROMPT_MODE:=default}"
: "${LONGVIDEOBENCH_ENABLE_THINKING:=false}"
: "${LONGVIDEOBENCH_TEMPERATURE:=0.0}"
: "${LONGVIDEOBENCH_TOP_P:=1.0}"
: "${LONGVIDEOBENCH_TOP_K:=-1}"
: "${LONGVIDEOBENCH_PRESENCE_PENALTY:=0.0}"
: "${LONGVIDEOBENCH_MIN_P:=0.0}"
: "${LONGVIDEOBENCH_MAX_SAMPLES:=0}"
: "${LONGVIDEOBENCH_SETTING:=longvideobench-val-f${LONGVIDEOBENCH_MAX_FRAMES}-fps${LONGVIDEOBENCH_FPS}-min${LONGVIDEOBENCH_VIDEO_MIN_PIXELS}-max${LONGVIDEOBENCH_VIDEO_MAX_PIXELS}-total${LONGVIDEOBENCH_VIDEO_TOTAL_PIXELS}-${LONGVIDEOBENCH_PROMPT_MODE}}"

# LVBench defaults (long-video multiple-choice; independent from LongVideoBench).
: "${LVBENCH_DATA_FILE:=${PROJECT_DIR}/eval/data/valid_data/lvbench.json}"
: "${LVBENCH_VIDEO_ROOT:=${PROJECT_DIR}/data/eval/lvbench}"
: "${LVBENCH_BATCH_SIZE:=4}"
: "${LVBENCH_MAX_MODEL_LEN:=65536}"
: "${LVBENCH_MAX_NEW_TOKENS:=128}"
: "${LVBENCH_MAX_NUM_BATCHED_TOKENS:=65536}"
: "${LVBENCH_GPU_MEMORY_UTILIZATION:=0.90}"
: "${LVBENCH_VIDEO_MIN_PIXELS:=4096}"
: "${LVBENCH_VIDEO_MAX_PIXELS:=262144}"
: "${LVBENCH_VIDEO_TOTAL_PIXELS:=0}"
: "${LVBENCH_MAX_FRAMES:=128}"
: "${LVBENCH_FPS:=2}"
: "${LVBENCH_PROMPT_MODE:=default}"
: "${LVBENCH_ENABLE_THINKING:=false}"
: "${LVBENCH_TEMPERATURE:=0.0}"
: "${LVBENCH_TOP_P:=1.0}"
: "${LVBENCH_TOP_K:=-1}"
: "${LVBENCH_PRESENCE_PENALTY:=0.0}"
: "${LVBENCH_MIN_P:=0.0}"
: "${LVBENCH_SETTING:=lvbench-f${LVBENCH_MAX_FRAMES}-fps${LVBENCH_FPS}-min${LVBENCH_VIDEO_MIN_PIXELS}-max${LVBENCH_VIDEO_MAX_PIXELS}-total${LVBENCH_VIDEO_TOTAL_PIXELS}-${LVBENCH_PROMPT_MODE}}"

# MLVU defaults (dev multiple-choice subset = 7 tasks, M-Avg).
# Build the MC JSONL once: python scripts/build_mlvu_mc.py
: "${MLVU_DATA_FILE:=${PROJECT_DIR}/eval/data/valid_data/mlvu_mc.jsonl}"
: "${MLVU_VIDEO_ROOT:=${PROJECT_DIR}/data/eval/mlvu}"
: "${MLVU_BATCH_SIZE:=1}"
: "${MLVU_MAX_MODEL_LEN:=65536}"
: "${MLVU_MAX_NEW_TOKENS:=128}"
: "${MLVU_MAX_NUM_BATCHED_TOKENS:=65536}"
: "${MLVU_GPU_MEMORY_UTILIZATION:=0.90}"
: "${MLVU_VIDEO_MIN_PIXELS:=4096}"
: "${MLVU_VIDEO_MAX_PIXELS:=262144}"
: "${MLVU_VIDEO_TOTAL_PIXELS:=0}"
: "${MLVU_MAX_FRAMES:=384}"
: "${MLVU_FPS:=2}"
: "${MLVU_PROMPT_MODE:=default}"
: "${MLVU_ENABLE_THINKING:=false}"
: "${MLVU_TEMPERATURE:=0.0}"
: "${MLVU_TOP_P:=1.0}"
: "${MLVU_TOP_K:=-1}"
: "${MLVU_PRESENCE_PENALTY:=0.0}"
: "${MLVU_MIN_P:=0.0}"
: "${MLVU_MAX_SAMPLES:=0}"
: "${MLVU_SETTING:=mlvu-mc-f${MLVU_MAX_FRAMES}-fps${MLVU_FPS}-min${MLVU_VIDEO_MIN_PIXELS}-max${MLVU_VIDEO_MAX_PIXELS}-total${MLVU_VIDEO_TOTAL_PIXELS}-${MLVU_PROMPT_MODE}}"

# VSI-Bench defaults.
: "${VSI_SETTING:=video128-16M-video-f128-fps2-min65536-maxnone-total16777216-vsibench_preprocessed_128f_16M}"
: "${VSI_DATA_FILE:=${PROJECT_DIR}/eval/data/valid_data/vsibench_preprocessed_128f_16M.jsonl}"
: "${VSI_PREPROCESSED_VIDEO_DIR:=${PROJECT_DIR}/data/eval/vsi/preprocessed_videos}"
: "${VSI_BATCH_SIZE:=16}"
: "${VSI_MAX_MODEL_LEN:=32768}"
: "${VSI_MAX_NEW_TOKENS:=1024}"
: "${VSI_GPU_MEMORY_UTILIZATION:=0.90}"
: "${VSI_EXPECTED_SAMPLES:=5130}"

# MMSI-Bench defaults. Transformers is used so every image can be retained;
# set MMSI_BACKEND=vllm and MMSI_MAX_IMAGES=8 for the legacy path.
: "${MMSI_BACKEND:=transformers}"
: "${MMSI_DATA_FILE:=${PROJECT_DIR}/data/eval/mmsi/MMSI_bench.tsv}"
: "${MMSI_IMAGE_MIN_PIXELS:=4096}"
: "${MMSI_IMAGE_MAX_PIXELS:=262144}"
if [[ "${MMSI_BACKEND}" == "transformers" ]]; then
  : "${MMSI_MAX_IMAGES:=0}"
else
  : "${MMSI_MAX_IMAGES:=8}"
fi
: "${MMSI_IMAGE_COUNT_TAG:=$([[ "${MMSI_MAX_IMAGES}" -le 0 ]] && echo all || echo "${MMSI_MAX_IMAGES}")}"
: "${MMSI_SETTING:=${MMSI_BACKEND}-img-min${MMSI_IMAGE_MIN_PIXELS}-max${MMSI_IMAGE_MAX_PIXELS}-n${MMSI_IMAGE_COUNT_TAG}-all-MMSI_bench}"
: "${MMSI_BATCH_SIZE:=16}"
: "${MMSI_MAX_MODEL_LEN:=32768}"
: "${MMSI_MAX_NEW_TOKENS:=1024}"
: "${MMSI_MAX_NUM_BATCHED_TOKENS:=32768}"
: "${MMSI_GPU_MEMORY_UTILIZATION:=0.90}"
: "${MMSI_MAX_SAMPLES:=0}"
: "${MMSI_CATEGORY:=}"
: "${MMSI_ENABLE_THINKING:=false}"
: "${MMSI_ATTN_IMPLEMENTATION:=flash_attention_2}"
if [[ -n "${MMSI_CATEGORY}" || "${MMSI_MAX_SAMPLES}" != "0" ]]; then
  : "${MMSI_EXPECTED_SAMPLES:=0}"
else
  : "${MMSI_EXPECTED_SAMPLES:=1000}"
fi

# MindCube-Tiny defaults.
: "${MINDCUBE_SETTING:=img-min4096-max262144-n4-official-tiny1050}"
: "${MINDCUBE_DATA_FILE:=${PROJECT_DIR}/data/eval/mindcube_official/combined-00000-of-00001.parquet}"
: "${MINDCUBE_MAX_IMAGES:=4}"
: "${MINDCUBE_EXPECTED_SAMPLES:=1050}"
: "${MINDCUBE_HF_CACHE_DIR:=}"
: "${MINDCUBE_BATCH_SIZE:=16}"
: "${MINDCUBE_MAX_MODEL_LEN:=32768}"
: "${MINDCUBE_MAX_NEW_TOKENS:=1024}"
: "${MINDCUBE_MAX_NUM_BATCHED_TOKENS:=32768}"
: "${MINDCUBE_GPU_MEMORY_UTILIZATION:=0.90}"

# ReVSI defaults. The all-frame/native-frame profile reproduces the 58.2 result
# reported for Video-ORA-9B; ReVSI remains excluded from the three-benchmark
# spatial-intelligence average.
: "${REVSI_DATA_FILE:=${PROJECT_DIR}/data/eval/revsi/all_frame/test-00000-of-00001.parquet}"
: "${REVSI_VIDEO_ROOT:=${PROJECT_DIR}/data/eval/revsi}"
: "${REVSI_FRAME_BUDGET:=all}"
: "${REVSI_MAX_FRAMES:=128}"
: "${REVSI_EXACT_NFRAMES:=true}"
: "${REVSI_FPS:=2}"
: "${REVSI_VIDEO_MIN_PIXELS:=65536}"
: "${REVSI_VIDEO_MAX_PIXELS:=}"
: "${REVSI_VIDEO_TOTAL_PIXELS:=16777216}"
: "${REVSI_BATCH_SIZE:=16}"
: "${REVSI_MAX_MODEL_LEN:=32768}"
: "${REVSI_MAX_NEW_TOKENS:=64}"
: "${REVSI_GPU_MEMORY_UTILIZATION:=0.90}"
: "${REVSI_MAX_SAMPLES:=0}"
: "${REVSI_EXPECTED_SAMPLES:=6808}"
: "${REVSI_TASK_FILTER:=}"
: "${REVSI_ENABLE_THINKING:=false}"
: "${REVSI_SETTING:=native-all-f${REVSI_MAX_FRAMES}-exact${REVSI_EXACT_NFRAMES}-fps${REVSI_FPS}-min${REVSI_VIDEO_MIN_PIXELS}-max${REVSI_VIDEO_MAX_PIXELS:-none}-total${REVSI_VIDEO_TOTAL_PIXELS}}"

# Spatial grounding defaults, aligned to data/joint/sft_joint_all.jsonl.
: "${SPATIAL_GROUNDING_DATASETS:=refcoco-val,refcoco-testA,refcoco-testB,refcoco+-val,refcoco+-testA,refcoco+-testB,refcocog-val,refcocog-test}"
: "${SPATIAL_GROUNDING_BENCH_DIR:=${PROJECT_DIR}/eval/task/spatial_grounding}"
: "${SPATIAL_GROUNDING_IMAGE_ROOT:=${PROJECT_DIR}/data/eval/spatial_grounding/images}"
: "${SPATIAL_GROUNDING_PROCESSOR_PATH:=$MODEL_PATH}"
: "${SPATIAL_GROUNDING_PROMPT_STYLE:=qwen_native}"
: "${SPATIAL_GROUNDING_COORD_SYSTEM:=norm1000}"
: "${SPATIAL_GROUNDING_BBOX_SELECT:=first}"
: "${SPATIAL_GROUNDING_MIN_TOKENS:=64}"
: "${SPATIAL_GROUNDING_TOTAL_TOKENS:=1024}"
: "${SPATIAL_GROUNDING_BATCH_SIZE:=64}"
: "${SPATIAL_GROUNDING_MAX_MODEL_LEN:=32768}"
: "${SPATIAL_GROUNDING_MAX_NEW_TOKENS:=1024}"
: "${SPATIAL_GROUNDING_MAX_NUM_BATCHED_TOKENS:=32768}"
: "${SPATIAL_GROUNDING_GPU_MEMORY_UTILIZATION:=0.85}"
: "${SPATIAL_GROUNDING_ENABLE_THINKING:=false}"
: "${SPATIAL_GROUNDING_MAX_SAMPLES:=0}"
: "${SPATIAL_GROUNDING_SETTING:=spatial-grounding-${SPATIAL_GROUNDING_DATASETS//,/_}-${SPATIAL_GROUNDING_PROMPT_STYLE}-min${SPATIAL_GROUNDING_MIN_TOKENS}-total${SPATIAL_GROUNDING_TOTAL_TOKENS}-new${SPATIAL_GROUNDING_MAX_NEW_TOKENS}}"

# Tracking defaults, aligned to data/joint/sft_joint_all.jsonl tracking videos.
: "${TRACKING_DATASETS:=eval_got10k}"
: "${TRACKING_BENCH_DIR:=${PROJECT_DIR}/data/eval/tracking}"
: "${TRACKING_BASE_PREFIX:=$TRACKING_BENCH_DIR}"
: "${TRACKING_PROCESSOR_PATH:=$MODEL_PATH}"
: "${TRACKING_VIDEO_MIN_PIXELS:=4096}"
: "${TRACKING_VIDEO_MAX_PIXELS:=786432}"
: "${TRACKING_VIDEO_TOTAL_PIXELS:=8388608}"
: "${TRACKING_MAX_FRAMES:=32}"
: "${TRACKING_FPS:=1}"
: "${TRACKING_MAX_MODEL_LEN:=32768}"
: "${TRACKING_MAX_NEW_TOKENS:=8192}"
: "${TRACKING_MAX_NUM_BATCHED_TOKENS:=32768}"
: "${TRACKING_BATCH_SIZE:=64}"
: "${TRACKING_GPU_MEMORY_UTILIZATION:=0.95}"
: "${TRACKING_ENABLE_THINKING:=false}"
: "${TRACKING_PROMPT_MODE:=default}"
: "${TRACKING_CHUNKED_REPROMPT:=0}"
: "${TRACKING_REPROMPT_USE_GT_FIRST_BOX:=0}"
: "${TRACKING_MAX_SAMPLES:=0}"
: "${TRACKING_SETTING:=tracking-${TRACKING_DATASETS//,/_}-f${TRACKING_MAX_FRAMES}-fps${TRACKING_FPS}-min${TRACKING_VIDEO_MIN_PIXELS}-max${TRACKING_VIDEO_MAX_PIXELS}-total${TRACKING_VIDEO_TOTAL_PIXELS}-new${TRACKING_MAX_NEW_TOKENS}}"

# STVG defaults, aligned to data/joint/sft_joint_all.jsonl stvg videos.
: "${STVG_DATASETS:=eval_stvg}"
: "${STVG_BENCH_DIR:=${PROJECT_DIR}/data/eval/stvg}"
: "${STVG_BASE_PREFIX:=$STVG_BENCH_DIR}"
: "${STVG_PROCESSOR_PATH:=$MODEL_PATH}"
: "${STVG_VIDEO_MIN_PIXELS:=65536}"
: "${STVG_VIDEO_MAX_PIXELS:=393216}"
: "${STVG_VIDEO_TOTAL_PIXELS:=10485760}"
: "${STVG_MAX_FRAMES:=128}"
: "${STVG_FPS:=2}"
: "${STVG_MAX_MODEL_LEN:=65536}"
: "${STVG_MAX_NEW_TOKENS:=2048}"
: "${STVG_MAX_NUM_BATCHED_TOKENS:=65536}"
: "${STVG_BATCH_SIZE:=64}"
: "${STVG_GPU_MEMORY_UTILIZATION:=0.85}"
: "${STVG_ENABLE_THINKING:=false}"
: "${STVG_PROMPT_MODE:=train_stvg}"
: "${STVG_MAX_SAMPLES:=0}"
: "${STVG_SETTING:=stvg-${STVG_DATASETS//,/_}-f${STVG_MAX_FRAMES}-fps${STVG_FPS}-min${STVG_VIDEO_MIN_PIXELS}-max${STVG_VIDEO_MAX_PIXELS}-total${STVG_VIDEO_TOTAL_PIXELS}-new${STVG_MAX_NEW_TOKENS}-${STVG_PROMPT_MODE}}"

# Segmentation defaults.
# This task is dispatched to eval/task/segmentation/run_eval_vllm.sh,
# which performs vLLM inference plus optional SAM2 post-processing.
: "${SEGMENTATION_DATASETS:=eval_seg_refcoco,eval_seg_refcocop,eval_seg_refcocog,eval_seg_mevis,eval_seg_reasonvos}"
: "${SEGMENTATION_BENCH_DIR:=${PROJECT_DIR}/data/eval/segmentation}"
: "${SEGMENTATION_DATA_ROOT:=$SEGMENTATION_BENCH_DIR}"
: "${SEGMENTATION_PROCESSOR_PATH:=$MODEL_PATH}"
: "${SEGMENTATION_DATA_TYPE:=all}"
# train_seg mirrors the prompt used in joint SFT training data
# (data/train/seg_image_nothink_evalaware.jsonl, seg_video_nothink.jsonl).
: "${SEGMENTATION_PROMPT_MODE:=train_seg}"
: "${SEGMENTATION_ENABLE_THINKING:=false}"
: "${SEGMENTATION_BATCH_SIZE:=16}"
: "${SEGMENTATION_MAX_NEW_TOKENS:=1024}"
: "${SEGMENTATION_MAX_MODEL_LEN:=32768}"
: "${SEGMENTATION_MAX_PIXELS_IMAGE:=1048576}"
: "${SEGMENTATION_MIN_PIXELS_IMAGE:=4096}"
: "${SEGMENTATION_VIDEO_MAX_PIXELS:=262144}"
: "${SEGMENTATION_VIDEO_MIN_PIXELS:=4096}"
: "${SEGMENTATION_VIDEO_TOTAL_PIXELS:=16777216}"
: "${SEGMENTATION_MAX_FRAMES:=128}"
: "${SEGMENTATION_FPS:=2}"
: "${SEGMENTATION_VIDEO_READER:=decord}"
: "${SEGMENTATION_GPU_MEM_UTIL:=0.85}"
: "${SEGMENTATION_SEED:=42}"
: "${SEGMENTATION_MAX_SAMPLES:=}"
: "${SEGMENTATION_RUN_SAM2:=false}"
: "${SEGMENTATION_SAM2_CKPT:=${PROJECT_DIR}/models/sam2/sam2.1_hiera_large.pt}"
: "${SEGMENTATION_SAM2_CFG:=configs/sam2.1/sam2.1_hiera_l.yaml}"
: "${SEGMENTATION_POSTPROCESSOR_PATH:=${PROJECT_DIR}/third_party/OneThinker/Evaluation/Eval/seg_post_sam2.py}"
# SAM2 post-processing parallelism. world_size = NUM_GPUS * WORKERS_PER_GPU.
# With a large SAM2_EPOCH_SIZE (single epoch) + maxtasksperchild=1, EACH worker
# loads the SAM2 model exactly ONCE at startup and then streams its whole slice
# (no repeated reloads). So WORKERS_PER_GPU only trades GPU utilization vs total
# process count: too high (32 -> 256 procs) exhausts CPU/RAM/ffmpeg/handles and
# aborts; too low (1) underutilizes the GPU. 4 per GPU is a balance (8 GPUs -> 32
# persistent workers, 32 one-time loads); bump to 6-8 if GPU is still not full.
# Leave NUM_GPUS empty to auto-use all visible GPUs.
: "${SEGMENTATION_SAM2_NUM_GPUS:=}"
: "${SEGMENTATION_SAM2_WORKERS_PER_GPU:=8}"
: "${SEGMENTATION_SETTING:=segmentation-${SEGMENTATION_DATASETS//,/_}-${SEGMENTATION_PROMPT_MODE}-f${SEGMENTATION_MAX_FRAMES}-fps${SEGMENTATION_FPS}-min${SEGMENTATION_VIDEO_MIN_PIXELS}-max${SEGMENTATION_VIDEO_MAX_PIXELS}-total${SEGMENTATION_VIDEO_TOTAL_PIXELS}-new${SEGMENTATION_MAX_NEW_TOKENS}}"

# TimeLens-Bench temporal grounding defaults.
# This task uses HuggingFace transformers, not vLLM.
: "${TIMELENS_DATASETS:=charades-timelens}"
: "${TIMELENS_BENCH_DIR:=${PROJECT_DIR}/data/eval/temporal_grounding}"
: "${TIMELENS_ENABLE_THINKING:=false}"
: "${TIMELENS_MIN_TOKENS:=1}"
: "${TIMELENS_TOTAL_TOKENS:=128000}"
: "${TIMELENS_MAX_PIXELS:=409600}"
: "${TIMELENS_MAX_FRAMES:=2048}"
: "${TIMELENS_FPS:=4}"
: "${TIMELENS_MAX_NEW_TOKENS:=128}"
: "${TIMELENS_REPETITION_PENALTY:=1.0}"
: "${TIMELENS_STOP_AFTER_ANSWER:=true}"
: "${TIMELENS_PROMPT_MODE:=same}"
: "${TIMELENS_NUM_WORKERS:=2}"
: "${TIMELENS_MAX_SAMPLES:=0}"

export MODEL_PATH TASKS GPUS TP_SIZE SHARD_MODE VIDEO_CACHE_SIZE PREFETCH_BATCHES
export FORCE_MERGE SKIP_MERGE MERGE_LOG_DIR
export VIDEOMME_SETTING VIDEOMME_DATA_FILE VIDEOMME_VIDEO_BASE VIDEOMME_VIDEO_DIR
export VIDEOMME_PREPROCESSED_VIDEO_DIR VIDEOMME_VIDEO_MIN_PIXELS VIDEOMME_VIDEO_MAX_PIXELS
export VIDEOMME_VIDEO_TOTAL_PIXELS VIDEOMME_MAX_FRAMES VIDEOMME_FPS
export VIDEOMME_BATCH_SIZE VIDEOMME_MAX_MODEL_LEN VIDEOMME_MAX_NEW_TOKENS
export VIDEOMME_MAX_NUM_BATCHED_TOKENS VIDEOMME_GPU_MEMORY_UTILIZATION
export VIDEOMME_MAX_SAMPLES
export VIDEOMMEV2_SETTING VIDEOMMEV2_DATA_FILE VIDEOMMEV2_VIDEO_BASE VIDEOMMEV2_VIDEO_DIR
export VIDEOMMEV2_PREPROCESSED_VIDEO_DIR VIDEOMMEV2_VIDEO_MIN_PIXELS VIDEOMMEV2_VIDEO_MAX_PIXELS
export VIDEOMMEV2_VIDEO_TOTAL_PIXELS VIDEOMMEV2_MAX_FRAMES VIDEOMMEV2_FPS
export VIDEOMMEV2_BATCH_SIZE VIDEOMMEV2_MAX_MODEL_LEN VIDEOMMEV2_MAX_NEW_TOKENS
export VIDEOMMEV2_MAX_NUM_BATCHED_TOKENS VIDEOMMEV2_GPU_MEMORY_UTILIZATION
export VIDEOMMEV2_PROMPT_MODE VIDEOMMEV2_ANSWER_FILTER VIDEOMMEV2_MAX_SAMPLES
export VIDEOMMMU_SETTING VIDEOMMMU_DATA_FILE VIDEOMMMU_VIDEO_ROOT
export VIDEOMMMU_BATCH_SIZE VIDEOMMMU_MAX_MODEL_LEN VIDEOMMMU_MAX_NEW_TOKENS
export VIDEOMMMU_MAX_NUM_BATCHED_TOKENS VIDEOMMMU_GPU_MEMORY_UTILIZATION
export VIDEOMMMU_VIDEO_MIN_PIXELS VIDEOMMMU_VIDEO_MAX_PIXELS VIDEOMMMU_VIDEO_TOTAL_PIXELS
export VIDEOMMMU_MAX_FRAMES VIDEOMMMU_FPS VIDEOMMMU_PROMPT_MODE VIDEOMMMU_IMAGE_MAX_PIXELS
export VIDEOMMMU_ENABLE_THINKING VIDEOMMMU_TEMPERATURE VIDEOMMMU_TOP_P VIDEOMMMU_TOP_K
export VIDEOMMMU_PRESENCE_PENALTY VIDEOMMMU_MIN_P
export MMVU_SETTING MMVU_DATA_FILE MMVU_VIDEO_ROOT MMVU_BATCH_SIZE
export MMVU_MAX_MODEL_LEN MMVU_MAX_NEW_TOKENS MMVU_MAX_NUM_BATCHED_TOKENS
export MMVU_GPU_MEMORY_UTILIZATION MMVU_VIDEO_MIN_PIXELS MMVU_VIDEO_MAX_PIXELS
export MMVU_VIDEO_TOTAL_PIXELS MMVU_MAX_FRAMES MMVU_FPS MMVU_PROMPT_MODE
export MMVU_ENABLE_THINKING MMVU_TEMPERATURE MMVU_TOP_P MMVU_TOP_K
export MMVU_PRESENCE_PENALTY MMVU_MIN_P MMVU_MAX_SAMPLES
export MVBENCH_SETTING MVBENCH_DATA_FILE MVBENCH_VIDEO_ROOT MVBENCH_BATCH_SIZE
export MVBENCH_MAX_MODEL_LEN MVBENCH_MAX_NEW_TOKENS MVBENCH_MAX_NUM_BATCHED_TOKENS
export MVBENCH_GPU_MEMORY_UTILIZATION MVBENCH_VIDEO_MIN_PIXELS MVBENCH_VIDEO_MAX_PIXELS
export MVBENCH_VIDEO_TOTAL_PIXELS MVBENCH_MAX_FRAMES MVBENCH_FPS MVBENCH_PROMPT_MODE
export MVBENCH_ENABLE_THINKING MVBENCH_TEMPERATURE MVBENCH_TOP_P MVBENCH_TOP_K
export MVBENCH_PRESENCE_PENALTY MVBENCH_MIN_P MVBENCH_MAX_SAMPLES
export VIDEOHOLMES_SETTING VIDEOHOLMES_DATA_FILE VIDEOHOLMES_VIDEO_ROOT VIDEOHOLMES_BATCH_SIZE
export VIDEOHOLMES_MAX_MODEL_LEN VIDEOHOLMES_MAX_NEW_TOKENS VIDEOHOLMES_MAX_NUM_BATCHED_TOKENS
export VIDEOHOLMES_GPU_MEMORY_UTILIZATION VIDEOHOLMES_VIDEO_MIN_PIXELS VIDEOHOLMES_VIDEO_MAX_PIXELS
export VIDEOHOLMES_VIDEO_TOTAL_PIXELS VIDEOHOLMES_MAX_FRAMES VIDEOHOLMES_FPS VIDEOHOLMES_PROMPT_MODE
export VIDEOHOLMES_ENABLE_THINKING VIDEOHOLMES_TEMPERATURE VIDEOHOLMES_TOP_P VIDEOHOLMES_TOP_K
export VIDEOHOLMES_PRESENCE_PENALTY VIDEOHOLMES_MIN_P VIDEOHOLMES_MAX_SAMPLES
export LONGVIDEOBENCH_SETTING LONGVIDEOBENCH_DATA_FILE LONGVIDEOBENCH_VIDEO_ROOT
export LONGVIDEOBENCH_SUBTITLE_ROOT LONGVIDEOBENCH_USE_SUBTITLES LONGVIDEOBENCH_BATCH_SIZE
export LONGVIDEOBENCH_MAX_MODEL_LEN LONGVIDEOBENCH_MAX_NEW_TOKENS LONGVIDEOBENCH_MAX_NUM_BATCHED_TOKENS
export LONGVIDEOBENCH_GPU_MEMORY_UTILIZATION LONGVIDEOBENCH_VIDEO_MIN_PIXELS LONGVIDEOBENCH_VIDEO_MAX_PIXELS
export LONGVIDEOBENCH_VIDEO_TOTAL_PIXELS LONGVIDEOBENCH_MAX_FRAMES LONGVIDEOBENCH_FPS LONGVIDEOBENCH_PROMPT_MODE
export LONGVIDEOBENCH_ENABLE_THINKING LONGVIDEOBENCH_TEMPERATURE LONGVIDEOBENCH_TOP_P LONGVIDEOBENCH_TOP_K
export LONGVIDEOBENCH_PRESENCE_PENALTY LONGVIDEOBENCH_MIN_P LONGVIDEOBENCH_MAX_SAMPLES
export LVBENCH_SETTING LVBENCH_DATA_FILE LVBENCH_VIDEO_ROOT LVBENCH_BATCH_SIZE
export LVBENCH_MAX_MODEL_LEN LVBENCH_MAX_NEW_TOKENS LVBENCH_MAX_NUM_BATCHED_TOKENS
export LVBENCH_GPU_MEMORY_UTILIZATION LVBENCH_VIDEO_MIN_PIXELS LVBENCH_VIDEO_MAX_PIXELS
export LVBENCH_VIDEO_TOTAL_PIXELS LVBENCH_MAX_FRAMES LVBENCH_FPS LVBENCH_PROMPT_MODE
export LVBENCH_ENABLE_THINKING LVBENCH_TEMPERATURE LVBENCH_TOP_P LVBENCH_TOP_K
export LVBENCH_PRESENCE_PENALTY LVBENCH_MIN_P
export MLVU_SETTING MLVU_DATA_FILE MLVU_VIDEO_ROOT MLVU_BATCH_SIZE
export MLVU_MAX_MODEL_LEN MLVU_MAX_NEW_TOKENS MLVU_MAX_NUM_BATCHED_TOKENS
export MLVU_GPU_MEMORY_UTILIZATION MLVU_VIDEO_MIN_PIXELS MLVU_VIDEO_MAX_PIXELS
export MLVU_VIDEO_TOTAL_PIXELS MLVU_MAX_FRAMES MLVU_FPS MLVU_PROMPT_MODE
export MLVU_ENABLE_THINKING MLVU_TEMPERATURE MLVU_TOP_P MLVU_TOP_K
export MLVU_PRESENCE_PENALTY MLVU_MIN_P MLVU_MAX_SAMPLES
export VSI_SETTING VSI_DATA_FILE VSI_PREPROCESSED_VIDEO_DIR
export VSI_BATCH_SIZE VSI_MAX_MODEL_LEN VSI_MAX_NEW_TOKENS VSI_GPU_MEMORY_UTILIZATION
export VSI_EXPECTED_SAMPLES
export MMSI_BACKEND MMSI_SETTING MMSI_DATA_FILE MMSI_MAX_IMAGES MMSI_BATCH_SIZE
export MMSI_IMAGE_MIN_PIXELS MMSI_IMAGE_MAX_PIXELS MMSI_IMAGE_COUNT_TAG
export MMSI_MAX_MODEL_LEN MMSI_MAX_NEW_TOKENS MMSI_MAX_NUM_BATCHED_TOKENS
export MMSI_GPU_MEMORY_UTILIZATION MMSI_MAX_SAMPLES MMSI_CATEGORY
export MMSI_ENABLE_THINKING MMSI_ATTN_IMPLEMENTATION MMSI_EXPECTED_SAMPLES
export MINDCUBE_SETTING MINDCUBE_DATA_FILE MINDCUBE_MAX_IMAGES MINDCUBE_BATCH_SIZE
export MINDCUBE_MAX_MODEL_LEN MINDCUBE_MAX_NEW_TOKENS MINDCUBE_MAX_NUM_BATCHED_TOKENS
export MINDCUBE_GPU_MEMORY_UTILIZATION MINDCUBE_EXPECTED_SAMPLES MINDCUBE_HF_CACHE_DIR
export REVSI_SETTING REVSI_DATA_FILE REVSI_VIDEO_ROOT REVSI_FRAME_BUDGET
export REVSI_MAX_FRAMES REVSI_EXACT_NFRAMES REVSI_FPS
export REVSI_VIDEO_MIN_PIXELS REVSI_VIDEO_MAX_PIXELS REVSI_VIDEO_TOTAL_PIXELS
export REVSI_BATCH_SIZE REVSI_MAX_MODEL_LEN REVSI_MAX_NEW_TOKENS
export REVSI_GPU_MEMORY_UTILIZATION REVSI_MAX_SAMPLES REVSI_EXPECTED_SAMPLES
export REVSI_TASK_FILTER REVSI_ENABLE_THINKING
export SPATIAL_GROUNDING_SETTING SPATIAL_GROUNDING_DATASETS SPATIAL_GROUNDING_BENCH_DIR
export SPATIAL_GROUNDING_IMAGE_ROOT SPATIAL_GROUNDING_PROCESSOR_PATH SPATIAL_GROUNDING_PROMPT_STYLE
export SPATIAL_GROUNDING_COORD_SYSTEM SPATIAL_GROUNDING_BBOX_SELECT
export SPATIAL_GROUNDING_MIN_TOKENS SPATIAL_GROUNDING_TOTAL_TOKENS
export SPATIAL_GROUNDING_BATCH_SIZE SPATIAL_GROUNDING_MAX_MODEL_LEN
export SPATIAL_GROUNDING_MAX_NEW_TOKENS SPATIAL_GROUNDING_MAX_NUM_BATCHED_TOKENS
export SPATIAL_GROUNDING_GPU_MEMORY_UTILIZATION SPATIAL_GROUNDING_ENABLE_THINKING
export SPATIAL_GROUNDING_MAX_SAMPLES
export TRACKING_SETTING TRACKING_DATASETS TRACKING_BENCH_DIR TRACKING_BASE_PREFIX
export TRACKING_PROCESSOR_PATH TRACKING_VIDEO_MIN_PIXELS TRACKING_VIDEO_MAX_PIXELS
export TRACKING_VIDEO_TOTAL_PIXELS TRACKING_MAX_FRAMES TRACKING_FPS
export TRACKING_MAX_MODEL_LEN TRACKING_MAX_NEW_TOKENS TRACKING_MAX_NUM_BATCHED_TOKENS
export TRACKING_BATCH_SIZE TRACKING_GPU_MEMORY_UTILIZATION TRACKING_ENABLE_THINKING
export TRACKING_PROMPT_MODE TRACKING_CHUNKED_REPROMPT TRACKING_REPROMPT_USE_GT_FIRST_BOX
export TRACKING_MAX_SAMPLES
export STVG_SETTING STVG_DATASETS STVG_BENCH_DIR STVG_BASE_PREFIX STVG_PROCESSOR_PATH
export STVG_VIDEO_MIN_PIXELS STVG_VIDEO_MAX_PIXELS STVG_VIDEO_TOTAL_PIXELS
export STVG_MAX_FRAMES STVG_FPS STVG_MAX_MODEL_LEN STVG_MAX_NEW_TOKENS
export STVG_MAX_NUM_BATCHED_TOKENS STVG_BATCH_SIZE STVG_GPU_MEMORY_UTILIZATION
export STVG_ENABLE_THINKING STVG_PROMPT_MODE STVG_MAX_SAMPLES
export SEGMENTATION_SETTING SEGMENTATION_DATASETS SEGMENTATION_BENCH_DIR SEGMENTATION_DATA_ROOT
export SEGMENTATION_PROCESSOR_PATH SEGMENTATION_DATA_TYPE SEGMENTATION_PROMPT_MODE
export SEGMENTATION_ENABLE_THINKING SEGMENTATION_BATCH_SIZE SEGMENTATION_MAX_NEW_TOKENS
export SEGMENTATION_MAX_MODEL_LEN SEGMENTATION_MAX_PIXELS_IMAGE SEGMENTATION_MIN_PIXELS_IMAGE
export SEGMENTATION_VIDEO_MAX_PIXELS SEGMENTATION_VIDEO_MIN_PIXELS SEGMENTATION_VIDEO_TOTAL_PIXELS
export SEGMENTATION_MAX_FRAMES SEGMENTATION_FPS SEGMENTATION_VIDEO_READER
export SEGMENTATION_GPU_MEM_UTIL SEGMENTATION_SEED
export SEGMENTATION_MAX_SAMPLES
export SEGMENTATION_RUN_SAM2 SEGMENTATION_SAM2_CKPT SEGMENTATION_SAM2_CFG
export SEGMENTATION_SAM2_NUM_GPUS SEGMENTATION_SAM2_WORKERS_PER_GPU
export TIMELENS_BENCH_DIR TIMELENS_MAX_SAMPLES

# Conda env used to run eval. By default, keep the currently active environment.
# Override with EVAL_CONDA_ENV=xxx only when an explicit env switch is desired.
: "${EVAL_CONDA_ENV:=keep}"
if [[ "${EVAL_CONDA_ENV}" != "keep" && "${CONDA_DEFAULT_ENV:-}" != "${EVAL_CONDA_ENV}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${EVAL_CONDA_ENV}"
  else
    echo "ERROR: conda not found and CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<unset>}" >&2
    exit 1
  fi
fi

cd "${PROJECT_DIR}"

resolve_model_path() {
  local input_path="${MODEL_PATH%/}"

  if ls "${input_path}"/model_world_size_*_rank_0.pt >/dev/null 2>&1; then
    local actor_dir="$input_path"
    local hf_dir="${MERGED_MODEL_PATH:-${actor_dir}/huggingface}"
    local need_merge=0

    if [[ "${SKIP_MERGE}" = "1" ]]; then
      need_merge=0
    elif [[ "${FORCE_MERGE}" = "1" ]]; then
      need_merge=1
    elif [[ ! -f "${hf_dir}/model.safetensors" ]]; then
      need_merge=1
    fi

    if [[ "${need_merge}" = "1" ]]; then
      mkdir -p "${MERGE_LOG_DIR}"
      local merge_log="${MERGE_LOG_DIR}/merge-$(basename "$(dirname "${actor_dir}")")-$(date +%Y%m%d_%H%M%S).log"
      echo "[merge] FSDP actor checkpoint detected: ${actor_dir}"
      echo "[merge] Merging shards -> ${hf_dir}"
      if [[ -n "${BASE_MODEL_PATH:-}" ]]; then
        python -u scripts/model_merger.py \
          --local_dir "${actor_dir}" \
          --base_model_path "${BASE_MODEL_PATH}" \
          2>&1 | tee "${merge_log}"
      else
        python -u scripts/model_merger.py \
          --local_dir "${actor_dir}" \
          2>&1 | tee "${merge_log}"
      fi
      if [[ ! -f "${hf_dir}/model.safetensors" ]]; then
        echo "ERROR: merge finished but ${hf_dir}/model.safetensors was not created. Log: ${merge_log}" >&2
        exit 1
      fi
    else
      echo "[merge] Using existing merged HF model: ${hf_dir}"
    fi

    MODEL_PATH="${hf_dir}"
    export MODEL_PATH

    if [[ "${TRACKING_PROCESSOR_PATH%/}" = "${input_path}" ]]; then
      TRACKING_PROCESSOR_PATH="${MODEL_PATH}"
      export TRACKING_PROCESSOR_PATH
    fi
    if [[ "${STVG_PROCESSOR_PATH%/}" = "${input_path}" ]]; then
      STVG_PROCESSOR_PATH="${MODEL_PATH}"
      export STVG_PROCESSOR_PATH
    fi
    if [[ "${SPATIAL_GROUNDING_PROCESSOR_PATH%/}" = "${input_path}" ]]; then
      SPATIAL_GROUNDING_PROCESSOR_PATH="${MODEL_PATH}"
      export SPATIAL_GROUNDING_PROCESSOR_PATH
    fi
    if [[ "${SEGMENTATION_PROCESSOR_PATH%/}" = "${input_path}" ]]; then
      SEGMENTATION_PROCESSOR_PATH="${MODEL_PATH}"
      export SEGMENTATION_PROCESSOR_PATH
    fi
  elif [[ -d "${input_path}/huggingface" && -f "${input_path}/huggingface/model.safetensors" && ! -f "${input_path}/config.json" ]]; then
    MODEL_PATH="${input_path}/huggingface"
    export MODEL_PATH
    echo "[merge] MODEL_PATH points to a checkpoint wrapper; using ${MODEL_PATH}"
  fi
}

resolve_model_path

VLLM_TASKS=""
RUN_TIMELENS=0
RUN_SEGMENTATION=0
RUN_MMSI_TRANSFORMERS=0
RUN_REVSI=0

if [[ "${MMSI_BACKEND}" != "transformers" && "${MMSI_BACKEND}" != "vllm" ]]; then
  echo "ERROR: MMSI_BACKEND must be 'transformers' or 'vllm', got '${MMSI_BACKEND}'" >&2
  exit 1
fi

append_vllm_task() {
  local task="$1"
  if [[ -z "$VLLM_TASKS" ]]; then
    VLLM_TASKS="$task"
  else
    VLLM_TASKS="${VLLM_TASKS},${task}"
  fi
}

model_family_tag() {
  local model_path="${1%/}"
  local name
  name="$(basename "$model_path")"
  local parent
  local grandparent
  parent="$(basename "$(dirname "$model_path")")"
  grandparent="$(basename "$(dirname "$(dirname "$model_path")")")"
  if [[ "$name" == "huggingface" && "$parent" == "actor" && "$grandparent" == global_step_* ]]; then
    basename "$(dirname "$(dirname "$(dirname "$model_path")")")"
    return
  fi
  if [[ "$name" == checkpoint-* ]]; then
    basename "$(dirname "$(dirname "$model_path")")"
  else
    echo "$name"
  fi
}

checkpoint_tag() {
  local model_path="${1%/}"
  local name
  name="$(basename "$model_path")"
  local parent
  local grandparent
  parent="$(basename "$(dirname "$model_path")")"
  grandparent="$(basename "$(dirname "$(dirname "$model_path")")")"
  if [[ "$name" == "huggingface" && "$parent" == "actor" && "$grandparent" == global_step_* ]]; then
    basename "$(dirname "$(dirname "$model_path")")"
    return
  fi
  if [[ "$name" == checkpoint-* ]]; then
    echo "$name"
  else
    echo "base"
  fi
}

IFS=',' read -ra TASK_LIST <<< "$TASKS"
for RAW_TASK in "${TASK_LIST[@]}"; do
  TASK="$(echo "$RAW_TASK" | xargs)"
  case "$TASK" in
    all)
      VLLM_TASKS="videomme,videommev2,videommmu,mmvu,mvbench,videoholmes,longvideobench,lvbench,mlvu,vsi,mindcube,spatial_grounding,tracking,stvg"
      if [[ "${MMSI_BACKEND}" == "transformers" ]]; then
        RUN_MMSI_TRANSFORMERS=1
      else
        append_vllm_task "mmsi"
      fi
      RUN_REVSI=1
      ;;
    videomme|videommev2|videommmu|mmvu|mvbench|videoholmes|longvideobench|lvbench|mlvu|vsi|mindcube|spatial_grounding|tracking|stvg)
      append_vllm_task "$TASK"
      ;;
    mmsi)
      if [[ "${MMSI_BACKEND}" == "transformers" ]]; then
        RUN_MMSI_TRANSFORMERS=1
      else
        append_vllm_task "mmsi"
      fi
      ;;
    revsi)
      RUN_REVSI=1
      ;;
    spatial|grounding|refcoco|spatial_grounding_eval)
      append_vllm_task "spatial_grounding"
      ;;
    spatial_temporal_grounding)
      append_vllm_task "stvg"
      ;;
    temporal_grounding|timelens|timelens_bench)
      RUN_TIMELENS=1
      ;;
    segmentation|seg|reasonseg|refseg)
      RUN_SEGMENTATION=1
      ;;
    "")
      ;;
    *)
      echo "ERROR: unknown TASK '${TASK}'. Supported: videomme,videommev2,videommmu,mmvu,mvbench,videoholmes,longvideobench,lvbench,mlvu,vsi,mmsi,mindcube,revsi,spatial_grounding,tracking,stvg,temporal_grounding,segmentation,all" >&2
      exit 1
      ;;
  esac
done

if [[ -n "$VLLM_TASKS" ]]; then
  TASKS="$VLLM_TASKS" python eval/task/eval_vllm.py "$@"
fi

if [[ "$RUN_MMSI_TRANSFORMERS" = "1" ]]; then
  MMSI_FAMILY="$(model_family_tag "$MODEL_PATH")"
  MMSI_CKPT="$(checkpoint_tag "$MODEL_PATH")"
  MMSI_RUN_ROOT="${PROJECT_DIR}/outputs/${MMSI_FAMILY}/spatial_intelligence/mmsi/${MMSI_CKPT}/${MMSI_SETTING}/$(date +%Y%m%d_%H%M%S)"
  CUDA_VISIBLE_DEVICES="$GPUS" \
  PROCESSOR_PATH="$MODEL_PATH" \
  DATA_FILE="$MMSI_DATA_FILE" \
  OUTPUT_DIR="$MMSI_RUN_ROOT" \
  IMAGE_MIN_PIXELS="$MMSI_IMAGE_MIN_PIXELS" \
  IMAGE_MAX_PIXELS="$MMSI_IMAGE_MAX_PIXELS" \
  MAX_IMAGES="$MMSI_MAX_IMAGES" \
  MAX_NEW_TOKENS="$MMSI_MAX_NEW_TOKENS" \
  MAX_SAMPLES="$MMSI_MAX_SAMPLES" \
  CATEGORY="$MMSI_CATEGORY" \
  ENABLE_THINKING="$MMSI_ENABLE_THINKING" \
  ATTN_IMPLEMENTATION="$MMSI_ATTN_IMPLEMENTATION" \
  EXPECTED_SAMPLES="$MMSI_EXPECTED_SAMPLES" \
  bash eval/task/mmsi/run_eval_transformers.sh "$MODEL_PATH"
fi

if [[ "$RUN_REVSI" = "1" ]]; then
  REVSI_FAMILY="$(model_family_tag "$MODEL_PATH")"
  REVSI_CKPT="$(checkpoint_tag "$MODEL_PATH")"
  REVSI_RUN_EXPECTED_SAMPLES="$REVSI_EXPECTED_SAMPLES"
  if [[ "$REVSI_MAX_SAMPLES" != "0" || -n "$REVSI_TASK_FILTER" ]]; then
    REVSI_RUN_EXPECTED_SAMPLES=0
  fi
  REVSI_RUN_ROOT="${PROJECT_DIR}/outputs/${REVSI_FAMILY}/spatial_intelligence/revsi/${REVSI_CKPT}/${REVSI_SETTING}/$(date +%Y%m%d_%H%M%S)"
  CUDA_VISIBLE_DEVICES="$GPUS" \
  REVSI_ROOT="$REVSI_VIDEO_ROOT" \
  QA_FILE="$REVSI_DATA_FILE" \
  VIDEO_ROOT="$REVSI_VIDEO_ROOT" \
  FRAME_BUDGET="$REVSI_FRAME_BUDGET" \
  OUTPUT_DIR="$REVSI_RUN_ROOT" \
  MAX_FRAMES="$REVSI_MAX_FRAMES" \
  EXACT_NFRAMES="$REVSI_EXACT_NFRAMES" \
  FPS="$REVSI_FPS" \
  VIDEO_MIN_PIXELS="$REVSI_VIDEO_MIN_PIXELS" \
  VIDEO_MAX_PIXELS="$REVSI_VIDEO_MAX_PIXELS" \
  VIDEO_TOTAL_PIXELS="$REVSI_VIDEO_TOTAL_PIXELS" \
  BATCH_SIZE="$REVSI_BATCH_SIZE" \
  MAX_MODEL_LEN="$REVSI_MAX_MODEL_LEN" \
  MAX_NEW_TOKENS="$REVSI_MAX_NEW_TOKENS" \
  GPU_MEM_UTIL="$REVSI_GPU_MEMORY_UTILIZATION" \
  MAX_SAMPLES="$REVSI_MAX_SAMPLES" \
  EXPECTED_SAMPLES="$REVSI_RUN_EXPECTED_SAMPLES" \
  TASK_FILTER="$REVSI_TASK_FILTER" \
  ENABLE_THINKING="$REVSI_ENABLE_THINKING" \
  TP_SIZE="$TP_SIZE" \
  bash eval/task/revsi/run_eval_vllm.sh "$MODEL_PATH"
fi

if [[ "$RUN_TIMELENS" = "1" ]]; then
  TIMELENS_FAMILY="$(model_family_tag "$MODEL_PATH")"
  TIMELENS_CKPT="$(checkpoint_tag "$MODEL_PATH")"
  TIMELENS_SETTING="timelens-fps${TIMELENS_FPS}-min${TIMELENS_MIN_TOKENS}-total${TIMELENS_TOTAL_TOKENS}-new${TIMELENS_MAX_NEW_TOKENS}-${TIMELENS_DATASETS//,/_}"
  TIMELENS_RUN_ROOT="${PROJECT_DIR}/outputs/${TIMELENS_FAMILY}/temporal_grounding/${TIMELENS_CKPT}/${TIMELENS_SETTING}/$(date +%Y%m%d_%H%M%S)"
  CUDA_VISIBLE_DEVICES="$GPUS" \
  DATASETS="$TIMELENS_DATASETS" \
  SETTING="$TIMELENS_SETTING" \
  OUTPUT_ROOT="$TIMELENS_RUN_ROOT" \
  TIMELENS_USE_OUTPUT_ROOT_DIRECT=1 \
  MIN_TOKENS="$TIMELENS_MIN_TOKENS" \
  TOTAL_TOKENS="$TIMELENS_TOTAL_TOKENS" \
  MAX_PIXELS="$TIMELENS_MAX_PIXELS" \
  MAX_FRAMES="$TIMELENS_MAX_FRAMES" \
  FPS="$TIMELENS_FPS" \
  MAX_NEW_TOKENS="$TIMELENS_MAX_NEW_TOKENS" \
  REPETITION_PENALTY="$TIMELENS_REPETITION_PENALTY" \
  STOP_AFTER_ANSWER="$TIMELENS_STOP_AFTER_ANSWER" \
  PROMPT_MODE="$TIMELENS_PROMPT_MODE" \
  NUM_WORKERS="$TIMELENS_NUM_WORKERS" \
  bash eval/task/temporal_grounding/run_eval.sh \
    "$MODEL_PATH" \
    "$TIMELENS_ENABLE_THINKING" \
    "$TIMELENS_RUN_ROOT"
fi

if [[ "$RUN_SEGMENTATION" = "1" ]]; then
  SEG_FAMILY="$(model_family_tag "$MODEL_PATH")"
  SEG_CKPT="$(checkpoint_tag "$MODEL_PATH")"
  SEG_RUN_ROOT="${PROJECT_DIR}/outputs/${SEG_FAMILY}/segmentation/${SEG_CKPT}/${SEGMENTATION_SETTING}/$(date +%Y%m%d_%H%M%S)"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPUS}" \
  PROCESSOR_PATH="$SEGMENTATION_PROCESSOR_PATH" \
  BENCH_DIR="$SEGMENTATION_BENCH_DIR" \
  DATA_ROOT="$SEGMENTATION_DATA_ROOT" \
  DATASETS="$SEGMENTATION_DATASETS" \
  DATA_TYPE="$SEGMENTATION_DATA_TYPE" \
  PROMPT_MODE="$SEGMENTATION_PROMPT_MODE" \
  ENABLE_THINKING="$SEGMENTATION_ENABLE_THINKING" \
  BATCH_SIZE="$SEGMENTATION_BATCH_SIZE" \
  MAX_NEW_TOKENS="$SEGMENTATION_MAX_NEW_TOKENS" \
  MAX_MODEL_LEN="$SEGMENTATION_MAX_MODEL_LEN" \
  MAX_PIXELS_IMAGE="$SEGMENTATION_MAX_PIXELS_IMAGE" \
  MIN_PIXELS_IMAGE="$SEGMENTATION_MIN_PIXELS_IMAGE" \
  VIDEO_MAX_PIXELS="$SEGMENTATION_VIDEO_MAX_PIXELS" \
  VIDEO_MIN_PIXELS="$SEGMENTATION_VIDEO_MIN_PIXELS" \
  VIDEO_TOTAL_PIXELS="$SEGMENTATION_VIDEO_TOTAL_PIXELS" \
  MAX_FRAMES="$SEGMENTATION_MAX_FRAMES" \
  FPS="$SEGMENTATION_FPS" \
  VIDEO_READER="$SEGMENTATION_VIDEO_READER" \
  TP_SIZE="$TP_SIZE" \
  GPU_MEM_UTIL="$SEGMENTATION_GPU_MEM_UTIL" \
  SEED="$SEGMENTATION_SEED" \
  MAX_SAMPLES="$SEGMENTATION_MAX_SAMPLES" \
  RUN_SAM2="$SEGMENTATION_RUN_SAM2" \
  SAM2_CKPT="$SEGMENTATION_SAM2_CKPT" \
  SAM2_CFG="$SEGMENTATION_SAM2_CFG" \
  ONETHINKER_SEG_POST="$SEGMENTATION_POSTPROCESSOR_PATH" \
  SAM2_NUM_GPUS="$SEGMENTATION_SAM2_NUM_GPUS" \
  SAM2_WORKERS_PER_GPU="$SEGMENTATION_SAM2_WORKERS_PER_GPU" \
  bash eval/task/segmentation/run_eval_vllm.sh \
    "$MODEL_PATH" \
    "$SEG_RUN_ROOT"
fi
