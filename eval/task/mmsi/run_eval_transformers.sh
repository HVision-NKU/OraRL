#!/usr/bin/env bash
# MMSI-Bench evaluation with one Transformers model replica per GPU.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
EVALUATOR="${SCRIPT_DIR}/eval_mmsi_transformers.py"

MODEL_PATH=${1:-${MODEL_PATH:-}}
PROCESSOR_PATH=${PROCESSOR_PATH:-${MODEL_PATH}}
DATA_FILE=${DATA_FILE:-"${PROJECT_DIR}/data/eval/mmsi/MMSI_bench.tsv"}
IMAGE_MIN_PIXELS=${IMAGE_MIN_PIXELS:-4096}
IMAGE_MAX_PIXELS=${IMAGE_MAX_PIXELS:-262144}
MAX_IMAGES=${MAX_IMAGES:-0}
PATCH_SIZE=${PATCH_SIZE:-16}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
MAX_SAMPLES=${MAX_SAMPLES:-0}
CATEGORY=${CATEGORY:-}
ENABLE_THINKING=${ENABLE_THINKING:-false}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
SAVE_INTERVAL=${SAVE_INTERVAL:-20}
if [[ -n "${CATEGORY}" || "${MAX_SAMPLES}" != "0" ]]; then
  EXPECTED_SAMPLES=${EXPECTED_SAMPLES:-0}
else
  EXPECTED_SAMPLES=${EXPECTED_SAMPLES:-1000}
fi
LAUNCH_DELAY=${LAUNCH_DELAY:-2}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${PROJECT_DIR}/outputs/mmsi"}

[[ -n "${MODEL_PATH}" ]] || {
  echo "[FATAL] MODEL_PATH is required." >&2
  exit 1
}
for path in "${EVALUATOR}" "${MODEL_PATH}/config.json" "${DATA_FILE}"; do
  [[ -f "${path}" ]] || {
    echo "[FATAL] Required input is unavailable: ${path}" >&2
    exit 1
  }
done
compgen -G "${MODEL_PATH}/*.safetensors" >/dev/null || {
  echo "[FATAL] No safetensors weights found under ${MODEL_PATH}." >&2
  exit 1
}

IFS=',' read -ra GPULIST <<<"${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $(($(nvidia-smi -L | wc -l) - 1)))}"
NUM_GPUS=${#GPULIST[@]}
((NUM_GPUS > 0)) || {
  echo "[FATAL] No visible GPUs." >&2
  exit 1
}
unset CUDA_VISIBLE_DEVICES

model_name=$(basename "${MODEL_PATH%/}")
model_parent=$(basename "$(dirname "${MODEL_PATH%/}")")
model_grandparent=$(basename "$(dirname "$(dirname "${MODEL_PATH%/}")")")
if [[ "${model_name}" == "huggingface" && "${model_parent}" == "actor" && "${model_grandparent}" == global_step_* ]]; then
  MODEL_FAMILY=$(basename "$(dirname "$(dirname "$(dirname "${MODEL_PATH%/}")")")")
  CHECKPOINT_TAG=${model_grandparent}
elif [[ "${model_name}" == checkpoint-* ]]; then
  MODEL_FAMILY=$(basename "$(dirname "$(dirname "${MODEL_PATH%/}")")")
  CHECKPOINT_TAG=${model_name}
else
  MODEL_FAMILY=${model_name}
  CHECKPOINT_TAG=base
fi

DATA_TAG=$(basename "${DATA_FILE}")
DATA_TAG=${DATA_TAG%.*}
IMAGE_COUNT_TAG=$([[ "${MAX_IMAGES}" -le 0 ]] && echo all || echo "${MAX_IMAGES}")
SETTING_TAG="transformers-img-min${IMAGE_MIN_PIXELS}-max${IMAGE_MAX_PIXELS}-n${IMAGE_COUNT_TAG}-all-${DATA_TAG}"
OUTPUT_DIR=${OUTPUT_DIR:-"${OUTPUT_ROOT}/${MODEL_FAMILY}/${CHECKPOINT_TAG}/${SETTING_TAG}/$(date +%Y%m%d_%H%M%S)"}
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "MMSI-Bench Evaluation (Transformers, data-parallel)"
echo "============================================================"
echo "Model:       ${MODEL_PATH}"
echo "Processor:   ${PROCESSOR_PATH}"
echo "Data:        ${DATA_FILE}"
echo "GPUs:        ${GPULIST[*]} (${NUM_GPUS} replicas)"
echo "Images:      $([[ "${MAX_IMAGES}" -le 0 ]] && echo all || echo "${MAX_IMAGES}")"
echo "Pixels:      min=${IMAGE_MIN_PIXELS} max=${IMAGE_MAX_PIXELS}"
echo "Max new:     ${MAX_NEW_TOKENS}"
echo "Thinking:    ${ENABLE_THINKING}"
echo "Output:      ${OUTPUT_DIR}"
echo "============================================================"

pids=()
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "${pids[@]:-}"; do
    [[ -n "${pid}" ]] && kill "${pid}" >/dev/null 2>&1 || true
  done
  wait >/dev/null 2>&1 || true
  exit "${status}"
}
trap cleanup EXIT INT TERM

for index in "${!GPULIST[@]}"; do
  gpu=${GPULIST[index]}
  args=(
    --model_path "${MODEL_PATH}"
    --processor_path "${PROCESSOR_PATH}"
    --data_file "${DATA_FILE}"
    --output_dir "${OUTPUT_DIR}"
    --image_min_pixels "${IMAGE_MIN_PIXELS}"
    --image_max_pixels "${IMAGE_MAX_PIXELS}"
    --max_images "${MAX_IMAGES}"
    --patch_size "${PATCH_SIZE}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --max_samples "${MAX_SAMPLES}"
    --category "${CATEGORY}"
    --chunk "${NUM_GPUS}"
    --index "${index}"
    --attn_implementation "${ATTN_IMPLEMENTATION}"
    --save_interval "${SAVE_INTERVAL}"
  )
  if [[ "${ENABLE_THINKING,,}" == "true" || "${ENABLE_THINKING}" == "1" ]]; then
    args+=(--enable_thinking)
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    python "${EVALUATOR}" "${args[@]}" \
    >"${OUTPUT_DIR}/worker_${index}.log" 2>&1 &
  pids+=("$!")
  echo "Launched shard ${index}/${NUM_GPUS} on GPU ${gpu} (PID ${pids[-1]})"
  if ((index + 1 < NUM_GPUS)) && ((LAUNCH_DELAY > 0)); then
    sleep "${LAUNCH_DELAY}"
  fi
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[index]}"; then
    echo "[FAIL] shard ${index}; inspect ${OUTPUT_DIR}/worker_${index}.log" >&2
    failed=1
  else
    echo "[DONE] shard ${index}"
  fi
done
pids=()
((failed == 0)) || exit 1

python - "${OUTPUT_DIR}" "${NUM_GPUS}" "${EXPECTED_SAMPLES}" "${MAX_IMAGES}" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

output_dir = Path(sys.argv[1])
num_shards = int(sys.argv[2])
expected = int(sys.argv[3])
max_images = int(sys.argv[4])
records_by_id = {}
for shard in range(num_shards):
    path = output_dir / f"results_mmsi_shard{shard}.json"
    if not path.is_file():
        raise RuntimeError(f"Missing MMSI result shard: {path}")
    for record in json.loads(path.read_text(encoding="utf-8")):
        records_by_id[str(record["id"])] = record

records = list(records_by_id.values())
if expected > 0 and len(records) != expected:
    raise RuntimeError(
        f"Expected {expected} unique MMSI samples, got {len(records)}"
    )

correct = sum(int(record.get("score", 0)) for record in records)
parsed = sum(bool(record.get("pred_answer")) for record in records)
errors = sum(bool(record.get("error")) for record in records)
groups = defaultdict(lambda: {"correct": 0, "total": 0})
for record in records:
    group = str(record.get("group") or "unknown")
    groups[group]["total"] += 1
    groups[group]["correct"] += int(record.get("score", 0))

def pct(value, total):
    return round(100.0 * value / total, 2) if total else 0.0

summary = {
    "backend": "transformers",
    "image_policy": "all" if max_images <= 0 else f"first_{max_images}",
    "num_samples": len(records),
    "correct": correct,
    "accuracy": pct(correct, len(records)),
    "parse_rate": pct(parsed, len(records)),
    "errors": errors,
    "by_category": {
        key: {
            "accuracy": pct(value["correct"], value["total"]),
            "correct": value["correct"],
            "total": value["total"],
        }
        for key, value in sorted(groups.items())
    },
}
(output_dir / "results_mmsi.json").write_text(
    json.dumps(records, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
(output_dir / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(
    f"MMSI Transformers: n={len(records)} "
    f"acc={summary['accuracy']:.2f}% "
    f"parse={summary['parse_rate']:.2f}% errors={errors}"
)
print(f"Summary: {output_dir / 'summary.json'}")
PY

chmod -R a+rX "${OUTPUT_DIR}" 2>/dev/null || true
echo "MMSI Transformers evaluation complete: ${OUTPUT_DIR}"
