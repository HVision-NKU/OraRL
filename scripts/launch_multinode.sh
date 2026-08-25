#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  launch_multinode.sh [launcher options] -- <orarl-train options>

Launcher options:
  --hosts LIST          Comma-separated SSH hosts. May also use HOSTS.
  --hostfile FILE       One SSH host per line. May also use HOSTFILE.
  --gpus-per-node N     GPUs per host (default: 8).
  --ray-port N          Ray head port (default: 6379).
  --strict-host-key-checking MODE
                        yes (default), accept-new, ask, or no.
  --run                 Start Ray and training. Default is a safe dry run.
  --dry-run             Print the launch plan without connecting.
  --help                Show this message.

The arguments after -- must include --config, --model, --train-data,
--val-data, and --output. The launcher supplies topology and --run.
EOF
}

HOSTS_VALUE="${HOSTS:-}"
HOSTFILE_VALUE="${HOSTFILE:-}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
RAY_PORT="${RAY_PORT:-6379}"
STRICT_MODE="${STRICT_HOST_KEY_CHECKING:-yes}"
EXECUTE=0

while (($# > 0)); do
  case "$1" in
    --hosts)
      [[ $# -ge 2 ]] || { echo "ERROR: --hosts requires a value" >&2; exit 2; }
      HOSTS_VALUE="$2"
      shift 2
      ;;
    --hostfile)
      [[ $# -ge 2 ]] || { echo "ERROR: --hostfile requires a value" >&2; exit 2; }
      HOSTFILE_VALUE="$2"
      shift 2
      ;;
    --gpus-per-node)
      [[ $# -ge 2 ]] || { echo "ERROR: --gpus-per-node requires a value" >&2; exit 2; }
      GPUS_PER_NODE="$2"
      shift 2
      ;;
    --ray-port)
      [[ $# -ge 2 ]] || { echo "ERROR: --ray-port requires a value" >&2; exit 2; }
      RAY_PORT="$2"
      shift 2
      ;;
    --strict-host-key-checking)
      [[ $# -ge 2 ]] || {
        echo "ERROR: --strict-host-key-checking requires a value" >&2
        exit 2
      }
      STRICT_MODE="$2"
      shift 2
      ;;
    --run)
      EXECUTE=1
      shift
      ;;
    --dry-run)
      EXECUTE=0
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
      echo "ERROR: unknown launcher option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

TRAIN_ARGS=("$@")
if ((${#TRAIN_ARGS[@]} == 0)); then
  echo "ERROR: pass orarl-train options after --" >&2
  exit 2
fi
if [[ -n "${HOSTS_VALUE}" && -n "${HOSTFILE_VALUE}" ]]; then
  echo "ERROR: use only one of HOSTS/--hosts and HOSTFILE/--hostfile" >&2
  exit 2
fi
if [[ ! "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: GPUs per node must be a positive integer" >&2
  exit 2
fi
if [[ ! "${RAY_PORT}" =~ ^[0-9]+$ ]] || ((RAY_PORT < 1 || RAY_PORT > 65535)); then
  echo "ERROR: Ray port must be between 1 and 65535" >&2
  exit 2
fi
case "${STRICT_MODE}" in
  yes|accept-new|ask|no) ;;
  *)
    echo "ERROR: strict host key checking must be yes, accept-new, ask, or no" >&2
    exit 2
    ;;
esac

for argument in "${TRAIN_ARGS[@]}"; do
  lowered="${argument,,}"
  if [[ "${lowered}" == "--run" || "${lowered}" == "--dry-run" ]]; then
    echo "ERROR: training execution mode is controlled by the launcher" >&2
    exit 2
  fi
  if [[ "${lowered}" =~ (^|[._-])(api[_-]?key|auth[_-]?token|access[_-]?token|secret|password|credential|private[_-]?key)(=|$) ]]; then
    echo "ERROR: credential-shaped arguments are not forwarded" >&2
    exit 2
  fi
  if [[ "${argument}" =~ ://[^/@[:space:]]+:[^@/[:space:]]+@ ]]; then
    echo "ERROR: URLs with embedded authentication are not forwarded" >&2
    exit 2
  fi
done

declare -a RAW_HOSTS=()
if [[ -n "${HOSTFILE_VALUE}" ]]; then
  [[ -f "${HOSTFILE_VALUE}" ]] || {
    echo "ERROR: host file does not exist: ${HOSTFILE_VALUE}" >&2
    exit 2
  }
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -n "${line}" ]] && RAW_HOSTS+=("${line}")
  done < "${HOSTFILE_VALUE}"
elif [[ -n "${HOSTS_VALUE}" ]]; then
  IFS=',' read -r -a RAW_HOSTS <<< "${HOSTS_VALUE}"
else
  echo "ERROR: set HOSTS, set HOSTFILE, or pass the matching option" >&2
  exit 2
fi

declare -a HOST_LIST=()
declare -A SEEN_HOSTS=()
for host in "${RAW_HOSTS[@]}"; do
  host="${host#"${host%%[![:space:]]*}"}"
  host="${host%"${host##*[![:space:]]}"}"
  if [[ ! "${host}" =~ ^([A-Za-z0-9._-]+@)?[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: invalid SSH host entry" >&2
    exit 2
  fi
  if [[ -z "${SEEN_HOSTS[${host}]:-}" ]]; then
    HOST_LIST+=("${host}")
    SEEN_HOSTS["${host}"]=1
  fi
done
if ((${#HOST_LIST[@]} == 0)); then
  echo "ERROR: no hosts were provided" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "ERROR: active Python executable was not found" >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
  echo "ERROR: OraRL multi-node launch requires Python 3.10 or newer." >&2
  echo "Set PYTHON_BIN to the compatible verl environment." >&2
  exit 2
fi
export PYTHONPATH="${RELEASE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

FORWARD_VARIABLES=(
  PATH
  PYTHONPATH
  LD_LIBRARY_PATH
  VIRTUAL_ENV
  CONDA_PREFIX
  CONDA_DEFAULT_ENV
  CUDA_VISIBLE_DEVICES
  NCCL_DEBUG
  NCCL_SOCKET_IFNAME
  NCCL_IB_HCA
  NCCL_IB_DISABLE
  NCCL_P2P_DISABLE
  TOKENIZERS_PARALLELISM
  HF_HOME
  TRANSFORMERS_CACHE
  XDG_CACHE_HOME
  ORARL_MEDIA_ROOT
  ORARL_PREPROCESSED_DIR
  ORARL_DATALOADER_WORKERS
  ORARL_VIDEO_MAX_PIXELS
  ORARL_VIDEO_MIN_PIXELS
  ORARL_VIDEO_MAX_FRAMES
  ORARL_VIDEO_FPS
  ORARL_VIDEO_TOTAL_PIXELS
  ORARL_IMAGE_MIN_PIXELS
  ORARL_IMAGE_MAX_PIXELS
  ORARL_VIDEO_SOURCE_MODE
  ORARL_INLINE_VIDEO_TENSORS
  ORARL_MAX_PROMPT_LENGTH
  ORARL_MAX_RESPONSE_LENGTH
  ORARL_ROLLOUT_BATCH_SIZE
  ORARL_VAL_BATCH_SIZE
  ORARL_ENABLE_THINKING
  ORARL_SEED
  ORARL_GLOBAL_BATCH_SIZE
  ORARL_UPDATE_MICRO_BATCH
  ORARL_EXPERIENCE_MICRO_BATCH
  ORARL_MAX_TOKEN_LEN_PER_GPU
  ORARL_LEARNING_RATE
  ORARL_ROLLOUT_BACKEND
)

SSH_OPTIONS=(
  -o "BatchMode=yes"
  -o "StrictHostKeyChecking=${STRICT_MODE}"
)

build_remote_command() {
  local -a parts=("env")
  local name
  for name in "${FORWARD_VARIABLES[@]}"; do
    if [[ -v "${name}" ]]; then
      parts+=("${name}=${!name}")
    fi
  done
  parts+=("$@")
  local rendered
  printf -v rendered '%q ' "${parts[@]}"
  printf '%s' "${rendered% }"
}

display_command() {
  local host="$1"
  local label="$2"
  shift 2
  local rendered
  printf -v rendered '%q ' "$@"
  printf '[dry-run] %s on %s: %s\n' "${label}" "${host}" "${rendered% }"
  printf '          forwards %d allowlisted environment names when set\n' \
    "${#FORWARD_VARIABLES[@]}"
}

run_remote() {
  local host="$1"
  local label="$2"
  shift 2
  if ((EXECUTE == 0)); then
    display_command "${host}" "${label}" "$@"
    return
  fi
  printf '[run] %s on %s\n' "${label}" "${host}"
  remote_command="$(build_remote_command "$@")"
  ssh "${SSH_OPTIONS[@]}" "${host}" "${remote_command}"
}

HEAD_HOST="${HOST_LIST[0]}"
HEAD_ADDRESS="${HEAD_HOST##*@}"
RAY_ADDRESS="${HEAD_ADDRESS}:${RAY_PORT}"
NODE_COUNT="${#HOST_LIST[@]}"

run_remote "${HEAD_HOST}" "Ray head" \
  "${PYTHON_BIN}" -m ray start \
  --head \
  "--port=${RAY_PORT}" \
  "--num-gpus=${GPUS_PER_NODE}" \
  --disable-usage-stats

for ((index = 1; index < NODE_COUNT; index++)); do
  run_remote "${HOST_LIST[index]}" "Ray worker" \
    "${PYTHON_BIN}" -m ray start \
    "--address=${RAY_ADDRESS}" \
    "--num-gpus=${GPUS_PER_NODE}" \
    --disable-usage-stats
done

TRAIN_COMMAND=(
  "${PYTHON_BIN}"
  -m
  orarl.cli.train
  "${TRAIN_ARGS[@]}"
  --nodes
  "${NODE_COUNT}"
  --gpus-per-node
  "${GPUS_PER_NODE}"
  --run
)

if ((EXECUTE == 0)); then
  display_command "${HEAD_HOST}" "trainer (RAY_ADDRESS set at execution)" \
    "${TRAIN_COMMAND[@]}"
else
  printf '[run] trainer on %s\n' "${HEAD_HOST}"
  remote_command="$(build_remote_command "RAY_ADDRESS=${RAY_ADDRESS}" "${TRAIN_COMMAND[@]}")"
  ssh "${SSH_OPTIONS[@]}" "${HEAD_HOST}" "${remote_command}"
fi
