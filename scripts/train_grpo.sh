#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_SIZE="${MODEL_SIZE:-4b}"

if [[ "${1:-}" == "--size" ]]; then
  [[ $# -ge 2 ]] || {
    echo "ERROR: --size requires 4b or 9b" >&2
    exit 2
  }
  MODEL_SIZE="${2,,}"
  shift 2
fi

MODEL_SIZE="${MODEL_SIZE,,}"
case "${MODEL_SIZE}" in
  4b|9b) ;;
  *)
    echo "ERROR: model size must be 4b or 9b" >&2
    exit 2
    ;;
esac

if ! "${PYTHON_BIN}" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
  echo "ERROR: OraRL training requires Python 3.10 or newer." >&2
  echo "Set PYTHON_BIN to the interpreter of the installed orarl environment." >&2
  exit 2
fi

export PYTHONPATH="${RELEASE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m orarl.cli.train \
  --config "${RELEASE_DIR}/configs/grpo_${MODEL_SIZE}.yaml" \
  "$@"
