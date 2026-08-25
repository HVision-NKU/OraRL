#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORARL_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "$#" -eq 0 ]]; then
  echo "Usage: $0 --config SOURCES.yaml --output TRAIN.jsonl [options]" >&2
  echo "No data is downloaded; all source and media paths must be supplied locally." >&2
  exit 2
fi

if ! "${PYTHON_BIN}" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
  echo "OraRL data preparation requires Python 3.10 or newer." >&2
  echo "Set PYTHON_BIN to a compatible interpreter." >&2
  exit 2
fi

export PYTHONPATH="${ORARL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m orarl.data.build "$@"
