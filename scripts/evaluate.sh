#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! "${PYTHON_BIN}" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
  echo "ERROR: OraRL evaluation requires Python 3.10 or newer." >&2
  echo "Set PYTHON_BIN to the compatible verl environment." >&2
  exit 2
fi

export PYTHONPATH="${RELEASE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m orarl.cli.evaluate "$@"
