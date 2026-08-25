#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" || ! -x "${CONDA_PREFIX}/bin/python" ]]; then
  echo "ERROR: activate the target Conda environment before installing the hook." >&2
  exit 1
fi

HOOK_DIR="${CONDA_PREFIX}/etc/conda"
mkdir -p "${HOOK_DIR}/activate.d" "${HOOK_DIR}/deactivate.d"

cat > "${HOOK_DIR}/activate.d/orarl-runtime.sh" <<'EOF'
export _ORARL_LD_LIBRARY_PATH_WAS_SET="${LD_LIBRARY_PATH+x}"
export _ORARL_SAVED_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"

_ORARL_NVIDIA_LIBS="$("${CONDA_PREFIX}/bin/python" -c '
from pathlib import Path
import sysconfig

root = Path(sysconfig.get_path("purelib")) / "nvidia"
print(":".join(str(path) for path in sorted(root.glob("*/lib")) if path.is_dir()))
')"
_ORARL_RUNTIME_LIBS="${_ORARL_NVIDIA_LIBS}"
if [[ -d "${CONDA_PREFIX}/lib" ]]; then
  _ORARL_RUNTIME_LIBS="${_ORARL_RUNTIME_LIBS:+${_ORARL_RUNTIME_LIBS}:}${CONDA_PREFIX}/lib"
fi
export LD_LIBRARY_PATH="${_ORARL_RUNTIME_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset _ORARL_NVIDIA_LIBS _ORARL_RUNTIME_LIBS
EOF

cat > "${HOOK_DIR}/deactivate.d/orarl-runtime.sh" <<'EOF'
if [[ "${_ORARL_LD_LIBRARY_PATH_WAS_SET:-}" == "x" ]]; then
  export LD_LIBRARY_PATH="${_ORARL_SAVED_LD_LIBRARY_PATH-}"
else
  unset LD_LIBRARY_PATH
fi
unset _ORARL_LD_LIBRARY_PATH_WAS_SET _ORARL_SAVED_LD_LIBRARY_PATH
EOF

chmod 0644 \
  "${HOOK_DIR}/activate.d/orarl-runtime.sh" \
  "${HOOK_DIR}/deactivate.d/orarl-runtime.sh"

echo "Installed OraRL Conda runtime hook in ${CONDA_PREFIX}."
echo "Reactivate the environment to apply it."
