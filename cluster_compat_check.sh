#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="./venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Error: embedded venv python not found at $PYTHON_BIN"
  echo "Run 'bash install.bash' first."
  exit 1
fi

"$PYTHON_BIN" "./cluster_compat_check.py" "$@"
