#!/bin/bash

export HF_HOME=huggingface
export PYTHONUTF8=1

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Error: python3/python not found"
    exit 1
  fi
fi

"$PYTHON_BIN" gui.py "$@"
