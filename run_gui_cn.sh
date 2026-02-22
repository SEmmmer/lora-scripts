#!/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$script_dir" || exit 1

export HF_HOME=huggingface
export HF_ENDPOINT=https://hf-mirror.com
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PYTHONUTF8=1

exec bash "$script_dir/run_gui.sh" "$@"

