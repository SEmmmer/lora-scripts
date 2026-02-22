#!/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$script_dir" || exit 1

export HF_HOME="huggingface"
export PYTHONUTF8="1"

ensure_venv_installed() {
  if [[ -x "$script_dir/venv/bin/python" ]]; then
    return 0
  fi

  echo "Detected missing virtual environment: ./venv"

  if [[ ! -t 0 ]]; then
    echo "No interactive terminal detected."
    echo "Run 'bash install.bash' manually to install embedded Python + dependencies."
    return 1
  fi

  read -r -p "venv is missing. Run install.bash now? [y/N] " reply
  case "$reply" in
    y|Y|yes|YES)
      if ! bash "$script_dir/install.bash"; then
        echo "install.bash failed."
        return 1
      fi
      ;;
    *)
      echo "Installation cancelled."
      return 1
      ;;
  esac

  if [[ ! -x "$script_dir/venv/bin/python" ]]; then
    echo "venv was not created successfully. Please check install logs."
    return 1
  fi

  return 0
}

if ! ensure_venv_installed "$@"; then
  exit 1
fi

"$script_dir/venv/bin/python" gui.py "$@"
