#!/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$script_dir" || exit 1

export HF_HOME=huggingface
export PYTHONUTF8=1

show_python_venv_install_hint() {
  echo "python3 venv support is not available."
  echo "Please install the python venv package first (package name may vary by distro)."
  if command -v apt-get >/dev/null 2>&1; then
    echo "  sudo apt update && sudo apt install -y python3-venv"
  elif command -v dnf >/dev/null 2>&1; then
    echo "  sudo dnf install -y python3-venv"
  elif command -v yum >/dev/null 2>&1; then
    echo "  sudo yum install -y python3-venv"
  elif command -v pacman >/dev/null 2>&1; then
    echo "  sudo pacman -S --needed python"
  elif command -v zypper >/dev/null 2>&1; then
    echo "  sudo zypper install python3-venv"
  elif command -v apk >/dev/null 2>&1; then
    echo "  sudo apk add python3 py3-pip"
  else
    echo "  Install python3-venv (or equivalent) with your system package manager."
  fi
}

check_python_venv_ready() {
  if ! command -v python3 >/dev/null 2>&1; then
    return 1
  fi

  local tmp_parent
  tmp_parent="$(mktemp -d 2>/dev/null)" || return 1

  if ! python3 -m venv "$tmp_parent/venv-check" >/dev/null 2>&1; then
    rm -rf "$tmp_parent"
    return 1
  fi

  rm -rf "$tmp_parent"
  return 0
}

ensure_venv_installed() {
  if [[ -x "./venv/bin/python" ]]; then
    return 0
  fi

  local is_help=0
  for arg in "$@"; do
    if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
      is_help=1
      break
    fi
  done

  if [[ "$is_help" -eq 1 ]]; then
    return 0
  fi

  echo "Detected missing virtual environment: ./venv"

  if ! check_python_venv_ready; then
    show_python_venv_install_hint
    return 1
  fi

  if [[ ! -t 0 ]]; then
    echo "No interactive terminal detected."
    echo "Run 'bash install.bash' manually to install dependencies."
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

  if [[ ! -x "./venv/bin/python" ]]; then
    echo "venv was not created successfully. Please check install logs."
    return 1
  fi

  return 0
}

if ! ensure_venv_installed "$@"; then
  exit 1
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "./venv/bin/python" ]]; then
    PYTHON_BIN="./venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

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
