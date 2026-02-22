#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$script_dir" || exit 1

create_venv=true
while [[ -n "${1:-}" ]]; do
  case "$1" in
    --disable-venv)
      create_venv=false
      shift
      ;;
    *)
      shift
      ;;
  esac
done

export HF_HOME="huggingface"
export PIP_DISABLE_PIP_VERSION_CHECK="1"
export PYTHONUTF8="1"

EMBEDDED_PYTHON_VERSION="${EMBEDDED_PYTHON_VERSION:-3.10}"
EMBEDDED_PYTHON_DIR="${EMBEDDED_PYTHON_DIR:-$script_dir/python}"
TOOLS_DIR="${EMBEDDED_TOOLS_DIR:-$script_dir/.tools}"
UV_BIN="$TOOLS_DIR/uv"

download_file() {
  local url="$1"
  local target="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -fL "$url" -o "$target"
    return
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -O "$target" "$url"
    return
  fi

  echo "Error: curl/wget not found."
  exit 1
}

get_python_major_minor() {
  local python_bin="$1"
  "$python_bin" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
}

install_uv() {
  if [[ -x "$UV_BIN" ]]; then
    return
  fi

  mkdir -p "$TOOLS_DIR"

  local os_name arch libc asset url tmp_dir uv_candidate
  os_name="$(uname -s)"
  arch="$(uname -m)"
  case "$arch" in
    x86_64|amd64) arch="x86_64" ;;
    aarch64|arm64) arch="aarch64" ;;
    *)
      echo "Unsupported architecture: $arch"
      exit 1
      ;;
  esac

  case "$os_name" in
    Linux)
      libc="gnu"
      if command -v ldd >/dev/null 2>&1 && ldd --version 2>&1 | grep -qi musl; then
        libc="musl"
      fi
      asset="uv-${arch}-unknown-linux-${libc}.tar.gz"
      ;;
    Darwin)
      asset="uv-${arch}-apple-darwin.tar.gz"
      ;;
    *)
      echo "Unsupported OS for install.bash: $os_name"
      exit 1
      ;;
  esac

  url="https://github.com/astral-sh/uv/releases/latest/download/${asset}"
  tmp_dir="$(mktemp -d)"

  echo "Downloading uv (${asset})..."
  download_file "$url" "$tmp_dir/uv.tar.gz"
  tar -xzf "$tmp_dir/uv.tar.gz" -C "$tmp_dir"

  uv_candidate="$(find "$tmp_dir" -type f -name uv | head -n1 || true)"
  if [[ -z "$uv_candidate" ]]; then
    rm -rf "$tmp_dir"
    echo "Failed to locate uv binary in archive."
    exit 1
  fi

  install -m 0755 "$uv_candidate" "$UV_BIN"
  rm -rf "$tmp_dir"
}

resolve_embedded_python_bin() {
  local py_bin
  py_bin="$(find "$EMBEDDED_PYTHON_DIR" -type f -path "*/bin/python3.10" | head -n1 || true)"
  if [[ -z "$py_bin" ]]; then
    py_bin="$(find "$EMBEDDED_PYTHON_DIR" -type f -path "*/bin/python3" | head -n1 || true)"
  fi
  if [[ -z "$py_bin" ]]; then
    py_bin="$(find "$EMBEDDED_PYTHON_DIR" -type f -path "*/bin/python" | head -n1 || true)"
  fi
  if [[ -z "$py_bin" ]]; then
    echo "Failed to find embedded Python binary under: $EMBEDDED_PYTHON_DIR"
    exit 1
  fi
  echo "$py_bin"
}

install_embedded_python() {
  mkdir -p "$EMBEDDED_PYTHON_DIR"
  if find "$EMBEDDED_PYTHON_DIR" -type f \( -path "*/bin/python3.10" -o -path "*/bin/python3" -o -path "*/bin/python" \) | grep -q .; then
    echo "Embedded Python already exists in $EMBEDDED_PYTHON_DIR, skip download."
    return
  fi
  echo "Installing embedded Python ${EMBEDDED_PYTHON_VERSION} to $EMBEDDED_PYTHON_DIR..."
  UV_PYTHON_INSTALL_DIR="$EMBEDDED_PYTHON_DIR" "$UV_BIN" python install "$EMBEDDED_PYTHON_VERSION"
}

install_uv
install_embedded_python

EMBEDDED_PYTHON_BIN="$(resolve_embedded_python_bin)"
echo "Using embedded python: $EMBEDDED_PYTHON_BIN"
EMBEDDED_PYTHON_RUNTIME_VERSION="$(get_python_major_minor "$EMBEDDED_PYTHON_BIN")"
echo "Embedded python version: $EMBEDDED_PYTHON_RUNTIME_VERSION"

if $create_venv; then
  if [[ -x "$script_dir/venv/bin/python" ]]; then
    VENV_PYTHON_VERSION="$(get_python_major_minor "$script_dir/venv/bin/python")"
    if [[ "$VENV_PYTHON_VERSION" != "$EMBEDDED_PYTHON_RUNTIME_VERSION" ]]; then
      echo "Existing venv python version $VENV_PYTHON_VERSION does not match embedded python $EMBEDDED_PYTHON_RUNTIME_VERSION. Recreating venv..."
      rm -rf "$script_dir/venv"
    fi
  fi

  if [[ ! -x "$script_dir/venv/bin/python" ]]; then
    echo "Creating python venv from embedded python..."
    "$EMBEDDED_PYTHON_BIN" -m venv "$script_dir/venv"
  fi
  PYTHON_BIN="$script_dir/venv/bin/python"
else
  PYTHON_BIN="$EMBEDDED_PYTHON_BIN"
fi

ACTIVE_PYTHON_VERSION="$(get_python_major_minor "$PYTHON_BIN")"
if [[ "$ACTIVE_PYTHON_VERSION" != "$EMBEDDED_PYTHON_RUNTIME_VERSION" ]]; then
  echo "Active python version $ACTIVE_PYTHON_VERSION does not match embedded python version $EMBEDDED_PYTHON_RUNTIME_VERSION"
  exit 1
fi
echo "Active python version: $ACTIVE_PYTHON_VERSION"

echo "Installing torch & xformers..."
cuda_version="$(nvidia-smi | grep -oiP 'CUDA Version: \K[\d\.]+' || true)"
if [[ -z "$cuda_version" ]]; then
  cuda_version="$(nvcc --version | grep -oiP 'release \K[\d\.]+' || true)"
fi

if [[ -z "$cuda_version" ]]; then
  echo "Unable to detect CUDA version from nvidia-smi or nvcc."
  exit 1
fi

cuda_major_version="$(echo "$cuda_version" | awk -F'.' '{print $1}')"
cuda_minor_version="$(echo "$cuda_version" | awk -F'.' '{print $2}')"
echo "CUDA Version: $cuda_version"

if (( cuda_major_version >= 12 )); then
  echo "Installing torch 2.10.0+cu128"
  "$PYTHON_BIN" -m pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
  "$PYTHON_BIN" -m pip install --no-deps xformers==0.0.35 --extra-index-url https://download.pytorch.org/whl/cu128
elif (( cuda_major_version == 11 && cuda_minor_version >= 8 )); then
  echo "Installing torch 2.4.0+cu118"
  "$PYTHON_BIN" -m pip install torch==2.4.0+cu118 torchvision==0.19.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
  "$PYTHON_BIN" -m pip install --no-deps xformers==0.0.27.post2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
elif (( cuda_major_version == 11 && cuda_minor_version >= 6 )); then
  echo "Installing torch 1.12.1+cu116"
  "$PYTHON_BIN" -m pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
  "$PYTHON_BIN" -m pip install --upgrade git+https://github.com/facebookresearch/xformers.git@0bad001ddd56c080524d37c84ff58d9cd030ebfd
  "$PYTHON_BIN" -m pip install triton==2.0.0.dev20221202
elif (( cuda_major_version == 11 && cuda_minor_version >= 2 )); then
  echo "Installing torch 1.12.1+cu113"
  "$PYTHON_BIN" -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu116
  "$PYTHON_BIN" -m pip install --upgrade git+https://github.com/facebookresearch/xformers.git@0bad001ddd56c080524d37c84ff58d9cd030ebfd
  "$PYTHON_BIN" -m pip install triton==2.0.0.dev20221202
else
  echo "Unsupported CUDA version: $cuda_version"
  exit 1
fi

echo "Installing deps..."
"$PYTHON_BIN" -m pip install --upgrade -r requirements.txt
echo "Install completed"
