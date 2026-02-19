#!/bin/bash
# Two-node launcher for Linux cross-machine LoRA training.
# Usage:
#   bash train_2node.sh main [train|toml]
#   bash train_2node.sh worker [train|toml]

set -euo pipefail

node_role="${1:-main}"  # main | worker
train_mode="${2:-train}" # train | toml

usage() {
  echo "Usage: bash train_2node.sh [main|worker] [train|toml]"
  echo "Examples:"
  echo "  bash train_2node.sh main train"
  echo "  bash train_2node.sh worker toml"
}

case "$node_role" in
  main|master|0)
    default_machine_rank=0
    ;;
  worker|1)
    default_machine_rank=1
    ;;
  *)
    usage
    exit 1
    ;;
esac

case "$train_mode" in
  train)
    entry_script="./train.sh"
    ;;
  toml)
    entry_script="./train_by_toml.sh"
    ;;
  *)
    usage
    exit 1
    ;;
esac

if [[ ! -f "$entry_script" ]]; then
  echo "Error: entry script not found: $entry_script"
  exit 1
fi

# Defaults for your two-node setup; all can be overridden by env.
export NUM_MACHINES="${NUM_MACHINES:-2}"
export NUM_PROCESSES_PER_MACHINE="${NUM_PROCESSES_PER_MACHINE:-1}"
export MACHINE_RANK="${MACHINE_RANK:-$default_machine_rank}"
export MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-192.168.50.219}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp11s0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp11s0}"

if (( MACHINE_RANK < 0 || MACHINE_RANK >= NUM_MACHINES )); then
  echo "Error: MACHINE_RANK must be in [0, NUM_MACHINES-1], current: $MACHINE_RANK"
  exit 1
fi

echo "Launching distributed training with:"
echo "  entry_script=$entry_script"
echo "  NUM_MACHINES=$NUM_MACHINES"
echo "  NUM_PROCESSES_PER_MACHINE=$NUM_PROCESSES_PER_MACHINE"
echo "  MACHINE_RANK=$MACHINE_RANK"
echo "  MAIN_PROCESS_IP=$MAIN_PROCESS_IP"
echo "  MAIN_PROCESS_PORT=$MAIN_PROCESS_PORT"
echo "  NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
echo "  GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"

bash "$entry_script"
