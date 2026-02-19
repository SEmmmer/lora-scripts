#!/bin/bash
# LoRA train script by @Akegarasu

config_file="./config/default.toml"          # config file | 使用 toml 文件指定训练参数
sample_prompts="./config/sample_prompts.txt" # prompt file for sample | 采样 prompts 文件, 留空则不启用采样功能

sdxl=0      # train sdxl LoRA | 训练 SDXL LoRA
multi_gpu=0 # multi gpu | 多显卡训练 该参数仅限在显卡数 >= 2 使用

# Cross-machine distributed training | 跨机器分布式训练
num_processes_per_machine="${NUM_PROCESSES_PER_MACHINE:-1}" # process count per machine, usually equals GPU count | 每台机器进程数，通常等于该机器 GPU 数
num_machines="${NUM_MACHINES:-1}"                           # machine count | 机器总数
machine_rank="${MACHINE_RANK:-0}"                           # this machine rank, main node = 0 | 当前机器 rank，主节点为 0
main_process_ip="${MAIN_PROCESS_IP:-192.168.50.219}"        # main node IP | 主节点 IP
main_process_port="${MAIN_PROCESS_PORT:-29500}"             # main node port | 主节点端口
nccl_socket_ifname="${NCCL_SOCKET_IFNAME:-enp11s0}"         # optional NIC for NCCL, e.g. eth0 / enp3s0
gloo_socket_ifname="${GLOO_SOCKET_IFNAME:-enp11s0}"         # optional NIC for GLOO, e.g. eth0 / enp3s0

# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================

export HF_HOME="huggingface"
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONUTF8=1
if [[ -n "$nccl_socket_ifname" ]]; then export NCCL_SOCKET_IFNAME="$nccl_socket_ifname"; fi
if [[ -n "$gloo_socket_ifname" ]]; then export GLOO_SOCKET_IFNAME="$gloo_socket_ifname"; fi

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

extArgs=()
launchArgs=()

total_num_processes=$((num_processes_per_machine * num_machines))

if (( num_machines > 1 )); then
  multi_gpu=1
  if [[ -z "$main_process_ip" ]]; then
    echo "Error: main_process_ip is required when num_machines > 1"
    exit 1
  fi
  if (( machine_rank < 0 || machine_rank >= num_machines )); then
    echo "Error: machine_rank must be in [0, num_machines-1], current: $machine_rank"
    exit 1
  fi
fi

if [[ $multi_gpu == 1 ]]; then
  if (( total_num_processes < 2 )); then
    echo "Error: total processes must be >= 2 for --multi_gpu (num_processes_per_machine=$num_processes_per_machine, num_machines=$num_machines)"
    exit 1
  fi
  launchArgs+=("--multi_gpu")
  launchArgs+=("--num_processes=$total_num_processes")
  if (( num_machines > 1 )); then
    launchArgs+=("--num_machines=$num_machines")
    launchArgs+=("--machine_rank=$machine_rank")
    launchArgs+=("--main_process_ip=$main_process_ip")
    launchArgs+=("--main_process_port=$main_process_port")
  fi
fi

# run train
if [[ $sdxl == 1 ]]; then
  script_name="./scripts/stable/sdxl_train_network.py"
else
  script_name="./scripts/stable/train_network.py"
fi

"$PYTHON_BIN" -m accelerate.commands.launch "${launchArgs[@]}" --num_cpu_threads_per_process=8 "$script_name" \
  --config_file="$config_file" \
  --sample_prompts="$sample_prompts" \
  "${extArgs[@]}"
