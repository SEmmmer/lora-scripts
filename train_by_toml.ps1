$ErrorActionPreference = "Stop"

# LoRA train script by @Akegarasu

$config_file = "./config/default.toml"
$sample_prompts = "./config/sample_prompts.txt"

$sdxl = 0
$multi_gpu = 0

# Cross-machine distributed training
$num_processes_per_machine = if ($env:NUM_PROCESSES_PER_MACHINE) { [int]$env:NUM_PROCESSES_PER_MACHINE } else { 1 }
$num_machines = if ($env:NUM_MACHINES) { [int]$env:NUM_MACHINES } else { 1 }
$machine_rank = if ($env:MACHINE_RANK) { [int]$env:MACHINE_RANK } else { 0 }
$main_process_ip = if ($env:MAIN_PROCESS_IP) { $env:MAIN_PROCESS_IP } else { "192.168.50.219" }
$main_process_port = if ($env:MAIN_PROCESS_PORT) { [int]$env:MAIN_PROCESS_PORT } else { 29500 }
$nccl_socket_ifname = if ($env:NCCL_SOCKET_IFNAME) { $env:NCCL_SOCKET_IFNAME } else { "" }
$gloo_socket_ifname = if ($env:GLOO_SOCKET_IFNAME) { $env:GLOO_SOCKET_IFNAME } else { "" }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

function Resolve-PythonBin {
    if (Test-Path ".\\venv\\Scripts\\python.exe") {
        return (Resolve-Path ".\\venv\\Scripts\\python.exe").Path
    }
    throw "embedded venv python not found (.\\venv\\Scripts\\python.exe). Run .\\install.ps1 first."
}

$pythonBin = Resolve-PythonBin

$env:HF_HOME = "huggingface"
$env:TF_CPP_MIN_LOG_LEVEL = "3"
$env:PYTHONUTF8 = "1"
$env:XFORMERS_FORCE_DISABLE_TRITON = "1"
if ($nccl_socket_ifname) { $env:NCCL_SOCKET_IFNAME = $nccl_socket_ifname }
if ($gloo_socket_ifname) { $env:GLOO_SOCKET_IFNAME = $gloo_socket_ifname }

$ext_args = [System.Collections.ArrayList]::new()
$launch_args = [System.Collections.ArrayList]::new()

$total_num_processes = $num_processes_per_machine * $num_machines

if ($num_machines -gt 1) {
    $multi_gpu = 1
    if ([string]::IsNullOrWhiteSpace($main_process_ip)) {
        throw "main_process_ip is required when num_machines > 1"
    }
    if ($machine_rank -lt 0 -or $machine_rank -ge $num_machines) {
        throw "machine_rank must be in [0, num_machines-1], current: $machine_rank"
    }
}

if ($multi_gpu -eq 1) {
    if ($total_num_processes -lt 2) {
        throw "total processes must be >= 2 for --multi_gpu (num_processes_per_machine=$num_processes_per_machine, num_machines=$num_machines)"
    }

    [void]$launch_args.Add("--multi_gpu")
    [void]$launch_args.Add("--num_processes=$total_num_processes")

    if ($num_machines -gt 1) {
        [void]$launch_args.Add("--num_machines=$num_machines")
        [void]$launch_args.Add("--machine_rank=$machine_rank")
        [void]$launch_args.Add("--main_process_ip=$main_process_ip")
        [void]$launch_args.Add("--main_process_port=$main_process_port")
    }

    $env:USE_LIBUV = "0"
    [void]$launch_args.Add("--rdzv_backend")
    [void]$launch_args.Add("c10d")
}

if ($sdxl -eq 1) {
    $script_name = "./scripts/stable/sdxl_train_network.py"
}
else {
    $script_name = "./scripts/stable/train_network.py"
}

& $pythonBin -m accelerate.commands.launch @launch_args --num_cpu_threads_per_process=8 $script_name `
  --config_file=$config_file `
  --sample_prompts=$sample_prompts `
  @ext_args

Write-Output "Train finished"
if ([Environment]::UserInteractive) {
    Read-Host | Out-Null
}
