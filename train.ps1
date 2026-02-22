$ErrorActionPreference = "Stop"

# LoRA train script by @Akegarasu

# Train data path
$pretrained_model = "./sd-models/model.ckpt"
$model_type = "sd1.5" # sd1.5 sd2.0 sdxl flux
$parameterization = 0

$train_data_dir = "./train/aki"
$reg_data_dir = ""

# Network settings
$network_module = "networks.lora"
$network_weights = ""
$network_dim = 32
$network_alpha = 32

# Train params
$resolution = "512,512"
$batch_size = 1
$max_train_epoches = 10
$save_every_n_epochs = 2

$train_unet_only = 0
$train_text_encoder_only = 0
$stop_text_encoder_training = 0

$noise_offset = 0
$keep_tokens = 0
$min_snr_gamma = 0

# Learning rate
$lr = "1e-4"
$unet_lr = "1e-4"
$text_encoder_lr = "1e-5"
$lr_scheduler = "cosine_with_restarts"
$lr_warmup_steps = 0
$lr_restart_cycles = 1

# Optimizer
$optimizer_type = "AdamW8bit"

# Output
$output_name = "aki"
$save_model_as = "safetensors"

# Resume
$save_state = 0
$resume = ""

# Other
$min_bucket_reso = 256
$max_bucket_reso = 1024
$persistent_data_loader_workers = 1
$clip_skip = 2
$multi_gpu = 0
$lowram = 0

# Cross-machine distributed training
$num_processes_per_machine = if ($env:NUM_PROCESSES_PER_MACHINE) { [int]$env:NUM_PROCESSES_PER_MACHINE } else { 1 }
$num_machines = if ($env:NUM_MACHINES) { [int]$env:NUM_MACHINES } else { 1 }
$machine_rank = if ($env:MACHINE_RANK) { [int]$env:MACHINE_RANK } else { 0 }
$main_process_ip = if ($env:MAIN_PROCESS_IP) { $env:MAIN_PROCESS_IP } else { "192.168.50.219" }
$main_process_port = if ($env:MAIN_PROCESS_PORT) { [int]$env:MAIN_PROCESS_PORT } else { 29500 }
$nccl_socket_ifname = if ($env:NCCL_SOCKET_IFNAME) { $env:NCCL_SOCKET_IFNAME } else { "" }
$gloo_socket_ifname = if ($env:GLOO_SOCKET_IFNAME) { $env:GLOO_SOCKET_IFNAME } else { "" }

# LyCORIS
$algo = "lora"
$conv_dim = 4
$conv_alpha = 4
$dropout = "0"

# Remote logging
$use_wandb = 0
$wandb_api_key = ""
$log_tracker_name = ""

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

function Resolve-PythonBin {
    if (Test-Path ".\venv\Scripts\python.exe") {
        return (Resolve-Path ".\venv\Scripts\python.exe").Path
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
$trainer_file = "./scripts/stable/train_network.py"

if ($model_type -eq "sd1.5") {
    [void]$ext_args.Add("--clip_skip=$clip_skip")
}
elseif ($model_type -eq "sd2.0") {
    [void]$ext_args.Add("--v2")
}
elseif ($model_type -eq "sdxl") {
    $trainer_file = "./scripts/stable/sdxl_train_network.py"
}
elseif ($model_type -eq "flux") {
    $trainer_file = "./scripts/dev/flux_train_network.py"
}

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

if ($lowram -eq 1) { [void]$ext_args.Add("--lowram") }
if ($parameterization -eq 1) { [void]$ext_args.Add("--v_parameterization") }
if ($train_unet_only -eq 1) { [void]$ext_args.Add("--network_train_unet_only") }
if ($train_text_encoder_only -eq 1) { [void]$ext_args.Add("--network_train_text_encoder_only") }
if ($network_weights) { [void]$ext_args.Add("--network_weights=$network_weights") }
if ($reg_data_dir) { [void]$ext_args.Add("--reg_data_dir=$reg_data_dir") }
if ($optimizer_type) { [void]$ext_args.Add("--optimizer_type=$optimizer_type") }
if ($optimizer_type -eq "DAdaptation") {
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("decouple=True")
}
if ($save_state -eq 1) { [void]$ext_args.Add("--save_state") }
if ($resume) { [void]$ext_args.Add("--resume=$resume") }
if ($persistent_data_loader_workers -eq 1) { [void]$ext_args.Add("--persistent_data_loader_workers") }
if ($network_module -eq "lycoris.kohya") {
    [void]$ext_args.Add("--network_args")
    [void]$ext_args.Add("conv_dim=$conv_dim")
    [void]$ext_args.Add("conv_alpha=$conv_alpha")
    [void]$ext_args.Add("algo=$algo")
    [void]$ext_args.Add("dropout=$dropout")
}
if ($stop_text_encoder_training -ne 0) { [void]$ext_args.Add("--stop_text_encoder_training=$stop_text_encoder_training") }
if ($noise_offset -ne 0) { [void]$ext_args.Add("--noise_offset=$noise_offset") }
if ($min_snr_gamma -ne 0) { [void]$ext_args.Add("--min_snr_gamma=$min_snr_gamma") }

if ($use_wandb -eq 1) {
    [void]$ext_args.Add("--log_with=all")
    if ($wandb_api_key) { [void]$ext_args.Add("--wandb_api_key=$wandb_api_key") }
    if ($log_tracker_name) { [void]$ext_args.Add("--log_tracker_name=$log_tracker_name") }
}
else {
    [void]$ext_args.Add("--log_with=tensorboard")
}

& $pythonBin -m accelerate.commands.launch @launch_args --num_cpu_threads_per_process=4 $trainer_file `
    --enable_bucket `
    --pretrained_model_name_or_path=$pretrained_model `
    --train_data_dir=$train_data_dir `
    --output_dir="./output" `
    --logging_dir="./logs" `
    --log_prefix=$output_name `
    --resolution=$resolution `
    --network_module=$network_module `
    --max_train_epochs=$max_train_epoches `
    --learning_rate=$lr `
    --unet_lr=$unet_lr `
    --text_encoder_lr=$text_encoder_lr `
    --lr_scheduler=$lr_scheduler `
    --lr_warmup_steps=$lr_warmup_steps `
    --lr_scheduler_num_cycles=$lr_restart_cycles `
    --network_dim=$network_dim `
    --network_alpha=$network_alpha `
    --output_name=$output_name `
    --train_batch_size=$batch_size `
    --save_every_n_epochs=$save_every_n_epochs `
    --mixed_precision="fp16" `
    --save_precision="fp16" `
    --seed="1337" `
    --cache_latents `
    --prior_loss_weight=1 `
    --max_token_length=225 `
    --caption_extension=".txt" `
    --save_model_as=$save_model_as `
    --min_bucket_reso=$min_bucket_reso `
    --max_bucket_reso=$max_bucket_reso `
    --keep_tokens=$keep_tokens `
    --xformers --shuffle_caption @ext_args

Write-Output "Train finished"
if ([Environment]::UserInteractive) {
    Read-Host | Out-Null
}
