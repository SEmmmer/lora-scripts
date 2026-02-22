$ErrorActionPreference = "Stop"

# tagger script by @bdsqlsz

$train_data_dir = "./input"
$repo_id = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
$model_dir = ""
$batch_size = 12
$max_data_loader_n_workers = 0
$thresh = 0.35
$general_threshold = 0.35
$character_threshold = 0.1
$remove_underscore = 0
$undesired_tags = ""
$recursive = 0
$frequency_tags = 0

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

function Resolve-PythonBin {
    if ($env:PYTHON_BIN) {
        if (Test-Path $env:PYTHON_BIN) {
            return (Resolve-Path $env:PYTHON_BIN).Path
        }
        if (Get-Command $env:PYTHON_BIN -ErrorAction SilentlyContinue) {
            return $env:PYTHON_BIN
        }
    }
    if (Test-Path ".\venv\Scripts\python.exe") {
        return (Resolve-Path ".\venv\Scripts\python.exe").Path
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return "python"
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return "py"
    }
    throw "python executable not found. Please install dependencies first."
}

$pythonBin = Resolve-PythonBin

$env:HF_HOME = "huggingface"
$env:TF_CPP_MIN_LOG_LEVEL = "3"
$env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$ext_args = [System.Collections.ArrayList]::new()

if ($repo_id) { [void]$ext_args.Add("--repo_id=$repo_id") }
if ($model_dir) { [void]$ext_args.Add("--model_dir=$model_dir") }
if ($batch_size -ne 0) { [void]$ext_args.Add("--batch_size=$batch_size") }
if ($null -ne $max_data_loader_n_workers -and "$max_data_loader_n_workers" -ne "") {
    [void]$ext_args.Add("--max_data_loader_n_workers=$max_data_loader_n_workers")
}
if ($null -ne $general_threshold -and "$general_threshold" -ne "") {
    [void]$ext_args.Add("--general_threshold=$general_threshold")
}
if ($null -ne $character_threshold -and "$character_threshold" -ne "") {
    [void]$ext_args.Add("--character_threshold=$character_threshold")
}
if ($remove_underscore -eq 1) { [void]$ext_args.Add("--remove_underscore") }
if ($undesired_tags) { [void]$ext_args.Add("--undesired_tags=$undesired_tags") }
if ($recursive -eq 1) { [void]$ext_args.Add("--recursive") }
if ($frequency_tags -eq 1) { [void]$ext_args.Add("--frequency_tags") }

& $pythonBin -m accelerate.commands.launch --num_cpu_threads_per_process=8 "./scripts/stable/finetune/tag_images_by_wd14_tagger.py" `
    $train_data_dir `
    --thresh=$thresh `
    --caption_extension .txt `
    @ext_args

Write-Output "Tagger finished"
if ([Environment]::UserInteractive) {
    Read-Host | Out-Null
}
