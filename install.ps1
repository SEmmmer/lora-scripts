param(
    [switch]$DisableVenv
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$Env:HF_HOME = "huggingface"
$Env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

function Resolve-SystemPython {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return "python"
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return "py"
    }
    throw "python executable not found. Please install Python 3 first."
}

function Invoke-Pip {
    param(
        [string]$PythonBin,
        [string[]]$PipArgs
    )
    & $PythonBin -m pip @PipArgs
}

function Get-CudaVersion {
    try {
        $nvidiaSmi = & nvidia-smi 2>$null
        if ($LASTEXITCODE -eq 0 -and $nvidiaSmi) {
            $m = [regex]::Match(($nvidiaSmi | Out-String), "CUDA Version:\s*([0-9]+\.[0-9]+)")
            if ($m.Success) { return $m.Groups[1].Value }
        }
    } catch {}

    try {
        $nvcc = & nvcc --version 2>$null
        if ($LASTEXITCODE -eq 0 -and $nvcc) {
            $m = [regex]::Match(($nvcc | Out-String), "release\s+([0-9]+\.[0-9]+)")
            if ($m.Success) { return $m.Groups[1].Value }
        }
    } catch {}

    return $null
}

$pythonBin = Resolve-SystemPython

if (-not $DisableVenv) {
    if (-not (Test-Path ".\venv\Scripts\python.exe")) {
        Write-Output "Creating python venv..."
        & $pythonBin -m venv venv
    }
    $pythonBin = (Resolve-Path ".\venv\Scripts\python.exe").Path
    Write-Output "Using venv python: $pythonBin"
}
else {
    Write-Output "Using system python (venv disabled)."
}

$cudaVersion = Get-CudaVersion
if (-not $cudaVersion) {
    throw "Unable to detect CUDA version from nvidia-smi or nvcc."
}

Write-Output "CUDA Version: $cudaVersion"
$cuda = [version]$cudaVersion

if ($cuda.Major -ge 12) {
    Write-Output "Installing torch 2.7.0+cu128"
    Invoke-Pip $pythonBin @("install", "torch==2.7.0+cu128", "torchvision==0.22.0+cu128", "--extra-index-url", "https://download.pytorch.org/whl/cu128")
    Invoke-Pip $pythonBin @("install", "--no-deps", "xformers==0.0.30", "--extra-index-url", "https://download.pytorch.org/whl/cu128")
}
elseif ($cuda.Major -eq 11 -and $cuda.Minor -ge 8) {
    Write-Output "Installing torch 2.4.0+cu118"
    Invoke-Pip $pythonBin @("install", "torch==2.4.0+cu118", "torchvision==0.19.0+cu118", "--extra-index-url", "https://download.pytorch.org/whl/cu118")
    Invoke-Pip $pythonBin @("install", "--no-deps", "xformers==0.0.27.post2+cu118", "--extra-index-url", "https://download.pytorch.org/whl/cu118")
}
elseif ($cuda.Major -eq 11 -and $cuda.Minor -ge 6) {
    Write-Output "Installing torch 1.12.1+cu116"
    Invoke-Pip $pythonBin @("install", "torch==1.12.1+cu116", "torchvision==0.13.1+cu116", "--extra-index-url", "https://download.pytorch.org/whl/cu116")
    Invoke-Pip $pythonBin @("install", "--upgrade", "git+https://github.com/facebookresearch/xformers.git@0bad001ddd56c080524d37c84ff58d9cd030ebfd")
    Invoke-Pip $pythonBin @("install", "triton==2.0.0.dev20221202")
}
elseif ($cuda.Major -eq 11 -and $cuda.Minor -ge 2) {
    Write-Output "Installing torch 1.12.1+cu113"
    Invoke-Pip $pythonBin @("install", "torch==1.12.1+cu113", "torchvision==0.13.1+cu113", "--extra-index-url", "https://download.pytorch.org/whl/cu116")
    Invoke-Pip $pythonBin @("install", "--upgrade", "git+https://github.com/facebookresearch/xformers.git@0bad001ddd56c080524d37c84ff58d9cd030ebfd")
    Invoke-Pip $pythonBin @("install", "triton==2.0.0.dev20221202")
}
else {
    throw "Unsupported CUDA version: $cudaVersion"
}

Write-Output "Installing deps..."
Invoke-Pip $pythonBin @("install", "--upgrade", "-r", "requirements.txt")

Write-Output "Install completed"
if ([Environment]::UserInteractive) {
    Read-Host | Out-Null
}
