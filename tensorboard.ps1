$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$Env:TF_CPP_MIN_LOG_LEVEL = "3"

$pythonPath = ".\venv\Scripts\python.exe"
if (-not (Test-Path $pythonPath)) {
    throw "embedded venv python not found ($pythonPath). Please run install.ps1 first."
}
$pythonBin = (Resolve-Path $pythonPath).Path

& $pythonBin -m mikazuki.tensorboard_launcher --logdir logs --host 127.0.0.1 --port 6006 @args
