$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$pythonBin = ".\venv\Scripts\python.exe"
if (-not (Test-Path $pythonBin)) {
    throw "embedded venv python not found at $pythonBin. Run .\install.ps1 first."
}

& $pythonBin ".\cluster_compat_check.py" @args
exit $LASTEXITCODE
