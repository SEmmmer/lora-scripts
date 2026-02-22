$ErrorActionPreference = "Stop"

$Env:HF_HOME = "huggingface"
$Env:PYTHONUTF8 = "1"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

function Resolve-PythonBin {
    if (Test-Path ".\venv\Scripts\python.exe") {
        return (Resolve-Path ".\venv\Scripts\python.exe").Path
    }
    throw "embedded venv python not found (.\\venv\\Scripts\\python.exe). Run .\\install.ps1 first."
}

function Ensure-VenvOrInstall {
    param([string[]]$CliArgs)

    if (Test-Path ".\venv\Scripts\python.exe") {
        return
    }

    Write-Host "Detected missing virtual environment: .\venv" -ForegroundColor Yellow

    if (-not [Environment]::UserInteractive) {
        throw "No interactive terminal detected. Run .\install.ps1 manually first."
    }

    $answer = Read-Host "venv is missing. Run install.ps1 now? [Y/n]"
    if ($answer -and ($answer -notmatch "^(?i:y|yes)$")) {
        throw "Installation cancelled."
    }

    & ".\install.ps1"

    if (-not (Test-Path ".\venv\Scripts\python.exe")) {
        throw "venv was not created successfully. Please check install.ps1 output."
    }
}

Ensure-VenvOrInstall $args
$pythonBin = Resolve-PythonBin
& $pythonBin "gui.py" @args
