param(
    [switch]$DisableVenv
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$Env:HF_HOME = "huggingface"
$Env:HF_ENDPOINT = "https://hf-mirror.com"
$Env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$Env:PIP_NO_CACHE_DIR = "1"
$Env:PIP_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
$Env:PYTHONUTF8 = "1"

$installScript = Join-Path $ScriptDir "install.ps1"
if (-not (Test-Path $installScript)) {
    throw "install.ps1 not found in project root"
}

& $installScript @PSBoundParameters
