param(
    [switch]$DisableVenv
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$Env:HF_HOME = "huggingface"
$Env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$Env:PYTHONUTF8 = "1"

$EmbeddedPythonVersion = if ($Env:EMBEDDED_PYTHON_VERSION) { $Env:EMBEDDED_PYTHON_VERSION } else { "3.10" }
$EmbeddedPythonDir = if ($Env:EMBEDDED_PYTHON_DIR) { $Env:EMBEDDED_PYTHON_DIR } else { Join-Path $ScriptDir "python" }
$ToolsDir = if ($Env:EMBEDDED_TOOLS_DIR) { $Env:EMBEDDED_TOOLS_DIR } else { Join-Path $ScriptDir ".tools" }
$UvExe = Join-Path $ToolsDir "uv.exe"

function Invoke-Pip {
    param(
        [string]$PythonBin,
        [string[]]$PipArgs
    )
    & $PythonBin -m pip @PipArgs
    if ($LASTEXITCODE -ne 0) {
        throw "pip command failed: python -m pip $($PipArgs -join ' ')"
    }
}

function Get-PythonMajorMinor {
    param(
        [string]$PythonBin
    )
    $versionOutput = & $PythonBin -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($LASTEXITCODE -ne 0 -or -not $versionOutput) {
        throw "failed to query python version from $PythonBin"
    }
    return ($versionOutput | Select-Object -First 1).Trim()
}

function Install-Uv {
    if (Test-Path $UvExe) {
        return
    }

    New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null

    if ($env:PROCESSOR_ARCHITECTURE -match "ARM64" -or $env:PROCESSOR_ARCHITEW6432 -match "ARM64") {
        $asset = "uv-aarch64-pc-windows-msvc.zip"
    }
    else {
        $asset = "uv-x86_64-pc-windows-msvc.zip"
    }

    $url = "https://github.com/astral-sh/uv/releases/latest/download/$asset"
    $tempDir = Join-Path $env:TEMP ("uv-download-" + [Guid]::NewGuid().ToString("N"))
    $zipPath = Join-Path $tempDir "uv.zip"

    try {
        New-Item -ItemType Directory -Force -Path $tempDir | Out-Null
        Write-Output "Downloading uv ($asset)..."
        Invoke-WebRequest -Uri $url -OutFile $zipPath
        Expand-Archive -Path $zipPath -DestinationPath $tempDir -Force

        $uvCandidate = Get-ChildItem -Path $tempDir -Filter "uv.exe" -Recurse | Select-Object -First 1
        if (-not $uvCandidate) {
            throw "failed to locate uv.exe in downloaded archive"
        }

        Copy-Item -Path $uvCandidate.FullName -Destination $UvExe -Force
    }
    finally {
        if (Test-Path $tempDir) {
            Remove-Item -Path $tempDir -Recurse -Force
        }
    }
}

function Resolve-EmbeddedPythonBin {
    $pattern = "cpython-$EmbeddedPythonVersion"
    $candidate = Get-ChildItem -Path $EmbeddedPythonDir -Filter "python.exe" -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -like "*$pattern*" } |
        Select-Object -First 1
    if (-not $candidate) {
        $candidate = Get-ChildItem -Path $EmbeddedPythonDir -Filter "python.exe" -Recurse -ErrorAction SilentlyContinue |
            Select-Object -First 1
    }
    if (-not $candidate) {
        throw "failed to locate embedded python under $EmbeddedPythonDir"
    }
    return $candidate.FullName
}

function Install-EmbeddedPython {
    New-Item -ItemType Directory -Force -Path $EmbeddedPythonDir | Out-Null
    $existing = Get-ChildItem -Path $EmbeddedPythonDir -Filter "python.exe" -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($existing) {
        Write-Output "Embedded Python already exists in $EmbeddedPythonDir, skip download."
        return
    }
    Write-Output "Installing embedded Python $EmbeddedPythonVersion to $EmbeddedPythonDir..."
    $Env:UV_PYTHON_INSTALL_DIR = $EmbeddedPythonDir
    & $UvExe python install $EmbeddedPythonVersion
    if ($LASTEXITCODE -ne 0) {
        throw "uv python install failed"
    }
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

try {
    Install-Uv
    Install-EmbeddedPython

    $embeddedPythonBin = Resolve-EmbeddedPythonBin
    Write-Output "Using embedded python: $embeddedPythonBin"
    $embeddedPythonRuntimeVersion = Get-PythonMajorMinor $embeddedPythonBin
    Write-Output "Embedded python version: $embeddedPythonRuntimeVersion"

    if (-not $DisableVenv) {
        $venvPythonPath = ".\venv\Scripts\python.exe"
        if (Test-Path $venvPythonPath) {
            $venvPythonVersion = Get-PythonMajorMinor (Resolve-Path $venvPythonPath).Path
            if ($venvPythonVersion -ne $embeddedPythonRuntimeVersion) {
                Write-Output "Existing venv python version $venvPythonVersion does not match embedded python $embeddedPythonRuntimeVersion. Recreating venv..."
                Remove-Item -Path ".\venv" -Recurse -Force
            }
        }

        if (-not (Test-Path $venvPythonPath)) {
            Write-Output "Creating python venv from embedded python..."
            & $embeddedPythonBin -m venv venv
            if ($LASTEXITCODE -ne 0) {
                throw "failed to create python virtual environment from embedded python"
            }
        }
        $pythonBin = (Resolve-Path $venvPythonPath).Path
        Write-Output "Using venv python: $pythonBin"
    }
    else {
        $pythonBin = $embeddedPythonBin
        Write-Output "Using embedded python (venv disabled): $pythonBin"
    }

    $activePythonVersion = Get-PythonMajorMinor $pythonBin
    if ($activePythonVersion -ne $embeddedPythonRuntimeVersion) {
        throw "active python version $activePythonVersion does not match embedded python version $embeddedPythonRuntimeVersion"
    }
    Write-Output "Active python version: $activePythonVersion"

    $cudaVersion = Get-CudaVersion
    if (-not $cudaVersion) {
        throw "Unable to detect CUDA version from nvidia-smi or nvcc."
    }

    Write-Output "CUDA Version: $cudaVersion"
    $cuda = [version]$cudaVersion

    if ($cuda.Major -ge 12) {
        Write-Output "Installing torch 2.10.0+cu128"
        Invoke-Pip $pythonBin @("install", "torch==2.10.0+cu128", "torchvision==0.25.0+cu128", "--extra-index-url", "https://download.pytorch.org/whl/cu128")
        Invoke-Pip $pythonBin @("install", "--no-deps", "xformers==0.0.35", "--extra-index-url", "https://download.pytorch.org/whl/cu128")
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
}
catch {
    $message = $_.Exception.Message
    if (-not $message) {
        $message = "unknown error"
    }
    Write-Output "Install failed: $message"
    exit 1
}
finally {
    if ([Environment]::UserInteractive) {
        Read-Host | Out-Null
    }
}
