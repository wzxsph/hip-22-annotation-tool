param(
    [string]$PythonLauncher = "py -3.12",
    [switch]$SkipChecks
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPath = Join-Path $ProjectRoot ".venv-win-cpu"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"

function Test-SupportedPython {
    param([string]$ExePath)
    $minor = & $ExePath -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to run Python at $ExePath"
    }
    return $minor -in @("3.10", "3.11", "3.12")
}

function New-BuildVenv {
    Write-Host "Creating Windows CPU build virtualenv with: $PythonLauncher"
    Invoke-Expression "$PythonLauncher -m venv `"$VenvPath`""
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $PythonExe)) {
        throw "Could not create $VenvPath. Install Python 3.10-3.12 and pass -PythonLauncher if needed."
    }
}

Set-Location $ProjectRoot

if (Test-Path $PythonExe) {
    if (-not (Test-SupportedPython $PythonExe)) {
        Write-Host "Existing .venv-win-cpu is not Python 3.10-3.12; recreating it."
        $resolved = (Resolve-Path -LiteralPath $VenvPath).Path
        if (-not $resolved.StartsWith($ProjectRoot + [System.IO.Path]::DirectorySeparatorChar)) {
            throw "Refusing to remove virtualenv outside the project: $resolved"
        }
        Remove-Item -LiteralPath $resolved -Recurse -Force
    }
}

if (-not (Test-Path $PythonExe)) {
    New-BuildVenv
}

if (-not (Test-SupportedPython $PythonExe)) {
    throw "Windows package builds require Python 3.10, 3.11, or 3.12."
}

Write-Host "Installing build tools and CPU inference dependencies..."
& $PythonExe -m pip install --upgrade pip setuptools wheel
& $PythonExe -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
& $PythonExe -m pip install -r (Join-Path $ProjectRoot "requirements-windows-cpu.txt")

if (-not $SkipChecks) {
    Write-Host "Running source checks..."
    & $PythonExe -m pytest -q
    & $PythonExe -m compileall annotation_tool
    $node = Get-Command node -ErrorAction SilentlyContinue
    if ($node) {
        & node --check static\app.js
    } else {
        Write-Warning "node is not available; skipped static/app.js syntax check."
    }
}

$env:HIP22_DEVICE = "cpu"
$env:HIP22_MODEL_DEVICE = "cpu"

Write-Host "Building Hip22AnnotationTool CPU package..."
& $PythonExe -m PyInstaller --noconfirm --clean (Join-Path $ProjectRoot "packaging\windows\hip22_windows_cpu.spec")

Write-Host ""
Write-Host "Build complete:"
Write-Host (Join-Path $ProjectRoot "dist\Hip22AnnotationTool\Hip22AnnotationTool.exe")
Write-Host "Distribute the whole dist\Hip22AnnotationTool folder, not just the exe."
