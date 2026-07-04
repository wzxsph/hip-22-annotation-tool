param(
    [switch]$SkipBuild,
    [string]$Version = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DistRoot = Join-Path $ProjectRoot "dist"
$BuildFolderName = "Hip22AnnotationTool"
$BuildFolder = Join-Path $DistRoot $BuildFolderName
$ExePath = Join-Path $BuildFolder "$BuildFolderName.exe"
$StagingRoot = Join-Path $DistRoot "_package_zip_stage"
$StagingFolder = Join-Path $StagingRoot $BuildFolderName
$ReadmeSource = Join-Path $ProjectRoot "packaging\windows\README.txt"
$WrapperSource = Join-Path $ProjectRoot "packaging\windows\Run-Hip22.bat"
$SmokeSource = Join-Path $ProjectRoot "scripts\smoke_windows_package.ps1"
$HospitalGuideSource = Join-Path $ProjectRoot "docs\hospital-user-guide.md"

Set-Location $ProjectRoot

if (-not $Version) {
    $match = Select-String -Path (Join-Path $ProjectRoot "pyproject.toml") -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1
    if (-not $match) {
        throw "Could not find version in pyproject.toml. Pass -Version explicitly."
    }
    $Version = $match.Matches[0].Groups[1].Value
    Write-Host "Detected version: $Version"
}

if (-not $SkipBuild) {
    & powershell -ExecutionPolicy Bypass -File (Join-Path $ProjectRoot "scripts\build_windows_cpu.ps1")
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE."
    }
}

if (-not (Test-Path $ExePath)) {
    throw "Expected exe not found: $ExePath"
}

if (Test-Path $StagingRoot) {
    Remove-Item $StagingRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $StagingFolder -Force | Out-Null

Write-Host "Staging build into $StagingFolder..."
Copy-Item -Path (Join-Path $BuildFolder "*") -Destination $StagingFolder -Recurse -Force

$readmeText = Get-Content -Path $ReadmeSource -Raw -Encoding UTF8
$readmeText = $readmeText.Replace("{VERSION}", $Version)
[System.IO.File]::WriteAllText((Join-Path $StagingFolder "README.txt"), $readmeText, [System.Text.UTF8Encoding]::new($false))
Copy-Item -Path $WrapperSource -Destination (Join-Path $StagingFolder "Run-Hip22.bat") -Force

$guideText = Get-Content -Path $HospitalGuideSource -Raw -Encoding UTF8
[System.IO.File]::WriteAllText((Join-Path $StagingFolder "Hospital-User-Guide-zh.md"), $guideText, [System.Text.UTF8Encoding]::new($false))

$ZipName = "$BuildFolderName-v$Version-win64-cpu.zip"
$ZipPath = Join-Path $DistRoot $ZipName
if (Test-Path $ZipPath) {
    Remove-Item $ZipPath -Force
}

Write-Host "Compressing archive..."
Compress-Archive -Path $StagingFolder -DestinationPath $ZipPath -CompressionLevel Optimal

Copy-Item -Path $SmokeSource -Destination (Join-Path $DistRoot "smoke_test.ps1") -Force

$zipSizeMb = [math]::Round((Get-Item $ZipPath).Length / 1MB, 1)
Write-Host ""
Write-Host "Package complete:"
Write-Host "  $ZipPath"
Write-Host "  Size: $zipSizeMb MB"
Write-Host "Run dist\smoke_test.ps1 before sharing the ZIP."

Remove-Item $StagingRoot -Recurse -Force
