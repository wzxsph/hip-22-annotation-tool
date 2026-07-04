param(
    [string]$ZipPath = "",
    [int]$Port = 8123,
    [switch]$SkipModelFlow
)

$ErrorActionPreference = "Stop"

function Get-ProjectRoot {
    $candidate = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    if (Test-Path (Join-Path $candidate "pyproject.toml")) {
        return $candidate
    }
    $candidate = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
    if (Test-Path (Join-Path $candidate "pyproject.toml")) {
        return $candidate
    }
    throw "Could not locate project root from $PSScriptRoot"
}

function Invoke-JsonPost {
    param([string]$Uri, [object]$Payload)
    $body = $Payload | ConvertTo-Json -Compress
    return Invoke-WebRequest -Uri $Uri -Method Post -Body $body -ContentType "application/json" -UseBasicParsing -TimeoutSec 15
}

function Remove-TreeWithRetry {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return
    }
    for ($i = 0; $i -lt 8; $i++) {
        try {
            Remove-Item $Path -Recurse -Force -ErrorAction Stop
            return
        } catch {
            Start-Sleep -Milliseconds 750
        }
    }
    Write-Warning "Could not remove temporary smoke-test directory: $Path"
}

$ProjectRoot = Get-ProjectRoot
$DistRoot = Join-Path $ProjectRoot "dist"
if (-not $ZipPath) {
    $ZipPath = Get-ChildItem -Path $DistRoot -Filter "Hip22AnnotationTool-v*-win64-cpu.zip" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $ZipPath -or -not (Test-Path $ZipPath)) {
    throw "ZIP not found. Pass -ZipPath or run scripts\package_zip.ps1 first."
}

$RunRoot = Join-Path $env:TEMP ("hip22-smoke-" + [guid]::NewGuid().ToString("N"))
$ExtractRoot = Join-Path $RunRoot "zip"
$RuntimeRoot = Join-Path $RunRoot "runtime"
$DatasetRoot = Join-Path $RunRoot "dataset"
New-Item -ItemType Directory -Path $ExtractRoot, $RuntimeRoot, $DatasetRoot -Force | Out-Null

try {
    Expand-Archive -Path $ZipPath -DestinationPath $ExtractRoot -Force
    $ExePath = Join-Path $ExtractRoot "Hip22AnnotationTool\Hip22AnnotationTool.exe"
    if (-not (Test-Path $ExePath)) {
        throw "Extracted exe not found: $ExePath"
    }

    $env:HIP22_NO_BROWSER = "1"
    $env:HIP22_RUNTIME_ROOT = $RuntimeRoot
    $env:HIP22_PORT = [string]$Port
    $env:HIP22_LOG_LEVEL = "warning"

    $proc = Start-Process -FilePath $ExePath -PassThru -WindowStyle Hidden
    try {
        $baseUrl = "http://127.0.0.1:$Port"
        $healthy = $false
        for ($i = 0; $i -lt 60; $i++) {
            Start-Sleep -Milliseconds 500
            try {
                $health = Invoke-WebRequest -Uri "$baseUrl/api/health" -UseBasicParsing -TimeoutSec 2
                if ($health.StatusCode -eq 200) {
                    $healthy = $true
                    break
                }
            } catch {
            }
        }
        if (-not $healthy) {
            throw "API health check failed after 30 seconds."
        }

        $index = Invoke-WebRequest -Uri "$baseUrl/" -UseBasicParsing -TimeoutSec 10
        $appJs = Invoke-WebRequest -Uri "$baseUrl/static/app.js" -UseBasicParsing -TimeoutSec 10
        $schema = Invoke-WebRequest -Uri "$baseUrl/api/annotation/schema" -UseBasicParsing -TimeoutSec 10
        if ($index.StatusCode -ne 200 -or $appJs.StatusCode -ne 200 -or $schema.StatusCode -ne 200) {
            throw "Basic HTTP probes failed."
        }

        if (-not $SkipModelFlow) {
            $demoImage = Get-ChildItem -Path (Join-Path $ProjectRoot "demo_picture") -Filter "*.jpg" | Select-Object -First 1
            if (-not $demoImage) {
                throw "No demo .jpg found under demo_picture. Pass -SkipModelFlow to test HTTP only."
            }
            Copy-Item -LiteralPath $demoImage.FullName -Destination (Join-Path $DatasetRoot $demoImage.Name)

            Invoke-JsonPost "$baseUrl/api/annotation/settings" @{ auto_detect = $true; autosave = $true; annotator = "smoke-test" } | Out-Null
            Invoke-JsonPost "$baseUrl/api/annotation/open-folder" @{ folder_path = $DatasetRoot; split = "train" } | Out-Null

            $statusPayload = $null
            for ($i = 0; $i -lt 180; $i++) {
                Start-Sleep -Milliseconds 500
                $statusPayload = (Invoke-WebRequest -Uri "$baseUrl/api/annotation/auto-detect/status" -UseBasicParsing -TimeoutSec 10).Content | ConvertFrom-Json
                if (-not $statusPayload.running -and -not $statusPayload.pending -and -not $statusPayload.processing) {
                    break
                }
            }
            if ($statusPayload.failed -gt 0) {
                throw "Auto-detect failed during smoke test."
            }

            $escaped = [System.Uri]::EscapeDataString($demoImage.Name)
            $annotationResponse = Invoke-WebRequest -Uri "$baseUrl/api/annotation/load-by-name?filename=$escaped" -UseBasicParsing -TimeoutSec 15
            $annotationJson = $annotationResponse.Content
            $annotation = $annotationJson | ConvertFrom-Json
            if ($annotation.auto_initialization.source -eq "model-unavailable") {
                $warnings = ($annotation.auto_initialization.warnings -join "; ")
                throw "Packaged model is unavailable: $warnings"
            }
            $visible = 0
            foreach ($prop in $annotation.keypoints.PSObject.Properties) {
                $point = $prop.Value
                if ($point.visible -and $null -ne $point.x -and $null -ne $point.y) {
                    $visible += 1
                }
            }
            if ($visible -le 0) {
                throw "Model flow produced no visible keypoints."
            }

            Invoke-WebRequest -Uri "$baseUrl/api/annotation/save" -Method Post -Body $annotationJson -ContentType "application/json" -UseBasicParsing -TimeoutSec 15 | Out-Null
            if (-not (Test-Path (Join-Path $DatasetRoot "annotations\$([IO.Path]::GetFileNameWithoutExtension($demoImage.Name)).json"))) {
                throw "Saved annotation JSON was not created."
            }
            if (-not (Test-Path (Join-Path $DatasetRoot "$([IO.Path]::GetFileNameWithoutExtension($demoImage.Name)).txt"))) {
                throw "Saved YOLO sidecar txt was not created."
            }
        }
    } finally {
        if ($proc -and -not $proc.HasExited) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            Wait-Process -Id $proc.Id -Timeout 10 -ErrorAction SilentlyContinue
        }
    }
} finally {
    Remove-Item Env:HIP22_NO_BROWSER -ErrorAction SilentlyContinue
    Remove-Item Env:HIP22_RUNTIME_ROOT -ErrorAction SilentlyContinue
    Remove-Item Env:HIP22_PORT -ErrorAction SilentlyContinue
    Remove-Item Env:HIP22_LOG_LEVEL -ErrorAction SilentlyContinue
    Remove-TreeWithRetry $RunRoot
}

Write-Host "Smoke test PASSED."
