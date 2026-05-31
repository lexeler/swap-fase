# =============================================================================
# swap-fase installer  —  Windows (PowerShell)
# -----------------------------------------------------------------------------
# Creates an ISOLATED project-local .venv, installs the cross-platform base deps
# + onnxruntime-directml (DmlExecutionProvider runs on ANY Windows GPU — NVIDIA /
# AMD / Intel — with a built-in CPU fallback), then fetches the model weights once.
#
# Usage (from a PowerShell prompt in the project root):
#   .\install.ps1                # detect Python 3.12, install, fetch models
#   .\install.ps1 -NoModels      # skip the one-time model download (offline)
#
# If you get "running scripts is disabled on this system", allow this session:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# then re-run .\install.ps1
#
# Safe to re-run (idempotent): reuses an existing .venv and upgrades in place.
# =============================================================================

[CmdletBinding()]
param(
    [switch]$NoModels
)

$ErrorActionPreference = 'Stop'

function Write-Banner($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Info($msg)   { Write-Host "    $msg" }
function Write-Warn($msg)   { Write-Host "[!] $msg" -ForegroundColor Yellow }

# Resolve the project root (this script's directory) and key paths.
$Here = Split-Path -Parent $MyInvocation.MyCommand.Definition
$Venv = Join-Path $Here '.venv'
$VPy  = Join-Path $Venv 'Scripts\python.exe'

Write-Banner 'swap-fase installer (Windows)'

# ----- find a Python 3.12 launcher -----
# Prefer the py launcher (`py -3.12`); fall back to a `python` that reports 3.12.
$PyCmd = $null
$PyArgs = @()

if (Get-Command py -ErrorAction SilentlyContinue) {
    try {
        & py -3.12 -c "import sys; assert sys.version_info[:2]==(3,12)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $PyCmd = 'py'
            $PyArgs = @('-3.12')
        }
    } catch { }
}

if (-not $PyCmd -and (Get-Command python -ErrorAction SilentlyContinue)) {
    try {
        & python -c "import sys; assert sys.version_info[:2]==(3,12)" 2>$null
        if ($LASTEXITCODE -eq 0) { $PyCmd = 'python' }
    } catch { }
}

if (-not $PyCmd) {
    Write-Warn 'Python 3.12 was not found.'
    Write-Info 'Install Python 3.12 from https://www.python.org/downloads/ (check'
    Write-Info '"Add python.exe to PATH"), or via: winget install Python.Python.3.12'
    Write-Info 'Then re-run .\install.ps1'
    exit 1
}
$verShown = & $PyCmd @PyArgs --version 2>&1
Write-Info "Python interpreter: $PyCmd $($PyArgs -join ' ') ($verShown)"

# ----- create the isolated project-local venv -----
Write-Banner 'Creating isolated project-local venv at .venv'
if (Test-Path $VPy) {
    Write-Info 'Reusing existing .venv (idempotent).'
} else {
    & $PyCmd @PyArgs -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create the .venv' }
}

# ----- upgrade pip, install base deps -----
Write-Banner 'Upgrading pip in the venv'
& $VPy -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }

Write-Banner 'Installing cross-platform base dependencies'
& $VPy -m pip install -r (Join-Path $Here 'requirements\base.txt')
if ($LASTEXITCODE -ne 0) { throw 'base dependency install failed' }

# ----- ONNX Runtime for Windows = DirectML -----
# DmlExecutionProvider works on any Windows GPU (NVIDIA / AMD / Intel) and falls
# back to CPU automatically. insightface may have pulled the CPU onnxruntime
# transitively — remove it so it never collides with onnxruntime-directml.
Write-Banner 'Installing ONNX Runtime (DirectML) for Windows'
& $VPy -m pip uninstall -y onnxruntime 2>$null | Out-Null
& $VPy -m pip install onnxruntime-directml
if ($LASTEXITCODE -ne 0) { throw 'onnxruntime-directml install failed' }
Write-Info 'Installed onnxruntime-directml (DmlExecutionProvider: NVIDIA/AMD/Intel + CPU fallback).'

# ----- one-time model fetch -----
if ($NoModels) {
    Write-Banner 'Skipping model fetch (-NoModels)'
    Write-Info "Run later with:"
    Write-Info "  $VPy -c `"import sys; sys.path.insert(0,'src'); from swapfase.bootstrap import ensure_models; ensure_models()`""
} else {
    Write-Banner 'Fetching model weights (one-time, ~880 MB: buffalo_l + inswapper_128)'
    Write-Info 'Downloads into project-local models\ (gitignored); offline thereafter.'
    & $VPy -c "import sys; sys.path.insert(0, 'src'); from swapfase.bootstrap import ensure_models; print('inswapper:', ensure_models())"
    if ($LASTEXITCODE -ne 0) { throw 'model fetch failed' }
}

# ----- virtual camera note (Windows) -----
Write-Banner 'Virtual camera (for Zoom / Meet / Discord)'
Write-Info 'On Windows the --vcam output uses the "OBS Virtual Camera" backend.'
Write-Info 'You MUST install OBS Studio (https://obsproject.com) once — it'
Write-Info 'registers the OBS Virtual Camera that pyvirtualcam writes to.'
Write-Info '(Alternatively install Unity Capture.) No need to keep OBS open.'

# ----- next steps -----
Write-Banner 'Done. Next steps:'
Write-Info 'Run the app:'
Write-Info '    .\.venv\Scripts\python run.py --target path\to\face.jpg'
Write-Info ''
Write-Info 'Output to the virtual camera for video calls:'
Write-Info '    .\.venv\Scripts\python run.py --target path\to\face.jpg --vcam'
Write-Info '  Then pick "OBS Virtual Camera" as your webcam in Zoom/Meet/Discord.'
Write-Host "`nInstall complete." -ForegroundColor Green
