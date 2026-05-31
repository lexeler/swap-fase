#!/usr/bin/env bash
# =============================================================================
# swap-fase installer  —  Linux / macOS
# -----------------------------------------------------------------------------
# Creates an ISOLATED project-local .venv (never the global/shared Python env),
# installs the cross-platform base deps + the ONE onnxruntime build that matches
# this machine's OS + accelerator, then fetches the model weights once.
#
#   Linux + NVIDIA   -> onnxruntime-gpu[cuda,cudnn]==1.22.0  (CUDA 12.x / cuDNN 9.x)
#   Linux + AMD      -> onnxruntime-rocm (best effort) else CPU onnxruntime
#   Linux Intel/none -> onnxruntime (CPU)
#   macOS            -> onnxruntime-silicon (CoreML, best effort) else onnxruntime
#
# Usage:
#   ./install.sh                # detect, install, fetch models
#   ./install.sh --no-models    # skip the one-time model download (offline)
#   ./install.sh --cpu          # force the CPU onnxruntime regardless of GPU
#
# Safe to re-run (idempotent): reuses an existing .venv and upgrades in place.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="$HERE/.venv"
PY_MIN="3.12"

# ----- args -----
FETCH_MODELS=1
FORCE_CPU=0
for arg in "$@"; do
    case "$arg" in
        --no-models) FETCH_MODELS=0 ;;
        --cpu)       FORCE_CPU=1 ;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg (try --help)" >&2
            exit 2
            ;;
    esac
done

banner() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
info()   { printf '    %s\n' "$*"; }
warn()   { printf '\033[1;33m[!] %s\033[0m\n' "$*" >&2; }

banner "swap-fase installer (Linux / macOS)"

# ----- detect OS -----
UNAME="$(uname -s)"
case "$UNAME" in
    Linux)  OS="linux" ;;
    Darwin) OS="macos" ;;
    *)
        warn "Unsupported OS '$UNAME'. This installer targets Linux/macOS."
        warn "On Windows use install.ps1 instead."
        exit 1
        ;;
esac
info "OS detected: $OS"

# ----- pick a Python 3.12 interpreter -----
pick_python() {
    for cand in python3.12 python3 python; do
        if command -v "$cand" >/dev/null 2>&1; then
            if "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) else 1)' 2>/dev/null; then
                echo "$cand"
                return 0
            fi
        fi
    done
    return 1
}
PYTHON="$(pick_python || true)"
if [ -z "${PYTHON:-}" ]; then
    warn "Python $PY_MIN not found on PATH. Install Python 3.12 and re-run."
    info "  Ubuntu/Debian: sudo apt install python3.12 python3.12-venv"
    info "  macOS (brew):  brew install python@3.12"
    exit 1
fi
info "Python interpreter: $PYTHON ($("$PYTHON" --version 2>&1))"

# ----- detect accelerator -----
# ACCEL is one of: nvidia | amd | cpu  (macOS is handled as a special case)
ACCEL="cpu"
if [ "$FORCE_CPU" -eq 1 ]; then
    ACCEL="cpu"
    info "Accelerator: forced CPU (--cpu)"
elif [ "$OS" = "macos" ]; then
    ACCEL="macos"
    info "Accelerator: macOS (will try CoreML via onnxruntime-silicon)"
else
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        ACCEL="nvidia"
        info "Accelerator: NVIDIA GPU detected (nvidia-smi)"
    elif command -v rocminfo >/dev/null 2>&1 || [ -d /opt/rocm ]; then
        ACCEL="amd"
        info "Accelerator: AMD ROCm detected (rocminfo / /opt/rocm)"
    else
        ACCEL="cpu"
        info "Accelerator: no GPU detected — using CPU onnxruntime"
    fi
fi

# ----- prefer uv for the venv + installs, else fall back to venv+pip -----
UV=""
if command -v uv >/dev/null 2>&1; then
    UV="$(command -v uv)"
elif [ -x "$HOME/.local/bin/uv" ]; then
    UV="$HOME/.local/bin/uv"
fi

VPY="$VENV/bin/python"

banner "Creating isolated project-local venv at .venv"
if [ -x "$VPY" ]; then
    info "Reusing existing .venv (idempotent)."
else
    if [ -n "$UV" ]; then
        info "Using uv: $UV"
        "$UV" venv --python "$PY_MIN" "$VENV"
    else
        warn "uv not found — falling back to '$PYTHON -m venv' (slower but works)."
        "$PYTHON" -m venv "$VENV"
    fi
fi

# pip front-end: `uv pip` when uv is present (faster, reliable on heavy CUDA wheels).
pip_install() {
    if [ -n "$UV" ]; then
        "$UV" pip install --python "$VPY" "$@"
    else
        "$VPY" -m pip install "$@"
    fi
}
pip_uninstall() {
    if [ -n "$UV" ]; then
        "$UV" pip uninstall --python "$VPY" "$@" || true
    else
        "$VPY" -m pip uninstall -y "$@" || true
    fi
}

if [ -z "$UV" ]; then
    banner "Upgrading pip in the venv"
    "$VPY" -m pip install --upgrade pip
fi

banner "Installing cross-platform base dependencies"
pip_install -r "$HERE/requirements/base.txt"

# ----- install the matching onnxruntime -----
banner "Installing ONNX Runtime for: $ACCEL"
case "$ACCEL" in
    nvidia)
        # insightface may have pulled the CPU onnxruntime transitively; remove it so
        # it never collides with onnxruntime-gpu (silent CPU fallback otherwise).
        pip_uninstall onnxruntime
        pip_install "onnxruntime-gpu[cuda,cudnn]==1.22.0"
        info "Installed onnxruntime-gpu 1.22.0 (CUDA 12.x + cuDNN 9.x, venv-local)."
        info "Launch via ./run.sh so the venv-local CUDA LD_LIBRARY_PATH is exported."
        ;;
    amd)
        info "Trying onnxruntime-rocm (ROCm build; may require a custom index)..."
        if pip_install onnxruntime-rocm; then
            info "Installed onnxruntime-rocm."
        else
            warn "onnxruntime-rocm install failed — falling back to CPU onnxruntime."
            warn "For GPU on AMD, install a ROCm wheel from AMD's index, e.g.:"
            warn "  $VPY -m pip install onnxruntime-rocm --index-url <amd-rocm-index>"
            pip_install onnxruntime
        fi
        ;;
    macos)
        info "Trying onnxruntime-silicon (CoreML build)..."
        if pip_install onnxruntime-silicon; then
            info "Installed onnxruntime-silicon (CoreML)."
        else
            warn "onnxruntime-silicon unavailable — falling back to CPU onnxruntime."
            pip_install onnxruntime
        fi
        ;;
    cpu|*)
        pip_install onnxruntime
        info "Installed CPU onnxruntime (degraded fallback — expect low FPS)."
        ;;
esac

# ----- one-time model fetch -----
if [ "$FETCH_MODELS" -eq 1 ]; then
    banner "Fetching model weights (one-time, ~880 MB: buffalo_l + inswapper_128)"
    info "Downloads into project-local models/ (gitignored); offline thereafter."
    "$VPY" -c "import sys; sys.path.insert(0, 'src'); from swapfase.bootstrap import ensure_models; print('inswapper:', ensure_models())"
else
    banner "Skipping model fetch (--no-models)"
    info "Run later with: $VPY -c \"import sys; sys.path.insert(0,'src'); from swapfase.bootstrap import ensure_models; ensure_models()\""
fi

# ----- next steps -----
banner "Done. Next steps:"
if [ "$OS" = "linux" ]; then
    info "Run the app (preferred launcher — exports venv-local CUDA libs):"
    info "    ./run.sh --target path/to/face.jpg"
    info ""
    info "Output to a virtual camera for Zoom/Meet/Discord:"
    info "    ./run.sh --target path/to/face.jpg --vcam"
    info "  Requires the v4l2loopback kernel module loaded with a 'DeepLiveCam'"
    info "  device (default /dev/video10). See README.md > Virtual camera."
else
    info "Run the app:"
    info "    $VPY run.py --target path/to/face.jpg"
    info ""
    info "Output to a virtual camera for Zoom/Meet/Discord:"
    info "    $VPY run.py --target path/to/face.jpg --vcam"
    info "  On macOS the virtual camera is provided by OBS Studio's 'OBS Virtual"
    info "  Camera' — install OBS Studio first (pyvirtualcam auto-detects it)."
fi
printf '\n\033[1;32mInstall complete.\033[0m\n'
