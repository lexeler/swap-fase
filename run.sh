#!/usr/bin/env bash
# Launcher: exports the venv-local CUDA LD_LIBRARY_PATH fallback before running
# the app, so onnxruntime's CUDA EP finds the pip-installed nvidia/*/lib shared
# objects (onnxruntime#25609 — ORT does NOT patch the loader path itself).
# Keeps CUDA strictly venv-local; never touches the system (Pitfall 4, "What NOT
# to Use" — no `sudo ldconfig`, no system CUDA toolkit).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="$HERE/.venv"
SITE="$VENV/lib/python3.12/site-packages"
# Resolved venv-local CUDA lib dirs (documents the literal paths SITE expands to):
#   $VENV/lib/python3.12/site-packages/nvidia/cudnn/lib
#   $VENV/lib/python3.12/site-packages/nvidia/cublas/lib
#   $VENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
#   $VENV/lib/python3.12/site-packages/nvidia/curand/lib
#   $VENV/lib/python3.12/site-packages/nvidia/cufft/lib
#   $VENV/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib
# curand/cufft/cuda_nvrtc are added (01-02 carry-forward) so the buffalo_l
# analyser warm-up binds CUDA warning-clean (no dlopen "cannot open" noise).
export LD_LIBRARY_PATH="$SITE/nvidia/cudnn/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/curand/lib:$SITE/nvidia/cufft/lib:$SITE/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
exec "$VENV/bin/python" "$HERE/run.py" "$@"
