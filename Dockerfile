# syntax=docker/dockerfile:1
# =============================================================================
# swap-fase — Docker image (Linux + NVIDIA ONLY)
# =============================================================================
# HONEST PLATFORM CAVEAT — READ THIS FIRST:
#   This image is the ONE realistic Docker target for swap-fase: a Linux host
#   with an NVIDIA GPU. Docker here needs ALL of the following ON THE HOST:
#     * an NVIDIA GPU + driver (>=535, CUDA 12.x capable)
#     * nvidia-container-toolkit  (so `--gpus all` / runtime: nvidia works)
#     * the v4l2loopback kernel MODULE loaded on the host (the virtual camera
#       /dev/video10 is a HOST kernel device — a container CANNOT load kernel
#       modules; it can only use a node the host already created). See
#       scripts/run-docker.sh for the modprobe line.
#     * an X11 display socket bind-mounted in (Qt GUI). Pure-Wayland sessions
#       must run XWayland or set QT_QPA_PLATFORM accordingly — see compose.
#   Windows / macOS / AMD / Intel / CPU-only: do NOT use Docker. Use the native
#   installer (install.ps1 on Windows, install.sh on Linux/macOS) instead — GPU
#   passthrough + a /dev/video* webcam + a GUI window do not work in Docker on
#   those platforms.
#
# WHY python:3.12-slim (not nvidia/cuda base):
#   We pull CUDA 12.x runtime + cuDNN 9.x INTO the image via pip, exactly as the
#   project's .venv does — `onnxruntime-gpu[cuda,cudnn]==1.22.0` ships the
#   nvidia-*-cu12 wheels. This keeps CUDA self-contained/in-image (no system CUDA
#   toolkit, mirroring CLAUDE.md's venv-local design) and keeps the base small.
#   The HOST driver is still required and is exposed by nvidia-container-toolkit.
# =============================================================================

FROM python:3.12-slim AS base

# --- OS packages: OpenCV (libGL/glib) + Qt6/PySide6 (xcb platform plugin) -----
# Qt's "xcb" platform plugin needs a fistful of libxcb-* shared objects or it
# fails at runtime with "could not load the Qt platform plugin 'xcb'". libgl1 +
# libglib2.0-0 are OpenCV's runtime needs. The rest cover Qt6 windowing/fonts.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libxkbcommon0 \
        libdbus-1-3 \
        libegl1 \
        libxrender1 \
        libxext6 \
        libsm6 \
        libfontconfig1 \
        libxcb1 \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-shm0 \
        libxcb-sync1 \
        libxcb-util1 \
        libxcb-xfixes0 \
        libxcb-xinerama0 \
        libxcb-xkb1 \
    && rm -rf /var/lib/apt/lists/*

# --- Python deps --------------------------------------------------------------
# Mirror the project layout: cross-platform BASE deps (NO onnxruntime) from
# requirements/base.txt, then the Linux+NVIDIA onnxruntime layered on top. This
# matches install.sh's "base + correct onnxruntime" contract exactly.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy ONLY the requirements first so the heavy pip layer caches across source
# edits (the .dockerignore keeps models/.venv/.git/etc. out of the build).
COPY requirements/ ./requirements/

RUN pip install --upgrade pip \
    && pip install -r requirements/base.txt \
    && pip install "onnxruntime-gpu[cuda,cudnn]==1.22.0" \
    # insightface may have dragged in the CPU `onnxruntime` transitively, which
    # COLLIDES with onnxruntime-gpu and forces a silent CPU fallback. Remove it
    # and reinstall the GPU build so the CUDA EP wins (CLAUDE.md "What NOT to Use";
    # same reconciliation install.sh performs).
    && pip uninstall -y onnxruntime || true \
    && pip install --force-reinstall --no-deps "onnxruntime-gpu[cuda,cudnn]==1.22.0"

# --- venv-local CUDA on the loader path ---------------------------------------
# onnxruntime does NOT patch the dynamic-loader path to its bundled nvidia/*/lib
# wheels (onnxruntime#25609) — run.sh exports LD_LIBRARY_PATH for the .venv. In
# the image the same nvidia-*-cu12 libs live under site-packages/nvidia/*/lib, so
# we point LD_LIBRARY_PATH there. Without this the CUDA EP fails to dlopen cuDNN
# and silently falls back to CPU. Computed once at build time for the known
# site-packages root of python:3.12-slim.
ENV NVIDIA_LIB=/usr/local/lib/python3.12/site-packages/nvidia
ENV LD_LIBRARY_PATH=${NVIDIA_LIB}/cudnn/lib:${NVIDIA_LIB}/cublas/lib:${NVIDIA_LIB}/cuda_runtime/lib:${NVIDIA_LIB}/curand/lib:${NVIDIA_LIB}/cufft/lib:${NVIDIA_LIB}/cuda_nvrtc/lib

# Let nvidia-container-toolkit expose the GPU + the libs the CUDA EP needs.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# --- App source ---------------------------------------------------------------
# Copy the rest of the repo (.dockerignore excludes models/, .venv/, .git/,
# images, *.onnx, *.pth, __pycache__, /tmp). Models are NOT baked in — they are
# bind-mounted at runtime into /app/models (see docker-compose.yml) and the app's
# bootstrap.ensure_models() downloads+verifies them there on first run.
COPY . /app

# bootstrap.py derives MODELS_DIR as <project>/models — i.e. /app/models here.
# Declare it a volume so it's clearly a mount point; compose binds ./models onto
# it so downloaded weights persist on the host and are never committed.
VOLUME ["/app/models"]

# Entrypoint: run the app directly (LD_LIBRARY_PATH is already exported above, so
# we do NOT need run.sh's wrapper). CMD args are appended — e.g.:
#   docker run ... swap-fase --target /app/face.jpg --vcam
ENTRYPOINT ["python", "run.py"]
# Default args show the required flag; override on `docker run`/compose `command`.
CMD ["--help"]
