#!/usr/bin/env bash
# =============================================================================
# swap-fase — Docker run helper (Linux + NVIDIA ONLY)
# =============================================================================
# Guarded wrapper around `docker compose`:
#   1. checks the Linux+NVIDIA prerequisites (toolkit, v4l2loopback, X11),
#   2. grants the container access to the host X11 display (`xhost +local:`),
#   3. builds the image and runs the app, passing your args through.
#
# HONEST CAVEAT: Docker for swap-fase is Linux + NVIDIA only. GPU passthrough +
# a /dev/video* webcam + a GUI window + a kernel virtual camera do NOT work in
# Docker on Windows, macOS, or for AMD/Intel/CPU. On those, use the native
# installer (install.ps1 / install.sh) instead.
#
# Usage:
#   scripts/run-docker.sh --target /app/face.jpg --vcam
#   scripts/run-docker.sh --target /app/face.jpg --device 0 --vcam --vcam-device /dev/video10
# Any args after the script name are forwarded verbatim to `python run.py` in the
# container. Your target photo must live under the project dir (mounted at /app),
# so reference it as /app/<relative-path>, e.g. ./face.jpg -> /app/face.jpg.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"   # project root
cd "$HERE"

warn()  { printf '\033[33m[warn]\033[0m  %s\n' "$*" >&2; }
info()  { printf '\033[36m[info]\033[0m  %s\n' "$*" >&2; }
fail()  { printf '\033[31m[fail]\033[0m  %s\n' "$*" >&2; exit 1; }

# --- 0. OS gate --------------------------------------------------------------
if [[ "$(uname -s)" != "Linux" ]]; then
  fail "Docker mode is Linux+NVIDIA only. On $(uname -s), use the native installer instead."
fi

# --- 1. docker + compose present --------------------------------------------
command -v docker >/dev/null 2>&1 || fail "docker not found. Install Docker Engine first."
if ! docker compose version >/dev/null 2>&1; then
  fail "'docker compose' (v2+) not found. Install the Compose plugin."
fi

# --- 2. nvidia-container-toolkit / GPU passthrough ---------------------------
# We can't fully prove the toolkit without running a GPU container; do cheap
# best-effort checks and WARN (don't hard-fail) so the user can still proceed.
if command -v nvidia-smi >/dev/null 2>&1; then
  info "host nvidia-smi present:"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/         /' >&2 || true
else
  warn "nvidia-smi not found on the host — is the NVIDIA driver installed? GPU passthrough will fail without it."
fi
if command -v nvidia-ctk >/dev/null 2>&1 || docker info 2>/dev/null | grep -qi 'Runtimes:.*nvidia'; then
  info "nvidia-container-toolkit appears configured."
else
  warn "nvidia-container-toolkit not detected. Install it and run 'sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker', or the GPU reservation will fail."
fi

# --- 3. v4l2loopback virtual-camera node (HOST kernel module) ----------------
# /dev/video10 (card 'DeepLiveCam') is a HOST device — a container cannot load
# the kernel module. Only matters if you pass --vcam; warn either way.
VCAM_DEV="/dev/video10"
if [[ ! -e "$VCAM_DEV" ]]; then
  warn "$VCAM_DEV not present. The --vcam virtual camera needs it. Load it on the HOST first:"
  warn "    sudo modprobe v4l2loopback video_nr=10 card_label=\"DeepLiveCam\" exclusive_caps=1"
else
  info "virtual-camera node $VCAM_DEV present."
fi

# --- 4. input webcam ---------------------------------------------------------
if [[ ! -e /dev/video0 ]]; then
  warn "/dev/video0 not present — no input webcam to capture from. Plug one in (or adjust devices in docker-compose.yml)."
else
  info "input webcam /dev/video0 present."
fi

# --- 5. X11 display for the Qt window ----------------------------------------
# The Qt preview needs the host X11 socket. `xhost +local:` lets local
# (container) clients connect. This loosens X access for local processes for the
# session — revoke afterwards with `xhost -local:` if you care.
if command -v xhost >/dev/null 2>&1; then
  warn "granting local X11 access (xhost +local:) so the container can open the Qt window."
  warn "  -> revoke later with: xhost -local:"
  xhost +local: >/dev/null 2>&1 || warn "xhost +local: failed (no X display?). The GUI may not appear."
else
  warn "xhost not found. On a Wayland-only session, ensure XWayland is running so DISPLAY/\$XAUTHORITY work; the Qt window uses the xcb backend over the mounted X11 socket."
fi
: "${DISPLAY:=:0}"
export DISPLAY
info "using DISPLAY=$DISPLAY"

# --- 6. build + run ----------------------------------------------------------
info "building image (docker compose build)..."
docker compose build

if [[ $# -eq 0 ]]; then
  warn "no args given. The app REQUIRES --target <photo>. Example:"
  warn "    scripts/run-docker.sh --target /app/face.jpg --vcam"
  warn "Showing --help inside the container instead:"
  exec docker compose run --rm swap-fase --help
fi

info "running: python run.py $*"
# `run --rm` (not `up`) so we can forward CLI args and auto-remove the container.
exec docker compose run --rm swap-fase "$@"
