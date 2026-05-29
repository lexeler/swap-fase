# Stack Research

**Domain:** Local real-time webcam face-swap desktop app (single-image / InsightFace inswapper)
**Researched:** 2026-05-29
**Confidence:** HIGH (versions verified against PyPI live metadata + ONNX Runtime official docs; driver diagnosis verified against the actual machine)

---

## TL;DR — The Prescriptive Stack

| Layer | Pick | Exact version |
|-------|------|---------------|
| Python | system 3.12.3, isolated via **uv venv** | 3.12.x |
| Face swap + analysis | **insightface** | `1.0.1` (auto-downloads `buffalo_l`; `inswapper_128.onnx` supplied manually) |
| Inference runtime | **onnxruntime-gpu** | `1.22.0` installed as `onnxruntime-gpu[cuda,cudnn]` |
| CUDA runtime (venv-local) | `nvidia-cuda-runtime-cu12` + friends | `~=12.x` (pulled by `[cuda]` extra) |
| cuDNN (venv-local) | `nvidia-cudnn-cu12` | `9.x` (pulled by `[cudnn]` extra) |
| Webcam + display | **opencv-python** | `4.13.0.92` |
| GUI window | **PySide6** (Qt6) | `6.8.x` (Wayland-native; see §display) |
| NVIDIA driver (SYSTEM, not venv) | `nvidia-driver-580-open` | `580.126.09` — **already installed; needs module rebuild for the running kernel** |

**The single most important fact about this machine:** the driver is NOT broken because it's missing — it is installed (580.126.09) but the kernel module is only built for `6.17.0-22-generic`, while the machine is booted into `6.17.0-29-generic`. See [§Driver](#nvidia-driver-system-level-not-venv) for the exact fix.

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| insightface | `1.0.1` | Face detection/alignment (`buffalo_l` pack) + face swap (`inswapper_128`) | The de-facto single-image swap stack. **1.0.x (released 2026-05-23) ships a lighter pure-Python default install that no longer needs a C++ build toolchain** — this removes the historic `pip install insightface` failures on fresh machines. Confirmed live on PyPI. |
| onnxruntime-gpu | `1.22.0` | ONNX inference on the GPU via the CUDAExecutionProvider | Runs both `inswapper_128.onnx` and `buffalo_l` models on the RTX 3080 Ti. 1.22.0 is chosen (over the 1.26.0 latest) because **its `[cuda]`/`[cudnn]` pip extras pull a clean, pinned `nvidia-*-cu12 ~=12.x / cudnn ~=9.x` set** (verified in package metadata) and it is the version FaceFusion-class projects converge on for stability. CUDA 12.x + cuDNN 9.x. |
| opencv-python | `4.13.0.92` | Webcam capture (V4L2, `/dev/video0`), frame BGR handling, image I/O | Universal in this domain; provides the V4L2 `VideoCapture` backend Linux needs. Matches FaceFusion's current pin exactly. |
| PySide6 (Qt6) | `6.8.x` | The application window (start/stop, photo picker, live preview) | **`cv2.imshow` is unreliable under GNOME/Wayland** (HighGUI's GTK/X11 path needs XWayland and gives a degraded window). A real Qt6 widget renders the BGR→RGB frame in a `QLabel`/`QImage` and is Wayland-native. PySide6 is LGPL (safe for personal use, no GPL contamination). |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | `2.2.x` (e.g. `2.2.1`) | Array math, frame buffers | Always. ORT 1.22 requires `numpy>=1.21.6`; insightface 1.0.1 and OpenCV 4.13 are numpy-2 compatible. Match FaceFusion's `2.2.1`. |
| onnx | `1.17+` | Model graph loading (transitive via insightface) | Pulled automatically; pin only if conflicts appear. |
| pillow | latest | Loading the user's uploaded target photo (PNG/JPEG) | Always (also a transitive dep). |
| tqdm | `4.67.x` | Progress bars for one-time model downloads | Optional/nice-to-have. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| **uv** | Environment + dependency manager | `uv venv .venv && uv pip install ...`. 10–100x faster than pip, resolves the heavy `nvidia-*-cu12` wheels reliably, keeps everything project-local. See [§Env manager](#environment-manager-uv). |
| `v4l2-ctl` (`v4l-utils`, apt) | Inspect `/dev/video0` formats/resolutions | System tool, not in venv. Use to confirm MJPG/YUYV modes and pick a fast capture resolution. |
| `nvidia-smi` / `nvtop` | Verify GPU is alive + watch utilization | System tools. `nvidia-smi` is the post-fix driver sanity check. |

---

## Installation

> Run from the project directory. Nothing here touches the system Python or any shared venv.

```bash
# 0. (one-time, system-level) install uv if not present
#    curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. Create an ISOLATED project-local venv on Python 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

# 2. ONNX Runtime GPU + its CUDA 12.x runtime + cuDNN 9.x — ALL into the venv via pip extras.
#    The [cuda] extra pulls nvidia-cuda-runtime-cu12, -cuda-nvrtc-cu12, -cufft-cu12, -curand-cu12 (~=12.x)
#    The [cudnn] extra pulls nvidia-cudnn-cu12 (~=9.x). NO system CUDA toolkit needed.
uv pip install "onnxruntime-gpu[cuda,cudnn]==1.22.0"

# 3. Face swap stack
uv pip install "insightface==1.0.1" "opencv-python==4.13.0.92" "numpy==2.2.1"

# 4. GUI
uv pip install "PySide6==6.8.*"
```

**Models (downloaded once, then offline):**

```bash
# buffalo_l face-analysis pack: auto-downloads on first FaceAnalysis(name='buffalo_l').prepare()
#   -> lands in ~/.insightface/models/buffalo_l/  (detection + recognition + landmarks)
# inswapper_128.onnx: MUST be placed manually (see §Models). Recommended location:
#   ~/.insightface/models/inswapper_128.onnx   (≈530 MB fp32)
```

**Library discovery (how the venv-local CUDA is found at runtime):**
The `nvidia-*-cu12` wheels install shared objects under `.venv/lib/python3.12/site-packages/nvidia/*/lib/`. ONNX Runtime ≥1.21 **auto-loads these from site-packages** (it preloads the bundled libs), so in the common case you do NOT need to set `LD_LIBRARY_PATH`. If you hit a `libcudnn.so.9 not found` / `CUDAExecutionProvider failed to load`, export the paths explicitly:

```bash
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:\
$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:\
$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cufft/lib:$LD_LIBRARY_PATH"
```

---

## onnxruntime-gpu ↔ CUDA ↔ cuDNN Compatibility Matrix

This is the make-or-break table. Source: ONNX Runtime official CUDA EP docs + live PyPI package metadata.

| onnxruntime-gpu | Built against CUDA | cuDNN | PyPI default CUDA major | Notes |
|-----------------|--------------------|-------|--------------------------|-------|
| 1.18.x | CUDA **11.8** (PyPI) / 12.x | cuDNN **8.x** | 11.x | Old default. cuDNN 8, NOT 9. |
| 1.19.x | CUDA **12.x** | cuDNN **9.x** | **12.x** | **Default flipped to CUDA 12 here.** |
| 1.20.x | CUDA 12.x | cuDNN 9.x | 12.x | |
| 1.21.x | CUDA 12.x | cuDNN 9.x | 12.x | Auto-preload of site-packages CUDA libs matures. |
| **1.22.0 (recommended)** | **CUDA 12.x** | **cuDNN 9.x** | **12.x** | `[cuda]` extra → `nvidia-cuda-runtime-cu12~=12.0`, `-cuda-nvrtc-cu12~=12.0`, `-cufft-cu12~=11.0`, `-curand-cu12~=10.0`; `[cudnn]` extra → `nvidia-cudnn-cu12~=9.0`. Verified in metadata. |
| 1.23–1.26 | CUDA 12.x | cuDNN 9.x | 12.x | Same family; 1.26.0 is latest. Newer = fine but less battle-tested for this exact use; 1.22 is the conservative pin. |

**CRITICAL pitfalls baked into this matrix:**
1. **CUDA major must match the build, minor does NOT.** ORT built for CUDA 12.x runs on any 12.x (12.0–12.9) thanks to NVIDIA minor-version compat. So the venv `nvidia-cuda-runtime-cu12==12.x` is fine regardless of the exact minor.
2. **cuDNN 8 vs 9 is a HARD break.** ORT ≥1.19 needs cuDNN **9.x**. Installing cuDNN 8 (the old `inswapper`/roop guides) silently fails to load the CUDA EP and falls back to CPU. The `[cudnn]` extra gets this right (`~=9.0`).
3. **The PyPI default is CUDA 12 (since 1.19).** Do not chase CUDA 11.8 guides — they apply to ORT ≤1.18 only.
4. **Driver must support CUDA 12.x.** Driver 580.126.09 supports CUDA 12.x/13.x → comfortably covers anything the venv pulls (see next section).

---

## NVIDIA Driver (SYSTEM-level, not venv)

**Hard architectural fact:** the GPU **kernel driver** (`nvidia.ko` + `nvidia-smi` + userspace libs) is a **system-level** component — it lives in the kernel and `/usr/lib`, and **cannot** be installed into a venv. Only the **CUDA *runtime*** (libcudart, libcudnn, etc.) can be venv-local via the `nvidia-*-cu12` pip wheels. The venv talks to the GPU through the system driver's `/dev/nvidia*` + `libcuda.so` (the "CUDA driver API", shipped by the driver, versioned with it).

**Driver ↔ max CUDA relationship:** the driver caps the **maximum** CUDA version it can run; CUDA has forward/backward compat within a major. Driver **580.x supports CUDA 12.x and 13.x**, so it easily runs the venv's CUDA-12.x runtime.

### Diagnosed root cause on THIS machine (verified, not guessed)

`nvidia-smi` fails with *"couldn't communicate with the NVIDIA driver"* — but the driver is **not** missing:

- Installed: `nvidia-driver-580-open 580.126.09` (+ all `libnvidia-*-580`, `nvidia-utils-580`).
- Secure Boot: **disabled** (no MOK signing blocker — good).
- PRIME mode: **on-demand** (Optimus hybrid; correct mode for a laptop — keeps Intel for display, NVIDIA for compute on request).
- **The problem:** the `nvidia.ko` module is built **only for kernel `6.17.0-22-generic`**, but the machine is **booted into `6.17.0-29-generic`** (kernel was upgraded; the NVIDIA module was never rebuilt for it). `lsmod` shows no nvidia module loaded; `modinfo nvidia` → "Module not found" for the running kernel.

### The fix (system-level — one-time, needs sudo)

```bash
# Install the DKMS/headers so the module builds for the CURRENT and future kernels:
sudo apt update
sudo apt install --reinstall nvidia-dkms-580-open linux-headers-$(uname -r)
# (or: sudo apt install nvidia-driver-580-open  to let the metapackage pull dkms + rebuild)
sudo dkms autoinstall          # build nvidia.ko for 6.17.0-29-generic
```

### Verify WITHOUT a reboot (if possible)

After the module builds for the running kernel, you may be able to load it live (no reboot):

```bash
sudo modprobe nvidia nvidia_uvm nvidia_modeset   # load the freshly-built modules
nvidia-smi                                        # should now list the RTX 3080 Ti
```

If `modprobe` complains the running kernel still lacks the module (e.g. DKMS built only for 6.17.0-22), the clean path is to **reboot into the kernel that has the module** OR rebuild for the running kernel as above. Confirm with:

```bash
lsmod | grep nvidia          # nvidia, nvidia_uvm, nvidia_modeset present
cat /proc/driver/nvidia/version
nvidia-smi                   # driver + CUDA-driver-API version shown
```

Then prove the venv sees it (the real success gate):

```python
import onnxruntime as ort
print(ort.get_available_providers())   # must contain 'CUDAExecutionProvider'
# and inswapper's session must actually run on CUDA, not silently fall to CPU
```

> Because this is an Optimus laptop in `on-demand` mode, compute jobs run on the NVIDIA GPU on request; you may prefix the app with `__NV_PRIME_RENDER_OFFLOAD=1` only for GL display, but **CUDA compute does not need PRIME offload env vars** — ORT's CUDA EP targets the NVIDIA device directly once the driver is up.

---

## Models (acquisition in 2025/2026 — availability has shifted)

| Model | What | How to get it (2026) | Caveats |
|-------|------|----------------------|---------|
| `buffalo_l` pack | Detection (RetinaFace-10GF) + recognition (R50@WebFace600K) + 2d106/3d68 landmarks. ~326 MB. | **Auto-downloads** on first `FaceAnalysis(name='buffalo_l').prepare(...)` into `~/.insightface/models/buffalo_l/`. Mirror: SourceForge `insightface.mirror v0.7/buffalo_l.zip` if auto-DL is blocked offline. | Free, bundled with insightface. No manual step needed if online once. |
| `inswapper_128.onnx` | The actual face-swap model (~530 MB fp32; fp16 ≈277 MB). | **NOT in the pip package and NOT auto-downloaded.** DeepInsight pulled the official public link years ago. In 2026 the practical sources are community mirrors: Hugging Face `ezioruan/inswapper_128.onnx`, `netrunner-exe/Insight-Swap-models` (fp16), or `facefusion/facefusion-assets` GitHub releases. | **License/availability caveat:** InsightFace restricts `inswapper` to **non-commercial / research** use; the original distribution was removed and is mirror-only. This is fine for the user's stated **strictly personal** use, but it means the URL must be treated as community-hosted and pinned/cached locally once downloaded. |

Place `inswapper_128.onnx` at `~/.insightface/models/inswapper_128.onnx` (or a project `models/` dir) and load via `insightface.model_zoo.get_model('.../inswapper_128.onnx', providers=['CUDAExecutionProvider','CPUExecutionProvider'])`.

---

## Webcam Capture + Display (Wayland specifics)

- **Capture:** OpenCV `cv2.VideoCapture(0, cv2.CAP_V4L2)` against `/dev/video0`. The machine exposes `/dev/video0..3` + `video10`; `video0` is the camera capture node. Force MJPG and a modest resolution for FPS headroom:
  ```python
  cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
  cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
  ```
- **Display under GNOME/Wayland — the gotcha:** `cv2.imshow()` from `opencv-python` uses HighGUI's GTK/X11 backend, which on a pure-Wayland session goes through XWayland and is flaky (no window, wrong DPI, frozen frames, or `Qt`/`GTK` plugin errors). **Recommendation: do not rely on `cv2.imshow` for the app window.** Render frames into a **PySide6** widget: convert BGR→RGB, wrap in `QImage`, show in a `QLabel` driven by a `QTimer` at ~30 Hz. This is Wayland-native, gives you the required start/stop + "change target photo without restart" UI naturally, and avoids the XWayland mess. (`opencv-python` headless is NOT needed — keep the full `opencv-python` for codecs/V4L2; just don't use its GUI.)

---

## Environment Manager (uv)

**Pick: `uv` (project-local venv).** Justification vs alternatives:

| Option | Verdict | Why |
|--------|---------|-----|
| **uv** ✅ | **Recommended** | Project-local `.venv`, dead-simple isolation, drop-in `pip` interface, and dramatically faster + more reliable resolution of the large `nvidia-*-cu12` wheels and the `onnxruntime-gpu[cuda,cudnn]` extras. Lockfile support for reproducibility. Honors the user's "isolated, don't touch the shared venv" constraint perfectly. |
| plain `venv` + pip | Acceptable fallback | Works identically and ships with Python 3.12. Slower, no lockfile, but zero extra install. Use if uv can't be installed. |
| conda / mamba | **Avoid here** | Conda's value is system-level CUDA toolkits — but the user explicitly wants **CUDA via pip into the venv**, which ORT's `[cuda,cudnn]` extras already do. Conda adds a heavier, separate ecosystem and tempts you toward conda-channel CUDA that conflicts with the pip CUDA. Unnecessary complexity for this goal. |

---

## Reference Open-Source Projects

| Project | Use it as | Notes |
|---------|-----------|-------|
| **Deep-Live-Cam** (`hacksider/Deep-Live-Cam`) | **Primary architectural reference** | Closest match to the goal: real-time **webcam** single-image swap with `inswapper`. Mirror its capture→detect→swap→display loop and execution-provider selection logic. Caveat: it pins Python **3.10/3.11** and older CUDA/cuDNN guidance — **do not copy its versions**; copy its *architecture*, use this doc's versions. |
| **FaceFusion** (`facefusion/facefusion`) | **Dependency-pinning reference** | Cleanest modern pins (verified live): `onnxruntime==1.24.4`, `opencv-python==4.13.0.92`, `numpy==2.2.1`, `scipy==1.17.1`. Excellent for "what versions actually coexist in 2026." Its full pipeline is heavier (many enhancers) than this MVP needs — borrow the version matrix, not the scope. |
| **roop / roop-unleashed** | Secondary reference | Original single-image swap pipeline; simplest to read for the core `insightface` + `inswapper` call. Largely unmaintained and uses old CUDA 11 / cuDNN 8 era deps — **reference for concept only**, not versions. |

**Best minimal architectural model:** start from **Deep-Live-Cam's** real-time loop (it's the only one of the three built specifically for live webcam), but pin dependencies per **FaceFusion's** current matrix and this document.

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| onnxruntime-gpu 1.22.0 | onnxruntime-gpu 1.24.4 / 1.26.0 | If you want to match FaceFusion exactly (1.24.4) or need a newer ORT feature. Same CUDA 12.x / cuDNN 9.x family — safe, just less conservative. |
| insightface 1.0.1 | insightface 0.7.3 | Only if 1.0.1 surfaces a regression; 0.7.3 is the long-standing pin BUT needs a C++ build toolchain (the historic install-pain version). Prefer 1.0.1. |
| CUDA via ORT `[cuda,cudnn]` extras | Explicit `nvidia-cuda-runtime-cu12` + `nvidia-cudnn-cu12==9.x` pins | If you need to lock exact CUDA/cuDNN builds for reproducibility, pin them explicitly instead of relying on the `~=` extras. |
| PySide6 window | `cv2.imshow` | Only acceptable on an X11 session, or for throwaway debugging via XWayland. Not for the shipped app on this Wayland machine. |
| uv | plain venv | If uv cannot be installed; functionally equivalent, slower. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **System-wide CUDA toolkit (apt `cuda-toolkit` / runfile)** | Violates the user's "venv-local CUDA, don't touch the system" constraint; pollutes `/usr/local/cuda`; risks version drift vs the venv. | `onnxruntime-gpu[cuda,cudnn]` pip extras — CUDA 12.x + cuDNN 9.x **inside the venv**. |
| **`onnxruntime` (CPU package)** | Has no CUDA EP — silently CPU-only (~5–15 FPS), defeating the real-time goal. Installing it alongside `onnxruntime-gpu` causes import collisions. | `onnxruntime-gpu` ONLY (never both in one venv). |
| **cuDNN 8.x** | ORT ≥1.19 needs cuDNN **9.x**; cuDNN 8 makes the CUDA EP fail to load and fall back to CPU with a confusing error. | `nvidia-cudnn-cu12~=9.0` (from the `[cudnn]` extra). |
| **CUDA 11.8 install guides (old roop/inswapper tutorials)** | Apply to ORT ≤1.18 only; wrong for the 2026 default (CUDA 12). | The CUDA-12 / cuDNN-9 path in this doc. |
| **`cv2.imshow` as the app window (on Wayland)** | Flaky/broken under GNOME-Wayland via XWayland. | PySide6 `QLabel` + `QImage`. |
| **Reinstalling/changing the NVIDIA driver from scratch** | The driver (580.126.09) is fine — only the kernel module is unbuilt for the running kernel. A full reinstall risks breaking a working install. | `dkms autoinstall` + `modprobe` (or reboot into the matching kernel). |
| **conda CUDA channels** | Conflicts with pip-installed venv CUDA; unnecessary given pip extras work. | uv/pip venv. |

---

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| onnxruntime-gpu 1.22.0 | CUDA 12.x runtime (`nvidia-cuda-runtime-cu12~=12.0`) | Any 12.x minor; matched by `[cuda]` extra. |
| onnxruntime-gpu 1.22.0 | cuDNN 9.x (`nvidia-cudnn-cu12~=9.0`) | cuDNN 8 is INCOMPATIBLE. |
| NVIDIA driver 580.126.09 | CUDA 12.x and 13.x | Driver max-CUDA ≥ venv CUDA → OK. Driver is system-level. |
| insightface 1.0.1 | numpy 2.2.x, onnxruntime-gpu 1.22 | 1.0.x is numpy-2 ready and needs no C++ build. |
| opencv-python 4.13.0.92 | numpy 2.2.x, Python 3.12 | Matches FaceFusion's live pin. |
| onnxruntime-gpu 1.22.0 | Python 3.12 | `requires_python >=3.10`; 3.12.3 supported. |
| PySide6 6.8.x | Python 3.12, Wayland | Qt6 native Wayland; no XWayland needed. |

---

## Sources

- PyPI live JSON metadata (queried 2026-05-29) — `onnxruntime-gpu` (latest 1.26.0; 1.22.0 `requires_dist` confirming `extra=="cuda"`→`nvidia-cuda-runtime-cu12~=12.0`/`-cuda-nvrtc-cu12`/`-cufft-cu12~=11.0`/`-curand-cu12~=10.0`, `extra=="cudnn"`→`nvidia-cudnn-cu12~=9.0`), `insightface` (1.0.1), `opencv-python` (4.13.0.92), `nvidia-cudnn-cu12` (9.x line), `nvidia-cuda-runtime-cu12` (12.x line). **HIGH**
- ONNX Runtime official docs — CUDA Execution Provider compatibility + Install pages: default CUDA 12.x since 1.19, cuDNN 8↔9 hard break, `LD_LIBRARY_PATH` requirement on Linux. https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html , https://onnxruntime.ai/docs/install/ — **HIGH**
- microsoft/onnxruntime release notes v1.22.0 — "GPU packages require CUDA 12.x; CUDA 11.x no longer published." https://github.com/microsoft/onnxruntime/releases/tag/v1.22.0 — **HIGH**
- FaceFusion `requirements.txt` (master) — live pins: `onnxruntime==1.24.4`, `opencv-python==4.13.0.92`, `numpy==2.2.1`, `scipy==1.17.1`, `gradio==5.44.1`. https://github.com/facefusion/facefusion — **HIGH**
- Deep-Live-Cam (`hacksider/Deep-Live-Cam`) docs/wiki — real-time webcam architecture, Python 3.10/3.11 + execution-provider selection. https://github.com/hacksider/Deep-Live-Cam — **MEDIUM** (architecture reference; its versions are intentionally not adopted)
- inswapper_128.onnx mirrors (2026): HF `ezioruan/inswapper_128.onnx`, `netrunner-exe/Insight-Swap-models` (fp16), `facefusion/facefusion-assets` releases; InsightFace non-commercial restriction. **MEDIUM** (community-hosted; original DeepInsight link removed)
- buffalo_l auto-download behavior + InsightFace 1.0 (released 2026-05-23, lighter pure-Python install). insightface.ai guides + GitHub. **HIGH**
- **Live machine diagnosis (this host, 2026-05-29):** `nvidia-driver-580-open 580.126.09` installed; running kernel `6.17.0-29-generic`; nvidia module built only for `6.17.0-22-generic`; Secure Boot disabled; PRIME `on-demand`; `/dev/video0` present; Wayland session. — **HIGH (directly observed)**

---
*Stack research for: local real-time webcam face-swap (single-image / InsightFace inswapper)*
*Researched: 2026-05-29*
