# Pitfalls Research

**Domain:** Local real-time webcam face-swap desktop app (InsightFace `inswapper` + onnxruntime-gpu + OpenCV) on a tricky Ubuntu/Wayland NVIDIA-Optimus laptop with a currently-broken NVIDIA driver
**Researched:** 2026-05-29
**Confidence:** HIGH for the two top suspects (broken driver path, silent CPU fallback) and the venv-CUDA library-discovery issue — all confirmed by official docs + multiple GitHub issues. MEDIUM for Wayland/cv2.imshow and Python 3.12 wheel specifics (community-sourced, version-dependent).

> **Reading order for this project:** the pitfalls are ordered by likelihood × impact **for this exact setup**. The first three WILL bite (driver is already down; venv-local CUDA + onnxruntime is a known silent-failure combo). Treat #1–#3 as the critical path of the env-setup phase — nothing downstream matters until GPU inference is *verified*, not just *available*.

---

## Critical Pitfalls

### Pitfall 1: NVIDIA driver not loaded — `nvidia-smi` fails (the current state)

**What goes wrong:**
`nvidia-smi` reports *"NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver."* This is the documented current state of the machine. Until this is fixed, there is **no CUDA at all** — onnxruntime will run CPU-only at ~5–15 fps and the project's core value (smooth real-time) is unreachable. On Optimus laptops this single symptom has **five common root causes**, and you must diagnose *which one* before installing anything.

**Why it happens (root causes, in order of likelihood on this box):**
1. **Secure Boot blocking the unsigned kernel module** (the classic Optimus/Ubuntu blocker). The DKMS-built `nvidia.ko` is unsigned; UEFI Secure Boot refuses to load it. The driver "installs fine" but never loads at boot.
2. **Kernel updated, DKMS didn't rebuild** the module for the new kernel (headers missing or mismatched). Very common after `apt upgrade` pulls a new kernel.
3. **`nouveau` (open-source driver) still loaded**, occupying the GPU so the proprietary module can't bind.
4. **`kernel-headers` mismatch** — `linux-headers-$(uname -r)` not installed, so DKMS can't build at all.
5. **Driver package half-installed / partially removed** (broken dpkg state).

**How to diagnose (run these first, do NOT reinstall blindly):**
```bash
mokutil --sb-state                       # "SecureBoot enabled" => cause #1 is live
lsmod | grep -E 'nvidia|nouveau'         # nouveau present + no nvidia => cause #3
dkms status                              # shows if nvidia module built for current kernel
uname -r                                 # current kernel
dpkg -l | grep -E 'linux-headers'        # headers for that exact kernel present?
sudo dmesg | grep -iE 'nvidia|nvrm|nouveau'   # load errors, "module verification failed"
ubuntu-drivers devices                   # recommended driver for the RTX 3080 Ti Mobile
```
The smoking gun for Secure Boot is `dmesg` showing **"module verification failed: signature and/or required key missing - tainting kernel"** or the module simply absent from `lsmod` while installed on disk.

**How to avoid / fix:**
- Install/repair via the distro path so DKMS + signing are wired up automatically:
  ```bash
  sudo ubuntu-drivers devices           # confirm the recommended driver branch
  sudo ubuntu-drivers autoinstall       # installs proprietary driver + DKMS hooks
  ```
- **If Secure Boot is enabled (cause #1):** during install Ubuntu prompts to create a MOK password. On the **next reboot** the blue MOK Manager screen appears — choose **Enroll MOK -> Continue -> enter the password**. Do NOT just "Continue boot" or the module stays blocked. Verify after reboot:
  ```bash
  mokutil --list-enrolled               # should list the NVIDIA/DKMS key
  ```
  Alternative: disable Secure Boot in UEFI (simpler, acceptable for a personal machine).
- **Blacklist nouveau** if still loaded (the proprietary install normally does this; verify `/etc/modprobe.d/blacklist-nvidia-nouveau.conf` exists). Reboot after.
- **Reboot is mandatory** after any of these — the module loads at boot.

**Warning signs (early detection):**
- `nvidia-smi` still fails after install -> almost always Secure Boot (you skipped MOK enrollment) or you didn't reboot.
- `dmesg` "tainting kernel / signature missing" -> MOK not enrolled.
- `dkms status` shows the module not built for `uname -r` -> headers missing.

**Phase to address:** **Env setup (Step 1, the gate).** Nothing else proceeds until `nvidia-smi` prints the RTX 3080 Ti Mobile.

---

### Pitfall 2: Treating PRIME render-offload env vars as required for CUDA (they are NOT)

**What goes wrong:**
Developers on Optimus laptops assume that because Intel drives the display, they must prefix their app with `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python app.py` to make the dGPU work — and then waste hours when it doesn't change anything, or conclude CUDA is "not using the GPU." Conversely, some never set the display offload and find the **OpenCV preview window** janky.

**Why it happens:**
`__NV_PRIME_RENDER_OFFLOAD` and `prime-select on-demand` govern **graphics rendering** (OpenGL/Vulkan -> which GPU draws windows). **CUDA compute is completely separate** — once the proprietary driver is loaded and `nvidia-smi` works, CUDA addresses the dGPU directly regardless of which GPU drives the display. onnxruntime's `CUDAExecutionProvider` does **not** need any PRIME env var.

**How to avoid:**
- For the **inference** (the whole point of this app): do **nothing** PRIME-related. Just confirm `nvidia-smi` works and that onnxruntime uses CUDA (Pitfall 3).
- Keep PRIME mode on **`on-demand`** (`prime-select on-demand` via `nvidia-prime`) so the dGPU isn't permanently driving the display and draining battery; it wakes for CUDA automatically.
- Only the **cv2 preview window** is GL/Wayland-rendered, and even that doesn't require the offload env vars to *function* — see Pitfall 7.

**Warning signs:**
- You're editing `__NV_PRIME_RENDER_OFFLOAD` to fix inference performance — wrong lever; the bottleneck is elsewhere (Pitfall 3 or 6).
- `nvidia-smi dmon` / `nvidia-smi -l 1` shows **0% GPU util during inference** -> that's a provider/CUDA problem (Pitfall 3), not a PRIME problem.

**Phase to address:** **Env setup** (set `on-demand` once); **explicitly do not** add PRIME vars to the run command.

---

### Pitfall 3: onnxruntime-gpu silently falls back to CPU (cuDNN/CUDA mismatch) — THE #1 functional bug

**What goes wrong:**
`onnxruntime.get_available_providers()` lists `'CUDAExecutionProvider'`, you pass it to `InferenceSession`, the app *runs* — but every frame is computed on the **CPU**. You get 5–15 fps and assume "GPU just isn't fast enough" or "the model is heavy." In reality the CUDA provider failed to load its native `.so` (cuDNN/cuBLAS version mismatch) and onnxruntime **silently dropped it** and fell through to CPU. This is the single most common functional failure for this stack.

**Why it happens:**
- `get_available_providers()` reports what was **compiled in**, not what can actually **load at runtime**.
- onnxruntime-gpu 1.18.1+ and all 1.19.x/1.20.x/1.21.x/1.22.x for CUDA 12 require **cuDNN 9.x**. onnxruntime 1.18.0 and 1.17.x require **cuDNN 8.x**. Installing the wrong cuDNN major -> `libcudnn_*.so.9` (or `.so.8`) not found -> `dlopen` fails -> provider dropped.
- Equivalent failure with cuBLAS: ORT wants `libcublasLt.so.12` but the loader finds `.so.13` (or vice versa) -> drop.
- ORT does **not** raise on this by default; it just removes the provider and uses the next one in the list (CPU).

**How to avoid (pin the matched set, then VERIFY):**
- Choose **one** coherent set. For CUDA 12.x on Python 3.12, a known-good combo is **onnxruntime-gpu 1.20.x or 1.19.x + cuDNN 9.x**. Match the onnxruntime version to its required cuDNN major from the official table (see Sources). Do not mix.
- Install the CUDA/cuDNN runtime **via pip into the venv** (satisfies the "venv-local CUDA, don't touch system" constraint):
  ```bash
  pip install onnxruntime-gpu        # pulls cu12 build
  pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12
  ```
  (or, on ORT >= 1.21, `pip install "onnxruntime-gpu[cuda,cudnn]"` which bundles them.)
- **VERIFY with a real inference + explicit failure check**, not `get_available_providers()`:
  ```python
  import onnxruntime as ort
  so = ort.InferenceSession("inswapper_128.onnx",
                            providers=["CUDAExecutionProvider"])
  assert so.get_providers()[0] == "CUDAExecutionProvider", \
      f"FELL BACK TO CPU: {so.get_providers()}"
  ```
  And raise on load failure so it can't pass silently:
  ```python
  sess_opts = ort.SessionOptions()
  # ORT >= 1.21:
  ort.set_default_logger_severity(1)            # surface the dlopen warnings
  # Best signal: run nvidia-smi -l 1 in another terminal during inference;
  # GPU util must spike. 0% util => CPU fallback regardless of get_providers().
  ```
- On ORT >= 1.21 call `onnxruntime.preload_dlls(cuda=True, cudnn=True)` at import to force-load the pip libs before session creation.

**Warning signs:**
- Console warning: *"Failed to create CUDAExecutionProvider"* / *"libcudnn_adv.so.9: cannot open shared object file"* / *"libonnxruntime_providers_cuda.so ... cannot open"*.
- `get_providers()` after creating the session returns `['CPUExecutionProvider']` (or CUDA is absent from it).
- `nvidia-smi -l 1` shows 0% GPU during the swap loop.
- fps stuck at single digits despite a working `nvidia-smi`.

**Phase to address:** **Env setup** (pin versions) **+ pipeline** (the assert-on-fallback check must be in the code, not a one-time manual test).

---

### Pitfall 4: venv-local CUDA libs not on the loader path (onnxruntime can't find pip's nvidia-* libs)

**What goes wrong:**
You installed `nvidia-cudnn-cu12` into the venv (correctly honoring the "don't touch system" constraint), but onnxruntime still can't load CUDA: *"libcudnn.so.9: cannot open shared object file: No such file or directory."* Even though the file physically exists under `venv/lib/python3.12/site-packages/nvidia/cudnn/lib/`.

**Why it happens:**
This is a **documented onnxruntime behavior**: unlike PyTorch, **onnxruntime does NOT patch `LD_LIBRARY_PATH` at runtime** to include the pip-installed `nvidia/*/lib` directories (GitHub onnxruntime #25609). The dynamic linker only searches system paths + `LD_LIBRARY_PATH`, so the venv's CUDA libs are invisible to `dlopen`. Result: silent CPU fallback (Pitfall 3 again, but from a *path* cause not a *version* cause).

**How to avoid (pick ONE, in preference order):**
1. **ORT >= 1.21:** call `onnxruntime.preload_dlls(cuda=True, cudnn=True)` before creating the session — it explicitly loads the libs from site-packages. Cleanest, no env vars.
2. **Set `LD_LIBRARY_PATH` from a launcher** so the constraint stays venv-local (no system changes):
   ```bash
   # in a run.sh wrapper or activate hook
   SITE=$(python -c "import site;print(site.getsitepackages()[0])")
   export LD_LIBRARY_PATH="$SITE/nvidia/cudnn/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
   ```
3. **In-process before importing onnxruntime** (most self-contained, survives without a wrapper):
   ```python
   import os, site, glob
   for sub in ("cudnn","cublas","cuda_runtime"):
       for p in glob.glob(os.path.join(site.getsitepackages()[0],"nvidia",sub,"lib")):
           os.add_dll_directory(p) if os.name=="nt" else None
   # On Linux, add_dll_directory doesn't apply; set LD_LIBRARY_PATH before python starts
   # OR use ctypes.CDLL to preload each .so explicitly.
   ```
   On Linux the reliable in-Python option is `ctypes.CDLL(".../libcudnn.so.9", mode=ctypes.RTLD_GLOBAL)` for each lib *before* `import onnxruntime`.

> **Do NOT** "fix" this with `sudo ldconfig` / writing to `/etc/ld.so.conf.d/` / installing system CUDA — that violates the hard constraint to keep everything venv-local. The launcher/`preload_dlls` route keeps it isolated.

**Warning signs:**
- The file exists (`find venv -name 'libcudnn.so.9'` succeeds) but ORT still says "cannot open shared object file" -> 100% a path problem, not a missing-package problem.
- `ldd` on `libonnxruntime_providers_cuda.so` shows `=> not found` for cudnn/cublas.

**Phase to address:** **Env setup** — bake the launcher / `preload_dlls` into the project entrypoint so it works on a clean machine, not just yours.

---

### Pitfall 5: inswapper_128.onnx acquisition — pulled from auto-download, license, and hash verification

**What goes wrong:**
The code calls `insightface.model_zoo.get_model('inswapper_128.onnx')` (or roop/facefusion-style auto-download) and it **fails / hangs / 404s**. InsightFace **removed the swapper models from automatic download** because they are **non-commercial research-only** licensed; the official auto-download no longer serves `inswapper_128.onnx`. Separately, some mirrored copies are corrupt or are different builds, leading to *"Protobuf parsing failed"* on load.

**Why it happens:**
- InsightFace's policy: training data and the trained models (incl. `inswapper_128.onnx`) are **for non-commercial research purposes only**; commercial use requires contacting insightface.ai. To enforce this they stopped hosting it for auto-download. The `buffalo_l` detection/recognition pack (the `FaceAnalysis` bundle) **still auto-downloads fine** — only the *swapper* must be obtained manually.
- Community mirrors exist on Hugging Face but vary in integrity/version.

**How to avoid:**
- **Download `inswapper_128.onnx` manually, once, offline-cache it** in a project models dir (matches the "models downloaded once, then offline" constraint). Known community mirrors: `ezioruan/inswapper_128.onnx`, `ApacheOne/insightface`, `Aitrepreneur/insightface` on Hugging Face.
- **Verify the SHA256 after download** — two widely-circulated hashes seen in the wild:
  `e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af` (the common ~554 MB build referenced by roop/facefusion lineage). Pin whichever you choose and assert it:
  ```bash
  sha256sum inswapper_128.onnx
  ```
- Let `FaceAnalysis(name='buffalo_l')` auto-download the **detector/recognizer** on first run (it still works), then point the swapper at your local file:
  ```python
  app = FaceAnalysis(name='buffalo_l')         # auto-downloads buffalo_l, fine
  swapper = insightface.model_zoo.get_model('models/inswapper_128.onnx',
                                             providers=[...])   # local path
  ```
- **Respect the license:** this project is explicitly personal/local/non-published (per PROJECT.md), which fits the non-commercial research terms. Document this so scope doesn't later drift into anything commercial.

**Warning signs:**
- Auto-download 404 / "model file not found" for the swapper specifically while buffalo_l downloads OK.
- `onnx` "Protobuf parsing failed" / size far from ~554 MB -> corrupt or wrong mirror; re-download and re-hash.

**Phase to address:** **Env setup / asset acquisition** — make model download + hash check a one-time scripted step with a pinned expected hash.

---

### Pitfall 6: Real-time latency accumulation — unbounded queues, per-frame detection, oversized inputs

**What goes wrong:**
The swap "works" but feels laggy and drifts further behind the longer it runs: you wave your hand and the displayed swap reacts a second later. Or fps is mediocre even on GPU. This is the difference between "technically works" and the project's actual success criterion ("плавно", 25–30+ fps).

**Why it happens (each is a separate, additive trap):**
1. **Unbounded frame queue** between the capture thread and the inference thread. The camera produces 30 fps; if inference is slower, frames pile up and you display ever-older frames -> growing latency. Classic producer/consumer mistake.
2. **Running face *detection* every frame.** `app.get()` (RetinaFace detect) is the expensive part. Detecting on every single frame doubles cost vs. detecting periodically and reusing/tracking the bbox.
3. **`det_size` too large.** `FaceAnalysis(..., det_size=(640,640))` is the default; dropping to `(320,320)` roughly quarters detection cost with little quality loss for a single close-up webcam face.
4. **Full-resolution capture.** 1080p frames mean huge CPU↔GPU transfers and detection area. Capture/process at 640×480 or 720p, upscale only for display if wanted.
5. **CPU↔GPU transfer overhead** per frame (numpy -> GPU -> numpy) — unavoidable per frame, but worsened by large frames (#4) and by re-creating sessions.
6. **Single-threaded loop** doing capture + detect + swap + imshow serially -> the UI freezes and fps = 1/(sum of all stages).

**How to avoid:**
- **Bounded queue of size 1–2, drop-oldest.** Capture thread overwrites the latest frame; inference always works on the freshest frame. This caps latency instead of letting it grow.
- **Decouple threads:** capture thread, inference thread, display on main thread (GUI toolkits require main-thread UI; cv2 `imshow`/`waitKey` must be on the main thread).
- **Detect every N frames** (e.g., re-detect every 5–10 frames or on a timer) and reuse the last face bbox/embedding between detections; the *swap* (inswapper) can still run every frame on the cached target.
- **`det_size=(320,320)`** and **capture at 640×480** as the starting point; raise only if quality demands it.
- **Create `FaceAnalysis` and the swapper once**, reuse across frames — never per-frame.
- The target-face embedding from the uploaded photo is computed **once** when the photo is loaded, not per frame.

**Warning signs:**
- Latency visibly *grows* over 10–30 s of runtime -> unbounded queue (#1).
- GPU util high but fps low and laggy -> per-frame detection (#2) or oversized det_size/resolution (#3/#4).
- UI freezes / `waitKey` unresponsive -> single-threaded blocking (#6).

**Phase to address:** **Pipeline + UI.** Bake bounded-queue + threading into the architecture from the start — retrofitting threading into a serial loop is a rewrite.

---

### Pitfall 7: Wayland — cv2.imshow window not appearing / GTK backend errors

**What goes wrong:**
On GNOME/Wayland (this machine), `cv2.imshow()` either shows nothing, hangs, throws *"Can't initialize GTK backend"*, or — with the newer experimental Wayland highgui backend — fails to create the window unless you call `namedWindow()` first (OpenCV issue #25497). The preview window is the only thing the user actually sees, so this silently kills the deliverable.

**Why it happens:**
- `pip install opencv-python` ships a build whose highgui talks to **GTK/X11**, not Wayland natively. Under GNOME-Wayland it relies on **XWayland**; if XWayland isn't available or the build lacks GTK, the window fails.
- OpenCV's native Wayland backend is newer and has the `namedWindow()`-required quirk and other rough edges.

**How to avoid:**
- Simplest reliable path: let it run through **XWayland**. `opencv-python` wheels generally work via XWayland on GNOME with no changes. If the window misbehaves, force a known platform:
  ```bash
  # if using a Qt-based opencv build:
  export QT_QPA_PLATFORM=xcb        # force XWayland instead of native wayland
  ```
- Always call `cv2.namedWindow("swap", cv2.WINDOW_NORMAL)` **before** the first `imshow` (works around #25497 and is harmless on X11).
- Keep `imshow` + `waitKey(1)` on the **main thread** (see Pitfall 6 #6).
- **Consider sidestepping cv2 highgui entirely**: render the frame in a real GUI toolkit (PyQt/PySide or GTK via the toolkit's image widget). The PROJECT requires start/stop and live photo-target swapping from the UI anyway, so a proper GUI is likely needed regardless — and it dodges the cv2/Wayland window mess.
- Note: this is a **camera** app, not screen-capture — no PipeWire/portal screencast permissions needed. Camera access on Linux is plain V4L2 (`/dev/video*`), not gated by Wayland portals.

**Warning signs:**
- `imshow` returns but no window; or "GTK backend" / "The function is not implemented. Rebuild the library with Window support" error.
- Window appears but is frozen / never repaints -> `waitKey` not called or not on main thread.

**Phase to address:** **UI.** Decide GUI toolkit vs cv2-highgui early; if cv2-highgui, verify a window actually paints on this Wayland session before building the pipeline around it.

---

### Pitfall 8: Python 3.12 — insightface/onnx wheel availability and build-from-source pain

**What goes wrong:**
`pip install insightface` on Python 3.12 tries to **build from source** and fails: *"ERROR: Could not build wheels for insightface"* — needs a C/C++ compiler, Cython, and a matching numpy, and the build breaks on toolchain/numpy ABI mismatches. Burns time at the very start of env setup.

**Why it happens:**
- insightface ships limited prebuilt wheels; on newer Pythons pip often has to compile its Cython extensions. Missing `build-essential`/`python3-dev`, an incompatible numpy, or PEP 517 build isolation pulling the wrong numpy all cause failures.
- Some downstream guidance pins **onnx 1.18.0 + insightface 0.7.3** as a known-good pairing for Python 3.12 stacks.

**How to avoid:**
- Be ready to provide the build toolchain (system-level, but build tools only — not CUDA): `build-essential`, `python3.12-dev`. If the constraint forbids even that, prefer a Python where wheels exist.
- **Pin known-good versions** rather than latest: e.g., `insightface==0.7.3`, a compatible `onnx`, `numpy<2` if ABI issues appear, and the matched `onnxruntime-gpu` from Pitfall 3. Install numpy first so the build picks it up.
- Use `--no-build-isolation` only if you've pre-installed Cython+numpy in the venv and isolation is fighting you.
- **Pragmatic fallback honoring the constraints:** if 3.12 wheels keep fighting, create the **project-local venv on Python 3.10 or 3.11** (where insightface/onnxruntime wheels are most plentiful). The constraint is "don't touch the global/shared venv" — using a different *interpreter version inside the isolated project venv* fully satisfies that and removes a whole class of build pain. System Python stays 3.12.3, untouched.

**Warning signs:**
- pip log shows "Building wheel for insightface (pyproject.toml) ... error" with gcc/Cython/numpy errors.
- `ImportError` about numpy ABI / `numpy.dtype size changed` at runtime -> numpy major mismatch with the compiled extension.

**Phase to address:** **Env setup** — decide interpreter version and pin the insightface/onnx/numpy set before anything else; it gates the whole stack.

---

### Pitfall 9: Wrong webcam device index — `/dev/video0` is metadata, real stream is elsewhere

**What goes wrong:**
`cv2.VideoCapture(0)` opens but returns no frames / `select() timeout` / black frames, even though the webcam works in other apps. PROJECT.md notes the camera shows up as `/dev/video0` plus `video1..3`.

**Why it happens:**
- Modern UVC webcams expose **multiple `/dev/video*` nodes per physical camera**: one (or more) is the actual capture stream, the others are **metadata-only** nodes (no capturable frames). With newer kernels the *first* node (`video0`) is sometimes the metadata node and the capturable stream is `video1`. Index ≠ physical camera count.
- A crashed/not-released previous open can also bump the camera to a different node.

**How to avoid:**
- Enumerate and check capabilities instead of guessing index 0:
  ```bash
  v4l2-ctl --list-devices
  v4l2-ctl -d /dev/video0 --all   # look for "Video Capture" cap; metadata nodes lack it
  ```
  The capturable node reports `Video Capture` in its capabilities; metadata nodes report `Metadata Capture`.
- In code, try indices until one yields a frame, and prefer the one whose first `read()` returns `ret=True`:
  ```python
  for idx in range(4):
      cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
      ok, _ = cap.read()
      if ok: break
      cap.release()
  ```
- Pass `cv2.CAP_V4L2` explicitly so OpenCV uses the V4L2 backend deterministically.

**Warning signs:**
- `VIDEOIO(V4L2:/dev/video0): select() timeout` or `read()` returns `ret=False` immediately while another app shows live video -> wrong node.
- "Device or resource busy" -> camera already in use (browser/Zoom/another instance) — see below.

**Phase to address:** **Pipeline (capture).** Probe-and-pick at startup; don't hardcode `0`.

---

### Pitfall 10: Webcam busy / held by another process

**What goes wrong:**
`VideoCapture` fails with *"Device or resource busy"* because a browser tab, video-call app, or a previous run of this app still holds `/dev/video0`. Most webcams allow only one consumer at a time.

**Why it happens:**
V4L2 capture devices are typically exclusive; a leaked handle from a crashed run keeps the node open.

**How to avoid:**
- Detect and report it clearly at startup instead of a cryptic OpenCV error.
- Find the holder:
  ```bash
  sudo fuser -v /dev/video0     # or: lsof /dev/video0
  ```
- Always `cap.release()` (and `cv2.destroyAllWindows()`) on exit/stop, including on exceptions (`try/finally`), so a crash doesn't leak the device.

**Warning signs:** "Device or resource busy"; camera works after closing the browser/restarting.

**Phase to address:** **Pipeline (capture lifecycle) + UI** (clean stop must release the device).

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Trust `get_available_providers()` instead of asserting on real inference | "GPU is available, ship it" | Silent CPU fallback ships; user blames "slow GPU"; hours of mis-diagnosis | **Never** — the assert-on-fallback check is cheap and mandatory |
| `cv2.VideoCapture(0)` hardcoded | One line, works on your box | Breaks on metadata-node-first kernels / reorders | Only a throwaway spike; production must probe |
| Single-threaded serial loop (capture->detect->swap->show) | Simplest to write | Latency grows, UI freezes, retrofitting threads = rewrite | OK for a first "does the swap work at all" smoke test, then refactor before tuning fps |
| `sudo ldconfig` / system CUDA to fix lib paths | Quick fix for "libcudnn not found" | **Violates the venv-local constraint**; pollutes the machine; non-reproducible | Never on this project — use `preload_dlls`/launcher `LD_LIBRARY_PATH` |
| Detect face every frame | Simplest, always-fresh bbox | Halves achievable fps | Acceptable only if fps target already met with margin |
| `pip install insightface` latest, unpinned | Less thinking now | Py3.12 build break / numpy ABI surprises later | Never — pin the known-good set |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| NVIDIA driver (Optimus) | Reinstalling driver blindly when `nvidia-smi` fails | Diagnose first (`mokutil --sb-state`, `dkms status`, `dmesg`); usually it's Secure-Boot/MOK, not a missing driver |
| onnxruntime-gpu + cuDNN | Installing latest cuDNN regardless of ORT version | Match ORT version to its required cuDNN major (ORT 1.18.1+/1.19/1.20 -> cuDNN 9; ORT 1.18.0/1.17 -> cuDNN 8) |
| pip nvidia-* libs in venv | Assuming ORT finds them like PyTorch does | ORT does NOT patch LD_LIBRARY_PATH; use `preload_dlls` (>=1.21) or set LD_LIBRARY_PATH to site-packages/nvidia/*/lib |
| insightface model zoo | Relying on auto-download for `inswapper_128.onnx` | Download manually (non-commercial-research, removed from auto-download), verify SHA256, cache locally; buffalo_l still auto-downloads |
| OpenCV highgui on Wayland | Expecting native Wayland window | Use XWayland (default wheels) / `QT_QPA_PLATFORM=xcb` / `namedWindow()` first, or use a real GUI toolkit |
| V4L2 camera | Assuming `/dev/video0` = the stream | Query caps; the capturable node has "Video Capture", metadata nodes don't |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Unbounded frame queue | Latency grows over runtime; displayed frame lags input by seconds | Bounded queue size 1, drop-oldest; always process newest frame | Immediately, whenever inference < camera fps (i.e., always at 30fps target) |
| Per-frame detection | High GPU util, low fps | Detect every N frames, reuse bbox; swap every frame | At any real-time target on a single face |
| `det_size=(640,640)` + 1080p capture | fps ~half of achievable; big CPU<->GPU transfers | `det_size=(320,320)`, capture 640x480 | Real-time; fine for offline single images |
| Re-creating InferenceSession/FaceAnalysis per frame | Stutter, GPU memory churn | Build once, reuse | Immediately |
| Display + inference on one thread | UI freezes, waitKey unresponsive | Capture thread + inference thread; UI on main thread | As soon as inference is non-trivial |

## Security / Privacy Mistakes

(Personal-use, fully-local app — most web-security concerns are N/A. Domain-specific items:)

| Mistake | Risk | Prevention |
|---------|------|------------|
| Treating inswapper as freely usable | License violation — it's **non-commercial research only** | Keep usage personal/local/non-published (already the project's stated scope); document it |
| Leaking webcam frames off-machine (telemetry, crash uploaders, cloud model fetch at runtime) | Privacy breach of live face video | Models cached locally once; run fully offline thereafter (matches PROJECT constraint); no network in the frame loop |
| Saving target photos / frames to disk without intent | Sensitive biometric data lingering | Keep target photo in memory or a clearly-scoped temp; don't write frames to disk (project explicitly does not record) |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| No clear CPU-fallback indicator | User sees 8 fps, thinks app is broken, can't tell GPU failed | Detect provider at startup; show "GPU (CUDA)" vs "CPU fallback" badge + the fps |
| No "camera busy / not found" message | Cryptic OpenCV error, user stuck | Friendly startup error: "Camera in use or not found — close other apps using the webcam" |
| Changing target photo requires restart | Friction; PROJECT explicitly wants live swap of target | Recompute only the target embedding on photo change; keep pipeline running |
| Window never appears (Wayland) | App "does nothing" | Verify a window paints on this exact session before shipping the loop; pick GUI toolkit if cv2 highgui is flaky |

## "Looks Done But Isn't" Checklist

- [ ] **GPU inference:** `get_providers()` says CUDA AND `nvidia-smi -l 1` shows util spike during the loop AND an assert fires if it ever falls back — verify all three, not just the first.
- [ ] **venv isolation:** confirm CUDA/cuDNN load from `venv/.../site-packages/nvidia/*/lib` (via `preload_dlls`/launcher), with **zero** system CUDA installed — verify `ldd` on the cuda provider .so resolves to venv paths.
- [ ] **Model integrity:** `inswapper_128.onnx` SHA256 matches the pinned value, file loads without Protobuf error — verify hash on a fresh download.
- [ ] **Latency stability:** run the swap 60 s; latency should stay flat, not grow — verify with a moving hand / clap-to-react test.
- [ ] **Camera robustness:** unplug/replug or open browser camera, restart app — verify it picks the right node and reports "busy" cleanly.
- [ ] **Wayland window:** verify the preview actually paints on this GNOME-Wayland session (not just "imshow returned").
- [ ] **Driver persistence:** reboot once after driver fix — verify `nvidia-smi` still works (MOK enrolled, DKMS survives) before building on top.
- [ ] **Live target change:** swap the source photo from the UI without restart — verify embedding updates and stream keeps running.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| `nvidia-smi` still failing | MEDIUM | `mokutil --sb-state`; if enabled, re-run install and **enroll MOK at reboot** (or disable Secure Boot in UEFI); verify `dkms status` + `linux-headers-$(uname -r)`; reboot |
| Silent CPU fallback | LOW–MEDIUM | Read the ORT warning (set logger severity 1); align cuDNN major to ORT version; ensure venv nvidia libs on loader path (`preload_dlls`/LD_LIBRARY_PATH); re-assert |
| venv CUDA libs not found | LOW | Add `site-packages/nvidia/*/lib` to LD_LIBRARY_PATH in launcher, or `onnxruntime.preload_dlls()` — no system changes |
| inswapper download fails | LOW | Manual download from a HF mirror; verify SHA256; place in local models dir; point swapper at the path |
| insightface won't build on 3.12 | LOW–MEDIUM | Pin `insightface==0.7.3` + compatible onnx/numpy; install build-essential/python3-dev; or recreate project venv on Python 3.10/3.11 |
| Wayland window dead | LOW | `QT_QPA_PLATFORM=xcb`, call `namedWindow()` first, or switch preview to a GUI toolkit widget |
| Latency creep | LOW | Replace queue with size-1 drop-oldest; move detection off the per-frame path |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1. NVIDIA driver not loaded (Secure Boot/DKMS/nouveau) | Env setup (gate) | `nvidia-smi` shows RTX 3080 Ti Mobile after reboot |
| 2. PRIME-vars-for-CUDA confusion | Env setup | Inference uses GPU with no PRIME env vars; `prime-select on-demand` set |
| 3. Silent CPU fallback (cuDNN/CUDA mismatch) | Env setup + pipeline | `get_providers()[0]==CUDA` AND nvidia-smi util spike AND assert in code |
| 4. venv CUDA libs not on loader path | Env setup | `ldd` on cuda provider .so resolves to venv nvidia/*/lib; `preload_dlls` succeeds |
| 5. inswapper acquisition + license + hash | Env setup / assets | SHA256 matches pinned value; model loads; buffalo_l auto-downloads |
| 6. Real-time latency traps | Pipeline + UI | 60 s run: flat latency, 25–30+ fps at 640x480/det 320 |
| 7. Wayland cv2.imshow window | UI | Preview paints on this GNOME-Wayland session |
| 8. Python 3.12 wheel/build pain | Env setup | insightface + onnxruntime import cleanly; pinned versions recorded |
| 9. Wrong webcam node index | Pipeline (capture) | Startup probe selects a node that returns frames |
| 10. Webcam busy/held | Pipeline + UI | Clean "busy" message; device released on stop/crash |

## Sources

- ONNX Runtime — CUDA Execution Provider (official compatibility matrix, `preload_dlls`): https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html — HIGH
- ONNX Runtime — Install guide: https://onnxruntime.ai/docs/install/ — HIGH
- "The Hidden Pitfalls of ONNXRuntime GPU Setup" (detection of silent fallback, exact versions): https://dev.to/deskpai/the-hidden-pitfalls-of-onnxruntime-gpu-setup-4kb7 — MEDIUM
- onnxruntime #25609 — CUDAEP fails: missing libcudnn.so.9 in LD_LIBRARY_PATH (ORT does not patch loader path): https://github.com/microsoft/onnxruntime/issues/25609 — HIGH
- onnxruntime discussion #22122 — why ORT 1.19.2 requires libcudnn.so.9: https://github.com/microsoft/onnxruntime/discussions/22122 — HIGH
- onnxruntime #21769 / #21825 — CUDA not working with 1.19 / cuDNN 9 RTX: https://github.com/microsoft/onnxruntime/issues/21769 , https://github.com/microsoft/onnxruntime/issues/21825 — MEDIUM
- NVIDIA Developer Forums — drivers not working with Secure Boot (Ubuntu): https://forums.developer.nvidia.com/t/nvidia-drivers-not-working-while-secure-boot-is-enabled-after-updating-to-ubuntu-24-04/305351 — HIGH
- "NVIDIA Drivers with Secure Boot on Ubuntu" (MOK enrollment): https://dev.to/gordinmitya/nvidia-drivers-with-secure-boot-on-ubuntu-59h4 — MEDIUM
- NVIDIA PRIME Render Offload README (render-offload vs CUDA distinction): https://download.nvidia.com/XFree86/Linux-x86_64/440.31/README/primerenderoffload.html — HIGH
- Arch Wiki — PRIME (on-demand, offload): https://wiki.archlinux.org/title/PRIME — MEDIUM
- insightface README + PyPI (non-commercial research license; buffalo_l): https://github.com/deepinsight/insightface/blob/master/README.md , https://pypi.org/project/insightface/ — HIGH
- insightface #2335 / #2294 — inswapper_128.onnx download removed from auto-download, mirrors: https://github.com/deepinsight/insightface/issues/2335 , https://github.com/deepinsight/insightface/issues/2294 — MEDIUM
- Hugging Face inswapper mirrors (hash references): https://huggingface.co/ezioruan/inswapper_128.onnx — MEDIUM
- insightface #2430 / ComfyUI #6826 / ReActor #166 — Python 3.12 build-from-source failures, pinning onnx 1.18.0 + insightface 0.7.3: https://github.com/deepinsight/insightface/issues/2430 , https://github.com/comfyanonymous/ComfyUI/issues/6826 , https://github.com/Gourieff/ComfyUI-ReActor/issues/166 — MEDIUM
- OpenCV #25497 — Wayland highgui: imshow needs namedWindow first: https://github.com/opencv/opencv/issues/25497 — HIGH
- OpenCV forum — GUI hangs on imshow with Wayland backend (Ubuntu 22.04): https://forum.opencv.org/t/gui-not-showing-hangs-on-imshow-with-wayland-backend-ubuntu-22-04/17287 — MEDIUM
- Linux kernel docs — VIDIOC_QUERYCAP (capture vs metadata nodes): https://www.kernel.org/doc/html/v4.9/media/uapi/v4l/vidioc-querycap.html — HIGH
- OpenCV forum — V4L2 select() timeout (wrong node symptom): https://forum.opencv.org/t/videoio-v4l2-dev-video0-select-timeout/8822 — MEDIUM
- Ubuntu/Optimus driver install (ubuntu-drivers autoinstall, nvidia-prime, nouveau blacklist): https://gist.github.com/a6ir/9496308804c71e382f91c92146013c1d — MEDIUM

---
*Pitfalls research for: local real-time webcam face-swap on Ubuntu/Wayland NVIDIA-Optimus laptop*
*Researched: 2026-05-29*
