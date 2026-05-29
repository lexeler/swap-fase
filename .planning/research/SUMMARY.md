# Project Research Summary

**Project:** Swap-Fase
**Domain:** Local real-time webcam face-swap desktop app (single-image / InsightFace `inswapper`)
**Researched:** 2026-05-29
**Confidence:** HIGH

## Executive Summary

Swap-Fase is a single-user, fully-local desktop app that swaps the user's live webcam face onto a single loaded photo and shows it in a window — "open app → webcam → my face becomes the photo's face, smoothly." Experts build exactly this with a settled stack: **Python 3.12 in an isolated venv, InsightFace `inswapper_128` for the swap + `buffalo_l` for detection/embedding, onnxruntime-gpu on the CUDA Execution Provider, OpenCV for V4L2 capture, and PySide6 (Qt6) for the window.** Reference projects (Deep-Live-Cam for the live-cam loop, FaceFusion for the modern version pins) converge on this exact pipeline. The defining architectural move is a **three-thread producer/consumer pipeline (capture / inference / UI) with a maxsize-1 keep-newest frame buffer** — fresh frames win, stale frames drop, so latency stays pinned to roughly one inference time instead of drifting seconds behind. The source photo's face embedding is computed **once** at load and reused every frame; the swapper and analyser are built once, never per-frame.

**The headline risk is not the AI — it is the GPU, and it is already broken in a known, fixable way.** `nvidia-smi` currently fails with "couldn't communicate with the NVIDIA driver," but the driver is NOT missing or broken hardware: `nvidia-driver-580-open 580.126.09` is installed, Secure Boot is disabled, and the only problem is that the `nvidia.ko` kernel module was built for kernel `6.17.0-22-generic` while the machine is booted into `6.17.0-29-generic` (kernel upgraded, module never rebuilt). The fix is a one-time, system-level `dkms autoinstall` + `modprobe` (verify live without reboot first; reboot into the matching kernel as fallback). **This is the first internal step and a hard GPU-verification gate: nothing in the pipeline matters until `nvidia-smi` shows the RTX 3080 Ti AND a warm-up inference proves `CUDAExecutionProvider` actually bound.** Driver aside, the second-biggest risk is the project's signature failure mode: onnxruntime **silently falls back to CPU** (~5–15 fps) when CUDA/cuDNN can't load, while `get_available_providers()` happily still lists CUDA. Detection requires a real warm-up inference + a `get_providers()[0]=="CUDAExecutionProvider"` assert + an `nvidia-smi` utilization check — not just availability.

The recommended approach is a **single-phase MVP** with a tightly ordered internal build sequence (driver/venv/provider-probe → models → FaceEngine on static images → capture+buffer → live smoke test → PySide6 UI → degradation polish), front-loading the highest-risk item and reaching an end-to-end "my face is swapped, live, smooth" smoke test as fast as possible. CUDA goes **into the venv via pip** (`onnxruntime-gpu[cuda,cudnn]==1.22.0`, which pulls CUDA 12.x + cuDNN 9.x wheels), the NVIDIA *driver* stays system-level (it cannot live in a venv), and the virtual-camera output is explicitly the next milestone — present only as a clean `FrameSink` seam, not built now.

## Key Findings

### Recommended Stack

The stack is fully pinned and verified against live PyPI metadata + ONNX Runtime official docs (see [STACK.md](STACK.md)). CUDA/cuDNN ship **inside the venv via pip extras** — no system CUDA toolkit — while the NVIDIA kernel driver remains a system-level component (it lives in the kernel + `/usr/lib` and physically cannot be venv-local; the venv reaches the GPU through the system driver's `libcuda.so` / `/dev/nvidia*`). Driver 580.x supports CUDA 12.x/13.x, comfortably covering the venv's CUDA-12.x runtime.

**Core technologies (exact pins):**
- **insightface `1.0.1`** — detection (`buffalo_l`, auto-downloads) + swap (`inswapper_128`, supplied manually); 1.0.x is pure-Python with no C++ build toolchain, removing the historic install pain.
- **onnxruntime-gpu `1.22.0`** installed as `onnxruntime-gpu[cuda,cudnn]` — GPU inference; the `[cuda]`/`[cudnn]` extras pull a clean pinned `nvidia-*-cu12 ~=12.x` + `nvidia-cudnn-cu12 ~=9.x` set.
- **opencv-python `4.13.0.92`** — V4L2 webcam capture (`/dev/video0`), BGR frame handling (matches FaceFusion's pin).
- **PySide6 `6.8.x`** (Qt6) — the app window; Wayland-native. **NOT `cv2.imshow`**, which is flaky under GNOME/Wayland via XWayland.
- **numpy `2.2.1`** — array math; numpy-2-compatible across the stack.
- Env manager: **uv** (project-local `.venv`, fast/reliable resolution of the heavy `nvidia-*-cu12` wheels); plain `venv`+pip is an acceptable fallback. **Avoid conda** (tempts conda-channel CUDA that conflicts with pip CUDA).

**The make-or-break compatibility chain:** onnxruntime-gpu ≥1.19 defaults to **CUDA 12.x** and **hard-requires cuDNN 9.x** (cuDNN 8 silently fails to load the CUDA EP → CPU fallback). Do not follow old roop/inswapper CUDA-11.8 / cuDNN-8 guides. `inswapper_128.onnx` is NOT auto-downloaded (InsightFace pulled it; non-commercial-research license) — acquire from a community mirror (HF `ezioruan/inswapper_128.onnx`, FaceFusion assets), **verify SHA256**, cache locally. Personal-only use fits the license.

**Resolved tension — venv CUDA library discovery (STACK/ARCH vs PITFALLS):** STACK.md and ARCHITECTURE.md state ORT ≥1.21 auto-loads the pip `nvidia-*` libs from site-packages (via `preload_dlls`), so `LD_LIBRARY_PATH` is usually unnecessary. PITFALLS.md cites onnxruntime issue #25609 — ORT does NOT patch `LD_LIBRARY_PATH` and can fail with "libcudnn.so.9: cannot open shared object file" even though the file exists in the venv. **Practical resolution (do not leave ambiguous):** use **ORT 1.22 and rely on its `onnxruntime.preload_dlls(cuda=True, cudnn=True)` auto-preload as the primary path**, but make the project robust regardless by (1) calling `preload_dlls` before session creation, (2) including the runtime provider-verification check that asserts CUDA actually bound, and (3) shipping an explicit **`LD_LIBRARY_PATH` fallback** in the launcher/entrypoint that points at `site-packages/nvidia/{cudnn,cublas,cuda_runtime}/lib`. Never "fix" this with system `ldconfig` / system CUDA — that violates the venv-local constraint.

### Expected Features

This MVP is the *minimal* slice of what Deep-Live-Cam/FaceFusion do — single photo, swap my own face live in a window, one phase (see [FEATURES.md](FEATURES.md)). Most of the reference apps' surface is deliberately out of scope.

**Must have (table stakes — fit in this one phase):**
- Load one target photo → detect → pick largest face → **cache source embedding once** (the hinge; reused every frame).
- Webcam capture from `/dev/video0` (V4L2; device index configurable even if defaulted).
- Per-frame detect → `inswapper` swap → paste back.
- Live display window with **Start/Stop**.
- **Change target photo without restart** (re-extract embedding under lock; loop keeps running).
- **CUDA execution with automatic CPU fallback + surface the active provider** (GPU vs CPU badge).
- **No-face-this-frame passthrough** (never freeze/crash when a face leaves frame).
- **On-screen FPS counter** (cheap; the primary signal for whether the GPU path is live).
- **Mirror/flip toggle + swap on/off toggle** (cheap UX wins).

**Should have (perf knobs — add in-phase only if fps falls short of 25–30):**
- Reduced-resolution / `det_size` (640→320) tuning — biggest fps lever after GPU.
- Detect-every-N-frames + bbox reuse — if detection is the bottleneck.
- Camera device selector UI — if the default node is wrong.
- Multi-face-in-frame policy (default: swap largest only).

**Defer (explicit later milestones / anti-features):**
- **Virtual camera output (v4l2loopback)** — the declared NEXT milestone; module already loaded. Build only the `FrameSink` seam now.
- **Face restoration (GFPGAN/CodeFormer)** — OFF/deferred; roughly halves fps, so it cannot be on-by-default against the smooth-real-time target.
- Multi-target face-mapping UI, recording/saving, streaming/OBS, web/Gradio UI, model training (DFM), mobile — all out of scope per PROJECT.md.

### Architecture Approach

Single-process, multi-threaded desktop app built around a **three-thread producer/consumer pipeline with a maxsize-1 keep-newest buffer** — the entire latency strategy (see [ARCHITECTURE.md](ARCHITECTURE.md)). Capture fills a 1-slot buffer (newest wins, stale drops), an inference thread does the GPU work, and the UI thread only converts BGR→QImage and paints via a thread-safe Qt queued signal. Bootstrap runs to completion *before* any thread starts: venv guard → provider probe (warm-up inference, not just availability) → model presence + hash check. Build it under `src/swapfase/` with `bootstrap.py`, `providers.py`, `engine.py` (FaceEngine), `capture.py`, `framebuffer.py`, `pipeline.py`, `sink.py`, `state.py`, `ui/`, wired in an `app.py` composition root.

**Major components:**
1. **Bootstrap / providers** — verify CUDA actually works + download/verify models once; fail loud and early.
2. **FaceEngine** — owns `FaceAnalysis(buffalo_l)` + `inswapper_128`; `embed(photo)→target_face` and `process(frame, target)→frame`.
3. **Capture thread + LatestFrameBuffer** — `cv2.VideoCapture(idx, CAP_V4L2)` pushing only the newest frame into a maxsize-1 keep-newest buffer.
4. **Pipeline / inference thread** — read-latest → detect → swap-with-cached-target → write to sink; tracks FPS.
5. **FrameSink (interface)** — `write(frame)`; one impl now (`PreviewSink` → Qt signal). **The future virtual-camera seam** (V4l2Sink/TeeSink are additive later, no pipeline change).
6. **PySide6 UI + AppState** — QLabel preview, controls, GPU/CPU + FPS badge; shared lock-guarded state (target embedding, running flag, device).

**Suggested internal build order (the single phase):** (1) driver fix + venv + provider-probe (highest risk first; smoke test: prints active provider = CUDA), (2) model download + hash verify (offline thereafter), (3) FaceEngine on two static JPEGs, (4) capture thread + keep-newest buffer (stable fps, no backlog), (5) pipeline wiring = the live end-to-end smoke test (Core Value), (6) PySide6 UI build-out + change-target-without-restart, (7) graceful-degradation polish (CPU path, badges, friendly camera/no-face errors). Dependencies: 1 gates the *quality* of 5 but not its existence (CPU still works); 3 needs 2; 5 needs 3+4; 6 needs 5; the `FrameSink` interface exists from step 5.

### Critical Pitfalls

Ordered by likelihood × impact for this exact machine; the first three WILL bite (see [PITFALLS.md](PITFALLS.md)).

1. **NVIDIA driver down — `nvidia-smi` fails (current state).** Root cause here is the kernel/module mismatch (module built for 6.17.0-22, booted into 6.17.0-29). Fix: `apt install --reinstall nvidia-dkms-580-open linux-headers-$(uname -r)` → `sudo dkms autoinstall` → `sudo modprobe nvidia nvidia_uvm nvidia_modeset` (verify with `nvidia-smi` and `lsmod | grep nvidia` without reboot; reboot into the matching kernel as fallback). Diagnose first; do NOT reinstall the working driver from scratch.
2. **Silent CPU fallback (cuDNN/CUDA mismatch) — the #1 functional bug.** `get_available_providers()` lists CUDA but every frame runs on CPU at 5–15 fps. Avoid: pin matched ORT 1.22 + cuDNN 9.x; `preload_dlls(cuda=True, cudnn=True)`; **assert `session.get_providers()[0]=="CUDAExecutionProvider"`** after a warm-up inference AND confirm `nvidia-smi -l 1` shows a util spike. The assert must live in code, not a one-time manual test.
3. **venv CUDA libs not on the loader path (#25609).** Library exists in site-packages but ORT can't `dlopen` it. Avoid via `preload_dlls` (primary) + launcher `LD_LIBRARY_PATH` fallback pointing at `site-packages/nvidia/*/lib`. Never use system `ldconfig`/system CUDA.
4. **Real-time latency accumulation.** Unbounded queue → display drifts seconds behind; per-frame detection / `det_size=640` / 1080p capture all sap fps. Avoid: maxsize-1 keep-newest buffer, three threads, build once/reuse, `det_size=320` + 640×480 capture as the starting point if needed.
5. **Wayland `cv2.imshow` window dead.** HighGUI's GTK/X11 path is flaky under GNOME-Wayland. Avoid entirely by using PySide6 for the window (Wayland-native; the app needs real controls anyway).

Also watch: `inswapper_128` acquisition + license + SHA256 (Pitfall 5); PRIME render-offload env vars are NOT needed for CUDA compute (Pitfall 2 — set `prime-select on-demand`, add no PRIME vars to the run command); wrong webcam node — `/dev/video0` may be metadata-only, probe for a node that returns frames (Pitfall 9); webcam-busy handling + always `release()` on stop/crash (Pitfall 10); Python 3.12 wheel pain is largely sidestepped by insightface 1.0.1 (pure-Python), with Py 3.10/3.11 in the *project* venv as the escape hatch if needed (Pitfall 8).

## Implications for Roadmap

PROJECT.md mandates **one phase** for the whole MVP (explicit user constraint, "уложить в одну фазу"). So the roadmap is a **single phase** with a strict internal build order, plus clearly-deferred future milestones. The build order below front-loads the GPU risk behind a hard verification gate and reaches the Core Value smoke test as early as possible.

### Phase 1: Real-time webcam face-swap MVP (the single phase)

**Rationale:** This is the entire project. The smart ordering inside it is risk-first: the NVIDIA driver and silent-CPU-fallback risks must be retired before any pipeline work, because they determine whether "smooth real-time" is even reachable — and the app still functions (degraded) on CPU, so the GPU step is risk-isolated but attempted first.

**Delivers:** Open app → webcam connects → user's live face swapped onto a loaded photo, shown smoothly in a PySide6 window with Start/Stop, change-photo-without-restart, GPU/CPU badge, FPS counter, mirror + swap toggles, and no-face passthrough.

**Internal build order (use as task ordering):**

1. **GPU/env gate (highest risk).** Fix the driver via `dkms autoinstall` + `modprobe` (verify without reboot; reboot fallback). Create isolated venv (uv, Python 3.12). Install `onnxruntime-gpu[cuda,cudnn]==1.22.0` + insightface 1.0.1 + opencv 4.13.0.92 + numpy 2.2.1 + PySide6 6.8. Write `providers.py` + `preload_dlls` + `LD_LIBRARY_PATH` fallback. **Hard gate:** a probe script prints active provider == CUDA, backed by an `nvidia-smi` util spike. *Avoids Pitfalls 1, 2, 3, 4(venv-path), 8.*
2. **Model management.** `bootstrap.py` lets `buffalo_l` auto-download, fetches `inswapper_128.onnx` from a mirror into project `models/`, **verifies SHA256**, then runs offline. *Avoids Pitfall 5.*
3. **FaceEngine on static images.** `embed(photo)` + `process(frame, target)` on two stills; logs confirm GPU. Validates the model path with zero threading.
4. **Capture + LatestFrameBuffer.** Capture thread → maxsize-1 keep-newest buffer; throwaway consumer prints stable fps. *Avoids Pitfalls 4, 9 (probe-and-pick the capturable node).*
5. **Pipeline wiring = the end-to-end live smoke test (Core Value).** Inference thread: read-latest → detect → swap-with-cached-target → `PreviewSink`. *This is the project's success criterion.*
6. **PySide6 UI build-out.** MainWindow, QLabel preview, Start/Stop, load-photo dialog, device picker, FPS + GPU/CPU badge, change-target-without-restart. *Avoids Pitfalls 5(window), 7.*
7. **Graceful-degradation polish.** CPU fallback through the same pipeline; surface real provider + fps; friendly "camera busy / no face in photo" errors; `release()` on stop/crash. *Avoids Pitfalls 3(visibility), 10.*

**Addresses (FEATURES.md):** all P1 table stakes + cheap P1/P2 toggles; perf knobs (reduced-res / det_size / detect-every-N) added in-phase only if fps < 25–30.

**Avoids (PITFALLS.md):** 1, 2, 3, 4, 5, 7, 9, 10 mapped to the steps above; latency design baked in from step 4 (retrofitting threads is a rewrite).

**Uses (STACK.md):** the full pinned stack; **Implements (ARCHITECTURE.md):** all six components, with `FrameSink` present as the seam from step 5.

### Future Milestones (NOT this phase — leave the seam, don't build)

- **Milestone 2 — Virtual camera output (v4l2loopback).** The declared next milestone; `v4l2loopback` already loaded. Additive: a new `V4l2Sink` (+ optional `TeeSink`) behind the existing `FrameSink` interface via `pyvirtualcam`, plus one wiring line. No pipeline/engine change.
- **Later quality polish:** optional GFPGAN/CodeFormer toggle (off by default, only if default swap quality disappoints and the GPU spares the budget), mouth-mask / face-parsing edge blending.

### Phase Ordering Rationale

- **One phase by user mandate**, but ordered risk-first internally: the GPU gate (driver + provider verification) precedes everything because it decides whether the Core Value ("smoothly") is achievable, while leaving a working CPU degradation path so the gate de-risks rather than blocks.
- **Dependency-driven:** cached embedding underpins every frame and "change photo without restart"; models gate FaceEngine; FaceEngine + capture gate the live pipeline; the pipeline gates the UI.
- **Architecture-driven grouping:** bootstrap/env is isolated as the first concern; the latency-critical capture↔inference boundary (maxsize-1 buffer) is built before UI; the `FrameSink` seam is introduced exactly when the pipeline first needs an output, making the vcam milestone purely additive.

### Research Flags

This domain is unusually well-documented for the chosen stack; **the single phase does NOT need a separate `/gsd-research-phase`** — STACK/FEATURES/ARCHITECTURE/PITFALLS already cover it at HIGH confidence with exact pins, reference code, and a verified machine diagnosis.

- **Standard patterns (skip research):** model loading, three-thread pipeline, provider selection/verification, PySide6 frame display, V4L2 capture — all have HIGH-confidence references (ONNX Runtime docs, InsightFace examples, Deep-Live-Cam, PySide6+OpenCV gist).
- **Validate during execution (not full research, but confirm on the box):** the live driver fix (verify-without-reboot vs reboot-fallback), exact `preload_dlls` vs `LD_LIBRARY_PATH` behavior on this ORT 1.22 install, and the `inswapper_128.onnx` mirror + SHA256 (community-hosted, integrity varies).

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Versions verified against live PyPI metadata + ONNX Runtime official docs; machine driver state directly observed. One MEDIUM spot: `inswapper_128` mirror/hash (community-hosted). |
| Features | HIGH | Deep-Live-Cam, FaceFusion, DeepFaceLive, roop-unleashed all converge on the same pipeline; restorer fps cost confirmed across sources. |
| Architecture | HIGH | Threading model, provider verification, model API verified against ORT docs + InsightFace + reference projects. MEDIUM only on exact local-CUDA-via-pip loader wiring (version-sensitive — resolved above). |
| Pitfalls | HIGH | Top-3 (driver, silent fallback, venv lib path) confirmed by official docs + multiple GitHub issues. MEDIUM on Wayland/cv2 + Py3.12 wheel specifics (community-sourced). |

**Overall confidence:** HIGH

### Gaps to Address

- **Driver fix verification path:** DKMS may build only for the kernel it's told to; if `modprobe` can't load into the running 6.17.0-29 kernel, reboot into the kernel with the module OR rebuild for the running kernel. Verify `nvidia-smi` survives a reboot (DKMS persists across future kernel upgrades) before building on top.
- **venv CUDA loader wiring (resolved but version-sensitive):** rely on ORT 1.22 `preload_dlls` as primary; keep the `LD_LIBRARY_PATH`-to-site-packages launcher fallback. Confirm `ldd` on the CUDA provider `.so` resolves to venv paths and zero system CUDA is involved.
- **`inswapper_128.onnx` integrity:** mirrors vary; pin the chosen SHA256 and assert it on download (a wrong/corrupt build throws "Protobuf parsing failed").
- **fps headroom unknown until measured:** whether the 3080 Ti Mobile hits 25–30+ fps at default res, or needs the det_size/resolution/frame-skip knobs, is only knowable after the live smoke test (step 5). Treat the perf knobs as in-phase contingencies.
- **Webcam capturable node:** `/dev/video0` may be metadata-only on this kernel; probe-and-pick at startup rather than hardcoding index 0.

## Sources

### Primary (HIGH confidence)
- **Live machine diagnosis (this host, 2026-05-29)** — driver 580.126.09 installed; running kernel 6.17.0-29 vs module built for 6.17.0-22; Secure Boot disabled; PRIME on-demand; `/dev/video0` present; Wayland.
- **PyPI live JSON metadata (2026-05-29)** — onnxruntime-gpu 1.22.0 `[cuda]`/`[cudnn]` extras (CUDA 12.x / cuDNN 9.x), insightface 1.0.1, opencv-python 4.13.0.92, nvidia-cudnn-cu12 9.x.
- **ONNX Runtime official docs** — CUDA Execution Provider compatibility, `preload_dlls`, silent fallback, install/compatibility pages; v1.22.0 release notes.
- **onnxruntime #25609** — ORT does NOT patch `LD_LIBRARY_PATH` for pip nvidia-* libs (the tension reconciled above).
- **InsightFace** — `FaceAnalysis(buffalo_l)` + `inswapper` example, provider/ctx_id usage; non-commercial-research license.
- **NVIDIA Developer Forums / PRIME Render Offload README** — driver+Secure Boot/MOK; render-offload vs CUDA-compute distinction.
- **Linux kernel VIDIOC_QUERYCAP** — capture vs metadata V4L2 nodes.
- **PySide6 + OpenCV** — QThread worker, BGR→QImage→QPixmap→QLabel.
- **DeepFaceLive / Deep-Live-Cam** — multi-stage pipeline, slowest-component limit, single-image live-cam loop.

### Secondary (MEDIUM confidence)
- **FaceFusion `requirements.txt`** — modern version pins reference (opencv 4.13.0.92, numpy 2.2.1).
- **Deep-Live-Cam GFPGAN cost notes** — disabling restorer ~doubles fps.
- **"Hidden Pitfalls of ONNXRuntime GPU Setup"** — silent CPU fallback detection.
- **NVIDIA/Optimus driver install guides, Arch Wiki PRIME** — autoinstall, nouveau blacklist, on-demand.
- **insightface #2335/#2294/#2430, ComfyUI/ReActor issues** — inswapper auto-download removal + mirrors; Py3.12 build pain.

### Tertiary (LOW confidence — validate during execution)
- **Hugging Face inswapper mirrors + circulated SHA256** — community-hosted; verify the exact hash for the build actually downloaded.

---
*Research completed: 2026-05-29*
*Ready for roadmap: yes*
