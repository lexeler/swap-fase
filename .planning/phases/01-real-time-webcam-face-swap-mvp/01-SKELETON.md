# Walking Skeleton — Swap-Fase

**Phase:** 1
**Generated:** 2026-05-30

## Capability Proven End-to-End

> One sentence: the smallest user-visible capability that exercises the full stack.

A user opens a PySide6 window, the webcam connects, and their live face is replaced in real time with the face from a hard-loaded target photo, displayed in the window — proving driver → venv-local CUDA → InsightFace `inswapper` on the GPU → V4L2 capture → keep-newest pipeline → Qt preview all work together end-to-end.

## Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Language / interpreter | Python 3.12 (system 3.12.3) in an **isolated project-local `.venv`** via `uv` | CLAUDE.md hard constraint: never touch the global Python env; uv resolves the heavy `nvidia-*-cu12` wheels fast/reliably (STACK.md §uv). Plain `venv`+pip is the documented fallback. |
| Face swap + analysis | `insightface==1.0.1` — `buffalo_l` analyser + `inswapper_128(_fp16).onnx` | De-facto single-image swap stack; 1.0.x is pure-Python (no C++ toolchain), removing historic install pain (STACK.md, D-11). |
| Inference runtime | `onnxruntime-gpu[cuda,cudnn]==1.22.0` (CUDA 12.x + cuDNN 9.x via pip extras) | The `[cuda]`/`[cudnn]` extras pull a clean pinned `nvidia-*-cu12 ~=12.x` + `nvidia-cudnn-cu12 ~=9.x` set **into the venv** — no system CUDA toolkit (D-03, D-11, D-12). cuDNN 8 is a hard break. |
| CUDA library discovery | `onnxruntime.preload_dlls(cuda=True, cudnn=True)` (primary) + `LD_LIBRARY_PATH`→`site-packages/nvidia/{cudnn,cublas,cuda_runtime}/lib` launcher fallback | ORT does NOT patch `LD_LIBRARY_PATH` for pip nvidia libs (#25609). Both routes keep CUDA venv-local; never `sudo ldconfig`/system CUDA (PITFALLS 3,4; D-14). |
| GPU driver | System `nvidia-driver-580-open 580.126.09`, fixed via `dkms autoinstall` + `modprobe` (NOT reinstalled) | Driver is installed and fine — only the kernel module is unbuilt for the running kernel (booted `6.17.0-29` vs module for `6.17.0-22`). Reinstalling risks breaking a working driver (D-13, PITFALLS 1). Driver stays system-level (cannot live in a venv). |
| Webcam capture | OpenCV `cv2.VideoCapture(idx, cv2.CAP_V4L2)` against probed `/dev/video*`, MJPG, 640×480 start | V4L2 is the Linux backend; `/dev/video0` may be metadata-only → probe-and-pick a node whose `read()` returns a frame (PITFALLS 9; D-15, LIVE-01). |
| GUI window | `PySide6==6.8.*` (Qt6) — `QMainWindow` + `QLabel` preview, frames via queued Qt signal | `cv2.imshow` is flaky under GNOME/Wayland via XWayland; PySide6 is Wayland-native and supplies the real controls the app needs (D-11, PITFALLS 7, ARCHITECTURE.md). |
| Concurrency model | 3 threads (capture / inference / Qt-UI) joined by a **maxsize-1 keep-newest `LatestFrameBuffer`** (drop stale) | Latency stays pinned to ~one inference time instead of drifting; baked in from the start (retrofitting threading is a rewrite) (D-15, ARCHITECTURE Pattern 1, PITFALLS 6, LIVE-02). |
| Source embedding | Computed **once** on photo load, cached as a `Face`, reused every frame; atomic swap under lock on photo change | Re-analysing the target 30×/s wastes GPU; caching makes change-target-without-restart a single lock-guarded cache replace (D-15, ARCHITECTURE Pattern 3, SWAP-01, UI-03). |
| Output abstraction | `FrameSink` protocol with one method `write(frame)`; one impl now (`PreviewSink`→Qt signal) | The future virtual-camera milestone is additive (`V4l2Sink`/`TeeSink` behind the same seam, one wiring line) — built as interface only now (D-domain, ARCHITECTURE Pattern 4). |
| Models location | Project-local `models/` (gitignored), `buffalo_l` auto-downloads with `root=models/`, `inswapper` fetched from a mirror + **SHA256-verified**, offline thereafter | Honors "models download once into project-local dir, then offline"; fail-closed on hash mismatch (D-16, ENV-04, PITFALLS 5). |
| Directory layout | `src/swapfase/` package + `run.py` entrypoint (single composition root in `app.py`) | ARCHITECTURE.md recommended structure; clean module boundaries for the 6 components. |

## Stack Touched in Phase 1

- [x] Project scaffold (uv venv, `src/swapfase/` package, `run.py` entrypoint, `requirements.txt`, `.gitignore`, `pyproject.toml`) — Plan 01 + 02
- [x] Real GPU pipeline — `onnxruntime-gpu` CUDA EP verified by warm-up inference + `nvidia-smi` util spike (NOT just "available") — Plan 01
- [x] Real model I/O — `buffalo_l` + `inswapper_128` loaded from project-local `models/`, hash-verified, one real static-image swap written to disk — Plan 02
- [x] Real capture + keep-newest buffer — V4L2 capture thread → `LatestFrameBuffer` → consumer prints stable fps — Plan 03
- [x] Real UI interaction wired to the pipeline — PySide6 window shows a **live swapped frame** from the camera (the Core-Value smoke test) — Plan 03
- [x] Documented local full-stack run command — `run.py` / `run.sh` launches the app with the `LD_LIBRARY_PATH` fallback wired — Plan 01 + 03

## Out of Scope (Deferred to Later Slices)

> Explicit so future milestones do not re-litigate Phase 1's minimalism.

- **Virtual camera output (v4l2loopback / VCAM-01, VCAM-02)** — next milestone; only the `FrameSink` seam is left in place now (no `V4l2Sink`).
- **Face restoration enhancer (GFPGAN/CodeFormer / QUAL-01)** — roughly halves fps; conflicts with "fps is king" (D-07). Deferred.
- **Mouth-mask / face-parsing edge blending (QUAL-02)** — quality polish, deferred.
- **"All faces" swap toggle** — locked to largest-only with no toggle (D-05, D-06). Engine keeps all detections internally so a future toggle is cheap, but it is not surfaced.
- **Target-photo persistence between runs / favorites / recents** — explicitly declined (D-09); the user loads a photo fresh each session.
- **Recording, streaming, model training, web/mobile UI** — out of scope per REQUIREMENTS.md.

## Subsequent Slice Plan

Within Phase 1 (after the skeleton, Plans 01–03, proves end-to-end), the remaining vertical slices refine the working skeleton without changing its architectural decisions:

- **Plan 04 (UI build-out slice):** real Start/Stop, load-photo dialog with change-target-without-restart, device picker, on-screen FPS counter, mirror + swap-on/off toggles — turning the skeleton's hard-loaded photo and minimal window into the full controllable app.
- **Plan 05 (robustness/degradation slice):** CPU graceful fallback through the same pipeline, the real GPU/CPU provider badge, friendly "camera busy / no face in photo" errors, guaranteed `release()` on stop/crash, plus the final human-verify checkpoint of the whole experience.

Future milestones (NOT Phase 1) add one vertical slice each on top of this same skeleton:

- **Milestone 2:** virtual camera output — additive `V4l2Sink`/`TeeSink` behind the existing `FrameSink`, selectable in Zoom/Meet/Discord.
- **Later:** optional GFPGAN/CodeFormer restoration toggle and mouth-mask/edge-blending quality polish.
