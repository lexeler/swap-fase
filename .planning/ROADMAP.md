# Roadmap: Swap-Fase

## Overview

Swap-Fase is a single, fully-local desktop app that swaps the user's live webcam face onto a single loaded photo and shows the result smoothly in a window. The entire MVP ships as ONE phase (explicit user mandate: "уложить в одну фазу"). The phase is delivered with a strict risk-first internal build order: retire the GPU risk first (broken NVIDIA driver + the silent-CPU-fallback trap), prove CUDA is really bound, then build the model/engine layer, the latency-critical three-thread capture→inference→display pipeline, and finally the PySide6 UI and graceful-degradation polish. The first reachable milestone inside the phase is the end-to-end live smoke test — "open app → webcam → my face is the photo's face, live and smooth" — which is the Core Value. Virtual-camera output and quality restorers are explicitly the next milestones, not built here (only the `FrameSink` seam is left in place).

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Real-Time Webcam Face Swap (MVP)** - Open app → webcam → live face-swapped onto a loaded photo, smoothly, in a PySide6 window with start/stop, change-photo-without-restart, GPU/CPU badge, FPS, and mirror/swap toggles.

## Phase Details

### Phase 1: Real-Time Webcam Face Swap (MVP)
**Goal**: Deliver the complete Core Value end-to-end — the user opens the app, the webcam connects, and their live face is swapped onto a single loaded target photo and shown smoothly in a desktop window, with all the controls and safeguards needed to make it usable and trustworthy on this specific machine.
**Mode:** mvp
**Depends on**: Nothing (first phase)
**Requirements**: ENV-01, ENV-02, ENV-03, ENV-04, ENV-05, SWAP-01, SWAP-02, SWAP-03, LIVE-01, LIVE-02, LIVE-03, LIVE-04, UI-01, UI-02, UI-03, UI-04, UI-05
**Success Criteria** (what must be TRUE):
  1. **GPU is verifiably used (hard gate):** `nvidia-smi` responds with the RTX 3080 Ti, and on startup a warm-up inference proves `session.get_providers()[0] == "CUDAExecutionProvider"` with a visible `nvidia-smi` utilization spike — i.e. the app is NOT silently running on CPU, and it reports the actual active provider. *(ENV-02, ENV-03, ENV-05)*
  2. **Live face swap is visible end-to-end:** with the webcam running, the user's face in each frame is detected and replaced in real time with the face from the loaded photo, displayed in the PySide6 window with working Start/Stop. *(SWAP-01, SWAP-02, LIVE-01, UI-01, UI-02)*
  3. **Smooth on GPU, survives on CPU:** on the working GPU the swap runs smoothly (target 25–30+ fps) with an on-screen FPS counter, and if the GPU is unavailable the app degrades to a working CPU fallback path (slower but functional) rather than crashing. *(LIVE-02, LIVE-03, LIVE-04, UI-05)*
  4. **Change target photo without restart:** the user loads a different target photo from the UI and the swapped face updates live, with the loop still running and no app restart required. *(SWAP-01, UI-03)*
  5. **Robust, isolated, and controllable:** when no face is present in a frame the frame passes through unchanged without crashing; mirror and swap-on/off toggles work; and all dependencies (CUDA/cuDNN, InsightFace, OpenCV, PySide6) plus the hash-verified models live in an isolated project-local venv + `models/` that never touches the global Python environment and runs offline after first download. *(ENV-01, ENV-04, SWAP-03, UI-04)*

**Internal build order** (risk-first — planner must sequence in this order):
  1. **GPU/env gate (highest risk).** Fix the NVIDIA driver kernel/module mismatch (kernel 6.17.0-29 vs `nvidia.ko` built for 6.17.0-22): `sudo dkms autoinstall` + `sudo modprobe nvidia nvidia_uvm nvidia_modeset`, verify `nvidia-smi` live without reboot; reboot into the matching kernel as fallback. Create the isolated project-local venv (uv, Python 3.12). *(ENV-02)*
  2. **Install + RUNTIME-VERIFY the stack (HARD GATE).** Install `onnxruntime-gpu[cuda,cudnn]==1.22.0` + insightface 1.0.1 + opencv-python 4.13.0.92 + numpy 2.2.1 + PySide6 6.8 into the venv. Wire `providers.py` with `preload_dlls(cuda=True, cudnn=True)` + an `LD_LIBRARY_PATH`-to-site-packages launcher fallback. **A probe must prove inference actually runs on `CUDAExecutionProvider` (not a silent CPU fallback) before any pipeline work proceeds.** *(ENV-01, ENV-03, ENV-05)*
  3. **Acquire + hash-verify models.** `bootstrap.py`: let `buffalo_l` auto-download; fetch `inswapper_128.onnx` from a mirror into project-local `models/`; verify SHA256; offline thereafter. *(ENV-04)*
  4. **FaceEngine on static images first.** Load `buffalo_l` + `inswapper`; `embed(photo)` (largest face, cache embedding once) + `process(frame, target)`; swap one still → save. Zero threading. *(SWAP-01, SWAP-02)*
  5. **Capture thread + keep-newest buffer.** `cv2.VideoCapture(idx, CAP_V4L2)` (probe-and-pick a capturable node, default `/dev/video0`) → maxsize-1 keep-newest `LatestFrameBuffer`; throwaway consumer prints stable fps. *(LIVE-01, LIVE-02)*
  6. **End-to-end live smoke test (Core Value).** Inference thread: read-latest → detect → swap-with-cached-target → `PreviewSink` → display. *This is the project's success criterion.* *(SWAP-02, LIVE-03, UI-01)*
  7. **PySide6 UI build-out.** MainWindow + QLabel preview, Start/Stop, load-photo dialog (change-target-without-restart under lock), device picker, FPS + GPU/CPU badge, mirror + swap toggles. *(UI-01, UI-02, UI-03, UI-04, UI-05, LIVE-01, LIVE-04)*
  8. **Graceful-degradation polish.** CPU fallback through the same pipeline; surface the real provider + fps; no-face passthrough; friendly "camera busy / no face in photo" errors; always `release()` on stop/crash. *(SWAP-03, LIVE-03, UI-05)*

**Performance contingencies (in-phase, only if fps < 25–30 after step 6):** reduce `det_size` (640→320), lower capture resolution (640×480), detect-every-N-frames + bbox reuse. Do not retrofit threading — it is baked in from step 5.

**Plans**: TBD
**UI hint**: yes

Plans:
- [ ] TBD (decomposed by /gsd-plan-phase 1)

## Future / Next Milestones (NOT this roadmap — leave the seam, don't build)

- **Milestone 2 — Virtual camera output (VCAM-01, VCAM-02).** Declared next milestone; `v4l2loopback` already loaded. Additive only: a new `V4l2Sink` (+ optional `TeeSink`) behind the existing `FrameSink` interface via `pyvirtualcam`, plus one wiring line, then select it in Zoom/Meet/Discord. No pipeline/engine change.
- **Later quality polish (QUAL-01, QUAL-02).** Optional GFPGAN/CodeFormer face restoration (OFF by default — roughly halves fps); mouth-mask / face-parsing edge blending for cleaner compositing. Added only if default swap quality disappoints and the GPU spares the budget.

## Progress

**Execution Order:**
Single phase: 1

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Real-Time Webcam Face Swap (MVP) | 0/TBD | Not started | - |
