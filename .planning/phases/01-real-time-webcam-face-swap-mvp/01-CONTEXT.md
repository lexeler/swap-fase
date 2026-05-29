# Phase 1: Real-Time Webcam Face Swap (MVP) - Context

**Gathered:** 2026-05-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver the complete working app end-to-end: a local desktop application (PySide6) that captures the webcam, swaps the user's live face onto a single user-loaded target photo using InsightFace `inswapper`, and shows the result smoothly in a window — with start/stop, change-photo-without-restart, GPU/CPU status, FPS counter, mirror + swap toggles, and graceful CPU fallback. All 17 v1 requirements (ENV-01..05, SWAP-01..03, LIVE-01..04, UI-01..05) live in this one phase.

**Not in this phase (next milestones):** virtual-camera output for video calls (VCAM), face-restoration enhancers (QUAL/GFPGAN), model training, recording/streaming. Leave only the `FrameSink` seam for the future virtual camera.
</domain>

<decisions>
## Implementation Decisions

### Launch behavior (Запуск и вид)
- **D-01:** App opens showing the **live webcam preview but with swap OFF**. The swap starts only when the user presses **Start** (manual gate — doesn't load the GPU until asked, predictable). *(UI-01, UI-02)*
- **D-02:** **No auto-load of a target photo on launch** — see D-09; the user picks a photo each session. Start is enabled only once a target photo is loaded (or prompts to load one).
- **D-03:** **Mirror is ON by default** (natural selfie view). Exposed as a toggle (UI-04) so it can be turned off.
- **D-04:** **Windowed, resizable** (not fullscreen). Standard desktop window.

### Multiple faces (Несколько лиц)
- **D-05:** Swap **only the single largest face** in the webcam frame (the user). Other faces are left untouched. Simpler, faster, predictable. *(SWAP-02)*
- **D-06:** **No "all faces" toggle in v1** — the user explicitly chose largest-only with no toggle. (Keep the engine able to return all detections so a future toggle is cheap, but do not surface it.)

### Quality vs FPS priority (Качество vs fps)
- **D-07:** 🔑 **FPS is the top priority** — user's words: "главное большой фпс". Maximize frame rate above visual fidelity. Target 25–30+ fps on GPU; it is acceptable to trade picture quality to hit it.
- **D-08:** Allowed FPS levers (apply as needed to hold the target, GPU first): use **`inswapper_128_fp16`** model; **reduce processing resolution** (e.g. capture 640×480, downscale detection input); **smaller `det_size`** (640→320); **detect-every-N-frames + reuse last bbox** between detections. No face-restoration enhancer (GFPGAN/CodeFormer) in this phase — it roughly halves fps and is a deferred milestone. Native `inswapper_128` output quality is accepted.

### Target photo handling (Память фото)
- **D-09:** **No persistence between runs** — the user loads a target photo fresh each session ("каждый раз заново"). No "remember last", no favorites/recents panel.
- **D-10:** **Change-target-without-restart still applies** (UI-03): during a session the user can pick a new photo via dialog and the live swap updates (atomic embedding-cache swap under lock). Persistence is what's excluded, not in-session switching.

### Carried forward — locked by PROJECT.md + research (planner: treat as decided, do NOT re-derive)
- **D-11:** Stack: Python 3.12 + InsightFace 1.0.1 (`buffalo_l` analyser + `inswapper_128[_fp16]`) + `onnxruntime-gpu[cuda,cudnn]==1.22.0` (CUDA 12.x / cuDNN 9.x) + OpenCV (V4L2) + **PySide6 for the window (NOT `cv2.imshow` — unreliable on Wayland)**.
- **D-12:** **Isolated project-local venv** (uv recommended); CUDA/cuDNN pulled into the venv via pip; the global/shared Python env is never touched. The NVIDIA *driver* stays system-level.
- **D-13:** **Driver fix is the first internal step:** kernel/module mismatch (booted `6.17.0-29` vs `nvidia.ko` built for `6.17.0-22`) → `sudo dkms autoinstall` + `sudo modprobe nvidia nvidia_uvm nvidia_modeset`; verify `nvidia-smi` live; reboot into matching kernel as fallback.
- **D-14:** **Hard GPU gate before any pipeline work:** runtime-verify inference actually runs on `CUDAExecutionProvider` (warm-up inference + `session.get_providers()[0]` check + `nvidia-smi` util spike), NOT just that the provider is "available". Use ORT 1.22 `preload_dlls(cuda=True, cudnn=True)` as primary, with an `LD_LIBRARY_PATH`→site-packages launcher fallback.
- **D-15:** **Real-time architecture:** 3 threads (capture / inference / UI) joined by a **maxsize-1 keep-newest frame buffer** (drop stale frames so latency never accumulates). Source embedding precomputed once on photo load, reused every frame.
- **D-16:** **Models in project-local `models/`**, `buffalo_l` auto-downloads, `inswapper_128(_fp16).onnx` fetched from a mirror and **SHA256-verified**, offline thereafter. Non-commercial model license is fine — strictly personal/local use, nothing published.
- **D-17:** **GPU→CPU graceful fallback:** if CUDA is unavailable the app runs on CPU (slower, still functional, no crash) and the UI badge shows the real active provider.
- **D-18:** **No-face passthrough:** frames with no detected face pass through unchanged without crashing. *(SWAP-03)*

### Claude's Discretion
- Exact UI layout/widget arrangement, button labels, device-picker presentation, where the FPS/provider badge sits.
- Camera device selection: probe `/dev/video0..3` and auto-pick the first capturable real camera; expose a picker (LIVE-01). Default `/dev/video0`.
- Threading primitives, queue/lock implementation details, error-message wording, project module layout.
- Whether to ship `inswapper_128_fp16` vs `inswapper_128` by default (lean fp16 for fps per D-07/D-08; confirm at the live smoke test).
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project intent & scope
- `.planning/PROJECT.md` — what this is, core value, constraints (isolated venv, GPU-first, personal/local), key decisions
- `.planning/REQUIREMENTS.md` — the 17 v1 requirements (ENV/SWAP/LIVE/UI) and their IDs; out-of-scope list
- `.planning/ROADMAP.md` §"Phase 1" — goal, success criteria, and the risk-first internal build order (8 steps) the planner must sequence to

### Research (HIGH confidence — exact pins, gotchas, build order)
- `.planning/research/SUMMARY.md` — reconciled decisions; READ FIRST (driver-is-module-mismatch headline, LD_LIBRARY_PATH resolution, build order)
- `.planning/research/STACK.md` — version pins, onnxruntime-gpu↔CUDA12↔cuDNN9 matrix, CUDA-in-venv pip packages, inswapper acquisition, PySide6/Wayland
- `.planning/research/PITFALLS.md` — driver diagnosis (dkms/Secure Boot/MOK), the silent-CPU-fallback trap + detection, venv CUDA loader path, real-time latency traps, Wayland/Py3.12 gotchas
- `.planning/research/ARCHITECTURE.md` — module boundaries, 3-thread + keep-newest design, provider verification, PySide6 frame display, the future `FrameSink` virtual-cam seam
- `.planning/research/FEATURES.md` — table-stakes vs anti-features, fps cost per feature, Deep-Live-Cam as the closest reference

No external ADRs — all decisions captured above and in the referenced planning docs.
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- None — greenfield project, no source code yet. The single git history so far is `.planning/` docs.

### Established Patterns
- None established yet. The architecture document defines the patterns to adopt (3-thread pipeline, keep-newest buffer, provider verification, `FrameSink` seam).

### Integration Points
- System: NVIDIA driver (system-level, currently a kernel/module mismatch — fix first), `/dev/video0` webcam, Wayland/GNOME display.
- Reference implementation to mirror: **Deep-Live-Cam** (same stack: InsightFace inswapper + onnxruntime + OpenCV, single-image, live webcam).
</code_context>

<specifics>
## Specific Ideas

- User framing of the goal: "открыл приложение → подключилась вебка → у меня свапнулось лицо на то, которое я загрузил" — a live preview window, not a CLI.
- FPS is explicitly king: "главное большой фпс."
- Target loaded fresh each run: "каждый раз заново."
- Reference behavior: like Deep-Live-Cam's live mode, but trimmed to the minimum (largest-face only, no restoration, no virtual cam).
</specifics>

<deferred>
## Deferred Ideas

- **Virtual camera output for calls (VCAM-01/02)** — next milestone. `v4l2loopback` already loaded. Additive via a `V4l2Sink`/`TeeSink` behind the `FrameSink` interface (`pyvirtualcam`). Do not build now; just leave the seam.
- **Face restoration enhancer (QUAL-01, GFPGAN/CodeFormer)** — deferred; roughly halves fps, conflicts with the "fps is king" decision (D-07).
- **Edge/mouth-mask blending (QUAL-02)** — deferred quality polish.
- **"All faces" swap toggle** — user chose largest-only with no toggle (D-06); revisit only if desired later.
- **Favorites/recents target-photo panel & remember-last-photo** — explicitly declined (D-09); could return as a convenience later.

None of these are in Phase 1 scope.
</deferred>

---

*Phase: 1-real-time-webcam-face-swap-mvp*
*Context gathered: 2026-05-30*
