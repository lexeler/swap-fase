# Architecture Research

**Domain:** Local real-time webcam face-swap desktop app (Python, InsightFace inswapper, onnxruntime-gpu, OpenCV)
**Researched:** 2026-05-29
**Confidence:** HIGH (threading model, provider selection, model API verified against onnxruntime docs + InsightFace examples + Deep-Live-Cam/DeepFaceLive reference projects; MEDIUM on exact local-CUDA-via-pip wiring, which has version-sensitive edge cases)

## Standard Architecture

### System Overview

The app is a single-process, multi-threaded desktop application. The crux is a **producer/consumer pipeline** where a capture thread fills a 1-slot "latest frame" buffer, an inference thread does the heavy GPU work, and the UI thread only paints. No frame ever waits in a long queue — that is the entire latency strategy.

```
┌──────────────────────────────────────────────────────────────────┐
│                          UI THREAD (Qt main)                       │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  MainWindow: preview QLabel + controls                     │   │
│  │  (load photo · start/stop · device picker · FPS / GPU badge) │   │
│  └───────────▲──────────────────────────────────────┬─────────┘   │
│              │ Qt signal (QImage, queued)            │ commands     │
└──────────────┼──────────────────────────────────────┼─────────────┘
               │ result frames                         │ set target / start / stop
   ┌───────────┴───────────┐              ┌────────────▼─────────────┐
   │   INFERENCE THREAD     │             │      CONTROL / STATE       │
   │  (FaceEngine.process)  │◀────────────│  (target embedding cache,  │
   │  detect → swap → emit  │  target emb │   running flag, device id) │
   └───────────▲───────────┘             └────────────────────────────┘
               │ get_latest() (drops stale)
   ┌───────────┴───────────┐
   │   LatestFrameBuffer    │  ← capacity 1, keep-newest, drop-old
   └───────────▲───────────┘
               │ put(frame)
   ┌───────────┴───────────┐
   │    CAPTURE THREAD      │
   │  cv2.VideoCapture loop │
   └───────────▲───────────┘
               │ /dev/video0
        ┌──────┴──────┐
        │   Webcam     │
        └─────────────┘

   ┌────────────────────────────────────────────────────────────┐
   │  BOOTSTRAP (runs once, before threads start)               │
   │  venv check · CUDA probe · model download/verify           │
   └────────────────────────────────────────────────────────────┘

   ┌────────────────────────────────────────────────────────────┐
   │  RENDER SINK (interface) — preview now; v4l2 LATER          │
   │  FrameSink.write(frame)  ← UI is one impl; vcam is future   │
   └────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **Bootstrap / env** | Verify venv is active & local; probe whether CUDA actually works (not just "available"); download + verify model files once, then offline | Plain Python module run at startup; `onnxruntime` provider probe; `hashlib` for integrity; `huggingface_hub`/`urllib` for download |
| **FaceEngine** | Own the `FaceAnalysis` (buffalo_l detector/embedding) + `inswapper_128` model; choose provider; expose `process(frame, target_face) -> frame` and `embed(image) -> target_face` | `insightface.app.FaceAnalysis` + `insightface.model_zoo.get_model`; ONNX Runtime sessions |
| **Capture** | Open `/dev/videoN`, read frames in a tight loop, push only the newest into the buffer | `cv2.VideoCapture(index, cv2.CAP_V4L2)` in a `threading.Thread` |
| **LatestFrameBuffer** | Decouple capture rate from inference rate; always hold at most the newest frame | `queue.Queue(maxsize=1)` with drain-on-put, or a lock + single slot + `Condition` |
| **Pipeline / loop** | Orchestrate read-latest → detect → swap → hand result to sink; track FPS; honor start/stop and target changes | Inference `threading.Thread` calling FaceEngine |
| **FrameSink (interface)** | Abstract "where a finished frame goes." Preview sink = emit Qt signal. Future vcam sink = push to v4l2loopback | `Protocol`/ABC with `write(frame)`; preview impl + (later) vcam impl |
| **UI** | Preview window, controls, status; convert BGR→QImage and paint; issue commands to control state | **PySide6 (Qt)** — `QMainWindow`, `QLabel` for preview, `QThread`/signals for thread-safe updates |
| **Control / state** | Hold target face embedding, running flag, selected device, current provider/FPS for display | Small thread-safe state object shared between UI and inference threads |

## Recommended Project Structure

```
swap-fase/
├── models/                    # model assets, downloaded once, gitignored
│   ├── inswapper_128.onnx
│   └── buffalo_l/             # detection+embedding pack (auto by insightface)
├── src/swapfase/
│   ├── __init__.py
│   ├── bootstrap.py           # venv guard, CUDA probe, model download+verify
│   ├── providers.py           # provider selection + runtime GPU verification
│   ├── engine.py              # FaceEngine: analyser + swapper, embed(), process()
│   ├── capture.py             # CaptureThread + device enumeration
│   ├── framebuffer.py         # LatestFrameBuffer (keep-newest)
│   ├── pipeline.py            # InferenceWorker: read→detect→swap→sink, FPS meter
│   ├── sink.py                # FrameSink protocol + PreviewSink (vcam sink LATER)
│   ├── state.py               # AppState: target embedding, running flag, device
│   ├── ui/
│   │   ├── main_window.py     # PySide6 window, controls, QImage paint
│   │   └── widgets.py         # status badge, fps label, device combo
│   └── app.py                 # composition root: wire bootstrap→engine→threads→UI
├── run.py                     # entry point: python run.py
├── requirements.txt
└── .planning/
```

### Structure Rationale

- **`models/` at project root:** Satisfies "models live in a project-local dir, download-once-then-offline." Point `insightface` at it via `root=` so nothing lands in `~/.insightface`. Gitignore it (large binaries).
- **`bootstrap.py` separate from `engine.py`:** Environment correctness (CUDA up? model present? hash valid?) is a distinct concern from inference. Bootstrap is the highest-risk task (broken NVIDIA driver) and must run/fail loudly *before* any thread starts.
- **`sink.py` as its own module:** This is the future virtual-camera seam. Keeping the sink interface separate from both pipeline and UI means the v4l2 milestone is a new file + one wiring line, not a refactor.
- **`framebuffer.py` tiny but standalone:** The latency control is so central it deserves to be testable in isolation.
- **`app.py` as composition root:** One place wires threads + queues + UI. Keeps each component free of global state and easy to reason about for a single-phase build.

## Architectural Patterns

### Pattern 1: Three-thread producer/consumer with keep-newest buffer (the crux)

**What:** Capture, inference, and UI each run on their own thread. They communicate through a **1-slot buffer that always holds only the newest frame** plus Qt's thread-safe signal queue for results.

**When to use:** Any real-time camera→GPU→display loop where inference (10–30 ms on GPU, 60–200 ms on CPU) is slower than capture (30–60 fps). This is the standard pattern; DeepFaceLive uses an equivalent multi-stage pipeline and notes "overall performance is limited by the slowest component."

**Trade-offs:**
- (+) UI never blocks on GPU; camera never blocks on inference; latency stays bounded to ~one inference time regardless of how slow the GPU step is.
- (+) On CPU fallback the app still works — it just shows fewer, older-by-one frames instead of building a growing backlog.
- (−) Some captured frames are intentionally discarded (correct behavior for live preview; you want *fresh*, not *all*).

**Example (the keep-newest buffer — drop-old, not drop-new):**
```python
# framebuffer.py
import queue

class LatestFrameBuffer:
    """Holds at most the newest frame. Producers never block; stale frames drop."""
    def __init__(self):
        self._q = queue.Queue(maxsize=1)

    def put(self, frame):
        try:
            self._q.get_nowait()       # discard the stale frame
        except queue.Empty:
            pass
        self._q.put_nowait(frame)      # newest wins

    def get(self, timeout=1.0):
        return self._q.get(timeout=timeout)   # blocks inference until a frame exists
```
```python
# pipeline.py (inference thread body, simplified)
while state.running:
    frame = buffer.get()                       # newest available frame
    target = state.target_face                  # precomputed embedding (see Pattern 3)
    if target is not None:
        faces = engine.detect(frame)            # buffalo_l
        for f in faces:
            frame = engine.swap(frame, f, target)  # inswapper, GPU
    fps_meter.tick()
    sink.write(frame)                           # -> Qt signal -> UI paint
```

### Pattern 2: Provider selection with real runtime verification + graceful degradation

**What:** Build the provider list `['CUDAExecutionProvider', 'CPUExecutionProvider']`, create the sessions, then **verify which provider actually got used** by reading `session.get_providers()` and running one warm-up inference. Surface GPU-vs-CPU to the user explicitly.

**When to use:** Always here — the NVIDIA driver is currently broken, so silent CPU fallback is the single most likely "why is it slow?" failure. ONNX Runtime does NOT raise if CUDA is missing; it silently drops to CPU. `ort.get_available_providers()` listing CUDA does *not* mean CUDA works (commonly missing cuDNN).

**Trade-offs:**
- (+) Honest status; user knows immediately whether they're on the 25–30 fps GPU path or the 5–15 fps CPU path.
- (+) App stays usable on CPU instead of crashing.
- (−) Requires a warm-up inference at startup (small one-time cost), and InsightFace's `FaceAnalysis` only exposes provider choice via the constructor + `ctx_id`, so verification reads the underlying ONNX sessions.

**Example:**
```python
# providers.py
import onnxruntime as ort

def select_providers(prefer_gpu=True):
    avail = ort.get_available_providers()
    if prefer_gpu and "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

def active_provider(session) -> str:
    # The provider that actually bound, not just what was requested.
    return session.get_providers()[0]
```
```python
# engine.py (selection + ctx_id wiring)
from insightface.app import FaceAnalysis
import insightface

class FaceEngine:
    def __init__(self, model_root, prefer_gpu=True):
        providers = select_providers(prefer_gpu)
        gpu = providers[0] == "CUDAExecutionProvider"
        self.analyser = FaceAnalysis(name="buffalo_l", root=model_root,
                                     providers=providers)
        self.analyser.prepare(ctx_id=0 if gpu else -1, det_size=(640, 640))
        self.swapper = insightface.model_zoo.get_model(
            f"{model_root}/inswapper_128.onnx", providers=providers)
        # after a warm-up inference, read self.swapper.session.get_providers()[0]
        self.using_gpu = gpu
```

### Pattern 3: Target-embedding precompute (compute once, not per frame)

**What:** When the user loads a target photo, run detection + embedding on it **once** and cache the resulting `Face` object (the 512-d embedding + bbox/kps). The per-frame loop only detects faces in the *webcam* frame and feeds them + the cached target into the swapper.

**When to use:** Always. The target photo does not change frame-to-frame; re-analysing it 30×/second wastes GPU and adds latency.

**Trade-offs:**
- (+) Removes one detection pass from the hot loop.
- (+) Makes "change target without restart" a single cache-replace operation under a lock — clean and cheap.
- (−) Must handle "photo has no detectable face" at load time (surface an error, keep the old target).

**Example:**
```python
# on "load photo" (UI thread → control state):
img = cv2.imread(path)
faces = engine.detect(img)
if not faces:
    raise NoFaceError(path)
state.target_face = max(faces, key=lambda f: f.det_score)  # cached embedding
```

### Pattern 4: FrameSink abstraction (the future virtual-camera seam)

**What:** Define a tiny `FrameSink` interface with one method, `write(frame: np.ndarray) -> None`. The pipeline writes finished frames to a sink; it does not know or care whether that sink paints to Qt or pushes to v4l2.

**When to use:** Now, as an interface only. Build exactly one implementation (`PreviewSink`) for this milestone. The virtual-camera milestone adds a second implementation and optionally a `TeeSink` (write to both).

**Trade-offs:**
- (+) The future v4l2 work is additive: new file `V4l2Sink`, no changes to pipeline/engine. Matches PROJECT.md's "leave a clean seam, don't build it."
- (+) Testable: a `NullSink` or recording sink lets you smoke-test the pipeline headless.
- (−) One layer of indirection on the output (negligible cost).

**Example (interface + the one impl built now; vcam impl is LATER, shown commented):**
```python
# sink.py
from typing import Protocol
import numpy as np

class FrameSink(Protocol):
    def write(self, frame: np.ndarray) -> None: ...

class PreviewSink:
    """Emits the frame to the Qt UI via a bound signal callback."""
    def __init__(self, emit_callable):
        self._emit = emit_callable          # MainWindow.frame_ready.emit
    def write(self, frame):
        self._emit(frame)                   # BGR ndarray; UI converts to QImage

# --- FUTURE MILESTONE ONLY — do NOT implement now ---
# class V4l2Sink:
#     """Pushes frames to /dev/videoN via pyvirtualcam (v4l2loopback already loaded)."""
#     def write(self, frame): ...           # cam.send(rgb_frame)
#
# class TeeSink:
#     def __init__(self, *sinks): self._sinks = sinks
#     def write(self, frame):
#         for s in self._sinks: s.write(frame)
```

## Data Flow

### Steady-state frame flow (running)

```
/dev/video0
   ↓  (capture thread: cv2.VideoCapture.read, ~30-60 fps)
LatestFrameBuffer.put(frame)     ── drops any stale frame, keeps newest
   ↓  (inference thread: blocking get of newest)
FaceEngine.detect(frame)         ── buffalo_l, GPU
   ↓
FaceEngine.swap(frame, face, cached_target)   ── inswapper_128, GPU  (per detected face)
   ↓
FrameSink.write(result)          ── PreviewSink
   ↓  (Qt queued signal: thread-safe hand-off to UI thread)
MainWindow.on_frame: BGR→RGB→QImage→QPixmap→QLabel.setPixmap
   ↓
Screen  (+ FPS / GPU-or-CPU badge updated)
```

### Control flow (commands)

```
[Load photo]  → UI thread → engine.detect(photo) → cache Face → state.target_face
[Start]       → UI thread → state.running=True → start capture+inference threads
[Stop]        → UI thread → state.running=False → join threads, release VideoCapture
[Pick device] → UI thread → stop → reopen cv2.VideoCapture(new_index) → start
[Change target] → same as Load photo; replaces cached embedding under lock, no restart
```

### Startup flow (bootstrap, once, before any thread)

```
venv guard (sys.prefix is project-local) 
   ↓
provider probe: ort.get_available_providers() + warm-up inference
   ↓ (CUDA works?) ──no──→ status = CPU fallback (warn user, continue)
   ↓ yes
model check: models/ present? hash matches? ──no──→ download once → verify hash
   ↓
construct FaceEngine → ready
```

## Scaling Considerations

This is a single-user local app; "scale" means **frame rate / latency**, not users. The realistic axis is GPU vs CPU and resolution.

| Scale | Architecture Adjustments |
|-------|--------------------------|
| GPU healthy (target) | inswapper_128 on CUDA ~10–30 ms/face → keep-newest buffer keeps preview at camera rate (25–30+ fps). No changes needed. |
| CPU fallback (degraded) | 60–200 ms/face. Buffer drops most frames; preview shows ~5–15 fps but stays *current* (no backlog). Optionally lower `det_size` (e.g. 320×320) and downscale capture to claw back fps. |
| Multiple faces in frame | Swap cost scales with detected face count. Cap to N faces or swap only the largest/highest-score face if fps suffers. |

### Scaling Priorities

1. **First bottleneck: the inference step itself.** Fix order — (a) get CUDA actually working (highest leverage, ~10–20×), (b) use `inswapper_128_fp16` if available, (c) reduce detector `det_size`, (d) downscale the frame before detect/swap then upscale for display.
2. **Second bottleneck: BGR→QImage→QPixmap on the UI thread.** Cheap relative to inference, but at high fps avoid extra copies; scale the QPixmap to the label size, not the other way around.

## Anti-Patterns

### Anti-Pattern 1: Unbounded frame queue (or queuing all frames)

**What people do:** Use a `queue.Queue()` with no maxsize between capture and inference so "no frames are lost."
**Why it's wrong:** Capture outruns inference, the queue grows without bound, and displayed video falls seconds behind reality and keeps drifting — the classic "lag buildup." For a live mirror, old frames are worthless.
**Do this instead:** A maxsize-1 keep-newest buffer that discards stale frames (Pattern 1). Latency stays pinned to roughly one inference time.

### Anti-Pattern 2: Running inference or capture on the UI thread

**What people do:** Put `VideoCapture.read()` and `swapper.get()` inside a Qt timer callback or the GUI event loop.
**Why it's wrong:** Blocks the Qt event loop → window freezes, can't be moved/closed, no responsive controls. Especially bad on Wayland where a stalled event loop looks like a hung app.
**Do this instead:** Capture thread + inference thread; the UI thread only converts and paints, fed by queued Qt signals.

### Anti-Pattern 3: Trusting `get_available_providers()` as proof CUDA works

**What people do:** See `CUDAExecutionProvider` in the available list and assume GPU is active.
**Why it's wrong:** ONNX Runtime silently falls back to CPU at session creation if cuDNN/CUDA runtime is actually missing or the driver is down — exactly this project's risk. You get full speed expectations and CPU performance.
**Do this instead:** After session creation, read `session.get_providers()[0]` and run a warm-up inference; report the *actual* provider in the UI badge.

### Anti-Pattern 4: Re-analysing the target photo every frame

**What people do:** Call `FaceAnalysis.get()` on both the webcam frame and the target photo inside the loop.
**Why it's wrong:** Doubles detection cost per frame for data that never changes.
**Do this instead:** Precompute and cache the target `Face` on load (Pattern 3).

### Anti-Pattern 5: Installing both onnxruntime and onnxruntime-gpu

**What people do:** `pip install onnxruntime` then later `onnxruntime-gpu` (or via transitive deps).
**Why it's wrong:** Having both installed commonly resolves to the CPU package and silently disables GPU — a documented, frequent foot-gun.
**Do this instead:** In the project venv install **only** `onnxruntime-gpu`; pin it (e.g. 1.20.x for CUDA 12.x + cuDNN 9). Verify with the runtime probe.

### Anti-Pattern 6: Using OpenCV HighGUI (`cv2.imshow`) as the app window

**What people do:** Ship `cv2.imshow` + `cv2.waitKey` as the UI because it's one line.
**Why it's wrong:** HighGUI has no real controls (load-photo dialog, device picker, status), and its windowing is unreliable under Wayland (often falls back through XWayland, focus/close quirks). It does not give you the responsive control surface this app needs.
**Do this instead:** Use **PySide6 (Qt6)** — first-class native Wayland support, proper widgets for all controls, and a clean QThread/signal model for feeding frames. Display a frame by converting the BGR ndarray to `QImage(Format_RGB888)` → `QPixmap` → `QLabel.setPixmap` on the UI thread.

## Integration Points

### External Services / Assets

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Model files (inswapper_128, buffalo_l) | Download once at first run into project `models/`, verify by hash, then run fully offline | Point InsightFace at `root=models/` so nothing writes to `~/.insightface`. inswapper_128 is ~ tens of MB; buffalo_l pack auto-downloads. Matches "download-once-then-offline" + "project-local". |
| CUDA / cuDNN runtime | Provide via pip into the venv (`nvidia-*-cu12` wheels pulled by onnxruntime-gpu / torch), **not** the broken system install; optionally `onnxruntime.preload_dlls()` / set `LD_LIBRARY_PATH` to the venv's nvidia libs | Honors "don't touch global venv; CUDA in project". Still needs a working **NVIDIA kernel driver** (`nvidia-smi`) — userspace wheels can't fix a down driver. This is the project's top risk. (MEDIUM confidence on exact wiring; version-sensitive.) |
| Webcam | `cv2.VideoCapture(index, cv2.CAP_V4L2)` on `/dev/video0..3` | Enumerate by probing indices; expose in device picker. |
| v4l2loopback (FUTURE) | Via `pyvirtualcam` `Camera.send(rgb_frame)` to `/dev/videoN`; reached through the `FrameSink` interface only | Module already loaded per PROJECT.md. **Not built this milestone** — interface seam only. |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| Capture ↔ Inference | `LatestFrameBuffer` (maxsize-1, keep-newest) | The latency-control boundary. No back-pressure on capture; stale frames dropped. |
| Inference ↔ UI | Qt queued signal carrying the result frame | Thread-safe hand-off; UI thread does conversion + paint only. |
| Inference → output | `FrameSink.write(frame)` | The future-vcam seam. One impl now (`PreviewSink`), additive later. |
| UI ↔ Control state | Shared `AppState` guarded by a lock (target_face, running, device) | "Change target without restart" = atomic cache swap; start/stop = flag + thread lifecycle. |
| Bootstrap → everything | Runs to completion before threads start; hands a ready `FaceEngine` + provider status to `app.py` | Fail loud and early if model missing or (optionally) if GPU unavailable and user demanded GPU. |

## Suggested Build Order (single phase)

Sequenced so the roadmapper can order tasks, front-loading the highest-risk item and reaching an end-to-end smoke test as fast as possible.

1. **Fix the NVIDIA driver + project venv + provider probe (highest risk first).** Get `nvidia-smi` responding; create the isolated venv; install `onnxruntime-gpu` (+ insightface, opencv-python, PySide6); write `providers.py` and prove `CUDAExecutionProvider` actually binds via a warm-up inference. *Smoke test: a script prints the active provider = CUDA.* If this slips, everything else still works on CPU, so it's risk-isolated but must be attempted first.
2. **Model management.** `bootstrap.py` downloads inswapper_128 + buffalo_l into `models/`, verifies hashes, runs offline thereafter. *Smoke test: delete network, app still loads models.*
3. **FaceEngine on static images.** `embed(photo)` + `process(frame, target)` working on two still JPEGs end-to-end (the InsightFace example flow). *Smoke test: swapped.jpg looks right; logs show GPU.* This validates the model path with zero threading complexity.
4. **Capture + LatestFrameBuffer.** Capture thread reading `/dev/video0` into the keep-newest buffer; a throwaway consumer prints fps. *Smoke test: stable fps, no growing backlog.*
5. **Pipeline wiring (the end-to-end smoke test).** Inference thread: read-latest → detect → swap-with-cached-target → `PreviewSink`. Drive it from a minimal window. *Smoke test: live webcam, your face swapped, smooth — this is the project's Core Value.*
6. **UI build-out (PySide6).** MainWindow with QLabel preview, start/stop, load-photo dialog, device picker, FPS + GPU/CPU badge; thread-safe frame signal. Change-target-without-restart.
7. **Graceful degradation polish.** Wire CPU fallback path through the same pipeline; surface the actual provider and fps in the badge; handle "no face in photo" and "camera busy" errors.

Dependencies: 1 gates the *quality* of 5 but not its existence (CPU works); 3 depends on 2; 5 depends on 3+4; 6 depends on 5; the `FrameSink` interface should exist from step 5 so the future vcam milestone is purely additive.

## Sources

- ONNX Runtime — Execution Providers & CUDA EP (provider list, silent fallback, get_providers verification, CUDA 12.x/cuDNN 9): https://onnxruntime.ai/docs/execution-providers/ , https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html , https://onnxruntime.ai/docs/api/python/api_summary.html — HIGH
- ONNX Runtime install + compatibility (onnxruntime-gpu 1.20, CUDA 12 default since 1.19, preload_dlls): https://onnxruntime.ai/docs/install/ , https://onnxruntime.ai/docs/reference/compatibility.html , https://pypi.org/project/onnxruntime-gpu/ — HIGH
- onnxruntime issue #21354 — get_providers() not showing CUDA (real-world silent-fallback gotcha): https://github.com/microsoft/onnxruntime/issues/21354 — MEDIUM
- "The Hidden Pitfalls of ONNXRuntime GPU Setup": https://dev.to/deskpai/the-hidden-pitfalls-of-onnxruntime-gpu-setup-4kb7 — MEDIUM
- InsightFace — FaceAnalysis(buffalo_l) + inswapper example, provider/ctx_id usage: https://github.com/deepinsight/insightface , https://github.com/deepinsight/insightface/blob/master/examples/in_swapper/inswapper_main.py , https://pypi.org/project/insightface/ — HIGH
- Deep-Live-Cam (closest reference: single-image inswapper live cam, models dir, execution-provider flag): https://github.com/hacksider/Deep-Live-Cam , https://github.com/hacksider/Deep-Live-Cam/issues/1353 — MEDIUM
- DeepFaceLive architecture (modular pipeline, slowest-component limit, multi-stage parallelism): https://deepwiki.com/iperov/DeepFaceLive — MEDIUM
- PySide6 + OpenCV webcam display (QThread worker, BGR→QImage→QPixmap→QLabel, producer/consumer overflow): https://gist.github.com/docPhil99/ca4da12c9d6f29b9cea137b617c7b8b1 , https://doc.qt.io/qtforpython-6/examples/example_external_opencv.html — HIGH
- pyvirtualcam (future v4l2loopback seam: Camera.send, /dev/videoN): https://github.com/letmaik/pyvirtualcam , https://letmaik.github.io/pyvirtualcam/ — HIGH

---
*Architecture research for: local real-time webcam face-swap desktop app*
*Researched: 2026-05-29*
