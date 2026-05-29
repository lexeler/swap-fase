# Feature Research

**Domain:** Local real-time webcam face-swap desktop app (single-image swap, personal use)
**Researched:** 2026-05-29
**Confidence:** HIGH (reference apps Deep-Live-Cam, FaceFusion, DeepFaceLive, roop-unleashed all converge on the same pipeline; fps cost of restorers confirmed across multiple sources)

> Scope note: This MVP is intentionally the *minimal* slice of what the reference apps do — single uploaded photo, swap my own face live in a window, fully local, one phase. The reference projects are streaming/production tools; most of their feature surface is explicitly out of scope here. Categorization below is calibrated to "swap my face live and it feels real-time," NOT to matching feature parity with Deep-Live-Cam.

## Feature Landscape

### Table Stakes (Required for "swap my face live")

Missing any of these and the core value ("open app → webcam → my face swapped to the loaded photo, smoothly") is not delivered.

| Feature | Why Expected | Complexity | Real-time / fps cost | Notes |
|---------|--------------|------------|----------------------|-------|
| Load a single target photo as source | This is the whole premise ("swap to the photo I loaded") | LOW | None (one-time) | File picker → `cv2.imread`. Validate the file is an image and contains a detectable face; show a clear error if not. |
| Face detection on the source photo + pick primary face | A photo may contain 0, 1, or several faces; need a deterministic pick | LOW | None (one-time) | Use InsightFace `FaceAnalysis.get()`. If multiple faces, pick **largest by bbox area** (industry-standard heuristic). |
| Cache the source embedding/face object | Recomputing source per frame wastes the frame budget | LOW | Saves cost every frame | Extract the source `Face` (embedding) **once** at load time, reuse across all frames. Reference apps all do this. Recompute only on photo change. |
| Webcam capture from `/dev/video0` | No input feed = no app | LOW | Capture itself is cheap | `cv2.VideoCapture(0, cv2.CAP_V4L2)`. Request a sane resolution (e.g. 640×480 or 1280×720). |
| Per-frame: detect my face → swap → composite back | The actual swap; without it nothing happens | MEDIUM | This IS the per-frame budget | Detect on the live frame, run `inswapper.get(frame, target_face, source_face, paste_back=True)`. inswapper internally aligns (norm_crop 128) and pastes back with its built-in blend mask. |
| Display swapped stream in a window | "shown in a window" is explicit in Core Value | LOW | Cheap | A persistent display window (OpenCV `imshow` loop, or a GUI canvas if a real GUI is chosen). Must update continuously, not freeze. |
| Start / Stop the live feed | Basic control; user must be able to stop the camera | LOW | None | Releasing the capture device on stop is important (frees `/dev/video0`). |
| Change target photo without restart | Explicitly an Active requirement in PROJECT.md | LOW | None (re-extract source embedding once) | On new photo: re-run source detection + re-cache embedding; keep the live loop running. |
| GPU (CUDA) execution path with CPU fallback | Explicit requirement; CPU-only is ~5–15 fps (not "smooth") | MEDIUM | Determines whether real-time is achievable at all | onnxruntime-gpu with `CUDAExecutionProvider`, fall back to `CPUExecutionProvider`. Surface which provider is active so the user knows if they are on the slow path. |
| Graceful handling of "no face this frame" | Faces leave frame / turn away constantly | LOW | None | If no face detected on a frame, pass the original frame through unchanged rather than crashing or freezing. |

### Differentiators (Optional for v1 — improve "looks decent" / "feels real-time")

Not required to satisfy the core value, but each meaningfully improves the experience. Several are cheap enough to be worth including in the single phase; the expensive ones (restoration) should be **off by default, toggleable**.

| Feature | Value Proposition | Complexity | Real-time / fps cost | Notes |
|---------|-------------------|------------|----------------------|-------|
| On-screen FPS counter | Lets the user see if they are on the GPU "smooth" path vs CPU fallback; primary feedback for "real-time feel" | LOW | Negligible | Rolling average over last N frames. Cheap, high diagnostic value — strongly recommended for v1 given the NVIDIA-driver risk in PROJECT.md. |
| Mirror / horizontal flip toggle | Webcam feed feels natural mirrored (like a mirror); Deep-Live-Cam ships `--live-mirror` | LOW | Negligible | `cv2.flip(frame, 1)`. Pure display concern. Cheap win. |
| Toggle swap on/off live | Quick A/B of original vs swapped face; lets user confirm the app is working | LOW | Saves cost when off | A keypress / button that bypasses the swap and shows the raw frame. |
| Detect every N frames + track/reuse bbox between | Detection is a large share of per-frame cost; reusing the last bbox for N-1 frames boosts fps | MEDIUM | **Boosts fps**; risk of lag/jitter if N too high or subject moves fast | Common "real-time feel" trick. Start simple (detect every frame); add this only if fps is short of the 25–30 target. Mark as a tuning knob, not core. |
| Process at reduced resolution | Detection + swap scale with frame size; downscaling input is the biggest fps lever after GPU | LOW–MEDIUM | **Boosts fps significantly** | Capture/operate at 640×480 or 720p. inswapper output is 128px anyway, so high capture res buys little. A resolution/quality selector is a natural performance knob. |
| Choose camera device (video0..video3) | PROJECT.md notes multiple `/dev/videoN`; nice if the default isn't the right cam | LOW | None | A device index/path selector. Low effort; defer if only one cam is ever used. |
| Face restoration enhancer (GFPGAN 1.4 / CodeFormer) — OFF by default | Sharpens the swapped face, reduces seams; what makes swaps look "HD" in the reference apps | MEDIUM | **HIGH — roughly halves fps.** Disabling GFPGAN ~doubles frame rate (Deep-Live-Cam). CodeFormer only hits ~25fps on high-end (~35 TFLOPS) GPUs; offline it's seconds/image. | **Deferrable / optional toggle, NOT table stakes.** inswapper's own paste-back already looks acceptable for personal use. If included, ship it disabled, behind a toggle, with a visible fps trade-off. The 3080 Ti *may* sustain it; treat as a stretch toggle, not a requirement. |
| Mouth mask (keep original mouth) | Preserves your real mouth movement/teeth for more natural talking; Deep-Live-Cam `--mouth-mask` | MEDIUM | LOW–MEDIUM (extra parsing/mask per frame) | Needs a face-parsing/landmark mask. Nice for talking-on-camera realism but not needed to "swap my face." Differentiator. |
| Face-parsing mask for blending edges | Cleaner hairline/jaw blend than inswapper's default rectangular-ish paste | MEDIUM–HIGH | MEDIUM (extra model per frame) | Adds another model to the per-frame path. Quality polish; defer. inswapper default blend is "good enough" for personal use. |
| Detection threshold / det_size tuning | Trade detection accuracy vs speed | LOW | Tuning lever | Expose `det_size` (e.g. 320 vs 640) as a perf knob if needed. |

### Anti-Features (Deliberately NOT Build in This One-Phase MVP)

| Feature | Why Requested / Why It Appears in Reference Apps | Why Problematic Here | Alternative |
|---------|--------------------------------------------------|----------------------|-------------|
| Virtual camera output (v4l2loopback) for Zoom/Meet/Discord | The "real" payoff for many users; module already loaded on this machine | **Explicitly the NEXT milestone** per PROJECT.md ("сначала надо своп сделать"). Adds output-routing complexity that competes with getting the swap solid. | Defer to next milestone. Architect the display sink so a virtual-cam sink can be added later, but build only the window now. |
| Model training (DFM / DeepFaceLive-style trained models) | Highest-fidelity persistent identity swap | Hours of training, datasets, GPU time; contradicts "single photo I loaded"; massively overkill for "поиграться" | Single-image `inswapper_128` (already the chosen path). No training. |
| Multi-target face-mapping UI (map face A→X, face B→Y) | Deep-Live-Cam `--map-faces`; useful for multi-person videos | Big UI surface (per-face source assignment) for a single-user "swap MY face" tool | Swap all detected faces to the one source, OR (better for solo use) swap only the largest/primary face. |
| Recording / saving output to disk | Standard in face-swap tools | Out of scope per PROJECT.md ("запись на диск — не нужно"); adds encoder/file-management code | None — live view only. |
| Streaming / OBS integration | DeepFaceLive's core use case | Out of scope; "ничего не стримится" | None. |
| Web UI / browser frontend (FaceFusion Gradio style) | Easy remote access, polished UI | Out of scope ("только локальный desktop"); adds a server, ports, browser dependency | Native local window only. |
| Mobile build | — | Out of scope (Linux desktop only) | None. |
| Multiple swapper models / model picker (hyperswap, inswapper variants, pixel-boost 512) | FaceFusion offers many | Decision overhead + VRAM/perf cost; one good default is enough for personal MVP | Ship `inswapper_128` (fp16 if it loads cleanly) as the single model. |
| Expression restorer / color-match / motion-blur sliders | DeepFaceLive real-time tuning sliders | Each adds a per-frame model or pass and a control; scope creep against "one phase" | Rely on inswapper defaults; revisit only if quality is unacceptable. |
| Audio handling (`--keep-audio` etc.) | Relevant for video-file processing | Irrelevant for a live preview window | None. |

## Feature Dependencies

```
Load target photo
    └──requires──> Face detection on source
                       └──requires──> Pick primary (largest) face
                                          └──requires──> Cache source embedding  ──used by──> Per-frame swap

Webcam capture (/dev/video0)
    └──requires──> Per-frame: detect my face ──> swap ──> composite
                       └──requires──> Cached source embedding (above)
                       └──requires──> Display window
                                          └──enhanced by──> Mirror/flip toggle
                                          └──enhanced by──> FPS counter
                                          └──enhanced by──> Swap on/off toggle

GPU (CUDA) provider ──enables──> "smooth" (25–30+ fps); falls back to ──> CPU (degraded ~5–15 fps)

Per-frame swap
    └──enhanced by (OPTIONAL, costly)──> Face restoration (GFPGAN/CodeFormer)   [~halves fps]
    └──enhanced by (OPTIONAL)─────────> Mouth mask / face-parsing blend          [extra per-frame model]
    └──tuned by──> Reduced-resolution processing      [boosts fps]
    └──tuned by──> Detect every N frames + track      [boosts fps]

Change photo without restart ──requires──> re-run (detect source + cache embedding) while live loop continues

[Virtual camera output] ──depends on──> stable per-frame swapped stream  (NEXT MILESTONE, not now)
```

### Dependency Notes

- **Cache source embedding is the hinge:** every live frame depends on it; it must be extracted at photo-load time and re-extracted on photo change. This is what makes "change photo without restart" cheap.
- **Pick-primary-face decouples source from "many faces in frame":** the source photo always resolves to exactly one face (largest); the *live* frame's multi-face handling is a separate, independent decision (swap-all-to-one vs swap-largest-only).
- **GPU path gates "real-time feel":** the resolution and frame-skip knobs are the fallback levers when GPU is unavailable or fps is short. They are not needed if the 3080 Ti delivers 25–30+ fps at default res.
- **Face restoration conflicts with the fps target:** it roughly halves fps, so it cannot be on-by-default without risking the "smooth" requirement. Hence: optional toggle, off by default.
- **Display sink is a seam for the next milestone:** keep the swapped-frame producer separate from the window consumer so a v4l2loopback sink can be added later without touching the pipeline.

## MVP Definition

### Launch With (v1 — the single phase)

- [ ] Load single target photo + detect + pick largest face + cache source embedding — the premise
- [ ] Webcam capture from `/dev/video0` (with device index configurable, even if hardcoded default) — input feed
- [ ] Per-frame detect → `inswapper` swap → paste back — the actual swap
- [ ] Live display window with Start/Stop — "shown in a window"
- [ ] Change target photo without restart — explicit requirement
- [ ] CUDA execution with automatic CPU fallback + surface active provider — explicit requirement / risk visibility
- [ ] FPS counter on screen — cheap, and the primary signal for the NVIDIA-driver risk
- [ ] Mirror/flip toggle + swap on/off toggle — cheap UX wins that make it usable
- [ ] No-face-this-frame passthrough — robustness so the window never freezes/crashes

### Add After Validation (v1.x — same milestone if time allows, else fast-follow)

- [ ] Reduced-resolution / det_size perf knob — add **if** fps falls short of 25–30
- [ ] Detect-every-N-frames + bbox reuse — add **if** detection is the bottleneck after the resolution knob
- [ ] Camera device selector UI — add **if** the default cam is wrong
- [ ] Multiple-faces-in-frame policy (swap-all-to-one) — add if more than one person commonly appears; default to swap-largest-only otherwise

### Future Consideration (explicitly later milestones)

- [ ] Virtual camera output (v4l2loopback) — **the declared next milestone**
- [ ] Face restoration enhancer (GFPGAN/CodeFormer) as an optional quality toggle — defer; only if default swap quality disappoints AND the GPU can spare the frame budget
- [ ] Mouth mask / face-parsing edge blending — quality polish for talking-on-camera
- [ ] Multi-target face-mapping UI — only if multi-person scenarios emerge (unlikely for personal use)

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Load photo + cache source embedding | HIGH | LOW | P1 |
| Webcam capture | HIGH | LOW | P1 |
| Per-frame detect + inswapper swap | HIGH | MEDIUM | P1 |
| Display window + Start/Stop | HIGH | LOW | P1 |
| Change photo without restart | HIGH | LOW | P1 |
| CUDA + CPU fallback (surface provider) | HIGH | MEDIUM | P1 |
| No-face passthrough robustness | HIGH | LOW | P1 |
| FPS counter | MEDIUM | LOW | P1 |
| Mirror/flip toggle | MEDIUM | LOW | P1/P2 |
| Swap on/off toggle | MEDIUM | LOW | P1/P2 |
| Reduced-res / frame-skip perf knobs | MEDIUM | LOW–MEDIUM | P2 (P1 if fps short) |
| Camera device selector | LOW–MEDIUM | LOW | P2 |
| Multi-face-in-frame policy | LOW–MEDIUM | LOW | P2 |
| Face restoration (GFPGAN/CodeFormer) | MEDIUM | MEDIUM | P3 (off by default) |
| Mouth mask / face-parsing blend | MEDIUM | MEDIUM–HIGH | P3 |
| Virtual camera output | HIGH (later) | MEDIUM | P3 — NEXT MILESTONE |
| Training / DFM | LOW (here) | HIGH | Anti-feature |
| Recording / streaming / web UI | LOW (here) | MEDIUM–HIGH | Anti-feature |

## Competitor Feature Analysis

| Feature | Deep-Live-Cam | FaceFusion (live) | DeepFaceLive | Our MVP |
|---------|---------------|-------------------|--------------|---------|
| Single-image swap | Yes (inswapper) | Yes (inswapper_128 + others) | Yes ("Insight" mode) | **Yes — only mode** |
| Training-based swap (DFM) | No | No | Yes (primary mode) | **No (anti-feature)** |
| Live webcam | Yes | Yes (~25fps@1080p) | Yes (≤25fps) | **Yes (target 25–30+ on GPU)** |
| GFPGAN/face enhancer | Yes (GFPGAN 1.4, toggle) | Yes (many models, toggle) | Color/align/blur sliders | **Optional toggle, off by default (defer)** |
| Mouth mask | Yes (`--mouth-mask`) | Has region masking | — | **Defer (differentiator)** |
| Mirror | Yes (`--live-mirror`) | — | — | **Yes (cheap)** |
| Multi-face / face mapping | Yes (`--many-faces`, `--map-faces`) | Yes | Per-face | **Swap-largest-only default; swap-all later** |
| Camera device select | Index-based | UDP/V4L2 | webcam/screen/file | **Configurable index** |
| FPS display | Not prominent in README | Implicit | Yes | **Yes (explicit)** |
| Virtual camera | Yes | Yes (V4L2) | Yes (OBS/Zoom/etc.) | **No — next milestone** |
| Recording/streaming | Yes (video) | Yes | OBS/stream | **No (anti-feature)** |
| Web/Gradio UI | Tkinter desktop | Gradio web UI | Qt desktop | **Native local window only** |

## Sources

- Deep-Live-Cam (closest reference: InsightFace inswapper + single image + live webcam; toggle list `--many-faces`/`--map-faces`/`--mouth-mask`/`--live-mirror`/`--live-resizable`, execution-provider/threads/max-memory) — https://github.com/hacksider/Deep-Live-Cam (HIGH)
- Deep-Live-Cam pipeline + GFPGAN cost note ("disabling it doubles frame rates", 24+fps needs ≤720p on weaker hardware) — https://starlog.is/articles/ai-dev-tools/hacksider-deep-live-cam/ , https://yuv.ai/blog/deep-live-cam (MEDIUM)
- FaceFusion live mode (~25fps@1080p, multiple swapper models, pixel boost, stacking processors multiplies time) — https://docs.facefusion.io/usage/cli-arguments/processors/face-swapper , https://docs.facefusion.io/usage/cli-arguments/processors/face-enhancer , https://magichour.ai/blog/how-to-use-facefusion (MEDIUM)
- DeepFaceLive (DFM training mode vs Insight single-image mode, ≤25fps, detectors, virtual cam, real-time sliders) — https://github.com/iperov/DeepFaceLive (HIGH)
- GFPGAN vs CodeFormer real-time cost (GFPGAN single forward pass; CodeFormer ~25fps only on ~35 TFLOPS GPU; seconds/image offline) — https://github.com/TencentARC/GFPGAN , https://www.technicalexplore.com/tech/gfpgan-vs-codeformer-the-ultimate-face-restoration-showdown (MEDIUM)
- InsightFace inswapper (source embedding extracted once and reused across frames; RetinaFace/YOLO detection + 106 landmarks; FaceAnalysis.get) — https://github.com/deepinsight/insightface , https://github.com/haofanwang/inswapper (HIGH)

---
*Feature research for: local real-time webcam single-image face-swap desktop app*
*Researched: 2026-05-29*
