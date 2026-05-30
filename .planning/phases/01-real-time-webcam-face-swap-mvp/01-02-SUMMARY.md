---
phase: 01-real-time-webcam-face-swap-mvp
plan: 02
subsystem: infra
tags: [insightface, inswapper, buffalo_l, onnxruntime-gpu, sha256, model-integrity, opencv, cuda]

# Dependency graph
requires:
  - phase: 01-01
    provides: "isolated .venv (onnxruntime-gpu 1.22.0, insightface 1.0.1, opencv 4.13, numpy 2.2.1), src/swapfase/providers.py (select_providers/active_provider/preload_cuda_libs/verify_gpu), verified GPU (RTX 3080 Ti, CUDAExecutionProvider)"
provides:
  - "src/swapfase/bootstrap.py — MODELS_DIR (project-local models/), ensure_models() (buffalo_l auto-download + inswapper mirror fetch + fail-closed SHA256 verify), ModelIntegrityError, EXPECTED_INSWAPPER_SHA256"
  - "src/swapfase/engine.py — FaceEngine (analyser + swapper built once), embed() (largest-face cache + NoFaceError), detect() (all faces, D-06), process() (swap largest + no-face passthrough, D-18); self.provider from active_provider()"
  - "scripts/swap_still.py — zero-threading still→still swap proving the model path end-to-end on GPU"
  - "tests/test_engine.py — behaviour tests for FaceEngine on real bundled face images"
  - "models/inswapper_128.onnx (fp32, SHA256-verified) + models/models/buffalo_l/ (project-local, gitignored)"
affects: [01-03 capture+pipeline, 01-04 UI, 01-05 robustness, live-pipeline, FaceEngine-consumers]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Fail-closed model integrity: pinned published SHA256, mismatch → delete + ModelIntegrityError (never load a tampered model)"
    - "Build-once inference sessions: analyser + swapper constructed once in __init__, reused every call (never per-frame)"
    - "Provider honesty: self.provider read from the live session via active_provider(), not the requested list"
    - "Largest-face pick by bbox area (D-05); detect() keeps all faces (D-06); no-face passthrough (D-18)"

key-files:
  created:
    - "src/swapfase/bootstrap.py"
    - "src/swapfase/engine.py"
    - "scripts/swap_still.py"
    - "tests/test_engine.py"
    - "models/.gitkeep"
  modified: []

key-decisions:
  - "Pinned the fp32 inswapper_128.onnx build (~554 MB) — NOT fp16 — because its SHA256 (e4a3f08c…16af) is a published, cross-mirror-verified value, giving GENUINE fail-closed integrity (D-16); fp16 remains a deferred FPS lever (D-08) that requires sourcing the fp16 build's own published checksum."
  - "Models live in project-local models/ via FaceAnalysis(root=MODELS_DIR); inswapper fetched with urllib (huggingface_hub not installed) over a mirror list, fail-closed by hash."
  - "Test fixtures use insightface's bundled t1.jpg (6 detectable faces); Tom_Hanks_54745.png is a 112×112 pre-aligned recognition crop the detector legitimately ignores, so it is not a detection fixture."

patterns-established:
  - "Model bootstrap separate from engine: integrity/acquisition is a distinct startup concern from inference."
  - "TDD for the engine: RED (failing import) → GREEN (implementation) on real GPU + real face images."

requirements-completed: [ENV-04, SWAP-01, SWAP-02]

# Metrics
duration: 8min
completed: 2026-05-30
---

# Phase 01 Plan 02: Models + FaceEngine + Still-Swap Summary

**Project-local SHA256-verified inswapper_128 (fp32) + buffalo_l, a build-once FaceEngine (largest-face embed, no-face passthrough) running on CUDA, and a still→still swap that writes a genuinely different image on the GPU.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-05-30T07:03:52Z
- **Completed:** 2026-05-30T07:12:00Z
- **Tasks:** 3 (Task 2 via TDD: RED → GREEN)
- **Files modified:** 5 created (4 source/test + models/.gitkeep)

## Accomplishments

- **Fail-closed model acquisition (ENV-04):** `bootstrap.ensure_models()` lands `buffalo_l` (5 onnx files) in project-local `models/models/buffalo_l/` (auto-download via `FaceAnalysis(root=MODELS_DIR)`) and fetches `inswapper_128.onnx` from a community mirror, then SHA256-verifies it against the pinned `EXPECTED_INSWAPPER_SHA256` and **deletes + raises `ModelIntegrityError` on mismatch**. Nothing lands in `~/.insightface`. Network only on first run.
- **FaceEngine on GPU (SWAP-01/02):** `engine.py` builds the buffalo_l analyser and the inswapper swapper exactly once, reads the real bound provider from the live session (`active_provider`), and exposes `embed()` (largest face by bbox area, cached source, `NoFaceError` on none — D-05), `detect()` (ALL faces — D-06), and `process()` (swap largest, no-face passthrough — D-18). 7/7 behaviour tests pass on `CUDAExecutionProvider`.
- **End-to-end still swap proven (SWAP-02):** `scripts/swap_still.py` ran on two distinct face inputs and wrote `swapped.jpg` that **differs from the input** (12,072 changed pixels in the swapped face region), printing `provider=CUDAExecutionProvider`, with an `nvidia-smi` utilization spike during the swap (peak 53 % util, ~1877 MiB device memory).

## Task Commits

1. **Task 1: bootstrap.py — model acquisition + SHA256 verify** - `d0a91b3` (feat)
2. **Task 2: engine.py — FaceEngine (TDD)** - `d9d8ca3` (test, RED) → `62e068e` (feat, GREEN)
3. **Task 3: swap_still.py — still→still swap on GPU** - `4562f36` (feat)

**Plan metadata:** committed after this summary (docs: complete plan)

_TDD gate sequence satisfied: `test(...)` (d9d8ca3) precedes `feat(...)` (62e068e); no REFACTOR needed (engine was clean at GREEN)._

## Model Build / Integrity Record (per plan `<output>`)

- **Build chosen:** `inswapper_128.onnx` **fp32** (554,253,681 bytes ≈ 529 MiB).
- **Pinned SHA256:** `e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af` (published roop/FaceFusion-lineage hash; cross-mirror-verified — Pitfall 5). `sha256sum models/inswapper_128.onnx` matches exactly.
- **Mirror used (first reachable):** `https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx` (mirror list also includes deepinsight/inswapper and facefusion/facefusion-assets as fallbacks).
- **fp16 note (D-07/D-08):** fp16 was NOT used. fp16 is a valid later FPS lever, but adopting it requires the fp16 build's own *published* checksum (re-hashing a downloaded file proves nothing). Integrity (D-16) is the hard gate; fp16 tuning is deferred to the live pipeline plans.
- **License:** non-commercial / research only; this project's strictly personal, local, non-published use fits the terms (documented in the bootstrap module docstring; threat T-01-09 accepted).

## swap_still.py invocation (recorded)

```bash
SITE=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
LD_LIBRARY_PATH="$SITE/nvidia/cudnn/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/curand/lib:\
$SITE/nvidia/cufft/lib:$SITE/nvidia/cuda_nvrtc/lib:$SITE/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}" \
  PYTHONPATH=src .venv/bin/python scripts/swap_still.py \
    --source <largest-face crop of t1.jpg, 60% margin, re-detectable as 1 face> \
    --target-frame <full t1.jpg, 6 faces> \
    --out swapped.jpg
# → provider=CUDAExecutionProvider ; wrote swapped.jpg ; output differs from input (12,072 px)
```

Inputs were derived from insightface's bundled `t1.jpg` (no external download needed): the largest detected face was cropped with margin to make a re-detectable single-face *source*, and the full group photo was the *target-frame*. A faceless source was also exercised → friendly `no face found in source photo` message, exit 1, no traceback, no output file.

## Files Created/Modified

- `src/swapfase/bootstrap.py` - MODELS_DIR, ensure_models(), _verify_or_raise (fail-closed SHA256), mirror download, ModelIntegrityError; license/scope docstring.
- `src/swapfase/engine.py` - FaceEngine (build-once analyser+swapper), embed/detect/process, NoFaceError, provider honesty via active_provider().
- `scripts/swap_still.py` - zero-threading CLI still→still swap; friendly errors on unreadable image + NoFaceError.
- `tests/test_engine.py` - 7 behaviour tests on real bundled face images (run with venv python; pytest intentionally not a runtime dep).
- `models/.gitkeep` - tracks the models/ dir (weights stay gitignored).

## Decisions Made

- **fp32 over fp16** for genuine published-hash integrity (see Model Build record). The integrity requirement (D-16) outranks the fp16 FPS lever (D-08) at this stage.
- **urllib mirror download** instead of `huggingface_hub` (not installed in the pinned venv) — keeps the dependency set unchanged; fail-closed by hash makes the untrusted mirror safe.
- **`LD_LIBRARY_PATH` extended** with `curand`, `cufft`, and `cuda_nvrtc` lib dirs (beyond run.sh's cudnn/cublas/cuda_runtime) to silence non-fatal `libcurand.so.10`/`libnvrtc.so.12` dlopen warnings during analyser warm-up. CUDA still bound without them (preload_dlls covers the EP); noted for the launcher in 01-03.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test fixture used a non-detectable image**
- **Found during:** Task 2 (engine.py GREEN run)
- **Issue:** The initial test used `Tom_Hanks_54745.png` as the single-face fixture; it is a 112×112 *pre-aligned recognition crop* with no surrounding context, so RetinaFace correctly does not detect a face — 3 tests failed with `NoFaceError`. The engine was correct; the fixture was wrong.
- **Fix:** Switched all face-bearing fixtures to insightface's bundled `t1.jpg` (6 detectable faces); the engine's largest-face pick + embedding + swap then verified genuinely.
- **Files modified:** tests/test_engine.py (committed with GREEN)
- **Verification:** 7/7 tests pass on `CUDAExecutionProvider`.
- **Committed in:** 62e068e (Task 2 GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 bug — test fixture). No production-code deviations.
**Impact on plan:** Engine implemented exactly as specified. The fixture fix made the TDD assertions exercise a real, detectable face. No scope creep.

## Issues Encountered

- **Non-fatal CUDA dlopen warnings** (`libcurand.so.10`, `libnvrtc.so.12` "cannot open shared object file") appeared during `FaceAnalysis.prepare()` warm-up when the LD_LIBRARY_PATH lacked those dirs. CUDA still bound (`preload_dlls` loads the EP); resolved by adding `curand`/`cufft`/`cuda_nvrtc` to the path for the verify/test/run commands. Flagged for the 01-03 launcher.

## User Setup Required

None required — the inswapper mirror download succeeded automatically (fail-closed by hash). If a future clean run hits a rate-limited/blocked mirror, manually place the fp32 `inswapper_128.onnx` (SHA256 `e4a3f08c…16af`) into `models/` and re-run; `ensure_models()` will verify and proceed.

## Next Phase Readiness

- **Ready for 01-03 (capture + keep-newest pipeline):** `FaceEngine.embed()`/`process()` and verified models are the inference core the live loop consumes. The model path is proven on static images with zero threading.
- **Carry-forward for 01-03:** extend the `run.sh` `LD_LIBRARY_PATH` to include `nvidia/{curand,cufft,cuda_nvrtc}/lib` so analyser warm-up is warning-clean.

## Self-Check: PASSED

- Files: bootstrap.py, engine.py, swap_still.py, test_engine.py, models/.gitkeep, 01-02-SUMMARY.md — all FOUND.
- Commits: d0a91b3 (Task 1), d9d8ca3 (Task 2 RED), 62e068e (Task 2 GREEN), 4562f36 (Task 3) — all FOUND.
- Models: models/inswapper_128.onnx (SHA256-verified fp32) + models/models/buffalo_l/ present in project-local models/, none in ~/.insightface.

---
*Phase: 01-real-time-webcam-face-swap-mvp*
*Completed: 2026-05-30*
