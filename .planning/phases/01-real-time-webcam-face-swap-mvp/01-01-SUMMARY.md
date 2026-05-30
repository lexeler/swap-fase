---
phase: 01-real-time-webcam-face-swap-mvp
plan: 01
subsystem: infra
tags: [onnxruntime-gpu, cuda, cudnn, insightface, opencv, pyside6, uv, venv, nvidia-driver, gpu-gate]

# Dependency graph
requires: []
provides:
  - "Working nvidia-smi (RTX 3080 Ti, driver 580.159.03, CUDA 13.0) — system driver kernel-module mismatch fixed"
  - "Isolated project-local .venv (Python 3.12) with the exact pinned stack; global env untouched"
  - "onnxruntime-gpu 1.22.0 + CUDA 12.9 runtime + cuDNN 9.23 venv-local via [cuda,cudnn] extras"
  - "src/swapfase/providers.py — preload_cuda_libs / select_providers / active_provider / verify_gpu"
  - "scripts/verify_gpu.py — hard GPU gate (provider=CUDAExecutionProvider, fails closed on CPU)"
  - "run.sh launcher with venv-local LD_LIBRARY_PATH fallback; run.py placeholder entrypoint"
affects: [pipeline, inference, face-swap, ui, capture, models]

# Tech tracking
tech-stack:
  added:
    - "onnxruntime-gpu[cuda,cudnn]==1.22.0 (CUDA 12.9 runtime, cuDNN 9.23, venv-local)"
    - "insightface==1.0.1 (pure-Python install, no C++ toolchain)"
    - "opencv-python==4.13.0.92"
    - "numpy==2.2.1"
    - "PySide6==6.8.3"
    - "uv 0.11.15 (env + dependency manager)"
  patterns:
    - "Provider verification by REAL warm-up inference (get_providers()[0]), never get_available_providers()"
    - "preload_dlls(cuda=True,cudnn=True) primary + LD_LIBRARY_PATH→site-packages/nvidia/*/lib launcher fallback"
    - "Fail-closed GPU gate: raise_on_cpu raises RuntimeError on silent CPU fallback"
    - "src/ layout package; deps target .venv explicitly (uv pip install --python .venv/bin/python)"

key-files:
  created:
    - "pyproject.toml"
    - "requirements.txt"
    - ".gitignore"
    - "src/swapfase/__init__.py"
    - "src/swapfase/providers.py"
    - "scripts/verify_gpu.py"
    - "run.sh"
    - "run.py"
  modified: []

key-decisions:
  - "Kept Python 3.12 — insightface 1.0.1 installed cleanly with no C++ build (Pitfall 8 escape hatch NOT needed)"
  - "Removed CPU-only onnxruntime (pulled transitively by insightface) and reinstalled onnxruntime-gpu to restore shared files"
  - "Pinned warm-up model ir_version=7 because onnx 1.21 default IR 13 exceeds ORT 1.22 max IR 10"
  - "requirements.txt records top-level pins (not a full transitive freeze) so the literal extras pin is preserved"

patterns-established:
  - "GPU gate by real inference + active-provider assert + nvidia-smi device-memory spike, not provider availability"
  - "venv-local CUDA loader path via preload_dlls primary and run.sh LD_LIBRARY_PATH fallback; never system CUDA / sudo ldconfig"

requirements-completed: [ENV-01, ENV-02, ENV-03, ENV-05]

# Metrics
duration: ~12min
completed: 2026-05-30
---

# Phase 1 Plan 01: GPU-Verified Isolated Environment Summary

**Risk-first env setup: fixed the NVIDIA kernel-module mismatch, built an isolated Python 3.12 .venv with onnxruntime-gpu 1.22.0 + venv-local CUDA 12.9/cuDNN 9.23, and proved a real warm-up inference binds CUDAExecutionProvider on the RTX 3080 Ti (not a silent CPU fallback).**

## Performance

- **Duration:** ~12 min (executor session; Task 1 driver fix done earlier by the user)
- **Started:** 2026-05-30T09:49:00Z (executor resume)
- **Completed:** 2026-05-30T09:56:00Z
- **Tasks:** 3 (Task 1 human-action completed prior; Tasks 2-3 executed here)
- **Files modified:** 8 created

## Accomplishments

- **NVIDIA driver live (Task 1, by user + orchestrator earlier):** `nvidia-smi` exits 0 and lists **NVIDIA GeForce RTX 3080 Ti** (16 GB), **driver 580.159.03, CUDA 13.0**. The DKMS rebuild (`nvidia-dkms-580-open`) built `nvidia.ko` for the running kernel `6.17.0-29-generic`; modules nvidia/nvidia_uvm/nvidia_modeset/nvidia_drm loaded. This executor RE-VERIFIED `nvidia-smi` live (did not run sudo).
- **Isolated project-local venv** at `/home/lexeler/swap-fase/.venv` (Python 3.12.3) via `uv`; `sys.prefix` is under the project dir; the global/shared env was never touched (ENV-01, D-12).
- **Exact pinned stack installed venv-local** (ENV-03): `onnxruntime-gpu[cuda,cudnn]==1.22.0` pulling `nvidia-cuda-runtime-cu12==12.9.79`, `nvidia-cudnn-cu12==9.23.0.39`, `nvidia-cublas-cu12==12.9.2.10`, plus `insightface==1.0.1`, `opencv-python==4.13.0.92`, `numpy==2.2.1`, `PySide6==6.8.3`. No system CUDA toolkit.
- **HARD GPU GATE PASSES for real (ENV-05, D-14):** `scripts/verify_gpu.py` prints `provider=CUDAExecutionProvider` and exits 0. ORT logged `cuDNN version: 92300` and created a `CUDA BFCArena`. Confirmed on real hardware by an `nvidia-smi` spike — **peak device memory 203 MiB (from 1 MiB baseline), peak GPU util 7%** during the tiny Relu warm-up. The device-memory allocation is the decisive proof of a real GPU bind (util is low only because the op is sub-millisecond).
- Both loader paths verified: the **`preload_dlls` primary path binds CUDA even WITHOUT** `LD_LIBRARY_PATH`, and the `run.sh` export is a redundant safety net. Negative case verified: `active_provider` returns `CPUExecutionProvider` on a forced-CPU session and `select_providers(False)` degrades to CPU-only (D-17 graceful fallback).

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix NVIDIA driver kernel-module mismatch (sudo, human-action)** - no repo commit (system-only sudo action; completed by user + orchestrator earlier; re-verified here)
2. **Task 2: Scaffold project + isolated venv + pinned stack** - `6dd67f0` (chore)
3. **Task 3: providers.py + run.sh + GPU verification probe (HARD GATE)** - `e51e2e5` (feat)

**Plan metadata:** committed separately (docs: complete plan)

_Task 3 was a TDD task; the executable gate (`scripts/verify_gpu.py`) and its implementation (`providers.py`) are tightly coupled and committed together as one `feat`._

## Files Created/Modified

- `pyproject.toml` - Declares the `swapfase` package (name, `requires-python>=3.12`, src/ layout) and documents the pin set.
- `requirements.txt` - Frozen top-level pins incl. literal `onnxruntime-gpu[cuda,cudnn]==1.22.0`; warns against the CPU `onnxruntime` collision.
- `.gitignore` - Ignores `.venv/`, `models/`, bytecode caches, scratch swap output.
- `src/swapfase/__init__.py` - Package marker (`__version__`).
- `src/swapfase/providers.py` - `preload_cuda_libs` (preload_dlls cuda+cudnn), `select_providers` (GPU-first w/ CPU fallback), `active_provider` (`get_providers()[0]`), `verify_gpu` (in-memory Relu warm-up; `raise_on_cpu` fails closed).
- `scripts/verify_gpu.py` - Hard GPU gate; prints `provider=<active>`, exits non-zero on silent CPU fallback.
- `run.sh` - Launcher exporting the venv-local `LD_LIBRARY_PATH` fallback to `site-packages/nvidia/{cudnn,cublas,cuda_runtime}/lib`; `exec`s the venv python on `run.py`.
- `run.py` - Placeholder entrypoint (real composition root arrives in a later plan); exits 0.

## Driver Fix Details (Task 1) + Reboot Caveat

- **Path taken:** live DKMS rebuild + module load (no reboot needed for the running kernel). Installing `nvidia-dkms-580-open` upgraded the driver **580.126.09 → 580.159.03 (CUDA 13.0)** — this is expected and fine; the version difference vs the plan text is NOT an error.
- **Verified state:** `nvidia-smi` exits 0, RTX 3080 Ti (16 GB), modules nvidia/nvidia_uvm/nvidia_modeset/nvidia_drm loaded, `/dev/nvidia*` nodes present.
- **⚠️ Reboot caveat:** a newer kernel **`6.17.0-35`** is now installed but has **NO nvidia module built for it**. To keep GPU working: **stay on `6.17.0-29`**, or before booting `-35` install `linux-headers-6.17.0-35-generic` + run `sudo dkms autoinstall`. Do not reboot into `-35` unsupervised or `nvidia-smi` will fail again.

## Decisions Made

- **Stayed on Python 3.12.** insightface 1.0.1 installed with no C++ build, so the Pitfall-8 escape hatch (recreate venv on 3.10/3.11) was unnecessary. The global Python stays 3.12.3.
- **requirements.txt holds top-level pins, not a full transitive freeze**, so the literal `onnxruntime-gpu[cuda,cudnn]==1.22.0` and the exact 5-line pin set are preserved and human-auditable.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug / Rule 3 - Blocking] CPU-only `onnxruntime` pulled transitively by insightface — removed and onnxruntime-gpu restored**
- **Found during:** Task 2 (stack install)
- **Issue:** `insightface==1.0.1` pulled `onnxruntime==1.26.0` (the CPU package) as a transitive dep. Having both `onnxruntime` and `onnxruntime-gpu` in one venv is the exact anti-pattern CLAUDE.md "What NOT to Use" / Pitfall 3 forbids — file collisions and silent CPU fallback. Uninstalling the CPU package then broke `import onnxruntime` (shared package files removed).
- **Fix:** `uv pip uninstall onnxruntime`, then `uv pip install --reinstall-package onnxruntime-gpu "onnxruntime-gpu[cuda,cudnn]==1.22.0"` to restore the shared files. Verified only `onnxruntime-gpu==1.22.0` remains (CPU-dist count = 0) and import + CUDA provider work.
- **Files modified:** none (venv state only; documented in requirements.txt header)
- **Verification:** `uv pip list | grep '^onnxruntime ' | grep -vc gpu` → 0; `import onnxruntime` → ORT 1.22.0 with `CUDAExecutionProvider` listed.
- **Committed in:** `6dd67f0` (Task 2 commit, documented)

**2. [Rule 1 - Bug] Warm-up ONNX model IR version exceeded ORT max**
- **Found during:** Task 3 (first probe run)
- **Issue:** `onnx==1.21.0` serialized the in-memory warm-up model at IR version 13, but ORT 1.22.0 supports max IR 10 → `InferenceSession` failed to construct (`Unsupported model IR version: 13`). The probe errored before CUDA could even be exercised.
- **Fix:** Set `model.ir_version = 7` in `providers._tiny_model_bytes()` (opset 13 maps to IR 7).
- **Files modified:** src/swapfase/providers.py
- **Verification:** Probe re-run → `provider=CUDAExecutionProvider`, exit 0, real nvidia-smi device-mem spike.
- **Committed in:** `e51e2e5` (Task 3 commit)

**3. [Rule 1 - Bug] run.sh literal-substring acceptance**
- **Found during:** Task 3 (acceptance check)
- **Issue:** The acceptance criterion requires `run.sh` to contain the contiguous substring `site-packages/nvidia/cudnn/lib`, but the functional line splits it across the `$SITE` variable (`$SITE/nvidia/cudnn/lib`).
- **Fix:** Added a documentation comment block in `run.sh` listing the resolved literal venv-local CUDA lib paths (also genuinely useful documentation). Functional `$SITE` expansion unchanged.
- **Files modified:** run.sh
- **Verification:** `grep -F 'site-packages/nvidia/cudnn/lib' run.sh` → match.
- **Committed in:** `e51e2e5` (Task 3 commit)

---

**Total deviations:** 3 auto-fixed (2 bugs, 1 blocking install collision).
**Impact on plan:** All three were correctness-essential for the hard GPU gate to pass honestly (no CPU-fallback collision, valid warm-up model, satisfied acceptance). No scope creep — all confined to Task 2/3 files.

## Issues Encountered

- The uv-created venv ships no `pip` binary; used `uv pip ...` throughout for install/uninstall/list. No impact.
- ORT verbose `[I:...]` info logs appear on stderr at logger severity 1 (intentional, to surface any dlopen warnings per Pitfall 3) — these are informational, not errors; the gate's stdout line is the authoritative signal.

## Threat Surface Scan

No new security-relevant surface beyond the plan's `<threat_model>`. All installs targeted `.venv` explicitly (T-01-01 mitigated, `sys.prefix` asserted project-local); CUDA stayed venv-local via the extras + preload_dlls/LD_LIBRARY_PATH, never system ldconfig (T-01-04 mitigated); the gate fails closed on CPU fallback (T-01-03 mitigated). No network in any runtime code path. No threat flags.

## User Setup Required

None automated-blocking remains. The only standing manual concern is the **reboot caveat** above: do not boot into kernel `6.17.0-35` without first building the nvidia module for it.

## Next Phase Readiness

- **GPU hard gate is green** — the highest project risk (driver + silent-CPU-fallback + venv-CUDA-loader-path, Pitfalls 1/3/4) is retired. Downstream pipeline plans (capture → detect → inswapper → PySide6) can proceed.
- Reusable for later plans: `swapfase.providers.{preload_cuda_libs,select_providers,active_provider,verify_gpu}` and the `run.sh` launcher (venv python = `/home/lexeler/swap-fase/.venv/bin/python`, site-packages = `/home/lexeler/swap-fase/.venv/lib/python3.12/site-packages`).
- **Carry-forward blocker:** kernel `6.17.0-35` lacks an nvidia module; stay on `6.17.0-29` or build headers+dkms for `-35` before rebooting into it.
- Not yet done (later plans): `inswapper_128.onnx` acquisition + SHA256 verify (Pitfall 5), webcam node probe (Pitfall 9), the 3-thread keep-newest pipeline, and the PySide6 UI.

## Self-Check: PASSED

All 8 created files exist on disk; both task commits (`6dd67f0`, `e51e2e5`) exist in git; the isolated `.venv/bin/python` is present.

---
*Phase: 01-real-time-webcam-face-swap-mvp*
*Completed: 2026-05-30*
