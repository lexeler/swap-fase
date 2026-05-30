---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 01-02-PLAN.md (models + FaceEngine + still-swap on GPU)
last_updated: "2026-05-30T07:14:44.832Z"
last_activity: 2026-05-30
progress:
  total_phases: 1
  completed_phases: 0
  total_plans: 5
  completed_plans: 2
  percent: 40
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-29)

**Core value:** Открыл приложение → подключилась вебка → моё лицо в реальном времени заменено на лицо с загруженного фото, и это идёт плавно.
**Current focus:** Phase 01 — real-time-webcam-face-swap-mvp

## Current Position

Phase: 01 (real-time-webcam-face-swap-mvp) — EXECUTING
Plan: 3 of 5
Status: Ready to execute
Last activity: 2026-05-30

Progress: [████░░░░░░] 40%

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: —
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**

- Last 5 plans: —
- Trend: —

*Updated after each plan completion*
| Phase 01 P01 | 12min | 3 tasks | 8 files |
| Phase 01 P02 | 8min | 3 tasks | 5 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Single-image swap (InsightFace `inswapper`), no training — fits "my face onto the photo I loaded"; fast and simple for real-time.
- GPU/CUDA is the primary path; fix the NVIDIA driver FIRST (kernel/module mismatch) before any pipeline work.
- Isolated project-local venv + CUDA via pip (`onnxruntime-gpu[cuda,cudnn]==1.22.0`); never touch the global Python env.
- Whole MVP in ONE phase (user mandate); risk-first internal build order.
- Virtual camera + quality restorers are next milestones — leave only the `FrameSink` seam now.
- [Phase ?]: Stayed on Python 3.12 — insightface 1.0.1 installed with no C++ build (Pitfall 8 escape hatch not needed)
- [Phase ?]: Removed CPU-only onnxruntime (pulled by insightface) and reinstalled onnxruntime-gpu — only onnxruntime-gpu 1.22.0 in the venv
- [Phase ?]: HARD GPU GATE PASSED on RTX 3080 Ti: provider=CUDAExecutionProvider, cuDNN 92300, 203 MiB device-mem spike — real GPU not CPU fallback
- [Phase ?]: [Phase 1 P02]: Pinned fp32 inswapper_128 (not fp16) — published SHA256 e4a3f08c…16af gives fail-closed integrity (D-16); fp16 deferred as FPS lever (D-08)
- [Phase ?]: [Phase 1 P02]: FaceEngine builds analyser+swapper once; embed() caches largest source face (D-05), detect() keeps all faces (D-06), process() no-face passthrough (D-18); provider from active_provider()
- [Phase ?]: [Phase 1 P02]: Models in project-local models/ (root=MODELS_DIR); urllib mirror fetch + fail-closed hash; nothing in ~/.insightface

### Pending Todos

[From .planning/todos/pending/ — ideas captured during sessions]

None yet.

### Blockers/Concerns

[Issues that affect future work]

- **GPU driver down (current state):** `nvidia-smi` fails — kernel/module mismatch (module built for 6.17.0-22, booted 6.17.0-29). Must be fixed first (build step 1); gates "smooth real-time."
- **Silent CPU fallback risk:** onnxruntime can run on CPU while still listing CUDA. Runtime provider-verification gate (build step 2) is mandatory before pipeline work.
- **inswapper_128.onnx integrity:** community-hosted mirror; pin + assert SHA256 on download.
- **Webcam node:** `/dev/video0` may be metadata-only; probe-and-pick a capturable node at startup.
- Kernel 6.17.0-35 installed but NO nvidia module built for it — stay on 6.17.0-29 or build headers+dkms for -35 before rebooting, else nvidia-smi fails again

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| VCAM | Virtual camera output (v4l2loopback) for calls | Next milestone | 2026-05-29 (init) |
| QUAL | GFPGAN/CodeFormer restoration + edge blending | Next milestone | 2026-05-29 (init) |

## Session Continuity

Last session: 2026-05-30T07:14:44.823Z
Stopped at: Completed 01-02-PLAN.md (models + FaceEngine + still-swap on GPU)
Resume file: None
