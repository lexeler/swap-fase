# Phase 1: Real-Time Webcam Face Swap (MVP) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-30
**Phase:** 1-real-time-webcam-face-swap-mvp
**Areas discussed:** Запуск и вид, Несколько лиц, Качество vs fps, Память фото

---

## Запуск и вид (Launch behavior)

| Option | Description | Selected |
|--------|-------------|----------|
| Старт по кнопке | Открывается с превью вебки, своп включаешь кнопкой «Старт». Не грузит GPU зря, предсказуемо. | ✓ |
| Своп сразу | Открыл — вебка и своп уже работают (если есть фото-цель из прошлого раза). | |
| Сначала выбор фото | При запуске сразу диалог выбора фото, потом превью и старт. | |

**User's choice:** Старт по кнопке
**Notes:** Defaults taken by Claude (stated, not asked): mirror ON by default (selfie view), windowed/resizable window. Both remain toggleable. Combined with "каждый раз заново" → no auto-loaded photo on launch.

---

## Несколько лиц (Multiple faces)

| Option | Description | Selected |
|--------|-------------|----------|
| Только крупнейшее | Свопится одно крупнейшее лицо (твоё), остальные не трогаем. Быстрее, предсказуемо. | ✓ |
| Крупнейшее + тумблер | По умолчанию крупнейшее, но есть переключатель «все лица». | |
| Все лица | Свопить все обнаруженные лица на одно целевое (дороже по fps). | |

**User's choice:** Только крупнейшее
**Notes:** No "all faces" toggle in v1. Engine may still return all detections internally so a future toggle is cheap, but it is not surfaced.

---

## Качество vs fps (Quality vs FPS)

| Option | Description | Selected |
|--------|-------------|----------|
| Плавность | Цель 25-30+ fps; при просадке понижаем разрешение обработки / детект реже. Качество — нативное inswapper_128. | |
| Баланс, fps-флор | Полное разрешение/детект каждый кадр, но не ниже ~20 fps. | |
| Качество важнее | Полное качество, fps как получится (GFPGAN — отдельной вехой). | |

**User's choice:** Free text — "главное большой фпс" (FPS is the top priority — stronger than the "Плавность" option)
**Notes:** Maximize frame rate above visual fidelity. Allowed levers: inswapper_128_fp16, reduced processing resolution, smaller det_size, detect-every-N + bbox reuse. No face-restoration enhancer in this phase.

---

## Память фото (Target photo persistence)

| Option | Description | Selected |
|--------|-------------|----------|
| Помнить последнее | При запуске подхватывает последнее использованное фото. | |
| Список избранных | Панель недавних/избранных фото для переключения в один клик. | |
| Каждый раз заново | Никакой памяти; при старте грузишь фото вручную. | ✓ |

**User's choice:** Каждый раз заново
**Notes:** No persistence between runs and no favorites panel. In-session change-target-without-restart (UI-03) still applies.

---

## Claude's Discretion

- Exact UI layout, widget arrangement, button labels, badge placement.
- Camera device probing/auto-pick (`/dev/video0..3`, default `video0`) with a picker.
- Threading/queue/lock implementation details, module layout, error wording.
- `inswapper_128_fp16` vs `inswapper_128` default (lean fp16 for fps; confirm at smoke test).

## Deferred Ideas

- Virtual camera output for calls (VCAM-01/02) — next milestone; leave `FrameSink` seam only.
- Face restoration enhancer (QUAL-01, GFPGAN/CodeFormer) — conflicts with "fps is king".
- Edge/mouth-mask blending (QUAL-02).
- "All faces" swap toggle.
- Favorites/recents photo panel and remember-last-photo.
