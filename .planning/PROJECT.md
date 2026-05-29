# Swap-Fase

## What This Is

Локальное desktop-приложение для замены лица (face swap) в реальном времени на видеопотоке с веб-камеры. Пользователь загружает одно фото-цель — и его лицо в кадре с вебки заменяется на лицо с фото; результат отображается в окне приложения. Только для личного использования и экспериментов; ничего никуда не публикуется и не стримится.

## Core Value

Открыл приложение → подключилась вебка → моё лицо в реальном времени заменено на лицо с загруженного фото, и это идёт плавно. Если работает только это одно — проект успешен.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

(None yet — ship to validate)

### Active

<!-- Current scope. Building toward these. -->

- [ ] Загрузка одного фото-цели как источника лица для свопа
- [ ] Захват видео с веб-камеры (`/dev/video0`) в реальном времени
- [ ] Детекция лица пользователя в кадре и замена на лицо-цель (InsightFace `inswapper`, single-image, без обучения)
- [ ] Отображение свопнутого видеопотока в окне приложения (старт/стоп)
- [ ] Смена фото-цели из интерфейса без перезапуска
- [ ] Работа на GPU (CUDA) для плавного real-time (цель 25–30+ fps), с понятным fallback на CPU, если GPU недоступен

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- Вывод в виртуальную камеру для созвонов (Zoom/Meet/Discord) — отложено на следующую веху; пользователь явно сказал «сначала надо своп сделать». `v4l2loopback` уже загружен в системе, так что это лёгкое расширение позже.
- Обучение модели на наборе фото (подход DeepFaceLive/DFM) — для «поиграться» избыточно; single-image `inswapper` проще, быстрее и соответствует «лицо на то, которое я загрузил».
- Публикация, стриминг, запись на диск — не нужно, всё только локально.
- Мобильная или веб-версия — только локальный desktop на этой Linux-машине.

## Context

- **Машина:** ноутбук, Ubuntu (GNOME, Wayland), CPU i9-12900H (20 потоков), 31 ГБ RAM, гибридная графика Intel Iris Xe + NVIDIA RTX 3080 Ti Mobile.
- **⚠️ Главный риск:** драйвер NVIDIA сейчас не отвечает (`nvidia-smi` падает: «couldn't communicate with the NVIDIA driver»). Поднять драйвер + CUDA — первая по риску задача; без неё real-time не будет плавным (останется только CPU-fallback ~5–15 fps).
- **✅ Виртуальная камера на будущее:** модуль `v4l2loopback` уже загружен — следующая веха (вывод в созвоны) не потребует доустановки модуля.
- **Вебка:** определяется как `/dev/video0` (плюс `video1..3`).
- **Python:** 3.12.3 в системе (для проекта будет отдельный изолированный venv).
- **Типовой стек для задачи:** Python + InsightFace (`inswapper_128`) + onnxruntime-gpu + OpenCV; родственные открытые проекты для референса — roop / facefusion / DeepFaceLive.

## Constraints

- **Окружение**: всё ставится в изолированный project-local venv + локальные CUDA/cuDNN-пакеты (через pip/в папку проекта) — НЕ трогать глобальный/общий Python-venv. *(Прямое требование пользователя.)*
- **Платформа**: только локально на этой Linux-машине; модели скачиваются один раз, дальше работа офлайн.
- **Performance**: цель — плавный real-time на GPU; CPU допускается только как деградированный fallback.
- **Tech stack**: Python 3.12, InsightFace + onnxruntime-gpu (CUDA 12.x), OpenCV.
- **Security/Privacy**: ничего не публикуется и не уходит с машины; использование строго личное.
- **Granularity**: уложить в одну фазу — без растягивания на много фаз *(прямое требование пользователя)*.

## Key Decisions

<!-- Decisions that constrain future work. Add throughout project lifecycle. -->

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Single-image своп (InsightFace `inswapper`), без обучения модели | Соответствует «лицо на то, которое я загрузил»; быстро, просто, годится для real-time | — Pending |
| GPU/CUDA как основной путь; сначала починить драйвер NVIDIA | RTX 3080 Ti есть, но драйвер лежит; CUDA нужен для плавного real-time | — Pending |
| Изолированный project-local venv + CUDA-пакеты в папке проекта | Прямое требование: не засорять общий venv | — Pending |
| Виртуальная камера для созвонов — отдельная веха, потом | «Сначала надо своп сделать», делаем в одну фазу | — Pending |
| Одна фаза на весь MVP | Запрос пользователя на минимальную нарезку | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-29 after initialization*
