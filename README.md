# swap-fase

**Real-time webcam face swap onto a photo of your choice — output as a virtual camera you can pick in Zoom, Google Meet, or Discord.**

swap-fase is a local desktop app. You pick one front-facing photo, point it at your webcam, and your face in the live preview is swapped for the face in that photo (InsightFace `inswapper_128` running on whatever GPU/CPU your machine has). The swapped stream is shown in a PySide6 window and, optionally, published to a virtual camera so any video-call app can use it as a regular webcam.

Everything runs **locally and offline** after a one-time model download. Nothing is published, uploaded, or streamed anywhere by the app itself.

```
   ┌─────────┐     ┌──────────────┐     ┌─────────────────┐
   │ webcam  │ ──▶ │  face swap   │ ──▶ │ preview window  │
   │ capture │     │ (inswapper)  │     │   +  virtual    │
   └─────────┘     └──────────────┘     │   camera (opt)  │
        ▲                 ▲             └─────────────────┘
        │            target photo               │
   /dev/videoN      (your chosen face)     "DeepLiveCam" / "OBS Virtual Camera"
                                            → selectable in Zoom / Meet / Discord
```

> Demo placeholder — no sample face is shipped in this repo. Supply your own photo (see [Usage](#usage)).

---

## Status / caveats

- This is a **GUI desktop application**, not a CLI tool or a service. It opens a window and a live webcam preview.
- It was **developed and is actively used on Linux + NVIDIA (CUDA)**. That path is well exercised.
- The **Windows install path is provided and follows the documented cross-platform contract, but it was authored on Linux** and has had less real-world testing. If something breaks on Windows, please [open an issue](../../issues) with the error output — it's the path most likely to need a fix.
- macOS support is **best-effort** (CoreML or CPU).
- The face-swap **models are not bundled** — they are downloaded at runtime and carry their own non-commercial license. See [License](#license).

---

## Quick install

swap-fase installs into an isolated, project-local virtual environment. The base
dependencies live in `requirements/base.txt` and deliberately **exclude
onnxruntime** — the platform installer picks the correct inference runtime for
your hardware (CUDA / DirectML / ROCm / CoreML / CPU). See
[Hardware & acceleration](#hardware--acceleration).

### Windows

Requires **Python 3.12** and, for the virtual camera, **OBS Studio's Virtual Camera**
(install [OBS Studio](https://obsproject.com/) once and start its Virtual Camera at least
once, or install [Unity Capture](https://github.com/schacluck/UnityCapture)).

```powershell
# from the project root, in PowerShell
.\install.ps1
.\.venv\Scripts\python run.py --target path\to\face.jpg
```

The Windows installer pulls `onnxruntime-directml`, which runs on **any** Windows GPU
(NVIDIA, AMD, or Intel) and falls back to CPU automatically.

### Linux

Requires **Python 3.12**. For the virtual camera you need the `v4l2loopback`
kernel module (see [Troubleshooting](#troubleshooting)).

```bash
# from the project root
./install.sh
./run.sh --target path/to/face.jpg
```

The Linux installer detects your GPU and installs the matching onnxruntime:
NVIDIA → `onnxruntime-gpu[cuda,cudnn]==1.22.0`, AMD → `onnxruntime-rocm`
(may need a custom index), Intel / no-GPU → CPU `onnxruntime`.

On Linux + NVIDIA, always launch via **`./run.sh`** (not `python run.py` directly):
it exports the venv-local CUDA `LD_LIBRARY_PATH` first so onnxruntime's CUDA
provider finds the pip-installed CUDA/cuDNN libraries.

### Docker (Linux + NVIDIA only)

```bash
# build
docker build -t swap-fase .

# run (requires the NVIDIA Container Toolkit; host webcam + virtual camera passed through)
docker run --rm -it --gpus all \
  --device /dev/video0 --device /dev/video10 \
  -v "$PWD/models:/app/models" \
  swap-fase --target /path/to/face.jpg --vcam
```

Docker support is **Linux + NVIDIA only** — it relies on the NVIDIA Container
Toolkit and host V4L2 device passthrough.

---

## Hardware & acceleration

swap-fase runs on whatever accelerator is best available. The onnxruntime
**execution-provider (EP)** priority is, in order:

```
TensorRT → CUDA → ROCm → OpenVINO → DirectML → CoreML → CPU
```

At startup the app filters this list to the providers your installed onnxruntime
wheel actually exposes, picks the best one present, and logs the chosen provider.
There is **no hard CPU failure** in the app path — if nothing better is available
it runs on CPU (slower, but it runs).

| Platform | Hardware | onnxruntime installed | Provider used |
| --- | --- | --- | --- |
| **Windows** | Any GPU (NVIDIA / AMD / Intel) | `onnxruntime-directml` | DirectML (CPU fallback built in) |
| **Linux** | NVIDIA | `onnxruntime-gpu[cuda,cudnn]==1.22.0` | CUDA (TensorRT if present) |
| **Linux** | AMD | `onnxruntime-rocm` (custom index) or CPU `onnxruntime` | ROCm or CPU |
| **Linux** | Intel / integrated | `onnxruntime` | OpenVINO if configured, else CPU |
| **Linux** | no GPU | `onnxruntime` | CPU |
| **macOS** | Apple Silicon | `onnxruntime-silicon` or `onnxruntime` | CoreML or CPU |

> **Performance:** GPU providers give smooth real-time; CPU is a usable but
> degraded fallback. If FPS is low, lower the capture resolution (see below).

---

## Usage

```bash
# Linux + NVIDIA
./run.sh --target path/to/face.jpg

# Windows
.\.venv\Scripts\python run.py --target path\to\face.jpg

# any platform, directly (no CUDA LD_LIBRARY_PATH export — fine for non-CUDA EPs)
python run.py --target path/to/face.jpg
```

### Picking a good target photo

- Use a **clear, well-lit, front-facing** photo with one obvious face.
- **Leave margin around the head.** A tightly-cropped portrait where the face
  fills the entire frame may not be detected — the detector needs some context
  around the face. If you get "no face found", try a less-cropped version.
- The photo's **largest** face is detected once at startup and cached.

### Flags

| Flag | Meaning |
| --- | --- |
| `--target PATH` | the photo whose face is worn (**required**; largest face detected once and cached). |
| `--device N` | input camera index to capture from (cross-platform integer index; default: first capturable camera). |
| `--vcam` | also publish the swapped stream to a virtual camera so call apps can use it. |
| `--vcam-device PATH` | virtual-camera device. Linux default `/dev/video10`; Windows/macOS default auto-picked by the OBS backend. |
| `--vcam-mirror` | mirror the virtual-camera output too (default: the call sees you un-mirrored / the right way round). |
| `--cpu` | force CPU (skip GPU) — for debugging the graceful-fallback path. |

### Capture resolution

The capture backend is chosen per OS automatically: **DirectShow/MSMF on
Windows, V4L2 on Linux, AVFoundation on macOS**. The resolution defaults to
**640×480** for FPS headroom and is overridable with environment variables:

```bash
# sharper picture (trades FPS) — e.g. 1280×720
SWAPFASE_CAP_W=1280 SWAPFASE_CAP_H=720 ./run.sh --target path/to/face.jpg
```

### Using the virtual camera in a video call

1. Run with `--vcam`.
2. In Zoom / Google Meet / Discord, open camera settings and select:
   - **Linux:** the camera named **"DeepLiveCam"** (the `/dev/video10` loopback).
   - **Windows / macOS:** **"OBS Virtual Camera"**.
3. Your face-swapped stream appears as that camera. The on-screen preview is a
   mirrored selfie view; the virtual camera is un-mirrored by default so other
   participants see you the right way round (add `--vcam-mirror` to flip it).

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| **Camera busy / "Device or resource busy"** | Another app (browser tab, another call app, a previous run) holds the webcam. Close it and retry. |
| **"No face found in target photo"** | The photo is too tightly cropped, too dark, or has no clear front-facing face. Use a clearer photo with margin around the head. |
| **Virtual camera not listed in a browser (Linux)** | `v4l2loopback` must be loaded with `exclusive_caps=1` for Chrome/Firefox to accept it: `sudo modprobe v4l2loopback exclusive_caps=1 card_label="DeepLiveCam" video_nr=10`. |
| **Virtual camera not available (Windows/macOS)** | Install **OBS Studio** and start its Virtual Camera once (it registers the backend `pyvirtualcam` uses), or install Unity Capture on Windows. |
| **Low FPS** | Lower the capture resolution: `SWAPFASE_CAP_W=480 SWAPFASE_CAP_H=360`. Confirm a GPU provider was chosen (check the startup log line). |
| **Falls back to CPU on Linux+NVIDIA** | Launch via `./run.sh`, not `python run.py`, so the venv-local CUDA libs are on `LD_LIBRARY_PATH`. Verify with `scripts/verify_gpu.py`. |

---

## Responsible use

Face-swapping technology can be misused. By using swap-fase you agree to use it
responsibly and lawfully. **Read [docs/RESPONSIBLE_USE.md](docs/RESPONSIBLE_USE.md)
before using this software** — it requires consent from anyone whose face you use
or swap onto, and prohibits impersonation, fraud, harassment, and any deceptive or
non-consensual use.

---

## License

- **The project code is licensed under the [MIT License](LICENSE).** © 2026 lexeler.
- **The models are NOT covered by the MIT license and are NOT bundled in this repo.**
  - `inswapper_128.onnx` and the InsightFace `buffalo_l` pack are distributed by
    InsightFace for **non-commercial / research use only**.
  - They are **downloaded separately at runtime** into a gitignored `models/`
    directory and are subject to their own licenses, which you must comply with.
  - swap-fase is intended for **strictly personal, local, experimental use**,
    which fits those terms. Do not use it commercially or in any published product.

See [LICENSE](LICENSE) for the full text and the model-license note.

---

## Credits

- **[InsightFace](https://github.com/deepinsight/insightface)** — the `buffalo_l`
  face analysis pack and the `inswapper_128` swap model that make this work.
- **[Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam)** — architectural
  inspiration for the real-time capture → swap → display loop.
- **[FaceFusion](https://github.com/facefusion/facefusion)** — reference for the
  modern, coexisting dependency pin set.
