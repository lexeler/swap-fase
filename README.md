# swap-fase

Local real-time webcam face-swap desktop app. Load one target photo and your face
in the live webcam preview is replaced with the photo's face (InsightFace
`inswapper` on the GPU, PySide6 preview window). Strictly personal / local —
nothing is published or streamed.

## Run

```bash
# capture the default probed camera, swap onto a target photo, show the preview window
./run.sh --target path/to/face.jpg
```

Always launch via `./run.sh` (not `python run.py` directly) so the venv-local CUDA
`LD_LIBRARY_PATH` is exported first and onnxruntime binds the GPU.

### Useful flags

| Flag | Meaning |
| ---- | ------- |
| `--target PATH` | the photo whose face is worn (required; its largest face is detected once and cached). |
| `--device N` | which INPUT camera index to capture from (default: first probed capturable `/dev/videoN`). Use it to pick a specific webcam. |
| `--cpu` | force CPU (skip CUDA) — for debugging the graceful-fallback path. |
| `--vcam` | ALSO output the swapped stream to a virtual camera (see below). |
| `--vcam-device PATH` | which v4l2loopback node to write to (default `/dev/video10`). |
| `--vcam-mirror` | mirror the virtual-camera output too (default: the call sees you the right way round). |

## Virtual camera for video calls

`swap-fase` can expose its face-swapped output as a regular webcam so you can use
it in **Zoom, Google Meet, or Discord**. It writes frames to a `v4l2loopback`
node (default `/dev/video10`, whose card name is **"DeepLiveCam"**).

```bash
# show the preview AND publish the swap to the virtual camera
./run.sh --target path/to/face.jpg --vcam

# pick a specific input camera and a specific loopback node
./run.sh --target path/to/face.jpg --device 0 --vcam --vcam-device /dev/video10
```

When `--vcam` is set the swapped frame is fanned out to BOTH the on-screen preview
**and** the virtual camera at the same time, so you see yourself locally while the
call sees the swap.

Then in your video-call app, open the camera/video settings and **choose the
camera named "DeepLiveCam"** (the `/dev/video10` loopback). Your face-swapped
stream appears as that camera.

Notes:
- The on-screen preview is mirrored (natural selfie view); the virtual camera is
  **un-mirrored by default** so call participants see you the right way round. Add
  `--vcam-mirror` if you want the selfie flip on the call too.
- Requires the `v4l2loopback` kernel module loaded with a device whose card name is
  "DeepLiveCam". Confirm it with `v4l2-ctl --list-devices` (look for
  `DeepLiveCam (platform:v4l2loopback-000)` → `/dev/video10`). If `pyvirtualcam`
  cannot open the node, the loopback may need `exclusive_caps=0`.
- Without `--vcam` the behaviour is unchanged: the preview window only.

## Setup

Dependencies are pinned in `requirements.txt` and installed into an isolated
project-local `.venv` (never the global env):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

CUDA 12.x + cuDNN 9.x come from the `onnxruntime-gpu[cuda,cudnn]` extras — no system
CUDA toolkit. The NVIDIA *driver* stays system-level.
