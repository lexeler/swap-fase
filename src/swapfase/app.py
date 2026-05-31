"""``app.main`` — the composition root: bootstrap → engine → threads → window.

This is the ONE place that wires the whole live skeleton together (ARCHITECTURE
"app.py as composition root"):

  1. ``ensure_models()`` — download+verify the models once (offline thereafter).
  2. build the ``FaceEngine`` once (analyser + swapper; CUDA-first, CPU fallback).
  3. probe for a capturable V4L2 node (Pitfall 9) and pick it.
  4. read the target photo, embed its largest face (D-05), cache it in ``AppState``.
  5. construct the keep-newest ``LatestFrameBuffer``, the ``MainWindow``, a
     ``PreviewSink`` bound to ``window.frame_ready.emit``, the ``CaptureThread``
     and the ``InferenceWorker``.
  6. start the threads, show the window, run the Qt event loop.
  7. on close: flip ``state.running`` off, stop+join both threads (the capture
     thread's finally releases the camera — Pitfall 10).

For the skeleton the target photo is supplied via ``--target <path>`` (the
load-photo dialog arrives in Plan 04). Launch via ``./run.sh --target <photo>`` so
the venv-local CUDA ``LD_LIBRARY_PATH`` is exported first (Plan 01).
"""

from __future__ import annotations

import argparse
import logging
import sys

import cv2

from .bootstrap import MODELS_DIR, ensure_models
from .capture import CaptureThread, list_capturable_devices
from .engine import FaceEngine, NoFaceError
from .framebuffer import LatestFrameBuffer
from .pipeline import InferenceWorker
from .platform_detect import os_name
from .sink import DEFAULT_VCAM_DEVICE, PreviewSink, TeeSink, VirtualCamSink
from .state import AppState

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="swap-fase",
        description="Live webcam face swap (skeleton): wears the --target photo's "
        "face on your live webcam stream.",
    )
    p.add_argument(
        "--target",
        required=True,
        help="path to the target photo whose face is WORN (its largest face is "
        "detected once and cached).",
    )
    p.add_argument(
        "--device",
        type=int,
        default=None,
        help="force the INPUT V4L2 camera index to capture from (default: first "
        "probed capturable node). Use this to choose which webcam to swap.",
    )
    p.add_argument(
        "--vcam",
        action="store_true",
        help="ALSO output the swapped stream to a virtual camera so "
        "Zoom/Meet/Discord can use it. Linux: a v4l2loopback node (pick "
        "'DeepLiveCam'). Windows/macOS: the OBS Virtual Camera (install OBS "
        "Studio and start its Virtual Camera first). The on-screen preview keeps "
        "working alongside it.",
    )
    p.add_argument(
        "--vcam-device",
        default=DEFAULT_VCAM_DEVICE,
        help="virtual-camera device to write to. Linux default: "
        f"{DEFAULT_VCAM_DEVICE!r} (v4l2loopback, card 'DeepLiveCam'). "
        "Windows/macOS default: None (pyvirtualcam auto-detects OBS Virtual "
        "Camera).",
    )
    p.add_argument(
        "--vcam-mirror",
        action="store_true",
        help="mirror the VIRTUAL CAMERA output too (default: the call sees you "
        "un-mirrored / the right way round, while the preview stays a selfie view).",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="force CPU (skip CUDA); for debugging the graceful-fallback path.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    logger.info("platform: %s (sys.platform=%s)", os_name(), sys.platform)

    # 1) Models present + verified (network only on first run).
    inswapper_path = ensure_models()

    # 2) Engine built ONCE (CUDA-first unless --cpu; honest provider recorded).
    engine = FaceEngine(
        model_root=MODELS_DIR,
        inswapper_path=inswapper_path,
        prefer_gpu=not args.cpu,
    )
    print(f"provider={engine.provider} using_gpu={engine.using_gpu}")

    # 3) Pick a capturable V4L2 node (probe — never trust index 0; Pitfall 9).
    devices = list_capturable_devices()
    if not devices:
        print(
            "error: no capturable webcam found (probed camera indices 0..3,10). "
            "Is the camera plugged in and not held by another app?",
            file=sys.stderr,
        )
        return 1
    device_index = args.device if args.device is not None else devices[0]
    print(f"capturable camera indices: {devices}; using index {device_index}")

    # 4) Read + embed the target photo's largest face (cached source — Pattern 3).
    target_img = cv2.imread(args.target)
    if target_img is None:
        print(f"error: could not read target photo: {args.target}", file=sys.stderr)
        return 2
    try:
        target_face = engine.embed(target_img)
    except NoFaceError:
        print(
            f"error: no face found in target photo: {args.target} "
            "(use a clear, front-facing photo)",
            file=sys.stderr,
        )
        return 1

    # 5) Shared state + the keep-newest pipeline wiring.
    state = AppState()
    state.device_index = device_index
    state.set_target(target_face)
    state.provider = engine.provider

    buffer = LatestFrameBuffer()

    # Qt application + window must be created on the main thread.
    from PySide6.QtWidgets import QApplication

    qt_app = QApplication.instance() or QApplication(sys.argv[:1])

    from .ui.main_window import MainWindow

    window = MainWindow()
    preview_sink = PreviewSink(window.frame_ready.emit)

    # Default: preview only (behaviour unchanged when --vcam is absent). With
    # --vcam, fan the swapped frame out to BOTH the preview AND a v4l2loopback
    # virtual camera via a TeeSink, so the user sees themselves on screen while
    # the video call sees the face-swapped stream ("DeepLiveCam").
    vcam_sink: VirtualCamSink | None = None
    if args.vcam:
        # Match the ACTUAL capture resolution (capture.py reads SWAPFASE_CAP_W/H,
        # default 640×480) so the virtual camera streams at full quality instead of
        # downscaling. VirtualCamSink resizes any off-size frame defensively. The
        # preview stays mirrored (D-03); the virtual camera is un-mirrored by default
        # so call participants see the user the right way round (opt back in with
        # --vcam-mirror). Device default is per-OS (Linux /dev/video10; Windows/macOS
        # None => pyvirtualcam auto-detects OBS Virtual Camera).
        from .capture import _CAP_HEIGHT, _CAP_WIDTH

        try:
            vcam_sink = VirtualCamSink(
                device=args.vcam_device,
                width=_CAP_WIDTH,
                height=_CAP_HEIGHT,
                fps=30.0,
                mirror=args.vcam_mirror,
            )
        except RuntimeError as exc:
            print(f"error: virtual camera could not start: {exc}", file=sys.stderr)
            return 1
        if os_name() == "linux":
            print(
                f"virtual camera ON -> {args.vcam_device} "
                "(pick 'DeepLiveCam' as your camera in Zoom/Meet/Discord)"
            )
        else:
            print(
                "virtual camera ON -> OBS Virtual Camera "
                "(pick 'OBS Virtual Camera' as your camera in Zoom/Meet/Discord)"
            )
        sink = TeeSink([preview_sink, vcam_sink])
    else:
        sink = preview_sink

    capture = CaptureThread(device_index=device_index, buffer=buffer, state=state)
    worker = InferenceWorker(engine=engine, buffer=buffer, sink=sink, state=state)

    # 6) Start threads (set running BEFORE start so the loops don't exit at once).
    state.running = True
    capture.start()
    worker.start()

    window.show()

    def _shutdown() -> None:
        # 7) Clean stop: flip the flag, signal + join both threads (capture's
        # finally releases the camera — Pitfall 10), so a re-run reopens cleanly.
        state.running = False
        worker.stop()
        capture.stop()
        worker.join(timeout=3.0)
        capture.join(timeout=3.0)
        # Release the virtual camera too (free /dev/video10 for the next run).
        if vcam_sink is not None:
            vcam_sink.close()
        logger.info("threads stopped; camera released")

    qt_app.aboutToQuit.connect(_shutdown)

    try:
        return qt_app.exec()
    finally:
        # Belt-and-braces: ensure shutdown ran even on an exec() exception path.
        if state.running:
            _shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
