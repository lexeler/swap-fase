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
from .sink import PreviewSink
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
        help="force a V4L2 device index (default: first probed capturable node).",
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
            "error: no capturable webcam found (probed /dev/video0..3,10). "
            "Is the camera plugged in and not held by another app?",
            file=sys.stderr,
        )
        return 1
    device_index = args.device if args.device is not None else devices[0]
    print(f"capturable nodes: {devices}; using /dev/video{device_index}")

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
    sink = PreviewSink(window.frame_ready.emit)

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
