#!/usr/bin/env python3
"""Raw webcam capture FPS smoke test (no swap, no GPU) — proves LIVE-01.

Probes for a capturable V4L2 node (Pitfall 9), opens it via ``CaptureThread`` into
the keep-newest ``LatestFrameBuffer``, drains the buffer for a few seconds, and
prints the measured capture FPS. This isolates the camera/V4L2 path from the
inference path so a low end-to-end FPS can be attributed correctly (camera vs GPU).

Run via the launcher's CUDA path is NOT required here (no GPU is touched), but the
camera IS — so run with the sandbox disabled and an attached webcam:

    PYTHONPATH=src .venv/bin/python scripts/capture_fps.py --seconds 5

Exits 0 if at least one frame was captured, 1 if no capturable device was found.
"""

from __future__ import annotations

import argparse
import queue
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from swapfase.capture import CaptureThread, list_capturable_devices  # noqa: E402
from swapfase.framebuffer import LatestFrameBuffer  # noqa: E402
from swapfase.state import AppState  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seconds", type=float, default=5.0, help="measurement window (default 5s)"
    )
    p.add_argument(
        "--device", type=int, default=None,
        help="force a V4L2 index (default: first probed capturable node)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    devices = list_capturable_devices()
    if not devices:
        print("no capturable webcam node found (probed 0..3,10)", file=sys.stderr)
        return 1
    index = args.device if args.device is not None else devices[0]
    print(f"capturable nodes: {devices}; using /dev/video{index}")

    state = AppState()
    state.running = True
    state.device_index = index
    buf = LatestFrameBuffer()
    cap = CaptureThread(device_index=index, buffer=buf, state=state)
    cap.start()

    frames = 0
    last_shape = None
    start = time.perf_counter()
    deadline = start + args.seconds
    try:
        while time.perf_counter() < deadline:
            try:
                frame = buf.get(timeout=1.0)
            except queue.Empty:
                continue
            frames += 1
            last_shape = frame.shape
    finally:
        state.running = False
        cap.stop()
        cap.join(timeout=3.0)

    elapsed = time.perf_counter() - start
    fps = frames / elapsed if elapsed > 0 else 0.0
    print(f"captured {frames} unique frames in {elapsed:.2f}s -> {fps:.1f} FPS")
    if last_shape is not None:
        print(f"frame shape: {last_shape}")
    # Note: this counts UNIQUE buffered frames the consumer drained, not the raw
    # sensor rate — stale frames are intentionally dropped by the keep-newest
    # buffer, so this reflects the live-preview cadence, which is what matters.
    return 0 if frames > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
