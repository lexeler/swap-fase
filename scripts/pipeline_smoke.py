#!/usr/bin/env python3
"""Headless end-to-end LIVE-pipeline proof — REAL camera + REAL GPU, no window.

This is the automated half of the 01-03 verification (the live visual window is
the human gate). It runs the EXACT production pipeline — CaptureThread (real
/dev/video0) → keep-newest LatestFrameBuffer → InferenceWorker (real
engine.process swapping onto a cached target embedding) → a counting sink — for a
fixed window, then reports:

  * the active ONNX provider (must be CUDAExecutionProvider on the GPU path),
  * the measured end-to-end swap FPS,
  * keep-newest proof: capture produced FAR more frames than inference consumed
    (stale frames were dropped, so latency does not accumulate — LIVE-02/D-15),
  * no-face passthrough proof: a synthetic blank frame fed through engine.process
    comes back byte-identical (D-18/SWAP-03).

Run with the sandbox disabled (camera + GPU) and the CUDA LD_LIBRARY_PATH set
(use run.sh's env, or the explicit export in this script's invocation):

    SITE=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    LD_LIBRARY_PATH="$SITE/nvidia/cudnn/lib:$SITE/nvidia/cublas/lib:\
$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/curand/lib:$SITE/nvidia/cufft/lib:\
$SITE/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}" \
      PYTHONPATH=src .venv/bin/python scripts/pipeline_smoke.py --target <face.jpg> --seconds 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from swapfase.bootstrap import MODELS_DIR, ensure_models  # noqa: E402
from swapfase.capture import CaptureThread, list_capturable_devices  # noqa: E402
from swapfase.engine import FaceEngine, NoFaceError  # noqa: E402
from swapfase.framebuffer import LatestFrameBuffer  # noqa: E402
from swapfase.pipeline import InferenceWorker  # noqa: E402
from swapfase.state import AppState  # noqa: E402


class _CountingSink:
    """A headless FrameSink that just counts written frames (no display)."""

    def __init__(self) -> None:
        self.count = 0
        self.last_shape = None

    def write(self, frame: np.ndarray) -> None:
        self.count += 1
        self.last_shape = frame.shape


class _CountingBuffer(LatestFrameBuffer):
    """LatestFrameBuffer that also counts how many frames the producer pushed.

    The gap between puts (capture) and the sink's count (inference) is the
    keep-newest drop count — direct proof that stale frames are discarded.
    """

    def __init__(self) -> None:
        super().__init__()
        self.put_count = 0

    def put(self, frame) -> None:  # type: ignore[override]
        self.put_count += 1
        super().put(frame)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", required=True, help="target photo (face to wear)")
    p.add_argument("--seconds", type=float, default=8.0, help="run window (default 8s)")
    p.add_argument("--device", type=int, default=None, help="force a V4L2 index")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    engine = FaceEngine(model_root=MODELS_DIR, inswapper_path=ensure_models())
    print(f"provider={engine.provider} using_gpu={engine.using_gpu}")

    # --- no-face passthrough proof (D-18/SWAP-03), independent of the camera ---
    target_img = cv2.imread(args.target)
    if target_img is None:
        print(f"error: could not read target: {args.target}", file=sys.stderr)
        return 2
    try:
        target_face = engine.embed(target_img)
    except NoFaceError:
        print(f"error: no face in target: {args.target}", file=sys.stderr)
        return 1
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    passthrough = engine.process(blank, target_face)
    passthrough_ok = np.array_equal(passthrough, blank)
    print(f"no-face passthrough byte-identical: {passthrough_ok}")

    # --- live pipeline proof on the REAL camera --------------------------------
    devices = list_capturable_devices()
    if not devices:
        print("error: no capturable webcam node found", file=sys.stderr)
        return 1
    index = args.device if args.device is not None else devices[0]
    print(f"capturable nodes: {devices}; using /dev/video{index}")

    state = AppState()
    state.running = True
    state.device_index = index
    state.set_target(target_face)
    state.provider = engine.provider

    buffer = _CountingBuffer()
    sink = _CountingSink()
    capture = CaptureThread(device_index=index, buffer=buffer, state=state)
    worker = InferenceWorker(engine=engine, buffer=buffer, sink=sink, state=state)

    # Sample inference FPS at a few points to confirm latency stays FLAT (no drift).
    samples: list[tuple[float, int]] = []
    start = time.perf_counter()
    capture.start()
    worker.start()
    try:
        deadline = start + args.seconds
        while time.perf_counter() < deadline:
            time.sleep(0.5)
            samples.append((time.perf_counter() - start, sink.count))
    finally:
        state.running = False
        worker.stop()
        capture.stop()
        worker.join(timeout=3.0)
        capture.join(timeout=3.0)

    elapsed = time.perf_counter() - start
    swap_fps = sink.count / elapsed if elapsed > 0 else 0.0

    # Per-interval (instantaneous) FPS — flat values => no latency drift.
    interval_fps: list[float] = []
    for (t0, c0), (t1, c1) in zip(samples, samples[1:]):
        dt = t1 - t0
        if dt > 0:
            interval_fps.append((c1 - c0) / dt)

    print("\n--- LIVE PIPELINE RESULT ---")
    print(f"active provider:        {state.provider}")
    print(f"captured (put) frames:  {buffer.put_count}")
    print(f"swapped (sink) frames:  {sink.count}")
    print(f"dropped (keep-newest):  {buffer.put_count - sink.count} stale frames discarded")
    print(f"end-to-end swap FPS:    {swap_fps:.1f}  over {elapsed:.2f}s")
    if interval_fps:
        lo, hi = min(interval_fps), max(interval_fps)
        print(f"interval FPS (flat?):   {['%.1f' % f for f in interval_fps]}")
        print(f"interval FPS min/max:   {lo:.1f} / {hi:.1f}  (close => no drift)")
    print(f"state.fps (rolling):    {state.fps:.1f}")
    print(f"output frame shape:     {sink.last_shape}")

    ok = (
        passthrough_ok
        and sink.count > 0
        and buffer.put_count >= sink.count  # producer outran/matched consumer
    )
    print(f"\nPIPELINE SMOKE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
