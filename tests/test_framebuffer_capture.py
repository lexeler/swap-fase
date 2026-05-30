"""Behaviour tests for ``framebuffer.LatestFrameBuffer`` + ``capture`` (Plan 01-03, Task 1).

Follows the 01-02 convention: pytest is intentionally NOT a runtime dep, so these
run directly with the venv python. The framebuffer tests are pure (no hardware);
the capture tests touch the REAL webcam and must be run with sandbox disabled:

    PYTHONPATH=src .venv/bin/python tests/test_framebuffer_capture.py

Behaviour under test (D-15, LIVE-01/02, Pitfall 6/9/10):
  * LatestFrameBuffer.put(a) then put(b) then get() -> b  (newest wins, a dropped);
    put never blocks the producer, never raises on a full slot.
  * LatestFrameBuffer.get(timeout) blocks until a frame exists, then returns it;
    get on an empty buffer times out (raises queue.Empty) rather than hanging.
  * list_capturable_devices() returns >=1 index on this machine, and each index
    re-opens with CAP_V4L2 and yields a real frame (a capturable node, not a
    metadata-only node — Pitfall 9).
  * CaptureThread.stop() always release()s the underlying VideoCapture, even when
    the read loop raised mid-run (Pitfall 10 — try/finally).
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swapfase.framebuffer import LatestFrameBuffer  # noqa: E402


# --- LatestFrameBuffer (pure, no hardware) -----------------------------------

def test_keep_newest() -> None:
    b = LatestFrameBuffer()
    b.put("a")
    b.put("b")
    assert b.get(timeout=0.5) == "b", "newest frame must win; stale 'a' dropped"
    print("  put(a); put(b); get() -> 'b' (keep-newest) OK")


def test_put_never_blocks_or_raises() -> None:
    b = LatestFrameBuffer()
    # Hammer put() many times with no consumer — must never block or raise even
    # though the slot is size-1 (producer is decoupled from inference rate).
    for i in range(1000):
        b.put(i)
    assert b.get(timeout=0.5) == 999, "only the newest of many puts survives"
    print("  1000x put() with no consumer never blocks/raises; get() -> 999 OK")


def test_get_blocks_until_frame() -> None:
    b = LatestFrameBuffer()
    result: list = []

    def consumer() -> None:
        result.append(b.get(timeout=2.0))

    t = threading.Thread(target=consumer)
    t.start()
    time.sleep(0.2)  # consumer is blocked in get() — buffer is empty
    assert not result, "get() must block while the buffer is empty"
    b.put("late")
    t.join(timeout=2.0)
    assert result == ["late"], "get() must return the frame once one arrives"
    print("  get() blocks on empty, returns the frame once put() arrives OK")


def test_get_times_out_when_empty() -> None:
    b = LatestFrameBuffer()
    raised = False
    try:
        b.get(timeout=0.2)
    except queue.Empty:
        raised = True
    assert raised, "get() on an empty buffer must time out (queue.Empty), not hang"
    print("  get(timeout) on empty -> queue.Empty (no hang) OK")


# --- capture (REAL /dev/video* — hardware) -----------------------------------

def test_list_capturable_devices_nonempty() -> None:
    from swapfase.capture import list_capturable_devices

    import cv2

    devices = list_capturable_devices()
    assert isinstance(devices, list)
    assert devices, "no capturable webcam node found (Pitfall 9 — probe failed)"
    # Each returned index must genuinely yield a frame via CAP_V4L2.
    for idx in devices:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        try:
            ok, frame = cap.read()
            assert ok and frame is not None, f"index {idx} is not actually capturable"
        finally:
            cap.release()
    print(f"  list_capturable_devices() -> {devices} (all yield real frames) OK")


def test_capture_thread_releases_so_device_reopens() -> None:
    """stop() must release() the device so it can be re-opened (Pitfall 10).

    ``cv2.VideoCapture.release`` is a read-only C-extension attribute (can't be
    monkeypatched), so the honest, behaviour-level proof that release ran is that
    the same exclusive V4L2 node re-opens cleanly right after stop(). A leaked
    handle would keep it busy and the re-open read() would fail.
    """
    import cv2

    from swapfase.capture import CaptureThread, list_capturable_devices
    from swapfase.framebuffer import LatestFrameBuffer as _LFB
    from swapfase.state import AppState

    devices = list_capturable_devices()
    assert devices, "need a capturable device for the release test"
    idx = devices[0]

    state = AppState()
    state.running = True
    state.device_index = idx
    buf = _LFB()
    thread = CaptureThread(device_index=idx, buffer=buf, state=state)
    thread.start()

    # Let it capture a few real frames.
    got = buf.get(timeout=3.0)
    assert got is not None, "capture thread produced no frame"

    # Stop and join — release() must run in run()'s finally.
    state.running = False
    thread.stop()
    thread.join(timeout=3.0)
    assert not thread.is_alive(), "capture thread did not stop"

    # Behaviour proof: the exclusive node re-opens + reads immediately afterwards.
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    try:
        assert cap.isOpened(), "device did not re-open — handle was leaked (Pitfall 10)"
        ok, frame = cap.read()
        assert ok and frame is not None, "device busy after stop — not released"
    finally:
        cap.release()
    print("  CaptureThread.stop() releases the device (re-opens cleanly) OK")


def _pure_tests() -> list:
    return [
        test_keep_newest,
        test_put_never_blocks_or_raises,
        test_get_blocks_until_frame,
        test_get_times_out_when_empty,
    ]


def _hardware_tests() -> list:
    return [
        test_list_capturable_devices_nonempty,
        test_capture_thread_releases_so_device_reopens,
    ]


def main() -> int:
    skip_hw = "--no-hardware" in sys.argv
    tests = _pure_tests() + ([] if skip_hw else _hardware_tests())
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001 — harness reports all failures
            failed += 1
            print(f"FAIL {t.__name__}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
