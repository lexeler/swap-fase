"""``InferenceWorker`` — the GPU thread: read-latest → mirror → swap → sink.

The consumer side of the keep-newest pipeline. It blocks for the newest webcam
frame, optionally mirrors it (selfie view, D-03), swaps the largest face onto the
cached source embedding (or passes a faceless frame through unchanged, D-18), and
writes the result to the ``FrameSink``. A rolling FPS meter records the live rate
and the active provider into ``AppState`` for the status badge (LIVE-04, surfaced
in Plan 04).

Latency control (Pitfall 6, D-15): exactly ONE ``buffer.get()`` per iteration —
the buffer already holds only the freshest frame, so reading once both gets the
latest and naturally drops the backlog. We never drain a queue in a loop.

Robustness (D-18, threat T-01-12): all per-frame work is wrapped so a single bad
frame (decode glitch, transient detector error) is logged and skipped — it never
kills the worker thread or crashes the app. The detector's own no-face case is
already a clean passthrough inside ``engine.process``.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import cv2

logger = logging.getLogger(__name__)

# How many recent frames to average for the rolling FPS read-out.
_FPS_WINDOW = 30


class InferenceWorker(threading.Thread):
    """Read newest frame → mirror → swap-largest-or-passthrough → sink, with FPS.

    Args:
        engine: a ready ``FaceEngine`` (analyser + swapper built once).
        buffer: the keep-newest ``LatestFrameBuffer`` the capture thread fills.
        sink: a ``FrameSink`` (``PreviewSink`` in the app; ``NullSink`` headless).
        state: shared ``AppState`` (mirror / swap_enabled / target_face / status).
    """

    def __init__(self, engine, buffer, sink, state) -> None:
        super().__init__(name="InferenceWorker", daemon=True)
        self._engine = engine
        self._buffer = buffer
        self._sink = sink
        self._state = state
        self._stop_flag = threading.Event()
        self._frame_times: list[float] = []

    def _update_fps(self) -> float:
        """Append now() and return the rolling FPS over the recent window."""
        now = time.perf_counter()
        self._frame_times.append(now)
        if len(self._frame_times) > _FPS_WINDOW:
            self._frame_times.pop(0)
        if len(self._frame_times) >= 2:
            span = self._frame_times[-1] - self._frame_times[0]
            if span > 0:
                return (len(self._frame_times) - 1) / span
        return 0.0

    def run(self) -> None:
        """Inference loop until stopped. Never dies on a single bad frame."""
        while getattr(self._state, "running", False) and not self._stop_flag.is_set():
            try:
                frame = self._buffer.get(timeout=0.5)  # NEWEST frame (drops backlog)
            except queue.Empty:
                continue  # no fresh frame yet — re-check the run flag and loop

            try:
                # Atomic, consistent per-frame view of the render inputs.
                mirror, swap_enabled, target = self._state.snapshot_render_flags()

                if mirror:
                    frame = cv2.flip(frame, 1)  # horizontal flip — selfie view (D-03)

                if swap_enabled and target is not None:
                    # Swaps the LARGEST face (D-05); a faceless frame is returned
                    # UNCHANGED by engine.process (D-18 passthrough, SWAP-03).
                    frame = self._engine.process(frame, target)

                fps = self._update_fps()
                self._state.set_status(getattr(self._engine, "provider", ""), fps)
                self._sink.write(frame)
            except Exception:  # noqa: BLE001 — one bad frame must not kill the loop
                logger.exception("inference skipped a bad frame; continuing")
                continue

    def stop(self) -> None:
        """Signal the loop to exit on its next iteration."""
        self._stop_flag.set()
