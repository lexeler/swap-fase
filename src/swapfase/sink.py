"""``FrameSink`` — the one-method output seam (preview now, virtual-camera LATER).

A finished (swapped) frame has to go *somewhere*. The pipeline writes it to a
``FrameSink`` and does not know or care whether that sink paints to the Qt window
or, in a FUTURE milestone, pushes to a v4l2loopback virtual camera. Keeping the
output behind this tiny interface means the virtual-camera work is purely additive
— a new ``V4l2Sink`` file + one wiring line — never a refactor of the pipeline or
engine (ARCHITECTURE Pattern 4; PROJECT.md "leave a clean seam, don't build it").

This milestone builds exactly ONE real sink, ``PreviewSink`` (emits the BGR frame
to the Qt UI via a bound signal callback), plus a trivial ``NullSink`` so the
pipeline can be smoke-tested headless. ``V4l2Sink`` / ``TeeSink`` are intentionally
NOT implemented here — they are the virtual-camera milestone behind this same seam.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np


class FrameSink(Protocol):
    """Where a finished frame goes. One method; that is the entire contract."""

    def write(self, frame: np.ndarray) -> None: ...


class PreviewSink:
    """Emits the finished BGR frame to the Qt UI via a bound signal callback.

    ``emit_callable`` is ``MainWindow.frame_ready.emit`` — a Qt signal emit, which
    is thread-safe and queues the frame onto the UI thread, where ``on_frame``
    converts BGR→RGB→QImage and paints it. The sink itself does NO conversion and
    NO Qt-object touching off the UI thread (Anti-Patterns 2/6).
    """

    def __init__(self, emit_callable: Callable[[np.ndarray], None]) -> None:
        self._emit = emit_callable

    def write(self, frame: np.ndarray) -> None:
        self._emit(frame)  # BGR ndarray handed to the UI thread via a queued signal


class NullSink:
    """Discards frames; for headless pipeline smoke tests (no UI, no display)."""

    def write(self, frame: np.ndarray) -> None:  # noqa: D401 — intentional no-op
        return None


# --- FUTURE VIRTUAL-CAMERA MILESTONE ONLY — do NOT implement now --------------
# Two future sinks live behind this SAME FrameSink seam; adding them is additive
# (a new file + one wiring line in app.py), never a pipeline change. They are
# described here (deliberately NOT written as real class definitions) so the
# seam's intent is unmistakable (ARCHITECTURE Pattern 4):
#
#   * a v4l2 virtual-camera sink — pushes RGB frames to /dev/videoN via
#     pyvirtualcam (v4l2loopback): its write() would call cam.send(rgb_frame).
#   * a tee/fan-out sink — wraps several sinks and forwards each frame to all of
#     them (e.g. preview + vcam at once), looping over its child sinks in write().
#
# Neither is built in this milestone (PROJECT.md: "leave a clean seam, don't
# build it").
