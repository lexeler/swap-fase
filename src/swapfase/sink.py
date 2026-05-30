"""``FrameSink`` — the one-method output seam (preview + virtual-camera).

A finished (swapped) frame has to go *somewhere*. The pipeline writes it to a
``FrameSink`` and does not know or care whether that sink paints to the Qt window
or pushes to a v4l2loopback virtual camera. Keeping the output behind this tiny
interface means the virtual-camera work is purely additive — a new ``V4l2Sink``
+ a ``TeeSink`` fan-out — never a refactor of the pipeline or engine (ARCHITECTURE
Pattern 4; PROJECT.md "leave a clean seam, don't build it").

Sinks implemented here:
  * ``PreviewSink`` — emits the BGR frame to the Qt UI via a bound signal callback.
  * ``NullSink`` — discards frames; for headless pipeline smoke tests.
  * ``V4l2Sink`` — pushes frames to a v4l2loopback node (``/dev/video10``,
    card name "DeepLiveCam") via ``pyvirtualcam`` so a video-call app can select
    the face-swapped stream as its camera (the VCAM milestone).
  * ``TeeSink`` — fans one frame out to several child sinks (preview AND vcam at
    once); an error in one child never starves the others.
"""

from __future__ import annotations

import logging
from typing import Callable, Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# The v4l2loopback node this machine exposes (card name "DeepLiveCam"). The user
# picks "DeepLiveCam" as their camera in Zoom/Meet/Discord; we write to this node.
DEFAULT_VCAM_DEVICE = "/dev/video10"


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


class V4l2Sink:
    """Pushes finished frames to a v4l2loopback node so a call app can select them.

    Opens the loopback device (default ``/dev/video10``, card "DeepLiveCam") via
    ``pyvirtualcam`` and, on each ``write(frame_bgr)``, sends the frame to it. The
    swapped stream then appears as a real webcam to Zoom/Meet/Discord — the user
    picks "DeepLiveCam" as their camera.

    Frame-size contract: the loopback is configured for ``width``×``height`` at
    construction; pyvirtualcam REQUIRES every sent frame to match that size exactly.
    The real pipeline captures 640×480, but a frame can differ (e.g. a target-photo
    passthrough on a faceless frame, or a future resolution change) — so ``write``
    resizes any off-size frame back to the configured dimensions rather than
    crashing the call.

    Mirroring: the on-screen preview is mirrored (selfie view, D-03), but a call's
    participants should see the user the RIGHT way round, so by default this sink
    UN-mirrors the incoming (already-mirrored) preview frame. Pass ``mirror=True``
    to keep the selfie flip on the virtual camera too.

    Robustness: construction fails LOUDLY with a clear message if the loopback node
    is missing or busy (so the user is told exactly what to fix), and per-frame send
    errors are logged-and-swallowed so one bad frame never kills the call stream.
    """

    def __init__(
        self,
        device: str = DEFAULT_VCAM_DEVICE,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        mirror: bool = False,
    ) -> None:
        self._device = device
        self._width = int(width)
        self._height = int(height)
        self._mirror = mirror
        # Imported lazily so the rest of the app (and tests) never hard-depend on
        # pyvirtualcam being importable; only --vcam users pay for it.
        try:
            import pyvirtualcam
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "virtual camera requested but pyvirtualcam is not installed. "
                "Install it into the project venv: "
                "uv pip install --python .venv/bin/python pyvirtualcam"
            ) from exc

        self._PixelFormat = pyvirtualcam.PixelFormat
        try:
            # backend='v4l2loopback' targets the exact node; BGR fmt matches our
            # OpenCV frames (the loopback's native pixel format here is BGR4), so
            # NO per-frame BGR→RGB conversion is needed on the hot path.
            self._cam = pyvirtualcam.Camera(
                width=self._width,
                height=self._height,
                fps=fps,
                device=device,
                backend="v4l2loopback",
                fmt=pyvirtualcam.PixelFormat.BGR,
            )
        except Exception as exc:  # noqa: BLE001 - surface a clear, catchable error
            raise RuntimeError(
                f"could not open virtual camera {device!r}. Is the v4l2loopback "
                "node present (v4l2-ctl --list-devices should show 'DeepLiveCam') "
                "and not held by another app? If pyvirtualcam reports an "
                "exclusive_caps error, the loopback may need exclusive_caps=0. "
                f"Underlying error: {exc}"
            ) from exc
        logger.info(
            "virtual camera open: %s (%dx%d @ %.0ffps, fmt=BGR, mirror=%s)",
            device, self._width, self._height, fps, mirror,
        )

    def write(self, frame: np.ndarray) -> None:
        """Send one BGR frame to the loopback; resize/flip as needed; never crash."""
        try:
            if frame is None:
                return
            if self._mirror:
                frame = cv2.flip(frame, 1)  # keep selfie flip on the call too
            h, w = frame.shape[:2]
            if w != self._width or h != self._height:
                # pyvirtualcam demands an exact match; resize any off-size frame.
                frame = cv2.resize(
                    frame, (self._width, self._height), interpolation=cv2.INTER_LINEAR
                )
            # pyvirtualcam wants a contiguous uint8 array of the configured size.
            self._cam.send(np.ascontiguousarray(frame))
            self._cam.sleep_until_next_frame()
        except Exception:  # noqa: BLE001 - one bad frame must not kill the stream
            logger.exception("virtual camera dropped a frame; continuing")

    def close(self) -> None:
        """Release the loopback device (idempotent)."""
        cam = getattr(self, "_cam", None)
        if cam is not None:
            try:
                cam.close()
            except Exception:  # noqa: BLE001 - best-effort release
                logger.exception("error closing virtual camera")
            self._cam = None
            logger.info("virtual camera closed: %s", self._device)

    def __enter__(self) -> "V4l2Sink":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class TeeSink:
    """Fans one frame out to several child sinks (e.g. preview + virtual camera).

    ``write`` forwards the frame to every child in turn. A failure in one child is
    logged and swallowed so the remaining sinks still receive the frame — the
    preview must never go dark because the virtual camera hiccuped, and vice versa.
    """

    def __init__(self, sinks: list[FrameSink]) -> None:
        self._sinks = list(sinks)

    def write(self, frame: np.ndarray) -> None:
        for sink in self._sinks:
            try:
                sink.write(frame)
            except Exception:  # noqa: BLE001 - one sink's error must not starve others
                logger.exception("a tee child sink failed; continuing to the rest")

    def close(self) -> None:
        """Close any child sink that exposes a close() (e.g. V4l2Sink)."""
        for sink in self._sinks:
            close = getattr(sink, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    logger.exception("error closing a tee child sink")
