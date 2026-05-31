"""``FrameSink`` — the one-method output seam (preview + virtual-camera).

A finished (swapped) frame has to go *somewhere*. The pipeline writes it to a
``FrameSink`` and does not know or care whether that sink paints to the Qt window
or pushes to a virtual camera. Keeping the output behind this tiny interface means
the virtual-camera work is purely additive — a ``VirtualCamSink`` + a ``TeeSink``
fan-out — never a refactor of the pipeline or engine (ARCHITECTURE Pattern 4;
PROJECT.md "leave a clean seam, don't build it").

Sinks implemented here:
  * ``PreviewSink`` — emits the BGR frame to the Qt UI via a bound signal callback.
  * ``NullSink`` — discards frames; for headless pipeline smoke tests.
  * ``VirtualCamSink`` — pushes frames to a virtual camera via ``pyvirtualcam`` so
    a video-call app can select the face-swapped stream as its camera (the VCAM
    milestone). Cross-platform:
      - Linux   -> v4l2loopback node (default ``/dev/video10``, card "DeepLiveCam")
      - Windows -> OBS Virtual Camera (or Unity Capture) backend, auto-detected
      - macOS   -> OBS Virtual Camera backend, auto-detected
    ``V4l2Sink`` is kept as a backward-compatible alias.
  * ``TeeSink`` — fans one frame out to several child sinks (preview AND vcam at
    once); an error in one child never starves the others.
"""

from __future__ import annotations

import logging
from typing import Callable, Protocol

import cv2
import numpy as np

from .platform_detect import default_vcam_device, is_linux

logger = logging.getLogger(__name__)

# The default virtual-camera device per OS. On Linux this is the v4l2loopback node
# (card "DeepLiveCam") the user selects in Zoom/Meet/Discord. On Windows/macOS it
# is ``None`` so pyvirtualcam auto-detects the OBS Virtual Camera backend (there is
# no ``/dev`` path). This mirrors ``platform_detect.default_vcam_device()``.
DEFAULT_VCAM_DEVICE: str | None = default_vcam_device()


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


class VirtualCamSink:
    """Pushes finished frames to a virtual camera so a call app can select them.

    Opens a virtual-camera device via ``pyvirtualcam`` and, on each
    ``write(frame_bgr)``, sends the frame to it. The swapped stream then appears as
    a real webcam to Zoom/Meet/Discord. Cross-platform backend selection:

      * Linux   -> ``backend='v4l2loopback'``, default device ``/dev/video10``
        (card "DeepLiveCam"). Requires v4l2loopback loaded with
        ``exclusive_caps=1``.
      * Windows -> pyvirtualcam auto-detects the **OBS Virtual Camera** backend
        (or Unity Capture). The user MUST have OBS Studio's Virtual Camera (or
        Unity Capture) installed. ``device`` defaults to ``None`` (auto-pick).
      * macOS   -> pyvirtualcam auto-detects the **OBS Virtual Camera** backend;
        OBS Studio's Virtual Camera must be installed. ``device`` defaults to
        ``None``.

    Frame-size contract: the virtual camera is configured for ``width``×``height``
    at construction; pyvirtualcam REQUIRES every sent frame to match that size
    exactly. The real pipeline captures 640×480, but a frame can differ (e.g. a
    target-photo passthrough on a faceless frame, or a future resolution change) —
    so ``write`` resizes any off-size frame back to the configured dimensions
    rather than crashing the call.

    Mirroring: the on-screen preview is mirrored (selfie view, D-03), but a call's
    participants should see the user the RIGHT way round, so by default this sink
    UN-mirrors the incoming (already-mirrored) preview frame. Pass ``mirror=True``
    to keep the selfie flip on the virtual camera too.

    Robustness: construction fails LOUDLY with a clear, OS-aware message if the
    virtual camera cannot open (so the user is told exactly what to fix), and
    per-frame send errors are logged-and-swallowed so one bad frame never kills the
    call stream.
    """

    def __init__(
        self,
        device: str | None = DEFAULT_VCAM_DEVICE,
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
                "Install it into the project venv (e.g. "
                "`pip install pyvirtualcam`)."
            ) from exc

        self._PixelFormat = pyvirtualcam.PixelFormat

        # Build the kwargs for pyvirtualcam.Camera. On Linux we pin the
        # v4l2loopback backend + the explicit /dev node so we hit "DeepLiveCam"
        # deterministically. On Windows/macOS we let pyvirtualcam auto-detect the
        # OBS Virtual Camera backend and only pass an explicit device if the caller
        # gave one (otherwise device=None lets it auto-pick). BGR fmt matches our
        # OpenCV frames, so NO per-frame BGR→RGB conversion is needed on the hot path.
        cam_kwargs: dict[str, object] = {
            "width": self._width,
            "height": self._height,
            "fps": fps,
            "fmt": pyvirtualcam.PixelFormat.BGR,
        }
        if is_linux():
            cam_kwargs["backend"] = "v4l2loopback"
            if device is not None:
                cam_kwargs["device"] = device
        else:
            # Windows/macOS: only pass device if explicitly provided; None => auto.
            if device is not None:
                cam_kwargs["device"] = device

        try:
            self._cam = pyvirtualcam.Camera(**cam_kwargs)
        except Exception as exc:  # noqa: BLE001 - surface a clear, catchable error
            raise RuntimeError(self._open_error_message(device, exc)) from exc

        # pyvirtualcam may auto-pick a device name; report the real one if exposed.
        actual = getattr(self._cam, "device", device)
        self._device = actual
        logger.info(
            "virtual camera open: %s (%dx%d @ %.0ffps, fmt=BGR, mirror=%s)",
            actual, self._width, self._height, fps, mirror,
        )

    @staticmethod
    def _open_error_message(device: str | None, exc: Exception) -> str:
        """OS-aware, actionable message for a failed virtual-camera open."""
        if is_linux():
            return (
                f"could not open virtual camera {device!r}. Is the v4l2loopback "
                "node present (`v4l2-ctl --list-devices` should show 'DeepLiveCam') "
                "and not held by another app? If pyvirtualcam reports an "
                "exclusive_caps error, the loopback may need exclusive_caps=1. "
                f"Underlying error: {exc}"
            )
        return (
            "could not open virtual camera. Install and enable the OBS Virtual "
            "Camera (OBS Studio -> 'Start Virtual Camera') — or Unity Capture on "
            "Windows — so pyvirtualcam has a backend to target, then retry. "
            f"Underlying error: {exc}"
        )

    def write(self, frame: np.ndarray) -> None:
        """Send one BGR frame to the virtual camera; resize/flip as needed; never crash."""
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
        """Release the virtual-camera device (idempotent)."""
        cam = getattr(self, "_cam", None)
        if cam is not None:
            try:
                cam.close()
            except Exception:  # noqa: BLE001 - best-effort release
                logger.exception("error closing virtual camera")
            self._cam = None
            logger.info("virtual camera closed: %s", self._device)

    def __enter__(self) -> "VirtualCamSink":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


# Backward-compatible alias: app.py and existing callers import ``V4l2Sink``.
# The canonical, cross-platform name is ``VirtualCamSink``; this keeps the old
# import working (and the name still reads correctly on the Linux v4l2loopback path).
V4l2Sink = VirtualCamSink


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
        """Close any child sink that exposes a close() (e.g. VirtualCamSink)."""
        for sink in self._sinks:
            close = getattr(sink, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    logger.exception("error closing a tee child sink")
