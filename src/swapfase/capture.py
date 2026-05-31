"""Cross-platform webcam capture: probe-and-pick a capturable device + the read thread.

Two responsibilities (LIVE-01, Pitfall 9/10):

  * ``list_capturable_devices()`` — webcams expose devices by integer INDEX on
    every OS. On Linux a single physical UVC camera can present several
    ``/dev/video*`` nodes, some of which are metadata-only (no frames) — index 0
    is sometimes such a node. On Windows/macOS the index simply selects a camera.
    So we PROBE each candidate index with the per-OS backend and keep only the
    ones whose first ``read()`` returns a real frame — never trust a hard-coded
    index (Pitfall 9).

  * ``CaptureThread`` — opens the chosen index with the per-OS backend
    (Windows ``CAP_DSHOW``→``CAP_MSMF``, Linux ``CAP_V4L2``, macOS
    ``CAP_AVFOUNDATION``), forces MJPG + 640×480 for fps headroom (D-08 lever,
    Pitfall 6 perf), and loops reading frames into the keep-newest
    ``LatestFrameBuffer`` while ``state.running``. The whole read loop is wrapped
    in try/finally so ``cap.release()`` ALWAYS runs on stop or crash — a leaked
    handle keeps the device "busy" for the next run (Pitfall 10).

The capture backend is chosen by OS; the ``/dev/video*`` naming is Linux-only and
guarded so import + use on Windows/macOS never crashes.

A "Device or resource busy" at open is raised as ``CameraBusyError`` (a clear,
catchable exception; the friendly UI message is wired in Plan 05).
"""

from __future__ import annotations

import logging
import os
import threading

import cv2

from .framebuffer import LatestFrameBuffer
from .platform_detect import default_capture_backend, is_linux, is_windows

logger = logging.getLogger(__name__)

# Candidate device indices to probe. On Linux these map to /dev/video0..3 plus the
# v4l2loopback-ish video10 this machine exposes; on Windows/macOS they are plain
# camera indices. Only the genuinely capturable ones survive the frame probe.
_PROBE_INDICES: tuple[int, ...] = (0, 1, 2, 3, 10)

# Capture format: MJPG. Default 640×480 for fps headroom (D-07/D-08, Pitfall 6);
# overridable via SWAPFASE_CAP_W / SWAPFASE_CAP_H env vars to trade fps for a
# sharper picture (quality lever — e.g. 1280×720). Cross-platform: the env knobs
# behave identically on every OS.
_CAP_WIDTH = int(os.environ.get("SWAPFASE_CAP_W", "640"))
_CAP_HEIGHT = int(os.environ.get("SWAPFASE_CAP_H", "480"))
_FOURCC_MJPG = cv2.VideoWriter_fourcc(*"MJPG")


class CameraBusyError(Exception):
    """Raised when a camera device cannot be opened because it is already held.

    Another consumer (browser tab, video-call app, or a leaked handle from a
    crashed run) holds the device — capture devices are typically exclusive on
    Linux V4L2 and DirectShow alike (Pitfall 10). Caught and shown as a friendly
    message by the UI (Plan 05).
    """


class CameraOpenError(Exception):
    """Raised when a device cannot be opened at all (unplugged / wrong index)."""


def _device_label(device_index: int) -> str:
    """Human-readable name for a device index (Linux ``/dev/videoN``, else ``#N``)."""
    if is_linux():
        return f"/dev/video{device_index}"
    return f"camera index {device_index}"


def _candidate_backends() -> list[int]:
    """Return the OpenCV API backends to try for opening a camera, best first.

    Windows tries DirectShow then falls back to Media Foundation (some cameras
    only enumerate on one of the two). Other OSes use their single native backend.
    """
    primary = default_capture_backend()
    if is_windows():
        # CAP_MSMF as the fallback after CAP_DSHOW.
        return [primary, cv2.CAP_MSMF]
    return [primary]


def _open_capture(device_index: int) -> cv2.VideoCapture:
    """Open ``device_index`` with the per-OS backend(s) and set the capture format.

    Tries each candidate backend in turn (Windows: DSHOW → MSMF); the first one
    that opens wins. Raises ``CameraBusyError`` if none opened (OpenCV does not
    cleanly distinguish "busy" from "absent" here; a busy device is the common
    cause on a re-run).
    """
    cap: cv2.VideoCapture | None = None
    for backend in _candidate_backends():
        candidate = cv2.VideoCapture(device_index, backend)
        if candidate.isOpened():
            cap = candidate
            break
        candidate.release()

    if cap is None:
        raise CameraBusyError(
            f"could not open {_device_label(device_index)}. It may be busy "
            "(held by another app or a previous run) or absent. Close other "
            "camera users and retry."
        )

    # MJPG + 640×480 BEFORE the first read so the driver negotiates the fast path.
    cap.set(cv2.CAP_PROP_FOURCC, _FOURCC_MJPG)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, _CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAP_HEIGHT)
    return cap


def list_capturable_devices() -> list[int]:
    """Probe candidate indices; return those that yield a real frame.

    Opens each candidate with the per-OS backend(s), attempts one ``read()``, and
    keeps the index only if ``ret`` is True and a frame came back — i.e. a genuine
    *capture* device, not a metadata-only node (Pitfall 9, mainly a Linux concern;
    harmless elsewhere). Every probe is released. Returns the indices in probe
    order; index 0 naturally comes first when it is capturable (the sensible
    default). Cross-platform: never references ``/dev`` paths.
    """
    capturable: list[int] = []
    for idx in _PROBE_INDICES:
        for backend in _candidate_backends():
            cap = cv2.VideoCapture(idx, backend)
            try:
                if not cap.isOpened():
                    continue
                ok, frame = cap.read()
                if ok and frame is not None:
                    capturable.append(idx)
                    break  # this index is good; don't try the other backend
            except cv2.error:  # a flaky node — skip it, never crash the probe
                continue
            finally:
                cap.release()
    logger.info("Capturable camera indices: %s", capturable)
    return capturable


class CaptureThread(threading.Thread):
    """Reads frames from a camera index into a keep-newest buffer until stopped.

    The producer side of the keep-newest pipeline: every successful ``read()`` is
    pushed via ``buffer.put`` (which drops any stale frame, so this thread never
    waits on the inference thread). ``cap.release()`` is guaranteed by try/finally
    on stop OR on any error (Pitfall 10). Backend selection is per-OS via
    ``_open_capture``.
    """

    def __init__(
        self,
        device_index: int,
        buffer: LatestFrameBuffer,
        state: "object",
    ) -> None:
        super().__init__(name="CaptureThread", daemon=True)
        self._device_index = device_index
        self._buffer = buffer
        self._state = state
        self._stop_flag = threading.Event()
        # Opened eagerly so a busy/absent device fails LOUDLY at construction
        # (before threads start), not silently inside run(). Tests also rely on
        # _cap existing so they can assert release() is called.
        self._cap = _open_capture(device_index)

    def run(self) -> None:
        """Capture loop: read → put-newest, until stopped; always release."""
        label = _device_label(self._device_index)
        try:
            while getattr(self._state, "running", False) and not self._stop_flag.is_set():
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    # Transient read miss (camera hiccup / unplugged). Don't busy-
                    # spin; log once-ish and keep trying so a brief glitch doesn't
                    # kill the stream. A persistent failure will be visible as a
                    # frozen preview (the UI can surface it in a later plan).
                    logger.debug("capture read() returned no frame on %s", label)
                    continue
                self._buffer.put(frame)
        except Exception:  # noqa: BLE001 — never let a capture error leak the device
            logger.exception("capture loop crashed on %s", label)
        finally:
            # ALWAYS release — a leaked handle keeps the device busy (Pitfall 10).
            self._cap.release()
            logger.info("released %s", label)

    def stop(self) -> None:
        """Signal the loop to exit; release happens in run()'s finally on join."""
        self._stop_flag.set()
