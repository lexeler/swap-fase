"""V4L2 webcam capture: probe-and-pick a capturable node + the capture thread.

Two responsibilities (LIVE-01, Pitfall 9/10):

  * ``list_capturable_devices()`` — modern UVC webcams expose several ``/dev/video*``
    nodes per physical camera; some are capture streams, others are metadata-only
    (no frames). Index 0 is sometimes the metadata node. So we PROBE each candidate
    index with the V4L2 backend and keep only the ones whose first ``read()`` returns
    a real frame — never trust a hard-coded index (Pitfall 9).

  * ``CaptureThread`` — opens the chosen node with ``cv2.CAP_V4L2``, forces MJPG +
    640×480 for fps headroom (D-08 lever, Pitfall 6 perf), and loops reading frames
    into the keep-newest ``LatestFrameBuffer`` while ``state.running``. The whole read
    loop is wrapped in try/finally so ``cap.release()`` ALWAYS runs on stop or crash —
    a leaked V4L2 handle keeps the device "busy" for the next run (Pitfall 10).

A "Device or resource busy" at open is raised as ``CameraBusyError`` (a clear,
catchable exception; the friendly UI message is wired in Plan 05).
"""

from __future__ import annotations

import logging
import threading

import cv2

from .framebuffer import LatestFrameBuffer

logger = logging.getLogger(__name__)

# Candidate V4L2 node indices to probe. /dev/video0..3 plus the v4l2loopback-ish
# video10 this machine exposes; only the genuinely capturable ones survive.
_PROBE_INDICES: tuple[int, ...] = (0, 1, 2, 3, 10)

# Capture format: MJPG @ 640×480 — modest resolution for fps headroom (D-07/D-08,
# Pitfall 6; det_size/resolution is the first perf lever if fps falls short).
_CAP_WIDTH = 640
_CAP_HEIGHT = 480
_FOURCC_MJPG = cv2.VideoWriter_fourcc(*"MJPG")


class CameraBusyError(Exception):
    """Raised when a V4L2 node cannot be opened because it is already held.

    Another consumer (browser tab, video-call app, or a leaked handle from a
    crashed run) holds the device — V4L2 capture nodes are typically exclusive
    (Pitfall 10). Caught and shown as a friendly message by the UI (Plan 05).
    """


class CameraOpenError(Exception):
    """Raised when a V4L2 node cannot be opened at all (unplugged / wrong index)."""


def _open_v4l2(device_index: int) -> cv2.VideoCapture:
    """Open ``device_index`` via the V4L2 backend with the capture format set."""
    cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        # OpenCV does not distinguish "busy" from "absent" cleanly here; surface a
        # clear, catchable error. A busy device is the common cause on a re-run.
        raise CameraBusyError(
            f"could not open /dev/video{device_index} (V4L2). It may be busy "
            "(held by another app or a previous run) or absent. Close other "
            "camera users and retry."
        )
    # MJPG + 640×480 BEFORE the first read so the driver negotiates the fast path.
    cap.set(cv2.CAP_PROP_FOURCC, _FOURCC_MJPG)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, _CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAP_HEIGHT)
    return cap


def list_capturable_devices() -> list[int]:
    """Probe candidate V4L2 indices; return those that yield a real frame.

    Opens each candidate with ``cv2.CAP_V4L2``, attempts one ``read()``, and keeps
    the index only if ``ret`` is True and a frame came back — i.e. a genuine
    *capture* node, not a metadata-only node (Pitfall 9). Every probe is released.
    Returns the indices in probe order; ``/dev/video0`` naturally comes first when
    it is capturable (the sensible default).
    """
    capturable: list[int] = []
    for idx in _PROBE_INDICES:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                continue
            ok, frame = cap.read()
            if ok and frame is not None:
                capturable.append(idx)
        except cv2.error:  # a flaky node — skip it, never crash the probe
            continue
        finally:
            cap.release()
    logger.info("Capturable V4L2 nodes: %s", capturable)
    return capturable


class CaptureThread(threading.Thread):
    """Reads frames from a V4L2 node into a keep-newest buffer until stopped.

    The producer side of the keep-newest pipeline: every successful ``read()`` is
    pushed via ``buffer.put`` (which drops any stale frame, so this thread never
    waits on the inference thread). ``cap.release()`` is guaranteed by try/finally
    on stop OR on any error (Pitfall 10).
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
        self._cap = _open_v4l2(device_index)

    def run(self) -> None:
        """Capture loop: read → put-newest, until stopped; always release."""
        try:
            while getattr(self._state, "running", False) and not self._stop_flag.is_set():
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    # Transient read miss (camera hiccup / unplugged). Don't busy-
                    # spin; log once-ish and keep trying so a brief glitch doesn't
                    # kill the stream. A persistent failure will be visible as a
                    # frozen preview (the UI can surface it in a later plan).
                    logger.debug("capture read() returned no frame on /dev/video%d",
                                 self._device_index)
                    continue
                self._buffer.put(frame)
        except Exception:  # noqa: BLE001 — never let a capture error leak the device
            logger.exception("capture loop crashed on /dev/video%d", self._device_index)
        finally:
            # ALWAYS release — a leaked handle keeps the device busy (Pitfall 10).
            self._cap.release()
            logger.info("released /dev/video%d", self._device_index)

    def stop(self) -> None:
        """Signal the loop to exit; release happens in run()'s finally on join."""
        self._stop_flag.set()
