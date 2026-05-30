"""``AppState`` — the small lock-guarded object shared across the three threads.

The capture thread, the inference thread, and the UI thread all read/write a few
shared fields (the running flag, the selected device, the cached source
embedding, the mirror/swap toggles, the live provider+fps for the status badge).
Concurrent access to those fields goes through a single ``threading.Lock`` so a
target swap mid-frame is atomic and a toggle flip is never seen half-applied.

Key design points:
  * ``mirror`` defaults to ``True`` — the natural selfie view (D-03); the UI-04
    toggle (Plan 04) flips it.
  * ``swap_enabled`` defaults to ``True`` (swap is on the moment Start is pressed).
  * ``set_target(face)`` replaces the cached source embedding ATOMICALLY under the
    lock — this is what makes change-target-photo-without-restart cheap (D-10,
    UI-03; the load-photo dialog that calls it arrives in Plan 04).
  * ``running`` is the cooperative stop flag the capture + inference loops poll.

Only simple field assignment is guarded; the heavy per-frame work (detect/swap)
happens OUTSIDE the lock so the GPU step never serialises the threads.
"""

from __future__ import annotations

import threading
from typing import Any


class AppState:
    """Lock-guarded shared state for the capture / inference / UI threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Cooperative run flag polled by the capture + inference loops.
        self.running: bool = False
        # Selected capturable V4L2 node (set by app.py after probing).
        self.device_index: int = 0
        # Mirror ON by default — natural selfie view (D-03); toggle in Plan 04.
        self.mirror: bool = True
        # Swap ON by default — active the instant the pipeline starts.
        self.swap_enabled: bool = True
        # Cached SOURCE face embedding (the face being WORN); None until a photo
        # is loaded. Replaced atomically via set_target (D-10/UI-03).
        self.target_face: Any | None = None
        # Live status surfaced in the UI badge (LIVE-04, Plan 04).
        self.provider: str = ""
        self.fps: float = 0.0

    def set_target(self, face: Any) -> None:
        """Atomically replace the cached source embedding (change-without-restart)."""
        with self._lock:
            self.target_face = face

    def get_target(self) -> Any | None:
        """Read the cached source embedding atomically."""
        with self._lock:
            return self.target_face

    def snapshot_render_flags(self) -> tuple[bool, bool, Any | None]:
        """Atomically read the per-frame render inputs (mirror, swap, target).

        Returning all three under one lock acquisition gives the inference loop a
        consistent view for a single frame even if the UI flips a toggle
        concurrently.
        """
        with self._lock:
            return self.mirror, self.swap_enabled, self.target_face

    def set_status(self, provider: str, fps: float) -> None:
        """Atomically record the live provider + fps for the status badge."""
        with self._lock:
            self.provider = provider
            self.fps = fps
