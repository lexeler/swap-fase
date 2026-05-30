"""``LatestFrameBuffer`` — the single keep-newest latency-control primitive.

This one tiny class is the crux of the real-time architecture (D-15, LIVE-02,
Pitfall 6). The capture thread produces webcam frames faster than the inference
thread can swap them on the GPU; if those frames piled up in an unbounded queue
the displayed face would lag the live input by seconds and grow worse over time.

The buffer holds AT MOST the newest frame: ``put`` discards any stale frame still
sitting in the slot before inserting the new one, so the producer NEVER blocks and
latency stays bounded to ~one inference time no matter how slow the GPU step is.
Some captured frames are intentionally dropped — for a live preview you want
*fresh*, not *all* (ARCHITECTURE Pattern 1).

Consumed by ``capture.CaptureThread`` (producer, ``put``) and
``pipeline.InferenceWorker`` (consumer, ``get``).
"""

from __future__ import annotations

import queue
from typing import Any


class LatestFrameBuffer:
    """A 1-slot, keep-newest, drop-old frame buffer.

    Producers never block (``put`` always returns immediately); stale frames are
    silently discarded so the consumer always reads the freshest available frame.
    """

    def __init__(self) -> None:
        # maxsize=1 is the whole point: at most one frame is ever buffered.
        self._q: "queue.Queue[Any]" = queue.Queue(maxsize=1)

    def put(self, frame: Any) -> None:
        """Insert ``frame`` as the newest; drop any stale frame first.

        Never blocks and never raises ``queue.Full`` — if the slot is occupied we
        drain the old frame (``get_nowait``) before inserting the new one. This is
        the drop-OLD (not drop-new) keep-newest discipline.
        """
        try:
            self._q.get_nowait()  # discard the stale frame, if any
        except queue.Empty:
            pass
        try:
            self._q.put_nowait(frame)  # newest wins
        except queue.Full:
            # A racing consumer just emptied-then-refilled the slot; the next
            # put() will replace it. Dropping here is correct keep-newest
            # behaviour (we never want to block the producer).
            pass

    def get(self, timeout: float = 1.0) -> Any:
        """Block until a frame is available (up to ``timeout``s), then return it.

        Raises ``queue.Empty`` if no frame arrived within ``timeout`` — the
        inference loop treats that as "no fresh frame yet" and loops again rather
        than hanging forever (lets the worker notice ``state.running`` flips).
        """
        return self._q.get(timeout=timeout)
