"""``MainWindow`` — the minimal PySide6 preview window (skeleton, Plan 01-03).

The UI thread owns exactly one job: take a finished BGR frame handed over from the
inference thread and paint it. It does NO frame capture and NO model inference
(Anti-Patterns 2/6 — running GPU work on the UI thread would freeze the Wayland
window). This module deliberately contains no swap call and no camera-open call;
all such work lives on the worker/capture threads.

Thread-safe hand-off: ``PreviewSink`` calls ``frame_ready.emit(frame)`` from the
inference thread. Because ``frame_ready`` is a Qt ``Signal`` and the slot lives on
an object owned by the UI thread, Qt queues the call onto the UI thread's event
loop (a queued connection) — so ``on_frame`` always runs on the UI thread, where
touching ``QImage``/``QPixmap`` is safe.

Paint path (steady state): BGR ndarray → RGB → ``QImage(Format_RGB888)`` →
``QPixmap`` → scaled into the ``QLabel``. The window is plain windowed + resizable
(D-04); the preview scales to fit.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QMainWindow


class MainWindow(QMainWindow):
    """A resizable window that paints the live swapped webcam stream.

    ``frame_ready`` carries a BGR ``np.ndarray`` (object payload). Connect it to a
    ``PreviewSink`` via ``PreviewSink(window.frame_ready.emit)`` — the inference
    thread emits, the UI thread paints.
    """

    # object payload = the raw BGR ndarray; a queued connection marshals it to UI.
    frame_ready = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("swap-fase — live")
        # Windowed + resizable (D-04); start at the 640×480 capture size.
        self.resize(640, 480)

        self._label = QLabel("waiting for camera…")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setMinimumSize(160, 120)
        self._label.setStyleSheet("background-color: #111; color: #888;")
        self.setCentralWidget(self._label)

        # Queued connection: emit() on the inference thread -> on_frame() on the UI
        # thread (Qt auto-detects the cross-thread call and queues it).
        self.frame_ready.connect(self.on_frame)

    def on_frame(self, frame: np.ndarray) -> None:
        """Paint one BGR frame. Runs on the UI thread (queued from the worker)."""
        if frame is None:
            return
        # BGR -> RGB without an OpenCV import here (keep the UI cv2-free): reverse
        # the last axis. ascontiguousarray so QImage's stride matches width*3.
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        # QImage shares the ndarray buffer; copy into the pixmap so the frame can be
        # freed/overwritten by the next iteration without corrupting what's shown.
        pixmap = QPixmap.fromImage(image.copy())
        self._label.setPixmap(
            pixmap.scaled(
                self._label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
