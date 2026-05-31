"""Tiny cross-platform helpers: OS detection + per-OS defaults.

Centralises the ``sys.platform`` branching so ``capture.py``, ``sink.py`` and
``app.py`` agree on exactly one definition of "are we on Windows / Linux / macOS"
and on the per-OS capture backend and virtual-camera defaults.

``sys.platform`` values used here:
  * ``"win32"``  -> Windows  (also covers 64-bit; Python reports ``win32`` on both)
  * ``"linux"``  -> Linux
  * ``"darwin"`` -> macOS

Nothing here touches a camera, a device node, or the network — it is pure,
import-safe branching usable on any OS.
"""

from __future__ import annotations

import sys


def is_windows() -> bool:
    """True on Windows (``sys.platform == 'win32'``, 32- and 64-bit alike)."""
    return sys.platform.startswith("win")


def is_linux() -> bool:
    """True on Linux."""
    return sys.platform.startswith("linux")


def is_macos() -> bool:
    """True on macOS (``sys.platform == 'darwin'``)."""
    return sys.platform == "darwin"


def os_name() -> str:
    """Short, stable OS label for logging: ``'windows'`` / ``'linux'`` / ``'macos'``."""
    if is_windows():
        return "windows"
    if is_macos():
        return "macos"
    if is_linux():
        return "linux"
    return sys.platform


def default_capture_backend() -> int:
    """Return the preferred OpenCV ``VideoCapture`` API backend for this OS.

      * Windows -> ``cv2.CAP_DSHOW`` (DirectShow; ``CAP_MSMF`` is the fallback,
        handled in ``capture.py``)
      * macOS   -> ``cv2.CAP_AVFOUNDATION``
      * Linux   -> ``cv2.CAP_V4L2``
      * unknown -> ``cv2.CAP_ANY`` (let OpenCV choose)

    Imported lazily so this module stays import-safe even if cv2 is unusual; cv2
    is a hard dependency of the app anyway.
    """
    import cv2

    if is_windows():
        return cv2.CAP_DSHOW
    if is_macos():
        return cv2.CAP_AVFOUNDATION
    if is_linux():
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def default_vcam_device() -> str | None:
    """Return the default virtual-camera device for this OS.

      * Linux -> ``"/dev/video10"`` (the v4l2loopback node, card "DeepLiveCam")
      * Windows / macOS -> ``None`` so pyvirtualcam auto-detects the OBS Virtual
        Camera (or Unity Capture on Windows) backend — there is no ``/dev`` path.
    """
    if is_linux():
        return "/dev/video10"
    return None
