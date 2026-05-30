"""``FaceEngine`` — the buffalo_l analyser + inswapper swapper, built once.

This is the inference core consumed by the still-image smoke test (Plan 01-02)
and the live pipeline (Plans 03-05). It owns exactly one ``FaceAnalysis``
(detection + embedding) and one ``inswapper_128`` swapper, both constructed
ONCE in ``__init__`` and reused for every call — never per frame (Pitfall 6,
Anti-Pattern 4; rebuilding sessions churns GPU memory and stutters).

Behavioural contract (D-05/D-06/D-18):
  * ``detect(image)``  → ALL detected faces (so a future "all faces" toggle is
    cheap — D-06), even though ``process`` only swaps the largest.
  * ``embed(image)``   → the LARGEST face by bbox area (the user — D-05); raises
    ``NoFaceError`` when the image has no detectable face. This is the cached
    *source* Face, computed once on photo load (Pattern 3).
  * ``process(frame, target_face)`` → swaps the largest face in ``frame`` onto
    ``target_face``; a frame with NO face passes through UNCHANGED and does not
    raise (D-18 passthrough).

Provider honesty (Pitfall 3): after a real warm-up inference the engine records
``self.provider`` = the EP that ACTUALLY bound (read from the live session via
``active_provider``), not merely the one requested. ``self.using_gpu`` reflects
the requested CUDA-first selection; if CUDA silently dropped to CPU,
``self.provider`` will say ``CPUExecutionProvider`` and the mismatch is logged.
"""

from __future__ import annotations

import logging

import insightface
import numpy as np
from insightface.app import FaceAnalysis

from .providers import active_provider, preload_cuda_libs, select_providers

logger = logging.getLogger(__name__)

CUDA = "CUDAExecutionProvider"


class NoFaceError(Exception):
    """Raised when an image expected to contain a face has none detectable."""


def _bbox_area(face) -> float:
    """Area of a detected face's bounding box (used to pick the largest)."""
    x1, y1, x2, y2 = face.bbox
    return float((x2 - x1) * (y2 - y1))


class FaceEngine:
    """Owns the analyser + swapper; embeds a source face and swaps frames.

    Args:
        model_root: project-local ``models/`` root (``bootstrap.MODELS_DIR``) —
            ``buffalo_l`` is loaded from ``<root>/models/buffalo_l/``.
        inswapper_path: absolute path to the SHA256-verified ``inswapper_128``
            ``.onnx`` (from ``bootstrap.ensure_models()``).
        prefer_gpu: try CUDA first (falls back to CPU gracefully — D-17).
        det_size: detector input size; a perf knob (D-08 allows dropping to
            ``(320, 320)`` later for fps). Default ``(640, 640)``.
    """

    def __init__(
        self,
        model_root: str,
        inswapper_path: str,
        prefer_gpu: bool = True,
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        # Force-load the venv-local CUDA/cuDNN .so's BEFORE any session is built,
        # so the CUDA EP can actually bind (ORT doesn't patch LD_LIBRARY_PATH —
        # Pitfall 4). No-op-safe if already preloaded.
        preload_cuda_libs()

        providers = select_providers(prefer_gpu)
        gpu = providers[0] == CUDA
        self.using_gpu = gpu

        # --- analyser (buffalo_l: detect + 512-d embedding) — built ONCE ------
        self.analyser = FaceAnalysis(
            name="buffalo_l", root=model_root, providers=providers
        )
        self.analyser.prepare(ctx_id=0 if gpu else -1, det_size=det_size)

        # --- swapper (inswapper_128) — built ONCE -----------------------------
        self.swapper = insightface.model_zoo.get_model(
            inswapper_path, providers=providers
        )

        # --- record the REAL bound provider (Pitfall 3) -----------------------
        # The swapper exposes its ONNX session as ``.session``; read the provider
        # that actually bound there (the analyser's det model is the fallback).
        session = getattr(self.swapper, "session", None)
        if session is None:
            det = getattr(self.analyser, "det_model", None)
            session = getattr(det, "session", None)
        self.provider = active_provider(session) if session is not None else (
            CUDA if gpu else "CPUExecutionProvider"
        )

        if gpu and self.provider != CUDA:
            logger.warning(
                "Requested CUDA but the session bound %s — running on CPU "
                "(silent-fallback; check cuDNN/CUDA libs on the loader path).",
                self.provider,
            )
        logger.info("FaceEngine ready: provider=%s using_gpu=%s", self.provider, gpu)

    def detect(self, image: np.ndarray) -> list:
        """Return ALL detected faces in ``image`` (D-06 keeps every detection)."""
        return self.analyser.get(image)

    def embed(self, image: np.ndarray):
        """Return the LARGEST face by bbox area (D-05); ``NoFaceError`` if none.

        This is the cached *source* Face — its ``normed_embedding`` is what the
        swapper paints onto frame faces. Computed once on photo load (Pattern 3).
        """
        faces = self.detect(image)
        if not faces:
            raise NoFaceError("no detectable face in the source image")
        return max(faces, key=_bbox_area)

    def process(self, frame: np.ndarray, target_face) -> np.ndarray:
        """Swap the largest face in ``frame`` onto ``target_face``.

        A frame with no detectable face is returned UNCHANGED (D-18 passthrough)
        — never raises, so the live loop keeps running through faceless frames.
        """
        faces = self.detect(frame)
        if not faces:
            return frame  # D-18: no-face passthrough
        largest = max(faces, key=_bbox_area)
        return self.swapper.get(frame, largest, target_face, paste_back=True)
