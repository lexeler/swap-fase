"""Behaviour tests for ``swapfase.engine.FaceEngine`` (Plan 01-02, Task 2).

Run directly with the venv python under the CUDA ``LD_LIBRARY_PATH`` fallback
(pytest is intentionally NOT a runtime dep — the pinned set stays minimal):

    SITE=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    LD_LIBRARY_PATH="$SITE/nvidia/cudnn/lib:$SITE/nvidia/cublas/lib:\
$SITE/nvidia/curand/lib:$SITE/nvidia/cufft/lib:$SITE/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}" \
      PYTHONPATH=src .venv/bin/python tests/test_engine.py

Real face images come bundled with insightface (``Tom_Hanks_54745.png`` — a
single face; ``t1.jpg`` — a group photo with multiple faces), so the tests
exercise the genuine detect→embed→swap path on the GPU, not mocks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import insightface
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swapfase.bootstrap import MODELS_DIR, ensure_models  # noqa: E402
from swapfase.engine import FaceEngine, NoFaceError  # noqa: E402

# Real bundled sample images (HIGH-confidence faces).
_IMG_DIR = Path(insightface.__file__).resolve().parent / "data" / "images"
SINGLE_FACE = str(_IMG_DIR / "Tom_Hanks_54745.png")  # one clear face
MULTI_FACE = str(_IMG_DIR / "t1.jpg")  # group photo, several faces


def _bbox_area(face) -> float:
    x1, y1, x2, y2 = face.bbox
    return (x2 - x1) * (y2 - y1)


def build_engine() -> FaceEngine:
    inswapper_path = ensure_models()
    return FaceEngine(model_root=MODELS_DIR, inswapper_path=inswapper_path)


def test_provider_is_set(engine: FaceEngine) -> None:
    assert isinstance(engine.provider, str) and engine.provider, "provider unset"
    assert isinstance(engine.using_gpu, bool)
    print(f"  provider={engine.provider} using_gpu={engine.using_gpu}")


def test_embed_single_face(engine: FaceEngine) -> None:
    img = cv2.imread(SINGLE_FACE)
    assert img is not None, f"could not read {SINGLE_FACE}"
    face = engine.embed(img)
    assert face is not None
    assert getattr(face, "normed_embedding", None) is not None, "embedding is None"
    print("  embed(single) -> Face with normed_embedding OK")


def test_embed_no_face_raises(engine: FaceEngine) -> None:
    blank = np.zeros((480, 640, 3), dtype=np.uint8)  # a black frame — no face
    raised = False
    try:
        engine.embed(blank)
    except NoFaceError:
        raised = True
    assert raised, "embed() must raise NoFaceError on a faceless image"
    print("  embed(no-face) -> NoFaceError OK")


def test_embed_picks_largest(engine: FaceEngine) -> None:
    img = cv2.imread(MULTI_FACE)
    assert img is not None, f"could not read {MULTI_FACE}"
    faces = engine.detect(img)
    assert len(faces) >= 2, f"expected a multi-face image, got {len(faces)} faces"
    chosen = engine.embed(img)
    largest = max(faces, key=_bbox_area)
    assert _bbox_area(chosen) == _bbox_area(largest), "embed did not pick the largest"
    print(f"  embed(multi: {len(faces)} faces) -> largest by bbox area OK")


def test_detect_returns_all(engine: FaceEngine) -> None:
    img = cv2.imread(MULTI_FACE)
    faces = engine.detect(img)
    assert len(faces) >= 2, "detect() must return ALL faces (D-06)"
    print(f"  detect(multi) -> {len(faces)} faces (all kept) OK")


def test_process_changes_pixels(engine: FaceEngine) -> None:
    target = engine.embed(cv2.imread(SINGLE_FACE))
    frame = cv2.imread(MULTI_FACE)
    out = engine.process(frame, target)
    assert out is not None and out.shape == frame.shape
    assert not np.array_equal(out, frame), "process() did not change any pixels"
    print("  process(face-frame) -> output differs from input OK")


def test_process_no_face_passthrough(engine: FaceEngine) -> None:
    target = engine.embed(cv2.imread(SINGLE_FACE))
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    out = engine.process(blank, target)
    assert np.array_equal(out, blank), "no-face frame must pass through unchanged (D-18)"
    print("  process(no-face) -> passthrough unchanged OK")


def main() -> int:
    engine = build_engine()
    tests = [
        test_provider_is_set,
        test_embed_single_face,
        test_embed_no_face_raises,
        test_embed_picks_largest,
        test_detect_returns_all,
        test_process_changes_pixels,
        test_process_no_face_passthrough,
    ]
    failed = 0
    for t in tests:
        try:
            t(engine)
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001 — test harness reports all failures
            failed += 1
            print(f"FAIL {t.__name__}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
