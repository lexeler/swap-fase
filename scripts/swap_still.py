#!/usr/bin/env python3
"""Zero-threading still→still face swap — proves the whole model path end-to-end.

This is the ARCHITECTURE build-order step-3 smoke test: it validates SWAP-01
(load photo → detect → pick the largest face → cache its embedding) and SWAP-02
(per-image detect + ``inswapper`` swap of the largest face) on STATIC images,
with no threads, before the live pipeline (Plans 03-05) depends on the engine.

Usage (run via the CUDA LD_LIBRARY_PATH fallback so the GPU binds — see run.sh):

    SITE=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    LD_LIBRARY_PATH="$SITE/nvidia/cudnn/lib:$SITE/nvidia/cublas/lib:\
$SITE/nvidia/curand/lib:$SITE/nvidia/cufft/lib:$SITE/nvidia/cuda_nvrtc/lib:\
$SITE/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}" \
      PYTHONPATH=src .venv/bin/python scripts/swap_still.py \
        --source path/to/face_to_wear.jpg \
        --target-frame path/to/scene.jpg \
        --out swapped.jpg

``--source`` is the photo whose face is *worn*; its largest face is detected and
its embedding cached once (Pattern 3). ``--target-frame`` is the image whose
largest face gets replaced. The output is written to ``--out`` (default
``swapped.jpg``) and differs from the target-frame input when a swap occurs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# Put src/ on sys.path so `import swapfase` works without installing the package.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from swapfase.bootstrap import MODELS_DIR, ensure_models  # noqa: E402
from swapfase.engine import FaceEngine, NoFaceError  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        required=True,
        help="photo containing the face to WEAR (its largest face is cached)",
    )
    p.add_argument(
        "--target-frame",
        required=True,
        help="photo whose largest face is REPLACED with the source face",
    )
    p.add_argument(
        "--out",
        default="swapped.jpg",
        help="output image path (default: swapped.jpg)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # 1) Ensure + verify the models, then build the engine ONCE.
    inswapper_path = ensure_models()
    engine = FaceEngine(model_root=MODELS_DIR, inswapper_path=inswapper_path)

    # 2) Read inputs (friendly errors on unreadable/missing files — threat T-01-07).
    source_img = cv2.imread(args.source)
    if source_img is None:
        print(f"error: could not read source image: {args.source}", file=sys.stderr)
        return 2
    frame = cv2.imread(args.target_frame)
    if frame is None:
        print(
            f"error: could not read target-frame image: {args.target_frame}",
            file=sys.stderr,
        )
        return 2

    # 3) Cache the source face embedding once (SWAP-01); friendly no-face message.
    try:
        source_face = engine.embed(source_img)
    except NoFaceError:
        print(
            f"no face found in source photo: {args.source} "
            "(use a clear, front-facing photo)",
            file=sys.stderr,
        )
        return 1

    # 4) Detect + swap the largest face in the target frame (SWAP-02), write out.
    result = engine.process(frame, source_face)
    if not cv2.imwrite(args.out, result):
        print(f"error: could not write output: {args.out}", file=sys.stderr)
        return 2

    print(f"provider={engine.provider}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
