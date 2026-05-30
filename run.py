#!/usr/bin/env python3
"""App entrypoint — launches the live webcam face-swap skeleton.

Wires the 3-thread keep-newest pipeline (capture → inference → PySide6 preview)
via ``swapfase.app.main``. The target photo whose face is worn is passed through:

    ./run.sh --target path/to/face.jpg

Launch via ``./run.sh`` so the venv-local CUDA ``LD_LIBRARY_PATH`` fallback is
exported first (so onnxruntime's CUDA EP finds the pip-installed nvidia/*/lib).
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from swapfase.app import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
