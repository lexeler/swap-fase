#!/usr/bin/env python3
"""App entrypoint placeholder.

The real composition root (camera capture -> face detect -> inswapper swap ->
PySide6 window, with the 3-thread keep-newest pipeline) is wired in a later plan.
For now this just confirms the launcher + venv-local CUDA path work end-to-end.

Launch via ``./run.sh`` so the LD_LIBRARY_PATH fallback to the venv's nvidia/*/lib
is exported first.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main() -> int:
    print("swap-fase app entrypoint — the live face-swap UI is wired in a later plan.")
    print("Run `PYTHONPATH=src .venv/bin/python scripts/verify_gpu.py` to check the GPU gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
