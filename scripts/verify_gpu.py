#!/usr/bin/env python3
"""Hard GPU gate: prove a real warm-up inference binds CUDAExecutionProvider.

Exits 0 and prints ``provider=CUDAExecutionProvider`` when onnxruntime actually
runs on the GPU. Exits non-zero (RuntimeError) on a silent CPU fallback — this
is the gate that blocks downstream pipeline work until the GPU is verified, not
just "available" (D-14 / ROADMAP step 2 / Pitfall 3).

Run via the venv python with the LD_LIBRARY_PATH fallback (run.sh wires this), e.g.:

    SITE=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    LD_LIBRARY_PATH="$SITE/nvidia/cudnn/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}" \
      PYTHONPATH=src .venv/bin/python scripts/verify_gpu.py
"""

import sys
from pathlib import Path

# Allow running as `scripts/verify_gpu.py` without installing the package: put
# the project's src/ on sys.path so `import swapfase` resolves.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from swapfase.providers import verify_gpu


def main() -> int:
    # raise_on_cpu=True -> RuntimeError (non-zero exit) on silent CPU fallback.
    provider = verify_gpu(raise_on_cpu=True)
    print(f"provider={provider}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
