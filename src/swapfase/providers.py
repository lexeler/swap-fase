"""ONNX Runtime execution-provider selection + the silent-CPU-fallback gate.

This module is the single source of truth for *how* onnxruntime binds to the GPU
on this machine, and for *proving* it actually did (rather than silently falling
back to CPU — the #1 functional bug for this stack, Pitfall 3).

Key facts baked in here:
  * onnxruntime does NOT patch ``LD_LIBRARY_PATH`` to find the pip-installed
    ``nvidia/*/lib`` shared objects (onnxruntime#25609). The primary remedy is
    ``onnxruntime.preload_dlls(cuda=True, cudnn=True)`` (ORT >= 1.21); the
    secondary remedy is the ``LD_LIBRARY_PATH`` export wired into ``run.sh``.
  * ``get_available_providers()`` reports what was COMPILED IN, not what can
    LOAD at runtime. The only trustworthy check is to build a session, run a
    warm-up inference, and read ``session.get_providers()[0]`` — the provider
    that ACTUALLY bound.

Downstream plans (02-05) consume ``preload_cuda_libs``, ``select_providers``,
``active_provider`` and ``verify_gpu`` from here.
"""

from __future__ import annotations

import onnxruntime as ort

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"


def preload_cuda_libs() -> None:
    """Force-load the venv-local CUDA/cuDNN shared objects before any session.

    Calls ``onnxruntime.preload_dlls(cuda=True, cudnn=True)`` (the D-14 primary
    path). Safe to call once at startup; no-op-safe to call repeatedly. If the
    running ORT predates ``preload_dlls`` (it shouldn't here — we pin 1.22.0),
    we degrade gracefully and rely on the ``run.sh`` ``LD_LIBRARY_PATH`` route.
    """
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        # Older ORT without the symbol: nothing to do here — the launcher's
        # LD_LIBRARY_PATH export is the fallback that makes the libs findable.
        return
    try:
        preload(cuda=True, cudnn=True)
    except Exception:
        # Best-effort: if preload raises (e.g. partial CUDA install), let the
        # session-creation path surface the real error / fall through to the
        # LD_LIBRARY_PATH fallback rather than crashing at import time.
        pass


def select_providers(prefer_gpu: bool = True) -> list[str]:
    """Return the ordered provider list to hand to an ``InferenceSession``.

    ``['CUDAExecutionProvider', 'CPUExecutionProvider']`` when CUDA is available
    and ``prefer_gpu`` is set; otherwise ``['CPUExecutionProvider']`` (the D-17
    graceful-fallback path — the app still runs on CPU, just slower).
    """
    available = ort.get_available_providers()
    if prefer_gpu and CUDA in available:
        return [CUDA, CPU]
    return [CPU]


def active_provider(session: ort.InferenceSession) -> str:
    """Return the provider that ACTUALLY bound (``get_providers()[0]``).

    This is the only honest signal of GPU vs CPU — a session created with a
    CUDA-first list can still report ``CPUExecutionProvider`` here if the CUDA
    provider failed to load its native libs and was silently dropped.
    """
    return session.get_providers()[0]


def _tiny_model_bytes() -> bytes:
    """Build a minimal single-op ONNX model (Relu) entirely in memory.

    Avoids any on-disk model dependency for the gate — the point is purely to
    create a real GPU session and run one inference, not to do useful work.
    """
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Relu", ["X"], ["Y"])
    graph = helper.make_graph([node], "warmup", [x], [y])
    # opset 13 is comfortably supported by ORT 1.22's CUDA EP.
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    # The onnx 1.21 default IR version (13) exceeds ORT 1.22's max (10); pin it
    # down so the session can construct. opset 13 maps to IR version 7.
    model.ir_version = 7
    return model.SerializeToString()


def verify_gpu(raise_on_cpu: bool = False) -> str:
    """Run a real warm-up inference and return the provider that bound.

    Steps: preload the venv-local CUDA libs, build a tiny in-memory ONNX session
    with a CUDA-first provider list, run one inference, then read the ACTIVE
    provider. If it fell back to ``CPUExecutionProvider`` and ``raise_on_cpu`` is
    set, raise ``RuntimeError`` so the hard gate (``scripts/verify_gpu.py``)
    exits non-zero instead of passing silently (D-14, Pitfall 3).

    Returns the active provider string (``"CUDAExecutionProvider"`` on a healthy
    GPU, ``"CPUExecutionProvider"`` on fallback).
    """
    import numpy as np

    # Surface dlopen warnings ("libcudnn.so.9 cannot open ...") instead of
    # swallowing them — severity 1 == WARNING (Pitfall 3).
    ort.set_default_logger_severity(1)

    preload_cuda_libs()

    providers = select_providers(prefer_gpu=True)
    session = ort.InferenceSession(_tiny_model_bytes(), providers=providers)

    # Real warm-up inference — actually exercises the bound EP on the device.
    session.run(None, {"X": np.array([[-1.0, 2.0, -3.0, 4.0]], dtype=np.float32)})

    provider = active_provider(session)
    if raise_on_cpu and provider == CPU:
        raise RuntimeError(f"FELL BACK TO CPU: {session.get_providers()}")
    return provider
