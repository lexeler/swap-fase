"""ONNX Runtime execution-provider selection (cross-platform) + the GPU probe gate.

This module is the single source of truth for *which* execution providers
onnxruntime binds to, on *any* platform/GPU, and for *proving* a chosen provider
actually ran (rather than silently falling back to CPU — the #1 functional bug
for this stack, Pitfall 3).

Cross-platform contract
-----------------------
The app must run on whatever inference backend is best available, depending on
which onnxruntime wheel the installer chose:

  * Windows                -> ``onnxruntime-directml``  (DmlExecutionProvider)
  * Linux + NVIDIA         -> ``onnxruntime-gpu[cuda,cudnn]`` (CUDAExecutionProvider)
  * Linux + AMD            -> ``onnxruntime-rocm``       (ROCMExecutionProvider)
  * Linux Intel / no-GPU   -> ``onnxruntime`` (CPU)
  * macOS                  -> ``onnxruntime-silicon`` (CoreMLExecutionProvider) / CPU

``select_providers`` picks the best provider that is ACTUALLY present in
``onnxruntime.get_available_providers()`` (which reflects the installed wheel),
following a fixed global priority order, and always appends a CPU fallback so the
app never hard-fails on the inference path. The Linux CUDA hard-gate survives
only as the dev probe (``verify_gpu(raise_on_cpu=True)`` via
``scripts/verify_gpu.py``).

Key facts baked in here
-----------------------
  * onnxruntime does NOT patch ``LD_LIBRARY_PATH`` to find the pip-installed
    ``nvidia/*/lib`` shared objects (onnxruntime#25609). The primary remedy is
    ``onnxruntime.preload_dlls(cuda=True, cudnn=True)`` (ORT >= 1.21); the
    secondary remedy is the ``LD_LIBRARY_PATH`` export wired into ``run.sh``. On
    non-CUDA wheels (DirectML/CoreML/CPU) ``preload_dlls`` is a harmless no-op or
    absent — it is wrapped in try/except and never affects those paths.
  * ``get_available_providers()`` reports what was COMPILED IN, not what can
    LOAD at runtime. The only trustworthy check is to build a session, run a
    warm-up inference, and read ``session.get_providers()[0]`` — the provider
    that ACTUALLY bound.

Downstream ``engine.py`` consumes ``preload_cuda_libs``, ``select_providers``,
``active_provider`` and ``verify_gpu`` from here.
"""

from __future__ import annotations

import logging

import onnxruntime as ort

logger = logging.getLogger(__name__)

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"

# Global hardware-acceleration preference order (the canonical cross-platform
# contract). The first entry that is actually present in
# ``onnxruntime.get_available_providers()`` (i.e. compiled into the installed
# wheel) AND not disabled wins; CPU is always available and is the guaranteed
# fallback.
#
#   TensorRT  -> NVIDIA, fastest but heaviest to set up (Linux, ORT TRT build)
#   CUDA      -> NVIDIA (Linux onnxruntime-gpu)
#   ROCm      -> AMD on Linux (onnxruntime-rocm)
#   OpenVINO  -> Intel CPU/GPU/NPU (onnxruntime-openvino)
#   DirectML  -> ANY Windows GPU: NVIDIA/AMD/Intel (onnxruntime-directml)
#   CoreML    -> Apple Silicon / macOS (onnxruntime-silicon)
#   CPU       -> universal fallback
PROVIDER_PRIORITY: tuple[str, ...] = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "OpenVINOExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
)

# The non-CPU providers that count as "GPU/accelerated" for downstream
# ctx-id/using-gpu decisions (InsightFace's ``prepare(ctx_id=0)`` etc).
_ACCELERATED = frozenset(PROVIDER_PRIORITY) - {CPU}

# TensorRT is in the canonical priority list, but on the ``onnxruntime-gpu`` wheel
# the TensorRT EP, when handed bare to an InferenceSession (no engine-cache / no
# warmup config — which is how InsightFace builds its sessions), causes very long
# first-inference stalls or noisy fallbacks and is NOT what the live Linux+NVIDIA
# setup uses. So we OMIT it from the selected list by default and let CUDA lead on
# NVIDIA. Opt in explicitly with ``SWAPFASE_ENABLE_TENSORRT=1`` (or
# ``allow_tensorrt=True``). This keeps the live CUDA path's
# ``providers[0] == 'CUDAExecutionProvider'`` GPU check intact while remaining
# fully cross-platform (DirectML/ROCm/CoreML/CPU are unaffected).
import os as _os

_ENABLE_TENSORRT_ENV = _os.environ.get("SWAPFASE_ENABLE_TENSORRT", "") not in (
    "",
    "0",
    "false",
    "False",
)


def preload_cuda_libs() -> None:
    """Best-effort: force-load venv-local CUDA/cuDNN shared objects before a session.

    On a CUDA wheel (Linux + ``onnxruntime-gpu``) this calls
    ``onnxruntime.preload_dlls(cuda=True, cudnn=True)`` so the CUDA EP can bind
    (ORT doesn't patch ``LD_LIBRARY_PATH`` — the ``run.sh`` export is the
    fallback). On a DirectML/CoreML/CPU wheel the symbol may be absent or a no-op;
    either way this is wrapped so it NEVER raises and NEVER affects non-CUDA paths.
    Safe to call once at startup and idempotent on repeat.
    """
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        # Older ORT, or a non-CUDA wheel without the symbol: nothing to do. On
        # Linux+CUDA the launcher's LD_LIBRARY_PATH export is the fallback.
        return
    try:
        preload(cuda=True, cudnn=True)
    except Exception:
        # Best-effort: if preload raises (partial CUDA install, or a wheel that
        # exposes the symbol but has no CUDA libs to load), let session creation
        # surface the real error / fall through to the CPU fallback rather than
        # crashing at import time.
        pass


def available_providers() -> list[str]:
    """Return the providers the installed onnxruntime wheel actually offers."""
    try:
        return list(ort.get_available_providers())
    except Exception:  # noqa: BLE001 - never let a probe crash the caller
        logger.exception("ort.get_available_providers() failed; assuming CPU only")
        return [CPU]


def select_providers(prefer_gpu: bool = True, allow_tensorrt: bool | None = None) -> list[str]:
    """Return the ordered provider list to hand to an ``InferenceSession``.

    Filters :data:`PROVIDER_PRIORITY` down to the providers actually present in
    the installed wheel, in priority order (best first), and always ensures a CPU
    fallback at the end so the app never hard-fails on the inference path. The
    chosen ordering is logged so the bound backend is visible in the logs.

    Examples (depending on the installed wheel):
      * Linux + ``onnxruntime-gpu``      -> ``['CUDAExecutionProvider', 'CPUExecutionProvider']``
      * Windows + ``onnxruntime-directml`` -> ``['DmlExecutionProvider', 'CPUExecutionProvider']``
      * macOS + ``onnxruntime-silicon``  -> ``['CoreMLExecutionProvider', 'CPUExecutionProvider']``
      * CPU-only wheel                   -> ``['CPUExecutionProvider']``

    TensorRT is omitted by default (see :data:`_ENABLE_TENSORRT_ENV`); set
    ``allow_tensorrt=True`` (or ``SWAPFASE_ENABLE_TENSORRT=1``) to include it when
    present. This keeps CUDA leading on the live Linux+NVIDIA setup.

    When ``prefer_gpu`` is False, hardware providers are skipped and the result is
    just ``['CPUExecutionProvider']`` (the ``--cpu`` debug / graceful-fallback path).
    """
    if allow_tensorrt is None:
        allow_tensorrt = _ENABLE_TENSORRT_ENV
    available = available_providers()

    if not prefer_gpu:
        chosen = [CPU]
    else:
        chosen = [
            p
            for p in PROVIDER_PRIORITY
            if p in available
            and (allow_tensorrt or p != "TensorrtExecutionProvider")
        ]
        # Guarantee a CPU fallback even on an exotic wheel that somehow omits it.
        if CPU not in chosen:
            chosen.append(CPU)

    logger.info(
        "execution providers: chose %s (available=%s, prefer_gpu=%s, allow_tensorrt=%s)",
        chosen, available, prefer_gpu, allow_tensorrt,
    )
    return chosen


def uses_gpu(providers: list[str]) -> bool:
    """True if the FIRST provider in ``providers`` is a hardware accelerator.

    A robust replacement for the brittle ``providers[0] == 'CUDAExecutionProvider'``
    check — returns True for CUDA/TensorRT/ROCm/DirectML/CoreML/OpenVINO leading
    the list, False when CPU leads. Provided for any caller that needs to derive a
    ctx-id / using-gpu flag cross-platform.
    """
    return bool(providers) and providers[0] in _ACCELERATED


def active_provider(session: ort.InferenceSession) -> str:
    """Return the provider that ACTUALLY bound (``get_providers()[0]``).

    This is the only honest signal of which backend is running — a session
    created with an accelerator-first list can still report
    ``CPUExecutionProvider`` here if that accelerator's native libs failed to load
    and the provider was silently dropped.
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

    Steps: preload the venv-local CUDA libs (no-op off CUDA), build a tiny
    in-memory ONNX session with the best-available provider list, run one
    inference, then read the ACTIVE provider. If it fell back to
    ``CPUExecutionProvider`` and ``raise_on_cpu`` is set, raise ``RuntimeError`` so
    the Linux dev hard-gate (``scripts/verify_gpu.py``) exits non-zero instead of
    passing silently (Pitfall 3).

    NOTE: the APP path must NEVER call this with ``raise_on_cpu=True`` — the app
    accepts any provider (CUDA/DML/CoreML/CPU). ``raise_on_cpu=True`` is the dev
    probe only.

    Returns the active provider string (e.g. ``"CUDAExecutionProvider"`` /
    ``"DmlExecutionProvider"`` on a healthy GPU, ``"CPUExecutionProvider"`` on
    fallback).
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
