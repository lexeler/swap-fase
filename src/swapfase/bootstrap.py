"""Model acquisition + integrity verification into a project-local ``models/``.

This module is the single startup gate for the face-swap model assets (ENV-04,
D-16). It guarantees that:

  * every model lives under ``<project>/models/`` — NEVER ``~/.insightface`` —
    so the project is self-contained and the binaries stay gitignored;
  * the ``inswapper_128`` swapper (fetched from an UNTRUSTED community mirror) is
    SHA256-verified against a pinned, KNOWN-GOOD published hash *before* it is
    ever loaded — a tampered/corrupt model is deleted and the run fails closed
    (``ModelIntegrityError``), never silently feeding a bad model into ONNX
    ("Protobuf parsing failed" is the classic downstream symptom; Pitfall 5);
  * the network is touched ONLY here, once, at first run — ``ensure_models()`` is
    a startup-only call. Nothing in the per-frame / ``embed`` / ``process`` path
    ever reaches the network (privacy constraint; threat T-01-08).

License / scope (CLAUDE.md, Pitfall 5 security note): ``inswapper_128.onnx`` is
distributed by InsightFace for **non-commercial / research use only**, and its
original public link was removed (it is mirror-hosted now). This project's use is
**strictly personal, local, and non-published**, which fits the non-commercial
research terms (threat T-01-09 — accepted). Do not let scope drift toward any
commercial or published use.

Model builds (D-07/D-08 trade-off, recorded honestly):
  * ``inswapper_128.onnx`` — the **fp32** build (~554 MB) is pinned here because
    it is the build whose SHA256 is published and widely cross-checked
    (roop/FaceFusion lineage), giving GENUINE fail-closed integrity rather than a
    hash recomputed from whatever a mirror happened to serve. fp16 is a valid
    later FPS lever (D-08), but switching to it requires sourcing the fp16 build's
    *published* checksum — not merely re-hashing a downloaded file. Integrity
    (D-16) is the hard gate; fp16 FPS tuning is deferred to the live pipeline.
  * ``buffalo_l`` — the detection/recognition/landmark pack — still
    auto-downloads via ``FaceAnalysis(..., root=MODELS_DIR).prepare()`` into
    ``MODELS_DIR/models/buffalo_l/`` (only the swapper was pulled from auto-DL).

Downstream plans (03-05) consume ``MODELS_DIR``, ``ensure_models``,
``EXPECTED_INSWAPPER_SHA256`` and ``ModelIntegrityError`` from here.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

# --- Project-local model root -------------------------------------------------

# <project>/models/  — absolute, derived from this file's location
# (this file is <project>/src/swapfase/bootstrap.py → parents[2] == <project>).
MODELS_DIR: str = str(Path(__file__).resolve().parents[2] / "models")

# The verified swapper filename inside MODELS_DIR.
INSWAPPER_FILENAME = "inswapper_128.onnx"

# Pinned KNOWN-GOOD SHA256 of the fp32 inswapper_128.onnx build (~554 MB).
# Source: the widely-circulated roop/FaceFusion-lineage hash (Pitfall 5,
# .planning/research/PITFALLS.md). This is a *published* value cross-checked
# across multiple mirrors — verifying against it proves the downloaded bytes are
# the canonical model, not merely that a (possibly tampered) download hashes to
# itself.
EXPECTED_INSWAPPER_SHA256: str = (
    "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af"
)

# Community mirrors for the fp32 build, tried in order. All host the same
# canonical fp32 file whose hash is EXPECTED_INSWAPPER_SHA256 (Pitfall 5;
# STACK.md §Models). The download is fail-closed by the hash check below, so a
# wrong/corrupt mirror copy is rejected rather than loaded.
_INSWAPPER_MIRRORS: tuple[str, ...] = (
    "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
    "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx",
    "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx",
)

# Read in 1 MiB chunks so a ~554 MB model never loads fully into memory.
_CHUNK = 1024 * 1024


class ModelIntegrityError(Exception):
    """Raised when a downloaded model's SHA256 does not match the pinned value.

    Fail-closed: the offending file is deleted and the run aborts rather than
    loading a tampered/corrupt model (threat T-01-06, D-16, Pitfall 5).
    """


def _sha256(path: str) -> str:
    """Stream a file through SHA256 without reading it all into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _verify_or_raise(path: str) -> None:
    """Compare ``path``'s SHA256 to the pin; on mismatch delete it and raise."""
    actual = _sha256(path)
    if actual != EXPECTED_INSWAPPER_SHA256:
        # Fail closed: never leave a bad model on disk where a later run might
        # load it without re-checking.
        try:
            os.remove(path)
        except OSError:
            pass
        raise ModelIntegrityError(
            "inswapper SHA256 mismatch — refusing to load a tampered/corrupt "
            f"model.\n  expected: {EXPECTED_INSWAPPER_SHA256}\n  actual:   {actual}\n"
            f"  (deleted {path}; re-download from a trusted mirror)"
        )


def _download(url: str, dest: str) -> None:
    """Download ``url`` to ``dest`` atomically (via a ``.part`` temp file)."""
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "swapfase/0.1"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:  # noqa: S310
        while True:
            chunk = resp.read(_CHUNK)
            if not chunk:
                break
            out.write(chunk)
    os.replace(tmp, dest)


def _fetch_inswapper(dest: str) -> None:
    """Download the inswapper model, trying each mirror until one succeeds."""
    last_err: Exception | None = None
    for url in _INSWAPPER_MIRRORS:
        try:
            _download(url, dest)
            return
        except Exception as exc:  # network / 404 / mirror down → try the next
            last_err = exc
            # Clean up a partial file before trying the next mirror.
            for stale in (dest, dest + ".part"):
                try:
                    os.remove(stale)
                except OSError:
                    pass
    raise ModelIntegrityError(
        "Could not download inswapper_128.onnx from any known mirror. "
        "Manually place the fp32 build (SHA256 "
        f"{EXPECTED_INSWAPPER_SHA256}) at "
        f"{os.path.join(MODELS_DIR, INSWAPPER_FILENAME)} and re-run. "
        f"Last error: {last_err!r}"
    )


def ensure_buffalo_l() -> None:
    """Trigger ``buffalo_l`` auto-download into project-local ``MODELS_DIR``.

    insightface still auto-downloads the analyser pack; we only need to point its
    ``root`` at ``MODELS_DIR`` so it lands in ``MODELS_DIR/models/buffalo_l/``
    instead of ``~/.insightface``. Constructing + ``prepare`` once is enough; the
    actual analyser used by inference is built in ``engine.FaceEngine``.
    """
    from insightface.app import FaceAnalysis

    # ctx_id=-1 keeps this presence-check cheap and provider-agnostic — it only
    # needs to materialise the model files on disk, not bind the GPU.
    app = FaceAnalysis(name="buffalo_l", root=MODELS_DIR)
    app.prepare(ctx_id=-1, det_size=(640, 640))


def ensure_models() -> str:
    """Ensure both models are present + verified in ``MODELS_DIR``; return path.

    1. ``buffalo_l`` auto-downloads into ``MODELS_DIR/models/buffalo_l/`` on first
       run (no hard-coded network call — insightface owns it).
    2. ``inswapper_128.onnx``: if absent, download once from a community mirror,
       then SHA256-verify against ``EXPECTED_INSWAPPER_SHA256`` (fail-closed —
       mismatch ⇒ delete + ``ModelIntegrityError``). An already-present file is
       re-verified every call (cheap insurance against on-disk corruption).

    Returns the absolute path to the verified inswapper ``.onnx``. Network is
    touched only on the first run; offline thereafter (threat T-01-08).
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1) buffalo_l analyser pack (auto-download into project-local models/).
    ensure_buffalo_l()

    # 2) inswapper swapper (manual mirror + pinned-hash fail-closed verify).
    inswapper_path = os.path.join(MODELS_DIR, INSWAPPER_FILENAME)
    if not os.path.isfile(inswapper_path):
        _fetch_inswapper(inswapper_path)
    _verify_or_raise(inswapper_path)

    return inswapper_path
