from __future__ import annotations

import hashlib

from garment_grader.core.schemas import NIRReading


def simulated_nir(garment_id: str) -> NIRReading:
    """Deterministic NIR placeholder.

    A real implementation would receive a calibrated spectrum from a sensor and
    run a small spectral model. This keeps the contract realistic without
    pretending we have the physical spectrometer in this repo.
    """
    b = hashlib.sha256((garment_id + "nir").encode()).digest()
    raw = [0.1 + b[i] / 255.0 for i in range(4)]
    total = sum(raw)
    p, c, w, o = [v / total for v in raw]
    quality = 0.65 + 0.30 * (b[5] / 255.0)
    return NIRReading(polyester=p, cotton=c, wool=w, other=o, signal_quality=quality)
