from __future__ import annotations

from typing import Protocol

import numpy as np

from garment_grader.core.schemas import VisionResult


class VisionProvider(Protocol):
    def infer(self, image_bgr: np.ndarray, garment_id: str) -> VisionResult:
        """Return normalized vision outputs independent of model/vendor format."""
        ...
