from __future__ import annotations

import numpy as np

from garment_grader.core.schemas import VisionResult


class TensorRTProvider:
    """Boundary for the real on-device TensorRT implementation.

    This repository intentionally does not fake a TensorRT parser without the exact
    exported model contract. The production implementation needs:
      1. serialized .engine built for target Jetson/GPU,
      2. input shape / normalization metadata,
      3. binding names and output decoder,
      4. numerical-equivalence tests against ONNX/PyTorch.

    The rest of the system already depends only on VisionResult, so replacing this
    class does not touch grading, routing, projection, storage, or telemetry.
    """

    def __init__(self, *_: object, **__: object) -> None:
        raise RuntimeError(
            "TensorRT provider requires the real trained/exported garment model. "
            "Use GG_VISION_PROVIDER=mock or roboflow for this portfolio build."
        )

    def infer(self, image_bgr: np.ndarray, garment_id: str) -> VisionResult:  # pragma: no cover
        raise NotImplementedError
