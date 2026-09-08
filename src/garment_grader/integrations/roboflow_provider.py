from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from garment_grader.core.schemas import Box, Defect, VisionResult


class RoboflowProvider:
    """Real adapter for a Roboflow Inference server.

    It normalizes a detector response into the application's stable VisionResult
    contract. Category/brand/attribute heads are left as explicit placeholders
    because their exact model IDs depend on the user's trained workspace models.
    """

    def __init__(self, api_url: str, model_id: str) -> None:
        try:
            from inference_sdk import InferenceHTTPClient
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install with: pip install -e .[roboflow]") from exc

        self.model_id = model_id
        self.client = InferenceHTTPClient(api_url=api_url, api_key=os.getenv("ROBOFLOW_API_KEY"))

    @staticmethod
    def _predictions(result: dict[str, Any]) -> list[dict[str, Any]]:
        # Common Roboflow detector response is {predictions: [...]}. Some endpoints
        # return a list with one item. Keep normalization defensive.
        if isinstance(result, list) and result:
            result = result[0]
        return list(result.get("predictions", []))

    def infer(self, image_bgr: np.ndarray, garment_id: str) -> VisionResult:
        t0 = time.perf_counter()
        result = self.client.infer(image_bgr, model_id=self.model_id)
        preds = self._predictions(result)

        defects: list[Defect] = []
        for p in preds:
            x, y = float(p["x"]), float(p["y"])
            w, h = float(p["width"]), float(p["height"])
            conf = float(p.get("confidence", 0.0))
            cls = str(p.get("class", "defect"))
            defects.append(
                Defect(
                    kind=cls,
                    confidence=conf,
                    severity=min(1.0, 0.25 + 0.75 * conf),
                    box=Box(x1=x - w / 2, y1=y - h / 2, x2=x + w / 2, y2=y + h / 2),
                )
            )

        latency = (time.perf_counter() - t0) * 1000
        max_conf = max((d.confidence for d in defects), default=0.86)
        return VisionResult(
            category="unknown",  # add a trained garment classifier model ID here
            category_confidence=max(0.75, max_conf),
            brand_top5=[("unknown", 0.80)],  # add embedding/retrieval head here
            attributes={"source": ("roboflow_detector", 0.99)},
            defects=defects,
            embedding_norm=1.0,
            ood_score=0.10,
            model_version=f"roboflow:{self.model_id}",
            inference_ms=latency,
        )
