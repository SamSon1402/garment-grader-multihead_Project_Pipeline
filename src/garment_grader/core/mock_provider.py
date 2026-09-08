from __future__ import annotations

import hashlib
import time

import numpy as np

from garment_grader.core.schemas import Box, Defect, VisionResult


class DeterministicMockProvider:
    """Runnable provider used for architecture demonstrations and tests.

    It is deliberately deterministic: the same garment id returns the same output.
    This lets reviewers exercise the full production pipeline without pretending
    this repository ships proprietary model weights.
    """

    CATEGORIES = ["tshirt", "shirt", "jeans", "jacket", "dress"]
    BRANDS = ["Patagonia", "Nike", "Levi's", "Adidas", "Uniqlo"]

    def infer(self, image_bgr: np.ndarray, garment_id: str) -> VisionResult:
        t0 = time.perf_counter()
        digest = hashlib.sha256(garment_id.encode()).digest()
        u = [b / 255.0 for b in digest[:12]]

        category = self.CATEGORIES[digest[0] % len(self.CATEGORIES)]
        brand = self.BRANDS[digest[1] % len(self.BRANDS)]
        category_conf = 0.82 + 0.17 * u[2]
        defect_severity = 0.08 + 0.82 * u[3]
        defect_conf = 0.76 + 0.22 * u[4]
        ood = 0.05 + 0.55 * u[5]

        h, w = image_bgr.shape[:2]
        x1 = 0.25 * w
        y1 = 0.35 * h
        x2 = 0.55 * w
        y2 = 0.58 * h
        defects = []
        if defect_severity > 0.15:
            defects.append(
                Defect(
                    kind="stain" if digest[6] % 2 == 0 else "tear",
                    confidence=defect_conf,
                    severity=defect_severity,
                    box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )

        inference_ms = (time.perf_counter() - t0) * 1000 + 7.5
        return VisionResult(
            category=category,
            category_confidence=category_conf,
            brand_top5=[
                (brand, 0.88 + 0.11 * u[7]),
                ("unknown", 0.20),
                ("second_candidate", 0.12),
            ],
            attributes={
                "color": ("dark", 0.91),
                "style": ("casual", 0.86 + 0.10 * u[8]),
                "sleeve": ("long" if digest[9] % 2 else "short", 0.83 + 0.12 * u[10]),
            },
            defects=defects,
            embedding_norm=18.0 + 4.0 * u[11],
            ood_score=ood,
            model_version="mock-multitask-0.3",
            inference_ms=inference_ms,
        )
