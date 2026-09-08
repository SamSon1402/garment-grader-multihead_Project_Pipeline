from __future__ import annotations

import hashlib

import cv2
import numpy as np


def make_demo_frame(garment_id: str, width: int = 960, height: int = 640) -> np.ndarray:
    """Create a synthetic conveyor frame for a dependency-free demo."""
    digest = hashlib.sha256(garment_id.encode()).digest()
    bg = int(105 + digest[0] % 70)
    image = np.full((height, width, 3), bg, dtype=np.uint8)

    # simple garment silhouette with texture, enough for quality checks + overlays
    center = (width // 2, height // 2)
    body = np.array([
        [center[0] - 150, center[1] - 180],
        [center[0] - 255, center[1] - 100],
        [center[0] - 185, center[1] - 30],
        [center[0] - 130, center[1] - 75],
        [center[0] - 125, center[1] + 220],
        [center[0] + 125, center[1] + 220],
        [center[0] + 130, center[1] - 75],
        [center[0] + 185, center[1] - 30],
        [center[0] + 255, center[1] - 100],
        [center[0] + 150, center[1] - 180],
    ], dtype=np.int32)
    color = (45 + digest[1] % 80, 55 + digest[2] % 80, 65 + digest[3] % 80)
    cv2.fillPoly(image, [body], color)
    cv2.circle(image, (center[0], center[1] - 145), 55, (bg, bg, bg), thickness=-1)
    cv2.line(image, (0, height - 35), (width, height - 35), (220, 220, 220), 3)
    cv2.putText(image, garment_id, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2)
    return image
