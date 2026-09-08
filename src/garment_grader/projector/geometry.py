from __future__ import annotations

import cv2
import numpy as np

from garment_grader.core.schemas import Box


class ProjectorCalibration:
    """Maps camera pixels to projector pixels with a planar homography."""

    def __init__(self, homography: np.ndarray | None = None) -> None:
        self.h = np.eye(3, dtype=np.float32) if homography is None else np.asarray(homography, dtype=np.float32)
        if self.h.shape != (3, 3):
            raise ValueError("homography must be 3x3")

    @classmethod
    def from_correspondences(cls, camera_points: np.ndarray, projector_points: np.ndarray) -> "ProjectorCalibration":
        if camera_points.shape[0] < 4 or projector_points.shape[0] < 4:
            raise ValueError("at least 4 point pairs are required")
        h, mask = cv2.findHomography(camera_points.astype(np.float32), projector_points.astype(np.float32), cv2.RANSAC)
        if h is None or mask is None:
            raise RuntimeError("could not estimate camera→projector homography")
        return cls(h)

    def point(self, x: float, y: float) -> tuple[float, float]:
        pts = np.array([[[x, y]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pts, self.h)[0, 0]
        return float(mapped[0]), float(mapped[1])

    def box(self, box: Box) -> Box:
        x1, y1 = self.point(box.x1, box.y1)
        x2, y2 = self.point(box.x2, box.y2)
        return Box(x1=min(x1, x2), y1=min(y1, y2), x2=max(x1, x2), y2=max(y1, y2))
