from __future__ import annotations

import cv2
import numpy as np

from garment_grader.core.schemas import ProjectionCommand


def render_projection(command: ProjectionCommand, width: int = 1280, height: int = 720) -> np.ndarray:
    """Render the image that a DLP projector would display in the calibrated zone."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if not command.enabled:
        return canvas

    cv2.putText(canvas, f"GRADE {command.grade.value}", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
    cv2.putText(canvas, command.route.upper(), (60, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    cv2.putText(canvas, command.label, (60, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

    for defect in command.projected_defects:
        if defect.box is None:
            continue
        b = defect.box
        cv2.rectangle(canvas, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (255, 255, 255), 4)
        cv2.putText(canvas, defect.kind.upper(), (int(b.x1), max(30, int(b.y1) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas
