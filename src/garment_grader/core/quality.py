from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from garment_grader.core.schemas import FrameQuality, NIRReading


@dataclass(frozen=True)
class QualityThresholds:
    min_brightness: float
    max_brightness: float
    min_blur_variance: float
    min_nir_signal: float
    max_nir_signal: float


def inspect_frame(image_bgr: np.ndarray, t: QualityThresholds) -> FrameQuality:
    """Cheap deterministic checks before spending GPU time.

    These checks do not prove the image is semantically correct. They catch common
    acquisition failures: dark/overexposed frames and severe blur/focus problems.
    """
    if image_bgr is None or image_bgr.size == 0:
        return FrameQuality(brightness=0, blur_variance=0, valid=False, reasons=["empty_frame"])

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    reasons: list[str] = []
    if brightness < t.min_brightness:
        reasons.append("underexposed")
    if brightness > t.max_brightness:
        reasons.append("overexposed")
    if blur_variance < t.min_blur_variance:
        reasons.append("blur_or_focus_failure")

    return FrameQuality(
        brightness=brightness,
        blur_variance=blur_variance,
        valid=not reasons,
        reasons=reasons,
    )


def validate_nir(nir: NIRReading, t: QualityThresholds) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if nir.signal_quality < t.min_nir_signal:
        reasons.append("nir_signal_too_low")
    if nir.signal_quality > t.max_nir_signal:
        reasons.append("nir_signal_saturated")
    return (not reasons, reasons)
