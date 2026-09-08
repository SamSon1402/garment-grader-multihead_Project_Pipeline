from __future__ import annotations

from garment_grader.core.config import Settings
from garment_grader.core.mock_provider import DeterministicMockProvider
from garment_grader.core.provider import VisionProvider


def build_provider(settings: Settings) -> VisionProvider:
    name = settings.vision_provider.lower().strip()
    if name == "mock":
        return DeterministicMockProvider()
    if name == "roboflow":
        from garment_grader.integrations.roboflow_provider import RoboflowProvider

        return RoboflowProvider(settings.roboflow_api_url, settings.roboflow_model_id)
    if name == "tensorrt":
        from garment_grader.integrations.tensorrt_provider import TensorRTProvider

        return TensorRTProvider()
    raise ValueError(f"unknown vision provider: {settings.vision_provider}")
