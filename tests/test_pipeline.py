from pathlib import Path

from garment_grader.core.config import Settings
from garment_grader.core.pipeline import GarmentPipeline
from garment_grader.core.policy import CustomerPolicy
from garment_grader.core.sample_image import make_demo_frame
from garment_grader.integrations.provider_factory import build_provider
from garment_grader.storage.jsonl_sink import JsonlEventSink


def test_pipeline_emits_event(tmp_path: Path):
    settings = Settings(event_log_path=tmp_path / "events.jsonl", vision_provider="mock")
    policy = CustomerPolicy.load("configs/retextil_demo.yaml")
    pipeline = GarmentPipeline(
        settings=settings,
        policy=policy,
        provider=build_provider(settings),
        sink=JsonlEventSink(settings.event_log_path),
    )
    event = pipeline.process("g-test-1", make_demo_frame("g-test-1"))
    assert event.garment_id == "g-test-1"
    assert event.route.route
    assert event.total_latency_ms >= 0
    assert settings.event_log_path.exists()
