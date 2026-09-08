from pathlib import Path

import cv2

from garment_grader.core.config import Settings
from garment_grader.core.pipeline import GarmentPipeline
from garment_grader.core.policy import CustomerPolicy
from garment_grader.core.sample_image import make_demo_frame
from garment_grader.integrations.provider_factory import build_provider
from garment_grader.projector.render import render_projection
from garment_grader.storage.jsonl_sink import JsonlEventSink


def main() -> None:
    settings = Settings()
    policy = CustomerPolicy.load(settings.policy_path)
    pipeline = GarmentPipeline(
        settings=settings,
        policy=policy,
        provider=build_provider(settings),
        sink=JsonlEventSink(settings.event_log_path),
    )

    out = Path("data/demo")
    out.mkdir(parents=True, exist_ok=True)

    for garment_id in ["g-1001", "g-1002", "g-1003", "g-1004", "g-1005"]:
        frame = make_demo_frame(garment_id)
        event = pipeline.process(garment_id, frame)
        overlay = render_projection(event.projection)
        cv2.imwrite(str(out / f"{garment_id}_frame.jpg"), frame)
        cv2.imwrite(str(out / f"{garment_id}_projection.png"), overlay)
        print(
            f"{garment_id}: category={event.vision.category:<7} "
            f"grade={event.grade.grade.value:<6} route={event.route.route:<16} "
            f"latency={event.total_latency_ms:.1f}ms projection_in={event.projection.fire_after_ms}ms"
        )


if __name__ == "__main__":
    main()
