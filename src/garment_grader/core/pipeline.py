from __future__ import annotations

import time

import numpy as np

from garment_grader.core.config import Settings
from garment_grader.core.nir import simulated_nir
from garment_grader.core.policy import CustomerPolicy
from garment_grader.core.provider import VisionProvider
from garment_grader.core.quality import inspect_frame, validate_nir
from garment_grader.core.schemas import Defect, GarmentEvent, ProjectionCommand
from garment_grader.projector.geometry import ProjectorCalibration
from garment_grader.projector.sync import projection_delay_ms
from garment_grader.storage.base import EventSink
from garment_grader.telemetry.metrics import EVENTS, INFER_LATENCY_MS, INPUT_FAILURES, TOTAL_LATENCY_MS


class GarmentPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        policy: CustomerPolicy,
        provider: VisionProvider,
        sink: EventSink,
        calibration: ProjectorCalibration | None = None,
    ) -> None:
        self.settings = settings
        self.policy = policy
        self.provider = provider
        self.sink = sink
        self.calibration = calibration or ProjectorCalibration()

    def process(self, garment_id: str, image_bgr: np.ndarray) -> GarmentEvent:
        start = time.perf_counter()
        q = inspect_frame(image_bgr, self.policy.quality_thresholds)
        for reason in q.reasons:
            INPUT_FAILURES.labels(sensor="rgb", reason=reason).inc()

        nir = simulated_nir(garment_id)
        nir_ok, nir_reasons = validate_nir(nir, self.policy.quality_thresholds)
        for reason in nir_reasons:
            INPUT_FAILURES.labels(sensor="nir", reason=reason).inc()

        # We still run the provider in this portfolio demo to make failure events
        # inspectable. A production line may skip expensive inference after a hard
        # camera failure and immediately route to review.
        vision = self.provider.infer(image_bgr, garment_id)
        INFER_LATENCY_MS.observe(vision.inference_ms)

        grade = self.policy.decide_grade(
            vision,
            frame_ok=q.valid,
            nir_ok=nir_ok,
            review_threshold=self.settings.review_threshold,
        )
        price = self.policy.price(vision, grade.grade)
        route = self.policy.route(grade.grade, price.estimated_eur, vision.max_defect_severity)

        processing_ms = (time.perf_counter() - start) * 1000
        projected: list[Defect] = []
        if self.policy.show_defects():
            for defect in vision.defects:
                if defect.box is None:
                    projected.append(defect)
                else:
                    projected.append(defect.model_copy(update={"box": self.calibration.box(defect.box)}))

        projection = ProjectionCommand(
            enabled=self.policy.should_project(grade.grade),
            fire_after_ms=projection_delay_ms(
                self.settings.projection_zone_m,
                self.settings.conveyor_speed_mps,
                processing_ms,
            ),
            grade=grade.grade,
            route=route.route,
            label="LOW CONFIDENCE" if route.requires_human else "AI DECISION",
            projected_defects=projected if self.policy.should_project(grade.grade) else [],
        )

        total_ms = (time.perf_counter() - start) * 1000
        event = GarmentEvent(
            garment_id=garment_id,
            customer=self.policy.customer,
            policy_version=self.policy.version,
            frame_quality=q,
            nir=nir,
            vision=vision,
            grade=grade,
            price=price,
            route=route,
            projection=projection,
            total_latency_ms=total_ms,
            metadata={
                "conveyor_speed_mps": self.settings.conveyor_speed_mps,
                "projection_zone_m": self.settings.projection_zone_m,
            },
        )
        self.sink.write(event)
        EVENTS.labels(grade=event.grade.grade.value, route=event.route.route).inc()
        TOTAL_LATENCY_MS.observe(total_ms)
        return event
