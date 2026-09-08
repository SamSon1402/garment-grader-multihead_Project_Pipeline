from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from garment_grader.core.config import get_settings
from garment_grader.core.pipeline import GarmentPipeline
from garment_grader.core.policy import CustomerPolicy
from garment_grader.core.sample_image import make_demo_frame
from garment_grader.integrations.provider_factory import build_provider
from garment_grader.storage.factory import build_event_sink
from garment_grader.telemetry.metrics import READY


class SimulateRequest(BaseModel):
    garment_id: str
    customer: str = "retextil_demo"


@lru_cache

def build_pipeline() -> GarmentPipeline:
    settings = get_settings()
    policy = CustomerPolicy.load(settings.policy_path)
    provider = build_provider(settings)
    sink = build_event_sink(settings)
    READY.set(1)
    return GarmentPipeline(settings=settings, policy=policy, provider=provider, sink=sink)


app = FastAPI(
    title="GarmentGrader Industrial",
    version="0.1.0",
    description="Edge CV grading/routing reference implementation",
)


@app.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/ready")
def ready() -> dict[str, str]:
    try:
        p = build_pipeline()
        return {"status": "ready", "provider": type(p.provider).__name__, "customer": p.policy.customer}
    except Exception as exc:
        READY.set(0)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/simulate")
def simulate(req: SimulateRequest):
    pipeline = build_pipeline()
    if req.customer != pipeline.policy.customer:
        raise HTTPException(
            status_code=400,
            detail=f"loaded policy is {pipeline.policy.customer}; set GG_POLICY_PATH for customer={req.customer}",
        )
    frame = make_demo_frame(req.garment_id)
    return pipeline.process(req.garment_id, frame)
