from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class Grade(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    REVIEW = "REVIEW"


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def ordered(self) -> "Box":
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("box must have positive width and height")
        return self


class Defect(BaseModel):
    kind: str
    confidence: float = Field(ge=0, le=1)
    severity: float = Field(ge=0, le=1)
    box: Box | None = None


class FrameQuality(BaseModel):
    brightness: float
    blur_variance: float
    valid: bool
    reasons: list[str] = Field(default_factory=list)


class NIRReading(BaseModel):
    polyester: float = Field(ge=0, le=1)
    cotton: float = Field(ge=0, le=1)
    wool: float = Field(ge=0, le=1)
    other: float = Field(ge=0, le=1)
    signal_quality: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def composition_is_reasonable(self) -> "NIRReading":
        total = self.polyester + self.cotton + self.wool + self.other
        if not 0.97 <= total <= 1.03:
            raise ValueError(f"fiber composition must sum to ~1.0, got {total:.3f}")
        return self


class VisionResult(BaseModel):
    category: str
    category_confidence: float = Field(ge=0, le=1)
    brand_top5: list[tuple[str, float]]
    attributes: dict[str, tuple[str, float]]
    defects: list[Defect]
    embedding_norm: float = Field(ge=0)
    ood_score: float = Field(ge=0, le=1)
    model_version: str
    inference_ms: float = Field(ge=0)

    @property
    def max_defect_severity(self) -> float:
        return max((d.severity for d in self.defects), default=0.0)

    @property
    def min_task_confidence(self) -> float:
        values = [self.category_confidence]
        values.extend(score for _, score in self.brand_top5[:1])
        values.extend(score for _, score in self.attributes.values())
        values.extend(d.confidence for d in self.defects)
        return min(values) if values else self.category_confidence


class GradeDecision(BaseModel):
    grade: Grade
    confidence: float = Field(ge=0, le=1)
    reason: str


class PriceDecision(BaseModel):
    estimated_eur: float = Field(ge=0)
    explanation: str


class RouteDecision(BaseModel):
    route: str
    reason: str
    requires_human: bool = False


class ProjectionCommand(BaseModel):
    enabled: bool
    fire_after_ms: int = Field(ge=0)
    grade: Grade
    route: str
    label: str
    projected_defects: list[Defect] = Field(default_factory=list)


class GarmentEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    garment_id: str
    customer: str
    policy_version: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    frame_quality: FrameQuality
    nir: NIRReading
    vision: VisionResult
    grade: GradeDecision
    price: PriceDecision
    route: RouteDecision
    projection: ProjectionCommand
    total_latency_ms: float = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
