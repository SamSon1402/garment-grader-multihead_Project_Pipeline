from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from garment_grader.core.quality import QualityThresholds
from garment_grader.core.schemas import Grade, GradeDecision, PriceDecision, RouteDecision, VisionResult


@dataclass(frozen=True)
class CustomerPolicy:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "CustomerPolicy":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))

    @property
    def customer(self) -> str:
        return str(self.raw["customer"])

    @property
    def version(self) -> int:
        return int(self.raw["version"])

    @property
    def quality_thresholds(self) -> QualityThresholds:
        q = self.raw["quality"]
        return QualityThresholds(**q)

    def decide_grade(
        self,
        vision: VisionResult,
        *,
        frame_ok: bool,
        nir_ok: bool,
        review_threshold: float,
    ) -> GradeDecision:
        if not frame_ok:
            return GradeDecision(grade=Grade.REVIEW, confidence=0.0, reason="camera_input_failed_quality_gate")
        if not nir_ok:
            return GradeDecision(grade=Grade.REVIEW, confidence=0.0, reason="nir_input_failed_quality_gate")
        if vision.ood_score >= 0.75:
            return GradeDecision(grade=Grade.REVIEW, confidence=1.0 - vision.ood_score, reason="out_of_distribution")

        confidence = min(vision.min_task_confidence, 1.0 - vision.ood_score)
        if confidence < review_threshold:
            return GradeDecision(grade=Grade.REVIEW, confidence=confidence, reason="low_model_confidence")

        severity = vision.max_defect_severity
        for grade in (Grade.A, Grade.B, Grade.C):
            rule = self.raw["grade"][grade.value]
            if severity <= float(rule["max_defect_severity"]) and confidence >= float(rule["min_confidence"]):
                return GradeDecision(
                    grade=grade,
                    confidence=confidence,
                    reason=f"severity={severity:.2f}, confidence={confidence:.2f} matched {grade.value}",
                )

        return GradeDecision(grade=Grade.REVIEW, confidence=confidence, reason="no_safe_grade_rule_matched")

    def price(self, vision: VisionResult, grade: Grade) -> PriceDecision:
        pricing = self.raw["pricing"]
        base = float(pricing["base_eur"].get(vision.category, pricing["base_eur"]["unknown"]))
        multiplier = float(pricing["grade_multiplier"][grade.value])

        # Small brand signal. Production code would use a calibrated valuation model.
        brand_conf = vision.brand_top5[0][1] if vision.brand_top5 else 0.0
        brand_boost = 1.10 if brand_conf >= 0.90 else 1.0
        value = round(base * multiplier * brand_boost, 2)
        return PriceDecision(
            estimated_eur=value,
            explanation=f"base={base:.2f} × grade={multiplier:.2f} × brand={brand_boost:.2f}",
        )

    def route(self, grade: Grade, price: float, severity: float) -> RouteDecision:
        r = self.raw["routing"]
        if grade is Grade.REVIEW:
            return RouteDecision(route="manual_review", reason="policy requires human review", requires_human=True)

        premium = r.get("premium_resale", {})
        if grade.value in premium.get("grades", []) and price >= float(premium.get("min_value_eur", 1e9)):
            return RouteDecision(route="premium_resale", reason="high-value Grade A item")

        resale = r.get("resale", {})
        if grade.value in resale.get("grades", []) and price >= float(resale.get("min_value_eur", 1e9)):
            return RouteDecision(route="resale", reason="meets resale grade/value rules")

        repair = r.get("repair", {})
        if grade.value in repair.get("grades", []) and severity <= float(repair.get("max_defect_severity", -1)):
            return RouteDecision(route="repair", reason="defect severity is inside repair envelope")

        recycling = r.get("recycling", {})
        if grade.value in recycling.get("grades", []) and severity >= float(recycling.get("min_defect_severity", 2)):
            return RouteDecision(route="recycling", reason="defect severity exceeds repair envelope")

        return RouteDecision(route="manual_review", reason="no routing rule matched safely", requires_human=True)

    def should_project(self, grade: Grade) -> bool:
        return grade.value in self.raw.get("projection", {}).get("enabled_for", [])

    def show_defects(self) -> bool:
        return bool(self.raw.get("projection", {}).get("show_defects", True))
