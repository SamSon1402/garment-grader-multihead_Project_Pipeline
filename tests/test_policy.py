from garment_grader.core.policy import CustomerPolicy
from garment_grader.core.schemas import Defect, VisionResult


def make_vision(severity: float = 0.1, confidence: float = 0.95, ood: float = 0.05) -> VisionResult:
    return VisionResult(
        category="jacket",
        category_confidence=confidence,
        brand_top5=[("Patagonia", confidence)],
        attributes={"style": ("outdoor", confidence)},
        defects=[Defect(kind="stain", confidence=confidence, severity=severity)],
        embedding_norm=20,
        ood_score=ood,
        model_version="test",
        inference_ms=10,
    )


def test_grade_a_for_clean_high_confidence_item():
    policy = CustomerPolicy.load("configs/retextil_demo.yaml")
    result = policy.decide_grade(make_vision(0.10, 0.95), frame_ok=True, nir_ok=True, review_threshold=0.72)
    assert result.grade.value == "A"


def test_ood_forces_review():
    policy = CustomerPolicy.load("configs/retextil_demo.yaml")
    result = policy.decide_grade(make_vision(0.05, 0.99, ood=0.90), frame_ok=True, nir_ok=True, review_threshold=0.72)
    assert result.grade.value == "REVIEW"
