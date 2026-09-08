from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

INFER_LATENCY_MS = Histogram(
    "garment_grader_inference_ms",
    "Vision provider inference latency",
    buckets=(5, 10, 20, 30, 50, 75, 100, 150, 250, 500),
)
TOTAL_LATENCY_MS = Histogram(
    "garment_grader_total_ms",
    "End-to-end decision latency",
    buckets=(10, 20, 30, 50, 75, 100, 150, 250, 500, 1000),
)
EVENTS = Counter("garment_grader_events_total", "Garment decisions", ["grade", "route"])
INPUT_FAILURES = Counter("garment_grader_input_failures_total", "Input quality failures", ["sensor", "reason"])
READY = Gauge("garment_grader_ready", "1 when service is ready")
