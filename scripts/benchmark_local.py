from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from garment_grader.core.config import Settings
from garment_grader.core.pipeline import GarmentPipeline
from garment_grader.core.policy import CustomerPolicy
from garment_grader.core.sample_image import make_demo_frame
from garment_grader.integrations.provider_factory import build_provider
from garment_grader.storage.jsonl_sink import JsonlEventSink


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    settings = Settings(event_log_path="data/benchmark-events.jsonl")
    policy = CustomerPolicy.load(settings.policy_path)
    pipeline = GarmentPipeline(
        settings=settings,
        policy=policy,
        provider=build_provider(settings),
        sink=JsonlEventSink(settings.event_log_path),
    )

    latencies: list[float] = []
    for i in range(args.n):
        frame = make_demo_frame(f"bench-{i}")
        start = time.perf_counter()
        pipeline.process(f"bench-{i}", frame)
        latencies.append((time.perf_counter() - start) * 1000)

    print(f"n={len(latencies)}")
    print(f"mean={statistics.mean(latencies):.2f} ms")
    print(f"p50={percentile(latencies, 50):.2f} ms")
    print(f"p95={percentile(latencies, 95):.2f} ms")
    print(f"p99={percentile(latencies, 99):.2f} ms")
    print("NOTE: mock-provider numbers are software plumbing only, not model/Jetson benchmarks.")


if __name__ == "__main__":
    main()
