from __future__ import annotations


def projection_delay_ms(distance_m: float, speed_mps: float, processing_ms: float) -> int:
    """Time until the garment reaches a fixed projection zone.

    The conveyor encoder is the source of truth in production. This calculation is
    a useful fallback / simulation and makes the latency budget explicit.
    """
    if distance_m < 0:
        raise ValueError("distance_m must be >= 0")
    if speed_mps <= 0:
        raise ValueError("speed_mps must be > 0")

    arrival_ms = distance_m / speed_mps * 1000.0
    return max(0, int(round(arrival_ms - processing_ms)))
