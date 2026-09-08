from __future__ import annotations

from typing import Protocol

from garment_grader.core.schemas import GarmentEvent


class EventSink(Protocol):
    def write(self, event: GarmentEvent) -> None:
        ...
