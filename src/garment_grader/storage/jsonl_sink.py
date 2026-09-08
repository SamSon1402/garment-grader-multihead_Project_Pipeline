from __future__ import annotations

from pathlib import Path
from threading import Lock

from garment_grader.core.schemas import GarmentEvent


class JsonlEventSink:
    """Small durable edge queue.

    Production code would rotate files / use SQLite or a durable local queue. The
    important design property is that cloud availability is not required for the
    grading decision.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def write(self, event: GarmentEvent) -> None:
        payload = event.model_dump_json()
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(payload + "\n")
