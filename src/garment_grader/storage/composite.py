from __future__ import annotations

import logging
from collections.abc import Iterable

from garment_grader.core.schemas import GarmentEvent
from garment_grader.storage.base import EventSink

log = logging.getLogger(__name__)


class CompositeEventSink:
    """Write to a required local sink and optional best-effort remote sinks."""

    def __init__(self, required: EventSink, optional: Iterable[EventSink] = ()) -> None:
        self.required = required
        self.optional = list(optional)

    def write(self, event: GarmentEvent) -> None:
        # Local durability is part of the decision contract.
        self.required.write(event)
        # Cloud/control-plane failures must not stop the conveyor.
        for sink in self.optional:
            try:
                sink.write(event)
            except Exception:
                log.exception("optional event sink failed; local event remains durable")
