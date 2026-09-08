from __future__ import annotations

from garment_grader.core.schemas import GarmentEvent


class SupabaseEventSink:
    """Optional cloud/control-plane sink.

    This adapter is deliberately not used as the only sink because network failure
    must not block the conveyor decision path.
    """

    def __init__(self, url: str, key: str, table: str = "garment_events") -> None:
        try:
            from supabase import create_client
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install with: pip install -e .[supabase]") from exc
        self.client = create_client(url, key)
        self.table = table

    def write(self, event: GarmentEvent) -> None:
        row = event.model_dump(mode="json")
        self.client.table(self.table).insert(row).execute()
