from __future__ import annotations

from garment_grader.core.config import Settings
from garment_grader.storage.composite import CompositeEventSink
from garment_grader.storage.jsonl_sink import JsonlEventSink


def build_event_sink(settings: Settings):
    local = JsonlEventSink(settings.event_log_path)
    optional = []
    if settings.supabase_url and settings.supabase_key:
        from garment_grader.storage.supabase_sink import SupabaseEventSink

        optional.append(
            SupabaseEventSink(
                url=settings.supabase_url,
                key=settings.supabase_key,
                table=settings.supabase_table,
            )
        )
    return CompositeEventSink(required=local, optional=optional)
