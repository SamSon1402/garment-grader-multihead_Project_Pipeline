from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings.

    Environment variables use the GG_ prefix. Secrets for third-party services
    intentionally remain outside this model where the vendor SDK expects them.
    """

    model_config = SettingsConfigDict(env_prefix="GG_", env_file=".env", extra="ignore")

    env: str = "dev"
    vision_provider: str = "mock"
    policy_path: Path = Path("configs/retextil_demo.yaml")
    event_log_path: Path = Path("data/events.jsonl")
    review_threshold: float = 0.72

    projection_zone_m: float = 1.8
    conveyor_speed_mps: float = 0.45

    roboflow_api_url: str = "http://127.0.0.1:9001"
    roboflow_model_id: str = "rfdetr-small"

    supabase_url: str | None = None
    supabase_key: str | None = None
    supabase_table: str = "garment_events"


@lru_cache

def get_settings() -> Settings:
    return Settings()
