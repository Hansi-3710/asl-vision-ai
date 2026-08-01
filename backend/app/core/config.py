"""
core/config.py
===============
Centralized application settings, loaded from environment variables (with
sensible local-dev defaults) via pydantic-settings. Every configurable
path/URL the app needs lives here -- nothing is hardcoded elsewhere.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    APP_NAME: str = "ASL Vision AI"
    ENVIRONMENT: str = "development"  # development | production
    DEBUG: bool = True

    # --- CORS ---
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # --- Database ---
    DATABASE_URL: str = "sqlite:///./data/asl_vision.db"

    # --- ML model ---
    ML_CONFIG_PATH: str = "ml_pipeline/configs/baseline.yaml"
    ML_CHECKPOINT_PATH: str = "ml_pipeline/checkpoints/efficientnet_v2_s_baseline/best.pt"
    # If no checkpoint exists yet (fresh clone, before the user has trained
    # a model), the API still starts up -- /predict returns a clear 503
    # instead of the whole process crashing at import time. See
    # app/ml/inference.py.
    REQUIRE_MODEL_ON_STARTUP: bool = False

    # --- Hand detection (MediaPipe) ---
    HAND_DETECTOR_MODEL_PATH: str = "ml_pipeline/checkpoints/mediapipe/hand_landmarker.task"
    # If the model file above hasn't been downloaded yet (see
    # app/ml/hand_detector.py's docstring for the one-time setup command),
    # the API falls back to classifying the full frame with no bounding
    # box, rather than failing to start. Set this True once you've
    # downloaded it, to require it (matches REQUIRE_MODEL_ON_STARTUP's
    # pattern).
    REQUIRE_HAND_DETECTOR_ON_STARTUP: bool = False

    # --- Uploads ---
    MAX_UPLOAD_SIZE_MB: int = 10
    UPLOAD_DIR: str = "data/uploads"

    # --- Prediction history ---
    HISTORY_DEFAULT_LIMIT: int = 50
    HISTORY_MAX_LIMIT: int = 500


@lru_cache
def get_settings() -> Settings:
    """Cached so Settings() is only constructed once per process (FastAPI's
    Depends(get_settings) pattern relies on this for cheap DI)."""
    return Settings()


def ensure_runtime_dirs(settings: Settings) -> None:
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    if settings.DATABASE_URL.startswith("sqlite"):
        db_path = settings.DATABASE_URL.replace("sqlite:///", "", 1)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
