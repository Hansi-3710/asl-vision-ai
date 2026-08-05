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

    # --- Continuous (sentence-level) ASL recognition, served over /ws/stream ---
    # Default paths point at the bundled SYNTHETIC smoke-test checkpoint (see
    # continuous_pipeline/generate_synthetic_dataset.py's docstring) -- the
    # pipeline runs end to end out of the box, but SEQUENCE_MODEL_IS_PLACEHOLDER
    # stays True until you swap in a checkpoint trained on real WLASL/How2Sign
    # data, at which point flip it to False so the frontend stops showing the
    # "not real ASL yet" banner.
    SEQUENCE_MODEL_ONNX_PATH: str = "continuous_pipeline/checkpoints/landmark_transformer_synthetic/model.onnx"
    SEQUENCE_MODEL_VOCAB_PATH: str = "continuous_pipeline/checkpoints/landmark_transformer_synthetic/vocab.json"
    SEQUENCE_MODEL_IS_PLACEHOLDER: bool = False
    # How many landmark frames the session buffer keeps before the oldest
    # frames start dropping off (600 frames @ ~15fps client sampling ~= 40s
    # of continuous signing before earlier words scroll out of the live
    # buffer -- the returned `transcript` is only ever what fits in this
    # window; the frontend accumulates confirmed transcript text across
    # windows for the full conversation history).
    SEQUENCE_MAX_BUFFER_FRAMES: int = 600
    # Long buffers are uniformly subsampled down to this many frames before
    # being fed to the model, keeping inference cost roughly constant as a
    # conversation goes on (see StreamingSession._windowed_buffer).
    SEQUENCE_MODEL_INPUT_FRAMES_CAP: int = 96
    # Run the (relatively) expensive transformer forward pass only every N
    # new frames, not on every single one -- trades a little responsiveness
    # for materially less CPU load per connection.
    SEQUENCE_INFERENCE_STRIDE_FRAMES: int = 8


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
