"""
ml/inference.py
================
Bridges the FastAPI app to `ml_pipeline` (a sibling directory, not a
package under `app/`, since it's also meant to be run standalone for
training/evaluation) and to the MediaPipe-based hand detector. Loads
everything ONCE at app startup and reuses it across requests -- a fresh
Predictor per-request would reload weights from disk on every single API
call, which is both slow and pointless since the model never changes
between requests.

Prediction flow per request:
    1. Decode input bytes to an RGB numpy array.
    2. If the hand detector is available, run it on the FULL frame.
       - Hand found: crop to its (padded) bounding box, classify the CROP
         (this matches the training data, which is close-up hand shots,
         not wide scenes), return hand_detected=True + the bounding box.
       - No hand found: classify the FULL frame anyway (better to give an
         answer than refuse), return hand_detected=False + no bounding box.
    3. If the hand detector isn't available (model file not downloaded --
       see hand_detector.py's docstring), classify the full frame and
       return hand_detected=None, distinguishing "we checked and found
       nothing" from "we never checked".
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# ml_pipeline is a sibling of app/, not a sub-package -- add it to sys.path
# once, here, so `from config import ExperimentConfig` / `from predict
# import Predictor` (ml_pipeline's own internal import style) resolve
# correctly regardless of the process's working directory.
_ML_PIPELINE_DIR = Path(__file__).resolve().parents[2] / "ml_pipeline"
if str(_ML_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_PIPELINE_DIR))


class ModelNotLoadedError(RuntimeError):
    """Raised by predict() when no checkpoint has been trained/placed yet."""


class InferenceService:
    """Singleton-ish wrapper (one instance created at app startup, stored
    on app.state) around ml_pipeline's Predictor and the hand detector."""

    def __init__(self):
        self._predictor = None
        self._hand_detector = None
        self._architecture: Optional[str] = None
        self._device: Optional[str] = None
        self._load_error: Optional[str] = None
        self._hand_detector_error: Optional[str] = None

    def load(self) -> None:
        settings = get_settings()
        self._load_predictor(settings)
        self._load_hand_detector(settings)

    def _load_predictor(self, settings) -> None:
        checkpoint_path = Path(settings.ML_CHECKPOINT_PATH)
        config_path = Path(settings.ML_CONFIG_PATH)

        if not checkpoint_path.exists():
            msg = (
                f"No trained checkpoint found at {checkpoint_path}. The API will "
                f"start, but /predict and /predict-webcam will return 503 until "
                f"you train a model (see ml_pipeline/README or the root README's "
                f"Training section) and place its checkpoint at this path, or "
                f"update ML_CHECKPOINT_PATH in your .env."
            )
            logger.warning(msg)
            self._load_error = msg
            if settings.REQUIRE_MODEL_ON_STARTUP:
                raise ModelNotLoadedError(msg)
            return

        try:
            from config import ExperimentConfig  # ml_pipeline's config.py
            from predict import Predictor  # ml_pipeline's predict.py

            cfg = ExperimentConfig.load(str(config_path))
            self._predictor = Predictor(cfg, str(checkpoint_path))
            self._architecture = cfg.model.architecture
            self._device = str(self._predictor.device)
            self._load_error = None
            logger.info(f"Loaded model '{self._architecture}' from {checkpoint_path} onto {self._device}")
        except Exception as e:
            msg = f"Failed to load model from {checkpoint_path}: {e}"
            logger.error(msg)
            self._load_error = msg
            if settings.REQUIRE_MODEL_ON_STARTUP:
                raise

    def _load_hand_detector(self, settings) -> None:
        try:
            from app.ml.hand_detector import HandDetector

            self._hand_detector = HandDetector(settings.HAND_DETECTOR_MODEL_PATH)
            self._hand_detector_error = None
            logger.info(f"Loaded hand detector from {settings.HAND_DETECTOR_MODEL_PATH}")
        except Exception as e:
            msg = (
                f"Hand detector not available ({e}). Predictions will classify the "
                f"full frame with no bounding box until the MediaPipe model file is "
                f"downloaded -- see the setup command in hand_detector.py's docstring."
            )
            logger.warning(msg)
            self._hand_detector_error = msg
            if settings.REQUIRE_HAND_DETECTOR_ON_STARTUP:
                raise

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    @property
    def architecture(self) -> Optional[str]:
        return self._architecture

    @property
    def device(self) -> Optional[str]:
        return self._device

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def hand_detector_available(self) -> bool:
        return self._hand_detector is not None

    def predict(self, image, top_k: int = 5) -> tuple[dict, float]:
        """Returns (result_dict, latency_ms). Raises ModelNotLoadedError if
        no classifier model has been loaded -- the API layer turns that
        into a 503. result_dict always includes predicted_class,
        confidence, top_k, bounding_box (nullable), and hand_detected
        (nullable tri-state -- see module docstring)."""
        if self._predictor is None:
            raise ModelNotLoadedError(self._load_error or "Model not loaded.")

        start = time.perf_counter()

        image_to_classify = image
        bounding_box = None
        hand_detected = None

        if self._hand_detector is not None:
            from predict import _to_numpy_rgb  # ml_pipeline's own normalization helper

            rgb_array = _to_numpy_rgb(image)
            box = self._hand_detector.detect(rgb_array)

            if box is not None:
                from app.ml.hand_detector import crop_to_bounding_box

                image_to_classify = crop_to_bounding_box(rgb_array, box)
                bounding_box = box.as_dict()
                hand_detected = True
            else:
                image_to_classify = rgb_array
                hand_detected = False

        result = self._predictor.predict(image_to_classify, top_k=top_k)
        result["bounding_box"] = bounding_box
        result["hand_detected"] = hand_detected

        latency_ms = (time.perf_counter() - start) * 1000
        return result, latency_ms
