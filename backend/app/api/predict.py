"""
api/predict.py
===============
POST /predict          -- multipart image upload
POST /predict-webcam   -- base64-encoded frame from the browser's <canvas>

Both endpoints funnel through the same InferenceService and record every
prediction to the database (Section "Database" requirement), differing
only in how the image arrives and whether it's saved to disk.
"""

from __future__ import annotations

import base64
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_inference_service
from app.core.config import Settings, get_settings
from app.core.logging_config import get_logger
from app.db.database import get_db
from app.db.models import Prediction
from app.ml.inference import ModelNotLoadedError
from app.schemas.prediction import PredictionResponse, WebcamFrameRequest

router = APIRouter(tags=["prediction"])
logger = get_logger(__name__)


def _save_prediction(db: Session, result: dict, source: str, latency_ms: float, image_path: str | None) -> Prediction:
    record = Prediction(
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        top_k=result["top_k"],
        source=source,
        latency_ms=latency_ms,
        image_path=image_path,
        bounding_box=result.get("bounding_box"),
        hand_detected=result.get("hand_detected"),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def _run_inference_or_503(inference_service, image, top_k: int = 5):
    try:
        return inference_service.predict(image, top_k=top_k)
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=503, detail=f"Model not available yet: {e}")
    except (OSError, ValueError) as e:
        # OSError covers PIL.UnidentifiedImageError (corrupted upload,
        # mislabeled content-type, or base64 that decodes to non-image
        # bytes) -- a client error, not a server error. ValueError covers
        # malformed-array edge cases from the numpy/Albumentations side.
        # Deliberately NOT a blanket `except Exception` here: that would
        # also catch genuine server-side bugs (a model runtime error, a
        # code bug) and mislabel them as the client's fault, hiding real
        # issues behind a generic 400 instead of surfacing them as 500s.
        logger.info(f"Rejecting request: input did not decode to a valid image ({e})")
        raise HTTPException(status_code=400, detail="Could not decode input as a valid image.")


@router.post("/predict", response_model=PredictionResponse)
async def predict_upload(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    inference_service=Depends(get_inference_service),
) -> PredictionResponse:
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")

    raw_bytes = await file.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit.")

    result, latency_ms = _run_inference_or_503(inference_service, raw_bytes)

    # Persist the uploaded image so the dashboard/history can reference it.
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    saved_name = f"{uuid.uuid4()}{suffix}"
    saved_path = upload_dir / saved_name
    saved_path.write_bytes(raw_bytes)

    record = _save_prediction(db, result, source="upload", latency_ms=latency_ms, image_path=str(saved_path))
    return PredictionResponse.model_validate(record)


@router.post("/predict-webcam", response_model=PredictionResponse)
async def predict_webcam(
    payload: WebcamFrameRequest,
    db: Session = Depends(get_db),
    inference_service=Depends(get_inference_service),
) -> PredictionResponse:
    raw_b64 = payload.image_base64
    if "," in raw_b64 and raw_b64.strip().startswith("data:"):
        # Strip a "data:image/jpeg;base64," prefix if the frontend sent a full data URL.
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        # validate=True is essential here: without it, b64decode SILENTLY
        # discards invalid characters instead of raising, which can turn
        # garbage input into "successfully decoded" garbage bytes that only
        # fail later (as an unhandled 500 from the image decoder) instead
        # of a clean 400 here.
        raw_bytes = base64.b64decode(raw_b64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="image_base64 is not valid base64.")

    result, latency_ms = _run_inference_or_503(inference_service, raw_bytes)

    # Webcam frames are NOT persisted to disk (there'd be one per frame at
    # several FPS -- that's a lot of throwaway images) -- only the
    # prediction record is stored, matching "Store every prediction" while
    # keeping disk usage sane. Change this if you want frame retention.
    record = _save_prediction(db, result, source="webcam", latency_ms=latency_ms, image_path=None)
    return PredictionResponse.model_validate(record)
