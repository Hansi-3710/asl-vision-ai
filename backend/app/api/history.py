"""api/history.py -- GET /history, paginated list of past predictions."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.database import get_db
from app.db.models import Prediction
from app.schemas.prediction import PredictionResponse

router = APIRouter(tags=["history"])


@router.get("/history", response_model=list[PredictionResponse])
def get_history(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    limit: int | None = Query(default=None, ge=1),
    offset: int = Query(default=0, ge=0),
    source: str | None = Query(default=None, description="Filter by 'upload' or 'webcam'"),
    predicted_class: str | None = Query(default=None, description="Filter by predicted letter/token"),
) -> list[PredictionResponse]:
    effective_limit = min(limit or settings.HISTORY_DEFAULT_LIMIT, settings.HISTORY_MAX_LIMIT)

    query = db.query(Prediction)
    if source is not None:
        query = query.filter(Prediction.source == source)
    if predicted_class is not None:
        query = query.filter(Prediction.predicted_class == predicted_class)

    records = (
        query.order_by(Prediction.created_at.desc())
        .offset(offset)
        .limit(effective_limit)
        .all()
    )
    return [PredictionResponse.model_validate(r) for r in records]
