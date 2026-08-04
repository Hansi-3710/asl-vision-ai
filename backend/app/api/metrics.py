"""api/metrics.py -- GET /metrics, aggregate stats for the dashboard."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Prediction
from app.schemas.metrics import LetterCount, MetricsResponse

router = APIRouter(tags=["metrics"])


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(db: Session = Depends(get_db)) -> MetricsResponse:
    total = db.query(func.count(Prediction.id)).scalar() or 0

    if total == 0:
        return MetricsResponse(
            total_predictions=0,
            average_confidence=0.0,
            average_latency_ms=0.0,
            most_predicted_letters=[],
            predictions_by_source={},
        )

    avg_confidence = db.query(func.avg(Prediction.confidence)).scalar() or 0.0
    avg_latency = db.query(func.avg(Prediction.latency_ms)).scalar() or 0.0

    letter_counts = (
        db.query(Prediction.predicted_class, func.count(Prediction.id).label("count"))
        .group_by(Prediction.predicted_class)
        .order_by(func.count(Prediction.id).desc())
        .limit(10)
        .all()
    )

    source_counts = (
        db.query(Prediction.source, func.count(Prediction.id))
        .group_by(Prediction.source)
        .all()
    )

    return MetricsResponse(
        total_predictions=total,
        average_confidence=round(float(avg_confidence), 4),
        average_latency_ms=round(float(avg_latency), 2),
        most_predicted_letters=[LetterCount(letter=letter, count=count) for letter, count in letter_counts],
        predictions_by_source={source: count for source, count in source_counts},
    )
