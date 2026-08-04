"""schemas/metrics.py -- response contracts for /metrics and /health."""

from __future__ import annotations

from pydantic import BaseModel


class LetterCount(BaseModel):
    letter: str
    count: int


class MetricsResponse(BaseModel):
    total_predictions: int
    average_confidence: float
    average_latency_ms: float
    most_predicted_letters: list[LetterCount]
    predictions_by_source: dict[str, int]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_architecture: str | None = None
    device: str | None = None
    # Continuous (sentence-level) recognition model, served over /ws/stream --
    # separate from the alphabet classifier fields above, since the two
    # models load independently and either can be ready while the other isn't.
    sequence_model_loaded: bool = False
    sequence_model_is_placeholder: bool = True
