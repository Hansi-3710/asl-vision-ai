"""api/health.py -- GET /health"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_inference_service, get_sequence_backend
from app.schemas.metrics import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check(
    inference_service=Depends(get_inference_service),
    sequence_backend=Depends(get_sequence_backend),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=inference_service.is_loaded,
        model_architecture=inference_service.architecture,
        device=inference_service.device,
        sequence_model_loaded=sequence_backend.is_ready,
        sequence_model_is_placeholder=sequence_backend.is_placeholder,
    )
