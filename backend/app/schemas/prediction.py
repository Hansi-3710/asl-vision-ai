"""schemas/prediction.py -- request/response contracts for the /predict* endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TopKEntry(BaseModel):
    class_: str = Field(alias="class")
    confidence: float

    model_config = {"populate_by_name": True}


class BoundingBoxSchema(BaseModel):
    """Normalized (0-1) hand bounding box -- fraction of frame width/height,
    so the frontend can draw it correctly at any displayed video size."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float


class PredictionResponse(BaseModel):
    id: str
    predicted_class: str
    confidence: float
    top_k: list[TopKEntry]
    source: str
    latency_ms: float
    image_path: str | None = None
    bounding_box: BoundingBoxSchema | None = None
    # True=found, False=ran but found none (full-frame fallback used), None=detector unavailable
    hand_detected: bool | None = None
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class WebcamFrameRequest(BaseModel):
    """The frontend sends webcam frames as a base64-encoded JPEG/PNG data
    URL (e.g. from HTMLCanvasElement.toDataURL()) rather than multipart
    file upload, since that's the natural format coming out of a <canvas>
    snapshot of a <video> element."""

    image_base64: str = Field(..., description="Base64-encoded image, with or without the data: URL prefix")
