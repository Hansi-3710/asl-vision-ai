"""
db/models.py
============
Single table: every prediction the API serves gets recorded here (Section
"Database" requirement -- prediction, confidence, timestamp, latency,
image path if uploaded).
"""

from __future__ import annotations

import uuid

from sqlalchemy import Boolean, Column, String, Float, DateTime, JSON
from sqlalchemy.sql import func

from app.db.database import Base


def _new_uuid() -> str:
    return str(uuid.uuid4())


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(String, primary_key=True, default=_new_uuid)
    predicted_class = Column(String, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    top_k = Column(JSON, nullable=False)  # list of {"class": str, "confidence": float}
    source = Column(String, nullable=False, default="upload")  # "upload" | "webcam"
    latency_ms = Column(Float, nullable=False)
    image_path = Column(String, nullable=True)  # relative path under UPLOAD_DIR, null for webcam frames
    bounding_box = Column(JSON, nullable=True)  # {"x_min", "y_min", "x_max", "y_max", "confidence"}, null if not detected
    hand_detected = Column(Boolean, nullable=True)  # NULL = hand detection wasn't available for this request
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "top_k": self.top_k,
            "source": self.source,
            "latency_ms": self.latency_ms,
            "image_path": self.image_path,
            "bounding_box": self.bounding_box,
            "hand_detected": self.hand_detected,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
