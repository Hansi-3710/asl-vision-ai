"""schemas/streaming.py -- message contracts for the /ws/stream WebSocket.

The protocol is deliberately tiny JSON messages, both directions, tagged
by a "type" field:

Client -> Server:
    {"type": "frame", "features": [258 floats], "timestamp_ms": 173...}
    {"type": "reset"}

Server -> Client:
    {"type": "ready", ...StreamReadyMessage}     -- once, right after connect
    {"type": "update", ...StreamUpdateMessage}   -- after each inference tick
    {"type": "error", "message": "..."}          -- malformed frame, doesn't close the connection
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LandmarkFrameMessage(BaseModel):
    type: str = Field("frame", frozen=True)
    features: list[float]
    timestamp_ms: int | None = None


class ResetMessage(BaseModel):
    type: str = Field("reset", frozen=True)


class StreamReadyMessage(BaseModel):
    type: str = Field("ready", frozen=True)
    model_ready: bool
    is_synthetic_placeholder: bool
    feature_dim: int
    vocab_size: int
    window_frames: int


class WordPrediction(BaseModel):
    word: str
    confidence: float


class StreamUpdateMessage(BaseModel):
    type: str = Field("update", frozen=True)
    transcript: str
    words: list[WordPrediction]
    buffer_frames: int
    latency_ms: float
    fps: float


class StreamErrorMessage(BaseModel):
    type: str = Field("error", frozen=True)
    message: str
