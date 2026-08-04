"""
api/stream.py
=============
WS /ws/stream

The continuous-recognition counterpart to predict.py's REST endpoints.
Protocol is documented in schemas/streaming.py. One StreamingSession is
created per connection (cheap: a deque of feature vectors + a short word
list) against the single shared SequenceModelBackend loaded once at
startup (expensive: the actual model weights) -- exactly the "one shared
model instance" backend responsibility called for in the source design
doc.

A malformed individual frame (wrong length, non-numeric, NaN) sends a
{"type": "error"} message and CONTINUES the connection rather than
closing it -- one bad frame from a flaky client shouldn't kill an
otherwise-fine session. Anything else unexpected also degrades to an
error message; only an actual WebSocketDisconnect ends the loop.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from app.api.deps import get_sequence_backend_ws
from app.core.config import Settings, get_settings
from app.core.logging_config import get_logger
from app.ml.landmarks import InvalidLandmarkFrame, parse_landmark_frame
from app.ml.sequence_recognizer import SequenceModelBackend, StreamingSession

router = APIRouter(tags=["streaming"])
logger = get_logger(__name__)


@router.websocket("/ws/stream")
async def stream_recognition(
    websocket: WebSocket,
    backend: SequenceModelBackend = Depends(get_sequence_backend_ws),
    settings: Settings = Depends(get_settings),
):
    await websocket.accept()

    session = StreamingSession(
        backend=backend,
        max_buffer_frames=settings.SEQUENCE_MAX_BUFFER_FRAMES,
        model_input_frames_cap=settings.SEQUENCE_MODEL_INPUT_FRAMES_CAP,
        inference_stride_frames=settings.SEQUENCE_INFERENCE_STRIDE_FRAMES,
    )

    await websocket.send_json({
        "type": "ready",
        "model_ready": backend.is_ready,
        "is_synthetic_placeholder": backend.is_placeholder,
        "feature_dim": 258,
        "vocab_size": len(backend.vocab) if backend.vocab else 0,
        "window_frames": settings.SEQUENCE_MODEL_INPUT_FRAMES_CAP,
    })

    try:
        while True:
            payload = await websocket.receive_json()
            msg_type = payload.get("type")

            if msg_type == "frame":
                try:
                    features = parse_landmark_frame(payload.get("features", []))
                except InvalidLandmarkFrame as e:
                    await websocket.send_json({"type": "error", "message": str(e)})
                    continue

                session.add_frame(features)
                if session.should_run_inference:
                    result = session.run_inference()
                    await websocket.send_json({"type": "update", **result})

            elif msg_type == "reset":
                session.reset()
                await websocket.send_json({"type": "update", **session.run_inference()})

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            else:
                await websocket.send_json({"type": "error", "message": f"Unknown message type: {msg_type!r}"})

    except WebSocketDisconnect:
        logger.info("Streaming client disconnected.")
