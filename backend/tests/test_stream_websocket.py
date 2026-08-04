"""tests/test_stream_websocket.py -- end-to-end /ws/stream protocol test.

Uses `app.dependency_overrides[get_sequence_backend_ws]` the same way
conftest.py's `client`/`client_no_model` fixtures override
get_inference_service -- swapping in a scripted fake backend so this test
doesn't depend on whether a real (or even placeholder) ONNX checkpoint
happens to be present on disk.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi import WebSocket
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.database import Base, get_db
from app.api.deps import get_inference_service, get_sequence_backend_ws

from continuous_pipeline.vocab import build_synthetic_vocab
from tests.conftest import FakeInferenceService


class ScriptedStreamBackend:
    """Deterministic fake: run() always reports the same two words, so the
    test can assert on the exact WebSocket response shape/content without
    depending on any trained model's actual (and here, synthetic/toy)
    predictions."""

    def __init__(self, vocab):
        self.vocab = vocab
        self.is_ready = True
        self.is_placeholder = True

    def run(self, features: np.ndarray) -> np.ndarray:
        seq_len = features.shape[0]
        vocab_size = len(self.vocab)
        log_probs = np.full((seq_len, vocab_size), -10.0, dtype=np.float32)
        log_probs[:, self.vocab.blank_id] = 0.0
        log_probs[0, self.vocab.encode("HELLO")] = 0.0
        log_probs[0, self.vocab.blank_id] = -10.0
        if seq_len > 2:
            log_probs[2, self.vocab.encode("YOU")] = 0.0
            log_probs[2, self.vocab.blank_id] = -10.0
        return log_probs


@pytest.fixture
def stream_client():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    from sqlalchemy.orm import sessionmaker
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    vocab = build_synthetic_vocab()
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_inference_service] = lambda: FakeInferenceService()
    # Deliberately a real dependency-shaped override (accepts the same
    # `websocket: WebSocket` parameter the real get_sequence_backend_ws
    # does) rather than a zero-arg lambda -- a zero-arg override is exactly
    # what let a Request-vs-WebSocket injection bug slip past this test
    # suite once already (the dependency worked in every TestClient test
    # here while being broken against a real running server). Matching the
    # real signature means this test exercises the same FastAPI dependency
    # injection path production traffic does. WebSocket must be imported at
    # MODULE level (see the import block above) -- with `from __future__
    # import annotations`, a locally-scoped import here would be invisible
    # to FastAPI's get_type_hints(..., globalns=fn.__globals__) resolution
    # and silently fall back to treating `websocket` as a query parameter.

    def override_get_sequence_backend_ws(websocket: WebSocket):
        return ScriptedStreamBackend(vocab)

    app.dependency_overrides[get_sequence_backend_ws] = override_get_sequence_backend_ws

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)


def _frame_message(feature_dim: int = 258) -> dict:
    return {"type": "frame", "features": [0.0] * feature_dim}


def test_stream_sends_ready_message_on_connect(stream_client):
    with stream_client.websocket_connect("/ws/stream") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["model_ready"] is True
        assert ready["feature_dim"] == 258
        assert ready["vocab_size"] > 0


def test_stream_returns_update_after_enough_frames(stream_client):
    with stream_client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()  # ready

        # default SEQUENCE_INFERENCE_STRIDE_FRAMES is 8 -- send exactly that many
        for _ in range(8):
            ws.send_json(_frame_message())

        update = ws.receive_json()
        assert update["type"] == "update"
        assert [w["word"] for w in update["words"]] == ["HELLO", "YOU"]
        assert all(0.0 <= w["confidence"] <= 1.0 for w in update["words"])
        assert update["transcript"] == "Hello You."
        assert update["buffer_frames"] == 8


def test_stream_rejects_malformed_frame_without_closing_connection(stream_client):
    with stream_client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()  # ready

        ws.send_json({"type": "frame", "features": [0.0] * 10})  # wrong length
        error = ws.receive_json()
        assert error["type"] == "error"
        assert "258" in error["message"]

        # connection should still be alive -- a valid frame afterward works normally
        for _ in range(8):
            ws.send_json(_frame_message())
        update = ws.receive_json()
        assert update["type"] == "update"


def test_stream_unknown_message_type_returns_error(stream_client):
    with stream_client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()  # ready
        ws.send_json({"type": "not_a_real_type"})
        error = ws.receive_json()
        assert error["type"] == "error"


def test_stream_reset_clears_transcript(stream_client):
    with stream_client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()  # ready
        for _ in range(8):
            ws.send_json(_frame_message())
        update = ws.receive_json()
        assert [w["word"] for w in update["words"]] == ["HELLO", "YOU"]

        ws.send_json({"type": "reset"})
        after_reset = ws.receive_json()
        assert after_reset["type"] == "update"
        assert after_reset["words"] == []
        assert after_reset["buffer_frames"] == 0


def test_stream_ping_pong(stream_client):
    with stream_client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()  # ready
        ws.send_json({"type": "ping"})
        pong = ws.receive_json()
        assert pong["type"] == "pong"
