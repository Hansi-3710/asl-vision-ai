"""
tests/conftest.py
==================
Shared pytest fixtures. Two important design choices:

1. Tests use an in-memory SQLite DB (not the real data/asl_vision.db), so
   test runs never touch or depend on real prediction history.
2. The InferenceService is replaced via `app.dependency_overrides` (see
   app/api/deps.py's `get_inference_service`) with a `FakeInferenceService`
   that returns deterministic, hand-specified predictions instead of
   loading a real trained model. This works regardless of whether the
   app's `lifespan` startup event manages to load a real checkpoint,
   because dependency_overrides intercepts the dependency call itself --
   an earlier version of this fixture set `app.state.inference_service`
   directly, which the real lifespan startup event then silently
   clobbered; using the dependency-override mechanism instead avoids that
   ordering bug entirely.

These are API/database/contract tests, not model accuracy tests -- model
behavior itself is covered by ml_pipeline's own test suite.

KNOWN LIMITATION: `TestClient(app)` still triggers the app's real
`lifespan` startup event, which calls `init_db()` against the PRODUCTION
DATABASE_URL (creating `data/asl_vision.db` if it doesn't exist yet) and
attempts to load a real model checkpoint (harmlessly logging a warning and
continuing if none exists, since REQUIRE_MODEL_ON_STARTUP defaults to
False). This is non-destructive -- `create_all` never drops or alters
existing tables/data -- but it is a real side effect of running this test
suite, not a fully hermetic setup. Fixing it properly would mean making
`lifespan` itself test-aware (e.g. skipping init_db()/model loading under
a TESTING env var), which is a reasonable follow-up if this starts to
matter for your CI setup.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.database import Base, get_db
from app.api.deps import get_inference_service
from app.ml.inference import ModelNotLoadedError


class FakeInferenceService:
    def __init__(self, loaded: bool = True):
        self._loaded = loaded
        self.architecture = "fake-arch"
        self.device = "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, image, top_k: int = 5):
        # Actually attempt to decode the image bytes, mirroring what a real
        # InferenceService/Predictor would do -- otherwise tests asserting
        # a 400 for "valid base64 but not an image" would pass vacuously
        # (this fake would return a fine result for ANY input otherwise).
        if isinstance(image, (bytes, bytearray)):
            from PIL import Image
            import io

            Image.open(io.BytesIO(image)).convert("RGB")  # raises for non-image bytes

        result = {
            "predicted_class": "A",
            "confidence": 0.987,
            "top_k": [
                {"class": "A", "confidence": 0.987},
                {"class": "B", "confidence": 0.008},
                {"class": "C", "confidence": 0.003},
            ][:top_k],
            "bounding_box": {"x_min": 0.3, "y_min": 0.2, "x_max": 0.7, "y_max": 0.8, "confidence": 0.95},
            "hand_detected": True,
        }
        return result, 12.3  # (result, latency_ms)


class NoHandFakeInferenceService(FakeInferenceService):
    """Simulates the hand detector running but finding nothing -- classifier
    still falls back to the full frame, per inference.py's design."""

    def predict(self, image, top_k: int = 5):
        result, latency_ms = super().predict(image, top_k=top_k)
        result["bounding_box"] = None
        result["hand_detected"] = False
        return result, latency_ms


class NoDetectorFakeInferenceService(FakeInferenceService):
    """Simulates the hand detector model file not being downloaded at all
    (hand_detected=None distinguishes this from 'ran and found nothing')."""

    def predict(self, image, top_k: int = 5):
        result, latency_ms = super().predict(image, top_k=top_k)
        result["bounding_box"] = None
        result["hand_detected"] = None
        return result, latency_ms


class UnloadedFakeInferenceService(FakeInferenceService):
    """Simulates the pre-training state (no checkpoint placed yet)."""

    def __init__(self):
        super().__init__(loaded=False)

    def predict(self, image, top_k: int = 5):
        raise ModelNotLoadedError("No checkpoint found (test double).")


@pytest.fixture
def test_engine():
    # StaticPool is essential here, not just a performance tweak: FastAPI's
    # TestClient dispatches sync path operations via run_in_threadpool, so
    # requests execute on a different thread than this fixture. Without
    # StaticPool, SQLAlchemy's default pooling for "sqlite:///:memory:"
    # hands out a SEPARATE physical connection per thread -- and each new
    # connection to an in-memory SQLite DB starts as its own private, empty
    # database. That silently orphans the tables create_all() just made,
    # producing "no such table: predictions" on the very first request.
    # StaticPool forces every connection (any thread) to reuse the one
    # physical connection this fixture created the tables on.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


def _make_client(test_engine, inference_service):
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_inference_service] = lambda: inference_service

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def client(test_engine):
    yield from _make_client(test_engine, FakeInferenceService())


@pytest.fixture
def client_no_model(test_engine):
    """Same as `client`, but simulates no model having been loaded yet --
    used to test the 503 path."""
    yield from _make_client(test_engine, UnloadedFakeInferenceService())


@pytest.fixture
def client_no_hand(test_engine):
    """Simulates a request where the hand detector ran but found no hand
    -- the classifier still falls back to classifying the full frame."""
    yield from _make_client(test_engine, NoHandFakeInferenceService())


@pytest.fixture
def client_no_detector(test_engine):
    """Simulates the MediaPipe model file not being downloaded at all."""
    yield from _make_client(test_engine, NoDetectorFakeInferenceService())
