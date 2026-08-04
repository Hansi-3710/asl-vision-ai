"""
api/deps.py
===========
Shared FastAPI dependencies. The inference service is exposed as a real
dependency function (not accessed via `request.app.state` directly in
route handlers) specifically so tests can swap it out cleanly with
`app.dependency_overrides[get_inference_service] = lambda: FakeService()`
-- this works regardless of what the app's `lifespan` startup event has
already put on `app.state`, avoiding any startup-vs-test-fixture ordering
issues.
"""

from __future__ import annotations

from fastapi import Request, WebSocket


def get_inference_service(request: Request):
    return request.app.state.inference_service


def get_sequence_backend(request: Request):
    """Same rationale as get_inference_service above, for the continuous
    (sentence-level) recognition model, used by REST endpoints (health.py).
    NOT usable by a WebSocket route -- FastAPI has no Request to inject
    into a websocket connection's dependency graph; use
    get_sequence_backend_ws (below) there instead. Both just read the
    same `app.state.sequence_backend` set once at startup."""
    return request.app.state.sequence_backend


def get_sequence_backend_ws(websocket: WebSocket):
    """WebSocket-route counterpart to get_sequence_backend -- same object,
    injected via the connection type a WebSocket route actually receives.
    See api/stream.py."""
    return websocket.app.state.sequence_backend
