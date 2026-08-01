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

from fastapi import Request


def get_inference_service(request: Request):
    return request.app.state.inference_service
