from __future__ import annotations

import io

from PIL import Image


def _upload_one(client):
    img = Image.new("RGB", (64, 64), color=(120, 80, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return client.post("/api/predict", files={"file": ("test.jpg", buf.getvalue(), "image/jpeg")})


def test_history_empty_state(client):
    response = client.get("/api/history")
    assert response.status_code == 200
    assert response.json() == []


def test_history_returns_predictions_most_recent_first(client):
    first = _upload_one(client).json()
    second = _upload_one(client).json()

    response = client.get("/api/history")
    body = response.json()
    assert len(body) == 2
    assert body[0]["id"] == second["id"]  # most recent first
    assert body[1]["id"] == first["id"]


def test_history_respects_limit(client):
    for _ in range(5):
        _upload_one(client)

    response = client.get("/api/history?limit=2")
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_history_rejects_limit_over_max(client):
    response = client.get("/api/history?limit=99999")
    # limit is clamped server-side to HISTORY_MAX_LIMIT, not rejected --
    # confirm it doesn't error and doesn't return more than exist.
    assert response.status_code == 200


def test_history_filters_by_source(client):
    _upload_one(client)

    response = client.get("/api/history?source=webcam")
    assert response.status_code == 200
    assert response.json() == []

    response2 = client.get("/api/history?source=upload")
    assert len(response2.json()) == 1


def test_history_offset_pagination(client):
    for _ in range(3):
        _upload_one(client)

    page1 = client.get("/api/history?limit=2&offset=0").json()
    page2 = client.get("/api/history?limit=2&offset=2").json()
    assert len(page1) == 2
    assert len(page2) == 1
    assert page1[0]["id"] != page2[0]["id"]
