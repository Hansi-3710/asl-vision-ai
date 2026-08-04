from __future__ import annotations

import base64
import io

from PIL import Image


def _upload_one(client, letter_confidence=0.987):
    img = Image.new("RGB", (64, 64), color=(120, 80, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return client.post("/api/predict", files={"file": ("test.jpg", buf.getvalue(), "image/jpeg")})


def test_metrics_empty_state(client):
    response = client.get("/api/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["total_predictions"] == 0
    assert body["average_confidence"] == 0.0
    assert body["most_predicted_letters"] == []
    assert body["predictions_by_source"] == {}


def test_metrics_after_predictions(client):
    _upload_one(client)
    _upload_one(client)

    img = Image.new("RGB", (64, 64), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    client.post("/api/predict-webcam", json={"image_base64": b64})

    response = client.get("/api/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["total_predictions"] == 3
    assert body["average_confidence"] == 0.987  # fake service always returns this
    assert body["most_predicted_letters"][0]["letter"] == "A"
    assert body["most_predicted_letters"][0]["count"] == 3
    assert body["predictions_by_source"] == {"upload": 2, "webcam": 1}
