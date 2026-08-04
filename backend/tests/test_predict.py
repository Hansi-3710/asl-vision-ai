from __future__ import annotations

import base64
import io

from PIL import Image


def _make_test_image_bytes() -> bytes:
    img = Image.new("RGB", (64, 64), color=(120, 80, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_predict_upload_returns_prediction_and_persists_it(client):
    image_bytes = _make_test_image_bytes()
    response = client.post(
        "/api/predict",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == "A"
    assert body["confidence"] == 0.987
    assert body["source"] == "upload"
    assert len(body["top_k"]) == 3
    assert body["top_k"][0]["class"] == "A"  # confirms the alias ("class", not "class_") round-trips correctly
    assert body["image_path"] is not None

    # Confirm it was actually persisted -- appears in history.
    history_response = client.get("/api/history")
    assert history_response.status_code == 200
    history = history_response.json()
    assert len(history) == 1
    assert history[0]["id"] == body["id"]


def test_predict_returns_bounding_box_when_hand_detected(client):
    image_bytes = _make_test_image_bytes()
    response = client.post("/api/predict", files={"file": ("test.jpg", image_bytes, "image/jpeg")})
    body = response.json()
    assert body["hand_detected"] is True
    assert body["bounding_box"] is not None
    assert 0.0 <= body["bounding_box"]["x_min"] < body["bounding_box"]["x_max"] <= 1.0


def test_predict_falls_back_to_full_frame_when_no_hand_detected(client_no_hand):
    """When the detector runs but finds nothing, the API should still
    return a real classification (of the full frame) rather than an
    error -- just with hand_detected=False and no bounding box."""
    image_bytes = _make_test_image_bytes()
    response = client_no_hand.post("/api/predict", files={"file": ("test.jpg", image_bytes, "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == "A"  # still got a real prediction
    assert body["hand_detected"] is False
    assert body["bounding_box"] is None


def test_predict_reports_none_when_hand_detector_unavailable(client_no_detector):
    """hand_detected=None should distinguish 'we never checked' (detector
    model file not downloaded) from 'we checked and found nothing'
    (hand_detected=False, see test above)."""
    image_bytes = _make_test_image_bytes()
    response = client_no_detector.post("/api/predict", files={"file": ("test.jpg", image_bytes, "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["hand_detected"] is None
    assert body["bounding_box"] is None


def test_predict_upload_rejects_unsupported_content_type(client):
    response = client.post(
        "/api/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_predict_upload_rejects_corrupted_file_with_valid_content_type(client):
    """A file that CLAIMS to be image/jpeg via its content-type header but
    isn't actually decodable image data (corrupted upload, wrong extension,
    etc.) should be a clean 400, not an unhandled 500 from the image
    decoder deep inside inference."""
    response = client.post(
        "/api/predict",
        files={"file": ("test.jpg", b"this is not really a jpeg", "image/jpeg")},
    )
    assert response.status_code == 400


def test_predict_upload_returns_503_when_model_not_loaded(client_no_model):
    image_bytes = _make_test_image_bytes()
    response = client_no_model.post(
        "/api/predict",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 503


def test_predict_webcam_accepts_plain_base64(client):
    image_bytes = _make_test_image_bytes()
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = client.post("/api/predict-webcam", json={"image_base64": b64})
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "webcam"
    assert body["image_path"] is None  # webcam frames are never persisted to disk


def test_predict_webcam_accepts_data_url_prefix(client):
    image_bytes = _make_test_image_bytes()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    response = client.post("/api/predict-webcam", json={"image_base64": data_url})
    assert response.status_code == 200
    assert response.json()["predicted_class"] == "A"


def test_predict_webcam_rejects_invalid_base64(client):
    response = client.post("/api/predict-webcam", json={"image_base64": "not-valid-base64!!!"})
    assert response.status_code == 400


def test_predict_webcam_rejects_garbage_that_naive_b64decode_would_silently_accept(client):
    """Regression test: base64.b64decode() without validate=True silently
    strips invalid characters instead of raising, so a string like this
    used to decode to garbage bytes and crash the image decoder as an
    unhandled 500 instead of a clean 400. See app/api/predict.py's use of
    validate=True."""
    response = client.post("/api/predict-webcam", json={"image_base64": "abc!@#defg=="})
    assert response.status_code == 400


def test_predict_webcam_rejects_valid_base64_that_is_not_an_image(client):
    """Valid base64, but the decoded bytes aren't a real image -- should be
    a clean 400 (client error), not an unhandled 500 from the image decoder."""
    import base64

    not_an_image = base64.b64encode(b"this is definitely not image data").decode("utf-8")
    response = client.post("/api/predict-webcam", json={"image_base64": not_an_image})
    assert response.status_code == 400
