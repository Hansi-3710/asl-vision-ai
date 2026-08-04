from __future__ import annotations


def test_health_returns_ok_and_model_loaded(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_architecture"] == "fake-arch"


def test_health_reports_model_not_loaded(client_no_model):
    response = client_no_model.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"  # the API itself is healthy even if the model isn't loaded
    assert body["model_loaded"] is False
