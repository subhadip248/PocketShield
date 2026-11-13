"""Focused tests for the PocketShield HTTP API."""

from fastapi.testclient import TestClient

from pocketshield import api
from pocketshield.phishing.url_detector import analyze_url
from pocketshield.sms.sms_detector import analyze_sms


client = TestClient(api.app)


def test_health_endpoint_returns_healthy_status():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_url_endpoint_returns_engine_response():
    response = client.post("/analyze/url", json={"url": "https://example.com"})

    assert response.status_code == 200
    assert response.json() == {
        "modality": "url",
        "analysis_status": "analyzed",
        "detector_result": analyze_url("https://example.com"),
    }


def test_sms_endpoint_returns_engine_response():
    message = "Your package is arriving today."
    response = client.post("/analyze/sms", json={"message": message})

    assert response.status_code == 200
    assert response.json() == {
        "modality": "sms",
        "analysis_status": "analyzed",
        "detector_result": analyze_sms(message),
    }


def test_detector_invalid_input_is_returned_as_structured_analysis():
    response = client.post("/analyze/url", json={"url": ""})

    assert response.status_code == 200
    assert response.json()["analysis_status"] == "invalid"
    assert response.json()["detector_result"]["risk_score"] is None


def test_pydantic_rejects_missing_or_non_string_inputs():
    missing = client.post("/analyze/url", json={})
    invalid_type = client.post("/analyze/sms", json={"message": 42})

    assert missing.status_code == 422
    assert invalid_type.status_code == 422


def test_unexpected_errors_return_safe_response(monkeypatch):
    def fail(_value):
        raise RuntimeError("internal path /secret/server/file.py")

    monkeypatch.setattr(api.engine, "analyze_url", fail)
    response = client.post("/analyze/url", json={"url": "https://example.com"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Analysis could not be completed."}
    assert "/secret" not in response.text
