"""Unit tests for the FinShield FastAPI endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_endpoint() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_loaded"] is True
    assert "vectorstore_loaded" in data
    assert "uptime_seconds" in data

def test_root_endpoint() -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "message" in data
    assert data["version"] == "1.0.0"

def _get_low_risk_payload() -> dict:
    payload = {f"V{i}": 0.0 for i in range(1, 29)}
    payload["time_hour"] = 14.0
    payload["amount"] = 10.0
    payload["is_night"] = 0
    payload["is_weekend"] = 0
    return payload

def test_predict_low_risk() -> None:
    payload = _get_low_risk_payload()
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

def test_predict_returns_required_fields() -> None:
    payload = _get_low_risk_payload()
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "fraud_probability" in data
    assert "risk_level" in data
    assert "is_fraud" in data
    assert "top_features" in data
    assert "explanation" in data
    assert "processing_time_ms" in data
    assert "transaction_id" in data
    assert "policy_reference" in data
    assert "model_version" in data

def test_predict_fast_endpoint() -> None:
    payload = _get_low_risk_payload()
    resp = client.post("/predict/fast", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["processing_time_ms"] < 500.0

def test_batch_predict() -> None:
    payload = _get_low_risk_payload()
    batch_payload = {
        "transactions": [payload, payload, payload]
    }
    resp = client.post("/predict/batch", json=batch_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 3

def test_model_info_endpoint() -> None:
    resp = client.get("/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_type" in data
    assert data["model_name"] == "xgboost_tuned"
