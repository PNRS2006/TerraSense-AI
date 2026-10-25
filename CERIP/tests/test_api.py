"""
CERIP — Tests: FastAPI Endpoints
Run after starting the server: uvicorn src.api.main:app --port 8000
Or run with TestClient (no server needed).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from src.api.main import app
    with TestClient(app) as c:
        yield c


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "CERIP" in r.json()["message"]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_heat_risk_get(client):
    r = client.get("/heat-risk")
    assert r.status_code in (200, 503)  # 503 if data not generated yet


def test_heat_risk_post_valid(client):
    payload = {"lat": 17.72, "lon": 83.30, "include_shap": False}
    r = client.post("/heat-risk", json=payload)
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        data = r.json()
        assert "heat_risk_class" in data
        assert 1 <= data["heat_risk_class"] <= 5


def test_heat_risk_post_out_of_bounds(client):
    payload = {"lat": 0.0, "lon": 0.0, "include_shap": False}
    r = client.post("/heat-risk", json=payload)
    assert r.status_code == 422  # validation error


def test_aqi_forecast_get(client):
    r = client.get("/aqi-forecast?horizon_hours=24")
    assert r.status_code in (200, 503)


def test_aqi_forecast_invalid_horizon(client):
    r = client.get("/aqi-forecast?horizon_hours=36")
    assert r.status_code == 400


def test_activity_recommendation(client):
    payload = {
        "activity": "Running",
        "user_profile": "Healthy Adult",
        "aqi_value": 85,
        "heat_risk_class": 3,
    }
    r = client.post("/activity-recommendation", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["safety_label"] in ("Safe", "Caution", "Avoid")
    assert "NOT MEDICAL ADVICE" in data["disclaimer"].upper() or \
           "informational" in data["disclaimer"].lower()


def test_activity_invalid_activity(client):
    payload = {
        "activity": "Skydiving",
        "user_profile": "Healthy Adult",
        "aqi_value": 80,
        "heat_risk_class": 2,
    }
    r = client.post("/activity-recommendation", json=payload)
    assert r.status_code == 422


def test_unified_index(client):
    payload = {"heat_risk_class": 4, "aqi_value": 165}
    r = client.post("/unified-index", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["uosi_category"] in ("Safe", "Caution", "Limit", "Avoid")
    assert data["compounding_applied"] is True  # 4+165 should trigger
    assert "formula_note" in data


def test_unified_index_disclaimer_present(client):
    payload = {"heat_risk_class": 1, "aqi_value": 30}
    r = client.post("/unified-index", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "disclaimer" in data
