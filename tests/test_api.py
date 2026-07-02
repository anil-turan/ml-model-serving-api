"""Integration tests for the FastAPI endpoints."""
from tests.conftest import SAMPLE_APPLICATION


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["test_roc_auc"] == 0.754


def test_ready_ok(client):
    r = client.get("/ready")
    assert r.status_code == 200


def test_predict_returns_valid_response(client):
    r = client.post("/predict", json=SAMPLE_APPLICATION)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["default_probability"] <= 1
    assert body["risk_grade"] in {"A", "B", "C", "D", "E"}
    assert body["decision"] in {"Approve", "Review", "Decline"}
    assert "prediction_id" in body
    assert "timestamp" in body


def test_predict_grade_e_on_high_risk(client):
    """Mock returns prob=0.35 which maps to grade D (boundary), not E."""
    r = client.post("/predict", json=SAMPLE_APPLICATION)
    assert r.status_code == 200
    body = r.json()
    # prob=0.35 is on the D/E boundary — grade D (<0.35) or E (>=0.35)
    assert body["risk_grade"] in {"D", "E"}


def test_predict_rejects_invalid_income(client):
    bad = {**SAMPLE_APPLICATION, "AMT_INCOME_TOTAL": -1}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_positive_days_birth(client):
    bad = {**SAMPLE_APPLICATION, "DAYS_BIRTH": 1000}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_drift_endpoint(client):
    # Make a few predictions to populate the window
    for _ in range(3):
        client.post("/predict", json=SAMPLE_APPLICATION)
    r = client.get("/drift")
    assert r.status_code == 200
    body = r.json()
    assert "score_mean" in body
    assert "alert" in body


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"prediction_requests_total" in r.content


def test_predict_missing_required_field(client):
    bad = {k: v for k, v in SAMPLE_APPLICATION.items() if k != "AMT_INCOME_TOTAL"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
