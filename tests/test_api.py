import pytest

from fraud_detection.config import MODELS_DIR

pytestmark = pytest.mark.skipif(not (MODELS_DIR / "model_card.json").exists(),
                                reason="train the models first: python scripts/train.py")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.api import app

    with TestClient(app) as c:
        yield c


BASE = {"user_id": 7, "amount": 2500.0, "transaction_type": "ATM", "merchant_category": "Travel",
        "country": "NG", "hour": 3}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and set(r.json()["models"]) == {"full", "hardmode"}


def test_score_with_vendor_scores_uses_full_model(client):
    r = client.post("/score", json={**BASE, "device_risk_score": 0.95, "ip_risk_score": 0.9})
    body = r.json()
    assert r.status_code == 200 and body["model_used"] == "full"
    assert body["decision"] == "block" and body["fraud_probability"] > 0.9
    assert len(body["reason_codes"]) == 3


def test_score_without_vendor_scores_falls_back(client):
    r = client.post("/score", json=BASE)
    assert r.status_code == 200 and r.json()["model_used"] == "hardmode"


def test_batch_and_validation(client):
    r = client.post("/score/batch", json=[BASE, {**BASE, "amount": 42.0, "hour": 14, "country": "DE"}])
    assert r.status_code == 200 and len(r.json()) == 2
    assert client.post("/score", json={**BASE, "hour": 25}).status_code == 422
