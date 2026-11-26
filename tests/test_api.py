import pytest
from fastapi.testclient import TestClient

from app.api import app
from src import config

pytestmark = pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="trained model artifact not present")

VALID = {
    "age": 40,
    "scholarship": 0,
    "hipertension": 1,
    "diabetes": 0,
    "alcoholism": 0,
    "scheduled_day": "2016-05-02T10:00:00Z",
    "appointment_day": "2016-05-09",
    "prior_appts": 2,
    "prior_noshows": 1,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_probability_and_version(client):
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["model_version"]


def test_rejects_invalid_age(client):
    assert client.post("/predict", json={**VALID, "age": -3}).status_code == 422


def test_rejects_booking_after_decision_point(client):
    late = {**VALID, "scheduled_day": "2016-05-08T18:30:00Z"}
    response = client.post("/predict", json=late)
    assert response.status_code == 422
    assert "decision point" in response.text


def test_rejects_more_noshows_than_appointments(client):
    assert client.post("/predict", json={**VALID, "prior_appts": 1, "prior_noshows": 3}).status_code == 422


def test_rejects_missing_field(client):
    bad = {k: v for k, v in VALID.items() if k != "age"}
    assert client.post("/predict", json=bad).status_code == 422
