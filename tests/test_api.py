"""Tests for the FastAPI service in src/api.py."""
import joblib
import pytest
from fastapi.testclient import TestClient

import api
from predict import SAMPLE_PATIENT


@pytest.fixture
def client(monkeypatch, tmp_path, trained_pipeline):
    path = tmp_path / "model.joblib"
    joblib.dump(trained_pipeline, path)
    monkeypatch.setattr(api, "MODEL_PATH", path)
    with TestClient(api.app) as c:  # the 'with' block runs startup (loads model)
        yield c


@pytest.fixture
def client_without_model(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.joblib")
    with TestClient(api.app) as c:
        yield c


def test_health_ok_when_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_503_when_model_missing(client_without_model):
    assert client_without_model.get("/health").status_code == 503


def test_predict_returns_prediction_and_confidence(client):
    resp = client.post("/predict", json=SAMPLE_PATIENT)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1
    assert 0 <= body["probability_disease"] <= 1
    assert body["label"] in ("heart disease", "no heart disease")


def test_predict_503_when_model_missing(client_without_model):
    resp = client_without_model.post("/predict", json=SAMPLE_PATIENT)
    assert resp.status_code == 503


def test_predict_rejects_missing_field(client):
    bad = {k: v for k, v in SAMPLE_PATIENT.items() if k != "age"}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_out_of_range_value(client):
    bad = {**SAMPLE_PATIENT, "age": 500}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_wrong_type(client):
    bad = {**SAMPLE_PATIENT, "sex": "male"}
    assert client.post("/predict", json=bad).status_code == 422


def test_responses_carry_request_id_header(client):
    assert client.get("/health").headers.get("X-Request-ID")


def test_requests_are_logged(client, caplog):
    with caplog.at_level("INFO", logger="heart-api"):
        client.post("/predict", json=SAMPLE_PATIENT)
    text = caplog.text
    assert "path=/predict" in text and "status=200" in text
