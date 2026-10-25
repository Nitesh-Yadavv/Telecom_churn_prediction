import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from churn_core import config

KEY = "test-key-123"
HEADERS = {"X-API-Key": KEY}
GOLDEN = json.loads((Path(__file__).parent / "golden_predictions.json").read_text())["rows"]


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", KEY)
    with TestClient(create_app()) as c:
        yield c


def test_health_is_open_and_names_the_model(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_version"] == config.MODEL_VERSION


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
@pytest.mark.parametrize("method, path", [("get", "/model-info"), ("post", "/predict")])
def test_protected_endpoints_need_the_key(client, headers, method, path):
    r = getattr(client, method)(path, headers=headers, **({"json": {}} if method == "post" else {}))
    assert r.status_code == 401


def test_the_server_refuses_to_start_without_a_key(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API_KEY"), TestClient(create_app()):
        pass


def test_model_info(client):
    info = client.get("/model-info", headers=HEADERS).json()
    assert info["cutoff"] == pytest.approx(1 / 3, abs=1e-12)
    assert info["assumptions"] == {"r": 0.5, "d": 0.2, "months": 12}
    assert info["bands"]["labels"] == ["Low", "Medium", "High", "Very high"] and len(info["bands"]["edges"]) == 3
    assert info["model_version"] == config.MODEL_VERSION


def test_predict_reproduces_the_notebook_probabilities_over_http(client):
    for row in GOLDEN:
        r = client.post("/predict", headers=HEADERS, json={"customer": row["customer"]})
        assert r.status_code == 200
        body = r.json()
        assert np.isclose(body["probability"], row["expected_probability"], atol=1e-6)
        assert body["flag"] == (body["probability"] >= body["cutoff"])
        assert body["model_version"] == config.MODEL_VERSION and body["band_description"]


def test_custom_assumptions_change_the_cutoff_not_the_probability(client):
    customer = GOLDEN[3]["customer"]
    base = client.post("/predict", headers=HEADERS, json={"customer": customer}).json()
    easy = client.post("/predict", headers=HEADERS,
                       json={"customer": customer, "assumptions": {"r": 1.0}}).json()
    assert easy["cutoff"] == pytest.approx(0.2) and easy["assumptions"]["r"] == 1.0
    assert easy["probability"] == base["probability"]


@pytest.mark.parametrize("change", [
    {"contract": "Weekly"}, {"tenure": -1}, {"monthly_charges": -1.0}, {"surprise": 1},
])
def test_bad_customers_are_rejected_with_422(client, change):
    r = client.post("/predict", headers=HEADERS, json={"customer": {**GOLDEN[0]["customer"], **change}})
    assert r.status_code == 422


def test_bad_assumptions_are_rejected(client):
    for bad in ({"r": 0}, {"d": 1.0}, {"months": 0}, {"unknown": 1}):
        r = client.post("/predict", headers=HEADERS,
                        json={"customer": GOLDEN[0]["customer"], "assumptions": bad})
        assert r.status_code == 422


def test_unusual_values_come_back_as_warnings_not_errors(client):
    odd = {**GOLDEN[0]["customer"], "tenure": 500}
    r = client.post("/predict", headers=HEADERS, json={"customer": odd})
    assert r.status_code == 200 and any("tenure" in w for w in r.json()["warnings"])


def test_every_response_carries_a_request_id(client):
    assert client.get("/health").headers["X-Request-ID"]
    assert client.get("/health", headers={"X-Request-ID": "abc"}).headers["X-Request-ID"] == "abc"