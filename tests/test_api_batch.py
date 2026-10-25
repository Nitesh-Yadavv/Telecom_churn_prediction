import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import MAX_ROWS, MAX_UPLOAD_BYTES, TEMPLATE_COLUMNS, create_app
from churn_core import checks
from churn_core.schema import CLEAN_TO_RAW, YES_NO_COLUMNS

KEY = "test-key-123"
HEADERS = {"X-API-Key": KEY}
GOLDEN = json.loads((Path(__file__).parent / "golden_predictions.json").read_text())["rows"]


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", KEY)
    with TestClient(create_app()) as c:
        yield c


def to_raw(clean: dict, customer_id: str | None) -> dict:
    """A clean-format customer as one row of the upload (Kaggle) format."""
    raw = {}
    for field, value in clean.items():
        if field == "senior_citizen":
            value = int(value)
        elif field in YES_NO_COLUMNS:
            value = "Yes" if value else "No"
        raw[CLEAN_TO_RAW[field]] = value
    if customer_id is not None:
        raw["customerID"] = customer_id
    return raw


def csv_bytes(rows: list[dict]) -> bytes:
    return pd.DataFrame(rows).to_csv(index=False).encode()


def upload(client, data: bytes, **params):
    return client.post("/predict_batch", headers=HEADERS, params=params,
                       files={"file": ("customers.csv", data, "text/csv")})


GOOD_ROWS = [to_raw(g["customer"], g["customer_id"]) for g in GOLDEN]


# ---------------- /explain ----------------
def test_explain_needs_the_key(client):
    assert client.post("/explain", json={"customer": GOLDEN[0]["customer"]}).status_code == 401


def test_explain_returns_consistent_reasons(client):
    for g in GOLDEN:
        r = client.post("/explain", headers=HEADERS, json={"customer": g["customer"]})
        assert r.status_code == 200
        body = r.json()
        assert np.isclose(body["probability"], g["expected_probability"], atol=1e-6)
        assert 1 <= len(body["drivers"]) <= 3
        total = body["base_logit"] + sum(w["effect"] for w in body["waterfall"])
        assert total == pytest.approx(body["raw_logit"], abs=1e-9)
        for d in body["drivers"]:
            assert d["direction"] in d["sentence"] and d["label"] in d["sentence"]
            assert d["lever"] is None or d["lever"]["status"] == "untested hypothesis"


# ---------------- /upload-template ----------------
def test_the_template_is_a_valid_upload(client):
    r = client.get("/upload-template", headers=HEADERS)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    df = pd.read_csv(io.StringIO(r.text), dtype=str)
    assert list(df.columns) == TEMPLATE_COLUMNS
    assert checks.check_upload(df).issues == []
    scored = upload(client, r.content)
    assert scored.status_code == 200 and scored.json()["n_scored"] == 1


# ---------------- /predict_batch ----------------
def test_batch_reproduces_the_notebook_probabilities(client):
    r = upload(client, csv_bytes(GOOD_ROWS))
    assert r.status_code == 200
    body = r.json()
    assert body["n_scored"] == len(GOLDEN) and body["summary"]["rows_with_errors"] == 0
    probs = [row["probability"] for row in body["results"]]
    np.testing.assert_allclose(probs, [g["expected_probability"] for g in GOLDEN], atol=1e-6)
    assert [row["row"] for row in body["results"]] == list(range(2, len(GOLDEN) + 2))
    assert [row["customer_id"] for row in body["results"]] == [g["customer_id"] for g in GOLDEN]
    assert all(row["reasons"] is None for row in body["results"])


def test_bad_rows_are_reported_and_left_out_but_good_rows_are_scored(client):
    rows = [dict(r) for r in GOOD_ROWS]
    rows[1]["Contract"] = "Weekly"          # spreadsheet row 3
    rows[4]["MonthlyCharges"] = -5          # spreadsheet row 6
    body = upload(client, csv_bytes(rows)).json()
    assert body["n_scored"] == len(rows) - 2
    assert body["summary"]["rows_with_errors"] == 2
    assert {i["row"] for i in body["issues"] if i["level"] == "error"} == {3, 6}
    assert [r["row"] for r in body["results"]] == [2, 4, 5, 7, 8, 9]


def test_reasons_are_added_only_when_asked(client):
    body = upload(client, csv_bytes(GOOD_ROWS), reasons="true").json()
    for row in body["results"]:
        assert 1 <= len(row["reasons"]) <= 3 and all(isinstance(s, str) for s in row["reasons"])


def test_assumptions_can_be_changed_with_query_parameters(client):
    body = upload(client, csv_bytes(GOOD_ROWS), r=1.0).json()
    assert body["cutoff"] == pytest.approx(0.2) and body["assumptions"]["r"] == 1.0


def test_customer_id_is_optional(client):
    rows = [{k: v for k, v in r.items() if k != "customerID"} for r in GOOD_ROWS]
    body = upload(client, csv_bytes(rows)).json()
    assert body["n_scored"] == len(rows) and all(r["customer_id"] is None for r in body["results"])


def test_nothing_scorable_returns_422_with_the_reasons(client):
    rows = [{**r, "Contract": "Weekly"} for r in GOOD_ROWS]
    r = upload(client, csv_bytes(rows))
    assert r.status_code == 422 and r.json()["n_scored"] == 0 and r.json()["issues"]
    missing = upload(client, csv_bytes([{k: v for k, v in GOOD_ROWS[0].items() if k != "tenure"}]))
    assert missing.status_code == 422 and any(i["column"] == "tenure" for i in missing.json()["issues"])


def test_limits_and_unreadable_files(client):
    assert upload(client, b"x" * (MAX_UPLOAD_BYTES + 1)).status_code == 413
    too_many = csv_bytes([GOOD_ROWS[0]] * (MAX_ROWS + 1))
    assert len(too_many) < MAX_UPLOAD_BYTES and upload(client, too_many).status_code == 413
    assert upload(client, b"").status_code == 400
    assert upload(client, b"\xff\xfe\x00\x81\x82").status_code == 400


def test_batch_needs_the_key(client):
    r = client.post("/predict_batch", files={"file": ("c.csv", csv_bytes(GOOD_ROWS), "text/csv")})
    assert r.status_code == 401