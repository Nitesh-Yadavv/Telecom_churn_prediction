import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from api.main import EXAMPLE_CUSTOMER, create_app
from api.prediction_log import PredictionLog

KEY = "test-key-123"
HEADERS = {"X-API-Key": KEY}
ROW = {"inputs": {"customer": {"tenure": 5}}, "probability": 0.6, "band": "High", "flag": True,
       "model_version": "v-test", "latency_ms": 12}


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def make_log(**kwargs):
    rows = []
    log = PredictionLog(write=lambda row, trim: rows.append((row, trim)), **kwargs)
    return log, rows


def test_it_does_nothing_unless_it_is_configured():
    log = PredictionLog()
    assert not log.enabled and log.record(**ROW) is False


def test_a_row_has_the_documented_fields_and_valid_json():
    log, rows = make_log()
    assert log.record(**ROW) is True
    row, trim = rows[0]
    assert set(row) == {"model_version", "probability", "band", "flag", "latency_ms", "inputs"}
    assert json.loads(row["inputs"]) == ROW["inputs"] and trim is None


def test_sampling():
    log, rows = make_log(sample_rate=0.5, rng=lambda: 0.7)
    assert log.record(**ROW) is False and rows == []
    log, rows = make_log(sample_rate=0.5, rng=lambda: 0.2)
    assert log.record(**ROW) is True and len(rows) == 1


def test_throttle_allows_a_burst_then_recovers():
    clock = Clock()
    log, rows = make_log(max_per_minute=3, clock=clock)
    assert [log.record(**ROW) for _ in range(5)] == [True, True, True, False, False]
    clock.now += 61
    assert log.record(**ROW) is True and len(rows) == 4


def test_a_failing_database_never_raises_and_pauses_logging():
    clock, calls = Clock(), []

    def broken(row, trim):
        calls.append(1)
        raise ConnectionError("database is down")

    log = PredictionLog(write=broken, pause_seconds=60, clock=clock)
    assert log.record(**ROW) is False and len(calls) == 1          # the failure was swallowed
    assert log.record(**ROW) is False and len(calls) == 1          # paused: the database is not even tried
    clock.now += 61
    assert log.record(**ROW) is False and len(calls) == 2          # tried again after the pause


def test_old_rows_are_trimmed_on_a_fixed_rhythm():
    log, rows = make_log(trim_every=3, keep_rows=100)
    for _ in range(7):
        log.record(**ROW)
    assert [trim for _, trim in rows] == [None, None, 100, None, None, 100, None]


# ---------------- through the API ----------------
@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", KEY)
    monkeypatch.delenv("PREDICTION_LOG_URL", raising=False)
    with TestClient(create_app()) as c:
        yield c


def test_predict_logs_one_anonymous_row_and_batch_and_explain_do_not(client):
    log, rows = make_log()
    client.app.state.prediction_log = log
    r = client.post("/predict", headers=HEADERS, json={"customer": EXAMPLE_CUSTOMER})
    assert r.status_code == 200 and len(rows) == 1
    row, _ = rows[0]
    assert row["probability"] == r.json()["probability"] and row["band"] == r.json()["band"]
    assert row["latency_ms"] >= 0 and "customer" in json.loads(row["inputs"])
    assert "customerID" not in row["inputs"] and "customer_id" not in row["inputs"]
    client.post("/explain", headers=HEADERS, json={"customer": EXAMPLE_CUSTOMER})
    assert len(rows) == 1


def test_a_broken_log_does_not_change_the_prediction(client):
    def broken(row, trim):
        raise RuntimeError("boom")

    client.app.state.prediction_log = PredictionLog(write=broken)
    r = client.post("/predict", headers=HEADERS, json={"customer": EXAMPLE_CUSTOMER})
    assert r.status_code == 200 and 0 <= r.json()["probability"] <= 1


def test_model_info_says_whether_logging_is_on(client):
    assert client.get("/model-info", headers=HEADERS).json()["prediction_log"] == {"enabled": False}
    client.app.state.prediction_log = make_log()[0]
    assert client.get("/model-info", headers=HEADERS).json()["prediction_log"] == {"enabled": True}


# ---------------- against a real PostgreSQL (only when TEST_DATABASE_URL is set) ----------------
URL = os.environ.get("TEST_DATABASE_URL")
SCHEMA = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"


@pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL is not set (needs a throwaway PostgreSQL)")
def test_rows_land_in_postgres_and_only_the_newest_are_kept(monkeypatch):
    assert "test" in (make_url(URL).database or "")
    eng = create_engine(URL)
    with eng.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        cur = conn.connection.cursor()
        cur.execute(SCHEMA.read_text())
        cur.close()
    monkeypatch.setenv("PREDICTION_LOG_URL", URL)
    monkeypatch.setenv("LOG_KEEP_ROWS", "5")
    monkeypatch.setenv("LOG_MAX_PER_MINUTE", "1000")
    log = PredictionLog.from_env()
    log.trim_every = 1
    for i in range(12):
        assert log.record(inputs={"n": i}, probability=0.5, band="High", flag=True,
                          model_version="v", latency_ms=i) is True
    log.close()
    with eng.connect() as conn:
        rows = conn.execute(text("SELECT inputs->>'n' AS n, latency_ms FROM predictions_log ORDER BY id")).all()
    assert [r.n for r in rows] == ["7", "8", "9", "10", "11"][-len(rows):] and len(rows) <= 6
    assert rows[-1].n == "11"                                   # the newest row is always kept
    eng.dispose()