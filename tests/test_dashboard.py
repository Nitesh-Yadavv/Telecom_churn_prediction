from unittest.mock import patch

import api_client
import pytest
from streamlit.testing.v1 import AppTest

INFO = {"model_version": "test-model", "cutoff": 1 / 3, "assumptions": {"r": 0.5, "d": 0.2, "months": 12}}
PREDICTION = {"probability": 0.54, "band": "High", "band_description": "Likely to leave", "flag": True,
              "expected_gain": 120.0, "cutoff": 1 / 3, "assumptions": {"r": 0.5, "d": 0.2, "months": 12},
              "warnings": ["tenure = 500 is outside the training range"], "model_version": "test-model"}
EXPLANATION = {
    "probability": 0.54, "band": "High", "base_logit": -1.0, "raw_logit": 0.2, "note": "not causes",
    "model_version": "test-model",
    "drivers": [{"feature": "contract", "label": "Contract type", "value": "Month-to-month", "effect": 0.6,
                 "direction": "raises", "sentence": "Contract type (Month-to-month) raises the risk",
                 "lever": {"text": "Offer a discount for a longer-term contract", "status": "untested hypothesis"}}],
    "waterfall": [{"feature": "contract", "label": "Contract type", "value": "Month-to-month", "effect": 0.6},
                  {"feature": "tenure", "label": "Time as a customer", "value": "5 months", "effect": -0.2}],
}


@pytest.fixture()
def app():
    from streamlit import cache_data
    cache_data.clear()
    with patch.object(api_client, "model_info", return_value=INFO), \
         patch.object(api_client, "predict", return_value=PREDICTION) as predict, \
         patch.object(api_client, "explain", return_value=EXPLANATION):
        at = AppTest.from_file("app/Home.py", default_timeout=30)
        at.predict_mock = predict
        yield at


def test_the_page_loads_with_the_saved_assumptions(app):
    app.run()
    assert not app.exception
    assert [s.value for s in app.sidebar.slider] == [0.5, 0.2, 12]
    assert app.sidebar.metric[0].value == "33.3%"                     # the derived break-even


def test_scoring_shows_the_result_the_reasons_and_the_warning(app):
    app.run()
    app.button[0].click().run()
    assert not app.exception
    shown = {m.label: m.value for m in app.metric}
    assert shown["Chance of leaving"] == "54.0%" and shown["Risk band"] == "High"
    assert any("Worth an offer" in s.value for s in app.success)
    assert any("outside the training range" in w.value for w in app.warning)
    assert any("untested hypothesis" in m.value for m in app.markdown)


def test_no_internet_forces_the_add_ons_to_match(app):
    app.run()
    app.selectbox(key="internet").select("No").run()
    app.button[0].click().run()
    sent = app.predict_mock.call_args.args[0]
    assert sent["internet_service"] == "No"
    assert all(sent[k] == "No internet service" for k in
               ["online_security", "online_backup", "device_protection", "tech_support",
                "streaming_tv", "streaming_movies"])


def test_no_phone_forces_multiple_lines_to_match(app):
    app.run()
    app.checkbox(key="phone").set_value(False).run()
    app.button[0].click().run()
    assert app.predict_mock.call_args.args[0]["multiple_lines"] == "No phone service"


def test_an_api_failure_shows_a_message_not_a_crash(app):
    app.run()
    with patch.object(api_client, "predict", side_effect=api_client.ApiError("Could not reach the service")):
        app.button[0].click().run()
    assert not app.exception and any("Could not reach" in e.value for e in app.error)


def test_the_page_stops_cleanly_if_the_service_is_down():
    from streamlit import cache_data
    cache_data.clear()
    with patch.object(api_client, "model_info", side_effect=api_client.ApiError("Service unavailable")):
        at = AppTest.from_file("app/Home.py", default_timeout=30).run()
    assert not at.exception and any("Service unavailable" in e.value for e in at.error)