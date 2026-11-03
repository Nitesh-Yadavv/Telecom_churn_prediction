from pathlib import Path
from unittest.mock import patch

import api_client
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"   # absolute: AppTest resolves relative paths against this file
PREDICT, ENTRY = str(APP / "views" / "predict.py"), str(APP / "streamlit_app.py")

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
    """The Predict page on its own, with the sidebar assumptions the entry point would have set."""
    with patch.object(api_client, "predict", return_value=PREDICTION) as predict, \
         patch.object(api_client, "explain", return_value=EXPLANATION):
        at = AppTest.from_file(PREDICT, default_timeout=30)
        at.session_state["r"], at.session_state["d"], at.session_state["months"] = 0.5, 0.2, 12
        at.predict_mock = predict
        yield at


def test_the_predict_page_loads(app):
    app.run()
    assert not app.exception
    assert app.title[0].value == "Predict one customer"


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


def test_changing_the_assumptions_after_scoring_updates_the_decision_without_calling_the_api_again(app):
    app.run()
    app.button[0].click().run()
    assert any("Worth an offer" in s.value for s in app.success)
    app.session_state["d"] = 0.5                              # discount 50%: break-even rises to 66.7%
    app.run()
    assert any("Not worth an offer" in i.value and "66.7%" in i.value for i in app.info)
    assert app.predict_mock.call_count == 1


def test_changing_the_customer_after_scoring_warns_that_the_result_is_out_of_date(app):
    app.run()
    app.button[0].click().run()
    assert not any("details changed" in w.value for w in app.warning)
    app.number_input(key="tenure").set_value(30).run()
    assert any("details changed" in w.value for w in app.warning)


def test_the_shared_sidebar_starts_from_the_saved_assumptions():
    at = AppTest.from_file(ENTRY, default_timeout=30).run()
    assert not at.exception
    assert [s.value for s in at.sidebar.slider] == [0.5, 0.2, 12]
    assert at.sidebar.metric[0].value == "33.3%"                     # the derived break-even


def test_moving_a_sidebar_slider_updates_the_break_even():
    at = AppTest.from_file(ENTRY, default_timeout=30).run()
    at.sidebar.slider[1].set_value(0.5).run()                          # discount 50%
    assert at.sidebar.metric[0].value == "66.7%"