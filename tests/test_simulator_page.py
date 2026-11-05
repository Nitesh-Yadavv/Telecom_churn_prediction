"""
tests/test_simulator_page.py

AppTest headless UI tests for app/views/simulator.py.
Verifies page rendering, KPI card outputs, view radio switching,
large fixed cost fallback to 0 contacts, and absence of unhandled exceptions.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
SIMULATOR = str(APP / "views" / "simulator.py")


def test_simulator_page_renders_defaults():
    at = AppTest.from_file(SIMULATOR, default_timeout=30)
    at.session_state["r"] = 0.5
    at.session_state["d"] = 0.2
    at.session_state["months"] = 12
    at.run()

    assert not at.exception
    assert at.title[0].value == "Retention campaign simulator"

    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Customers to contact"] == "496"
    assert metrics["Net gain"] == "60,674"
    assert metrics["Gain per 1,000 customers"] == "43,062"
    assert "+17,595" in metrics["Versus the naive rule"]


def test_simulator_page_switching_to_expected_view():
    at = AppTest.from_file(SIMULATOR, default_timeout=30)
    at.session_state["r"] = 0.5
    at.session_state["d"] = 0.2
    at.session_state["months"] = 12
    at.run()

    # Switch radio to "Held-out customers, model's expectation"
    at.radio(key="sim_view").set_value("Held-out customers, model's expectation").run()
    assert not at.exception

    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Net gain"] == "67,009"


def test_simulator_page_large_fixed_cost_zero_contacts():
    at = AppTest.from_file(SIMULATOR, default_timeout=30)
    at.session_state["r"] = 0.5
    at.session_state["d"] = 0.2
    at.session_state["months"] = 12
    at.run()

    # Set fixed cost very high (1,000,000) so policy_ranked declines campaign
    at.number_input(key="sim_fixed").set_value(1_000_000.0).run()
    assert not at.exception

    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Customers to contact"] == "0"
    assert metrics["Net gain"] == "0"
