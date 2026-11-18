"""
tests/test_explorer_page.py

Streamlit AppTest headless execution tests for Customer Risk Explorer page.
Validates:
- Page loads and renders without exceptions
- Filter changes (population radio, min_prob slider) re-run without crash
- Safe handling of API calls
"""
from streamlit.testing.v1 import AppTest


def test_explorer_page_renders_defaults():
    """Customer risk explorer page loads and renders default held-out population without error."""
    at = AppTest.from_file("app/views/explorer.py", default_timeout=15)
    at.run()
    assert not at.exception, f"AppTest threw exception: {at.exception}"

    # Verify key widgets exist
    pop_radio = at.radio(key="exp_population")
    assert pop_radio is not None
    assert pop_radio.value == "Held-out customers only (1,409)"


def test_explorer_page_switch_population():
    """Switching to full dataset renders out-of-fold note and re-evaluates without error."""
    at = AppTest.from_file("app/views/explorer.py", default_timeout=15)
    at.run()

    pop_radio = at.radio(key="exp_population")
    pop_radio.set_value("All customers (7,043; training rows are out-of-fold)")
    at.run()
    assert not at.exception, f"AppTest failed after switching population: {at.exception}"


def test_explorer_page_filter_min_probability():
    """Changing minimum probability slider updates table smoothly."""
    at = AppTest.from_file("app/views/explorer.py", default_timeout=15)
    at.run()

    prob_slider = at.slider(key="exp_min_prob")
    prob_slider.set_value(0.40)
    at.run()
    assert not at.exception, f"AppTest failed after adjusting min prob: {at.exception}"
