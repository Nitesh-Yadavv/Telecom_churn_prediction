"""
tests/test_ui.py

Unit tests for app/ui.py design helpers, card containers, KPI components,
Plotly theme application, and number formatters.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import plotly.graph_objects as go
import pytest

from app import ui


def test_palette_and_risk_band_constants():
    assert ui.BAND_ORDER == ("Low", "Medium", "High", "Very high")
    assert ui.BAND_COLORS["Low"] == "#2F9E67"
    assert ui.BAND_COLORS["Medium"] == "#3F73D6"
    assert ui.BAND_COLORS["High"] == "#E5710C"
    assert ui.BAND_COLORS["Very high"] == "#D14B52"
    assert ui.PRIMARY_TEAL == "#0B7F77"
    assert ui.DEEP_TEAL == "#0B6B64"
    assert ui.CARD_BG == "#FFFFFF"
    assert ui.PAGE_BG == "#F6F6F3"
    assert ui.BORDER_COLOR == "#DCE6E3"
    assert ui.GRID_COLOR == "#E6ECEA"
    assert ui.RAISES_RISK == "#D14B52"
    assert ui.LOWERS_RISK == "#2F9E67"


def test_card_context_manager():
    with patch("streamlit.container") as mock_container:
        mock_cm = MagicMock()
        mock_container.return_value = mock_cm

        with ui.card("kpi_summary"):
            pass

        mock_container.assert_called_once_with(border=True, key="card_kpi_summary")


def test_kpi_component():
    with patch("streamlit.metric") as mock_metric, patch("streamlit.caption") as mock_caption:
        ui.kpi(label="Net gain", value=60674, help="Calculated value", caption="Over 12 months")
        mock_metric.assert_called_once_with(label="Net gain", value="60,674", help="Calculated value")
        mock_caption.assert_called_once_with("Over 12 months")


def test_kpi_component_formatting_varieties():
    with patch("streamlit.metric") as mock_metric:
        ui.kpi(label="Preformatted", value="60,674")
        assert mock_metric.call_args[1]["value"] == "60,674"

    with patch("streamlit.metric") as mock_metric:
        ui.kpi(label="Rate", value=0.456)
        assert mock_metric.call_args[1]["value"] == "0.46"

    with patch("streamlit.metric") as mock_metric:
        ui.kpi(label="Count", value=1409)
        assert mock_metric.call_args[1]["value"] == "1,409"

    with patch("streamlit.metric") as mock_metric:
        ui.kpi(label="Numpy Int", value=np.int64(496))
        assert mock_metric.call_args[1]["value"] == "496"

    with patch("streamlit.metric") as mock_metric:
        ui.kpi(label="Numpy Float", value=np.float64(120.5))
        assert mock_metric.call_args[1]["value"] == "120.50"

    with patch("streamlit.metric") as mock_metric:
        ui.kpi(label="Missing", value=None)
        assert mock_metric.call_args[1]["value"] == "—"

    with patch("streamlit.metric") as mock_metric:
        ui.kpi(label="NaN", value=float("nan"))
        assert mock_metric.call_args[1]["value"] == "—"

    with patch("streamlit.metric") as mock_metric:
        ui.kpi(label="Infinity", value=float("inf"))
        assert mock_metric.call_args[1]["value"] == "—"


def test_apply_plotly_theme():
    fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6], mode="lines", name="Baseline")])
    themed = ui.apply_plotly_theme(fig)

    assert themed.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert themed.layout.plot_bgcolor == "rgba(0,0,0,0)"
    assert themed.layout.font.family == "Inter, sans-serif"
    assert themed.layout.font.color == ui.PRIMARY_TEXT
    assert themed.layout.title.text is None
    assert themed.layout.xaxis.gridcolor == ui.GRID_COLOR
    assert themed.layout.yaxis.gridcolor == ui.GRID_COLOR
    assert len(themed.data) == 1
    assert list(themed.data[0].x) == [1, 2, 3]
    assert list(themed.data[0].y) == [4, 5, 6]


def test_apply_plotly_theme_preserves_axis_configs():
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["A", "B"], y=[10, 20]))
    fig.update_layout(
        xaxis={"title": {"text": "Custom X Title"}, "range": [0, 5]},
        yaxis={"title": {"text": "Custom Y Title"}, "tickformat": ".1%"},
    )

    themed = ui.apply_plotly_theme(fig)

    assert themed.layout.xaxis.title.text == "Custom X Title"
    assert list(themed.layout.xaxis.range) == [0, 5]
    assert themed.layout.yaxis.title.text == "Custom Y Title"
    assert themed.layout.yaxis.tickformat == ".1%"
    assert themed.layout.paper_bgcolor == "rgba(0,0,0,0)"


@pytest.mark.parametrize(
    "input_val, expected",
    [
        (60674.16, "60,674"),
        (0, "0"),
        (-17132, "-17,132"),
        (np.float64(450.0), "450"),
        (np.int32(1000), "1,000"),
        (None, "—"),
        (float("nan"), "—"),
        (float("inf"), "—"),
        (float("-inf"), "—"),
    ],
)
def test_format_currency(input_val, expected):
    assert ui.format_currency(input_val) == expected
    assert ui.fmt_currency(input_val) == expected


@pytest.mark.parametrize(
    "input_val, decimals, expected",
    [
        (0.3333, 1, "33.3%"),
        (0.7246, 1, "72.5%"),
        (0.5, 0, "50%"),
        (0.0, 1, "0.0%"),
        (np.float32(0.85), 1, "85.0%"),
        (None, 1, "—"),
        (float("nan"), 1, "—"),
        (float("inf"), 1, "—"),
    ],
)
def test_format_pct(input_val, decimals, expected):
    assert ui.format_pct(input_val, decimals=decimals) == expected
    assert ui.fmt_pct(input_val, decimals=decimals) == expected


@pytest.mark.parametrize(
    "input_val, expected",
    [
        (7043, "7,043"),
        (1409.0, "1,409"),
        (0, "0"),
        (np.int64(5634), "5,634"),
        (None, "—"),
        (float("nan"), "—"),
        (float("inf"), "—"),
    ],
)
def test_format_count(input_val, expected):
    assert ui.format_count(input_val) == expected
    assert ui.fmt_count(input_val) == expected
