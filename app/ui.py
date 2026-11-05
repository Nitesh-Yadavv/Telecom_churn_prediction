"""
app/ui.py

Reusable UI helpers and design system tokens for the Telecom Churn dashboard.
Implements card wrappers, KPI metrics, Plotly chart theming, and number formatting
using the project's light palette (Inter and Source Serif 4 fonts).
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ----------------- Design System Tokens -----------------
PAGE_BG = "#F6F6F3"
CARD_BG = "#FFFFFF"
MINT_TINT = "#E6F3F0"
PRIMARY_TEAL = "#0B7F77"
DEEP_TEAL = "#0B6B64"
PRIMARY_TEXT = "#0F2A36"
MUTED_TEXT = "#4F6674"
BORDER_COLOR = "#DCE6E3"
GRID_COLOR = "#E6ECEA"

# Risk Band Palette
BAND_ORDER = ("Low", "Medium", "High", "Very high")
BAND_COLORS: dict[str, str] = {
    "Low": "#2F9E67",
    "Medium": "#3F73D6",
    "High": "#E5710C",
    "Very high": "#D14B52",
}

# Directional Indicators
RAISES_RISK = "#D14B52"
LOWERS_RISK = "#2F9E67"


# ----------------- Card Container -----------------
@contextmanager
def card(name: str):
    """Render content inside a card container with styling hooked to key card_<name>."""
    with st.container(border=True, key=f"card_{name}"):
        yield


# ----------------- KPI Component -----------------
def kpi(label: str, value: Any, help: str | None = None, caption: str | None = None) -> None:
    """Render a styled KPI metric block with an optional tooltip and descriptive caption."""
    display_value: str
    if value is None:
        display_value = "—"
    elif isinstance(value, str):
        # Strings that are already formatted or text (e.g. "60,674", "33.3%") must not be altered
        display_value = value
    elif isinstance(value, (float, np.floating)):
        display_value = "—" if (math.isnan(value) or math.isinf(value)) else f"{float(value):,.2f}"
    elif isinstance(value, (int, np.integer)):
        display_value = f"{int(value):,}"
    else:
        try:
            num = float(value)
            display_value = "—" if (math.isnan(num) or math.isinf(num)) else f"{num:,.2f}"
        except (ValueError, TypeError):
            display_value = str(value)

    st.metric(label=label, value=display_value, help=help)
    if caption:
        st.caption(caption)


# ----------------- Plotly Theming Helper -----------------
def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    """Apply the project's light theme, typography, transparent background, and grid styling."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": PRIMARY_TEXT, "size": 12},
        margin={"l": 40, "r": 20, "t": 20, "b": 40},
        
        legend={
            "font": {"family": "Inter, sans-serif", "color": PRIMARY_TEXT, "size": 11},
            "bgcolor": "rgba(0,0,0,0)",
        },
        hoverlabel={
            "font": {"family": "Inter, sans-serif", "size": 12},
            "bgcolor": CARD_BG,
            "bordercolor": BORDER_COLOR,
        },
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        linecolor=BORDER_COLOR,
        tickfont={"family": "Inter, sans-serif", "color": PRIMARY_TEXT, "size": 11},
        title_font={"family": "Inter, sans-serif", "color": PRIMARY_TEXT, "size": 12},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        linecolor=BORDER_COLOR,
        tickfont={"family": "Inter, sans-serif", "color": PRIMARY_TEXT, "size": 11},
        title_font={"family": "Inter, sans-serif", "color": PRIMARY_TEXT, "size": 12},
    )
    return fig


# ----------------- Number Formatters -----------------
def _is_invalid_number(val: Any) -> bool:
    """Check if a numeric scalar is None, NaN, or infinite."""
    if val is None:
        return True
    if isinstance(val, (float, np.floating)):
        return math.isnan(val) or math.isinf(val)
    return False


def format_currency(x: float | np.number | None) -> str:
    """Format a monetary amount with thousands separators and no decimal places (e.g. 60,674)."""
    if _is_invalid_number(x):
        return "—"
    try:
        val = float(x)  # type: ignore[arg-type]
        if math.isnan(val) or math.isinf(val):
            return "—"
        return f"{round(val):,.0f}"
    except (ValueError, TypeError):
        return "—"


def format_pct(x: float | np.number | None, decimals: int = 1) -> str:
    """Format a decimal proportion as a percentage string (e.g. 0.333 -> 33.3%)."""
    if _is_invalid_number(x):
        return "—"
    try:
        val = float(x)  # type: ignore[arg-type]
        if math.isnan(val) or math.isinf(val):
            return "—"
        return f"{val:.{decimals}%}"
    except (ValueError, TypeError):
        return "—"


def format_count(n: float | np.number | None) -> str:
    """Format an integer count with thousands separators (e.g. 7,043)."""
    if _is_invalid_number(n):
        return "—"
    try:
        val = float(n)  # type: ignore[arg-type]
        if math.isnan(val) or math.isinf(val):
            return "—"
        return f"{round(val):,}"
    except (ValueError, TypeError):
        return "—"


# Clean concise aliases
fmt_currency = format_currency
fmt_pct = format_pct
fmt_count = format_count
