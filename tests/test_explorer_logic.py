"""
tests/test_explorer_logic.py

Unit tests for pure explorer business logic and security defenses.
Validates:
- Formula injection defense (neutralise_value, neutralise_frame, neutralise_csv)
- Customer population, contract, tenure, and band filtering
- Expected gain ranking and sorting
- Summary metric calculations
- Batch scoring response parsing
"""
import pandas as pd
import pytest

from app.data import load_dashboard_data
from app.explorer_logic import (
    compute_explorer_summary,
    filter_customers,
    format_batch_results,
    neutralise_csv,
    neutralise_frame,
    neutralise_value,
    rank_by_expected_gain,
)


@pytest.fixture
def full_df():
    return load_dashboard_data()


def test_neutralise_formula_injection():
    """Formulas starting with =, +, -, @, tab must be prefixed with single quote."""
    dangerous = [
        "=SUM(A1:A10)",
        "+12345",
        "-2+3*cmd|' /C calc'!A0",
        "@SUM(B1)",
        "\tmalicious_tab",
        "\rmalicious_cr",
    ]
    for val in dangerous:
        neutral = neutralise_value(val)
        assert neutral.startswith("'"), f"Failed to neutralize: {val}"
        assert neutral[1:] == val

    harmless = ["Normal Text", "123", "Customer-001", "Two year", 45.6, None]
    for val in harmless:
        assert neutralise_value(val) == val


def test_neutralise_frame_and_csv():
    """DataFrames exported to CSV must safely neutralize string injection vectors."""
    df = pd.DataFrame({
        "customer_id": ["=1+1", "@calc", "SAFE-01"],
        "charges": [50.0, 75.0, 100.0],
        "notes": ["+bonus", "regular", "\ttabbed"],
    })
    safe_df = neutralise_frame(df)
    assert safe_df.loc[0, "customer_id"] == "'=1+1"
    assert safe_df.loc[1, "customer_id"] == "'@calc"
    assert safe_df.loc[2, "customer_id"] == "SAFE-01"
    assert safe_df.loc[0, "notes"] == "'+bonus"
    assert safe_df.loc[2, "notes"] == "'\ttabbed"

    csv_text = neutralise_csv(df)
    assert "'=1+1" in csv_text
    assert "'+bonus" in csv_text


def test_filter_customers_population(full_df):
    """Held-out population must return 1,409 rows; all customers returns 7,043."""
    test_pop = filter_customers(full_df, population="Held-out customers only (1,409)")
    assert len(test_pop) == 1409
    assert set(test_pop["split"]) == {"test"}

    all_pop = filter_customers(full_df, population="All customers (7,043; training rows are out-of-fold)")
    assert len(all_pop) == 7043


def test_filter_customers_by_contract_and_bands(full_df):
    """Filtering by specific contracts and bands restricts the subset accurately."""
    res = filter_customers(
        full_df,
        population="Held-out customers only (1,409)",
        bands=["High", "Very high"],
        contracts=["Month-to-month"],
    )
    assert not res.empty
    assert set(res["contract"]) == {"Month-to-month"}
    assert set(res["band"]).issubset({"High", "Very high"})


def test_filter_customers_min_prob(full_df):
    """Filtering by minimum probability filters out lower probabilities."""
    res = filter_customers(
        full_df,
        population="Held-out customers only (1,409)",
        min_prob=0.50,
    )
    assert not res.empty
    assert (res["probability"] >= 0.50).all()


def test_rank_by_expected_gain(full_df):
    """Ranking by expected gain computes positive gain correctly and sorts descending."""
    test_df = filter_customers(full_df, population="Held-out customers only (1,409)")
    ranked = rank_by_expected_gain(test_df, r=0.5, d=0.2, months=12.0)

    assert "expected_gain" in ranked.columns
    assert "contact" in ranked.columns
    # Must be sorted descending
    assert ranked["expected_gain"].is_monotonic_decreasing

    # At r=0.5, d=0.2, cutoff is 1/3; contact must match probability >= 1/3
    cutoff = 0.2 / (0.2 + 0.5 * 0.8)
    assert (ranked["contact"] == (ranked["probability"] >= cutoff)).all()


def test_compute_explorer_summary(full_df):
    """Summary calculations match expected totals for held-out customers."""
    test_df = filter_customers(full_df, population="Held-out customers only (1,409)")
    ranked = rank_by_expected_gain(test_df, r=0.5, d=0.2, months=12.0)
    summary = compute_explorer_summary(ranked)

    assert summary["n_matching"] == 1409
    assert summary["n_contact"] == 496
    # Default net expected gain is ~$67,009
    assert summary["total_gain"] == pytest.approx(67009, abs=10.0)


def test_format_batch_results():
    """Batch scoring parser sorts errors before warnings and counts risk bands."""
    fake_resp = {
        "n_scored": 2,
        "n_errors": 1,
        "n_warnings": 1,
        "issues": [
            {"row": 2, "column": "monthly_charges", "level": "warning", "message": "High charge"},
            {"row": 1, "column": "tenure", "level": "error", "message": "Invalid negative tenure"},
        ],
        "scored": [
            {"row": 2, "customer_id": "CUST-02", "probability": 0.45, "band": "High", "flag": True, "expected_gain": 85.0},
            {"row": 3, "customer_id": "CUST-03", "probability": 0.10, "band": "Low", "flag": False, "expected_gain": -20.0},
        ],
        "issues_truncated": False,
    }
    parsed = format_batch_results(fake_resp)

    assert parsed["n_scored"] == 2
    assert parsed["n_errors"] == 1
    assert parsed["n_warnings"] == 1
    assert parsed["n_contact"] == 1
    assert parsed["total_gain"] == 85.0

    # Errors must come first in issues_df
    issues_df = parsed["issues_df"]
    assert issues_df.iloc[0]["level"] == "error"
    assert issues_df.iloc[1]["level"] == "warning"

    assert parsed["band_counts"]["High"] == 1
    assert parsed["band_counts"]["Low"] == 1
    assert parsed["band_counts"]["Medium"] == 0
