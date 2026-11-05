"""
tests/test_simulator_logic.py

Unit tests for app/simulator_logic.py against the Phase 6 benchmark numbers.
Tests defaults, capacity limits, contact costs, bootstrap intervals,
sensitivity sweeps, and pooled population outcomes.
"""
import pandas as pd
import pytest

from app.data import load_dashboard_data
from app.simulator_logic import (
    compute_band_breakdown,
    compute_budget_curves,
    compute_sensitivity_table,
    evaluate_strategies,
    run_paired_bootstrap,
)


@pytest.fixture
def test_df() -> pd.DataFrame:
    df = load_dashboard_data()
    return df[df["split"] == "test"].copy().reset_index(drop=True)


@pytest.fixture
def full_df() -> pd.DataFrame:
    return load_dashboard_data().copy().reset_index(drop=True)


def test_strategies_held_out_defaults(test_df):
    strategies = evaluate_strategies(test_df, r=0.5, d=0.2, months=12, c=0.0, fixed_cost=0.0)
    strat_dict = {s["strategy"]: s for s in strategies}

    # 1. Model, ranked by value
    ranked = strat_dict["Model, ranked by value"]
    assert ranked["contacted"] == 496
    assert pytest.approx(ranked["net_gain"], abs=1.0) == 60674

    # 2. Contact everyone
    everyone = strat_dict["Contact everyone"]
    assert everyone["contacted"] == 1409
    assert pytest.approx(everyone["net_gain"], abs=1.0) == -17132

    # 3. Naive rule
    naive = strat_dict["Naive rule (month-to-month, tenure < 1 yr)"]
    assert naive["contacted"] == 385
    assert pytest.approx(naive["net_gain"], abs=1.0) == 43079

    # 4. Random customers (same count 496)
    rand = strat_dict["Random customers"]
    assert rand["contacted"] == 496
    assert pytest.approx(rand["net_gain"], abs=1.0) == -6031

    # 5. Model minus naive
    diff = ranked["net_gain"] - naive["net_gain"]
    assert pytest.approx(diff, abs=1.0) == 17595


def test_strategies_expected_value_view(test_df):
    strategies = evaluate_strategies(
        test_df, r=0.5, d=0.2, months=12, c=0.0, fixed_cost=0.0, use_expected_value=True
    )
    strat_dict = {s["strategy"]: s for s in strategies}
    ranked = strat_dict["Model, ranked by value"]
    assert ranked["contacted"] == 496
    assert pytest.approx(ranked["net_gain"], abs=1.0) == 67009


def test_capacity_300_limit(test_df):
    strategies = evaluate_strategies(
        test_df, r=0.5, d=0.2, months=12, c=0.0, fixed_cost=0.0, capacity=300
    )
    strat_dict = {s["strategy"]: s for s in strategies}
    ranked = strat_dict["Model, ranked by value"]
    assert ranked["contacted"] == 300
    assert pytest.approx(ranked["net_gain"], abs=1.0) == 60354


@pytest.mark.parametrize(
    "c, exp_thresh_n, exp_thresh_val, exp_rk_n, exp_rk_val",
    [
        (5.0, 496, 58194, 464, 58105),
        (10.0, 496, 55714, 455, 56400),
        (20.0, 496, 50754, 434, 51725),
        (40.0, 496, 40834, 389, 45681),
    ],
)
def test_contact_cost_sweep(test_df, c, exp_thresh_n, exp_thresh_val, exp_rk_n, exp_rk_val):
    strategies = evaluate_strategies(test_df, r=0.5, d=0.2, months=12, c=c)
    strat_dict = {s["strategy"]: s for s in strategies}

    rk = strat_dict["Model, ranked by value"]
    thresh = strat_dict["Model, plain cutoff"]

    assert rk["contacted"] == exp_rk_n
    assert pytest.approx(rk["net_gain"], abs=1.0) == exp_rk_val
    assert thresh["contacted"] == exp_thresh_n
    assert pytest.approx(thresh["net_gain"], abs=1.0) == exp_thresh_val


def test_paired_bootstrap_deterministic(test_df):
    res1 = run_paired_bootstrap(test_df, r=0.5, d=0.2, months=12, seed=42)
    res2 = run_paired_bootstrap(test_df, r=0.5, d=0.2, months=12, seed=42)
    assert res1 == res2
    assert pytest.approx(res1["point"], abs=1.0) == 17595
    assert res1["ci_low"] < res1["point"] < res1["ci_high"]
    assert pytest.approx(res1["ci_low"], abs=100.0) == 8676
    assert pytest.approx(res1["ci_high"], abs=100.0) == 27360


def test_sensitivity_table_and_breakeven(test_df):
    sens_df, break_even_r = compute_sensitivity_table(test_df, sidebar_r=0.5, d=0.2, months=12)

    r_01 = sens_df[sens_df["r"] == 0.10].iloc[0]
    assert pytest.approx(r_01["saving_adapted"], abs=5.0) == 2553
    assert pytest.approx(r_01["saving_kept"], abs=5.0) == -21134

    r_07 = sens_df[sens_df["r"] == 0.70].iloc[0]
    assert pytest.approx(r_07["saving_adapted"], abs=5.0) == 104828
    assert pytest.approx(r_07["saving_kept"], abs=5.0) == 101578

    assert break_even_r is not None
    assert pytest.approx(break_even_r, abs=0.02) == 0.20


def test_pooled_population_numbers(full_df):
    strategies = evaluate_strategies(full_df, r=0.5, d=0.2, months=12)
    strat_dict = {s["strategy"]: s for s in strategies}
    ranked = strat_dict["Model, ranked by value"]
    naive = strat_dict["Naive rule (month-to-month, tenure < 1 yr)"]

    assert pytest.approx(ranked["net_gain"], abs=1.0) == 311928
    assert pytest.approx(naive["net_gain"], abs=1.0) == 207145


def test_budget_curves(test_df):
    curves = compute_budget_curves(test_df, r=0.5, d=0.2, months=12)
    assert "x" in curves
    assert "model_ranked" in curves
    assert "model_prob" in curves
    assert len(curves["x"]) > 50
    assert curves["x"][0] == 0
    assert curves["x"][-1] == len(test_df)


def test_band_breakdown(test_df):
    bands = compute_band_breakdown(test_df, r=0.5, d=0.2, months=12)
    assert len(bands) == 4
    # Low and Medium should lose money on average, High and Very High earn it
    assert bands[0]["net_gain"] < 0
    assert bands[1]["net_gain"] < 0
    assert bands[2]["net_gain"] > 0
    assert bands[3]["net_gain"] > 0
