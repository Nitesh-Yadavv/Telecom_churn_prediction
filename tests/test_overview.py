"""
tests/test_overview.py

Tests for Business Overview dashboard page and its business logic.
Validates:
- KPI values equal overview_numbers.json
- Segment shares within each dimension sum to 1.0 (within 0.002)
- Top contract segment is 'Month-to-month' with share ~0.869
- Q1 to Q3 churn rates match benchmark specification
- Q5 net gain equals 60,674 (within 1)
- Streamlit AppTest renders without exception for every dimension option
"""
import pytest
from streamlit.testing.v1 import AppTest

from app.data import (
    load_dashboard_data,
    load_overview_numbers,
    load_segment_revenue,
)
from app.overview_logic import (
    compute_q1_contract,
    compute_q2_quartiles,
    compute_q3_tenure,
    compute_q4_services,
    compute_q5_net_gain,
    get_overview_kpis,
    get_segment_data,
    get_top_segment_insight,
)


@pytest.fixture
def overview_data():
    return load_overview_numbers()


@pytest.fixture
def segment_df():
    return load_segment_revenue()


@pytest.fixture
def full_df():
    return load_dashboard_data()


def test_kpis_equal_overview_numbers(overview_data):
    """KPI values must match overview_numbers.json."""
    kpis = get_overview_kpis(overview_data)
    assert kpis["total_revenue"]["value"] == pytest.approx(overview_data["total_monthly_revenue"], abs=0.1)
    assert kpis["revenue_lost"]["value"] == pytest.approx(overview_data["revenue_lost_monthly"], abs=0.1)
    assert kpis["current_customers"]["value"] == overview_data["n_current_customers"]
    assert kpis["expected_at_risk"]["value"] == pytest.approx(overview_data["expected_revenue_at_risk_monthly"], abs=0.1)


def test_segment_shares_sum_to_one(segment_df):
    """Segment shares within each dimension must sum to 1.0 (within 0.002)."""
    dimensions = ["contract", "internet_service", "payment_method", "tenure_band"]
    for dim in dimensions:
        dim_slice = segment_df[segment_df["dimension"] == dim]
        assert not dim_slice.empty
        total_share = dim_slice["share_of_lost"].sum()
        assert total_share == pytest.approx(1.0, abs=0.002), f"Dimension {dim} share sum is {total_share}"


def test_top_contract_segment(segment_df):
    """The top Contract segment must be 'Month-to-month' with share ~0.869."""
    contract_df = get_segment_data(segment_df, "contract")
    top_seg = contract_df.iloc[0]
    assert top_seg["segment"] == "Month-to-month"
    assert top_seg["share_of_lost"] == pytest.approx(0.869, abs=0.002)

    insight = get_top_segment_insight(contract_df, "Contract")
    assert "Month-to-month contracts account for 86.9% of lost revenue." in insight


def test_q1_churn_rates(full_df):
    """Q1 contract churn rates match: Month-to-month 42.7%, One year 11.3%, Two year 2.8%."""
    q1 = compute_q1_contract(full_df).set_index("contract")
    assert q1.loc["Month-to-month", "churn_rate"] == pytest.approx(0.427, abs=0.001)
    assert q1.loc["One year", "churn_rate"] == pytest.approx(0.113, abs=0.001)
    assert q1.loc["Two year", "churn_rate"] == pytest.approx(0.028, abs=0.001)


def test_q2_quartiles(full_df):
    """Q2 monthly charge quartiles compute successfully and show churn variation."""
    q2, labels = compute_q2_quartiles(full_df)
    assert len(q2) == 4
    assert len(labels) == 4
    # Quartile 1 (< ~$35.50) has lowest churn (~11.2%)
    assert q2.iloc[0]["churn_rate"] == pytest.approx(0.112, abs=0.002)
    # Quartile 3 has highest churn (~37.5%)
    assert q2.iloc[2]["churn_rate"] == pytest.approx(0.375, abs=0.002)


def test_q3_tenure(full_df):
    """Q3 tenure band churn rates match: 0-11m 48.3%, 12-35m 26.1%, 36-72m 12.1%."""
    q3 = compute_q3_tenure(full_df).set_index("tenure_band")
    assert q3.loc["0-11", "churn_rate"] == pytest.approx(0.483, abs=0.001)
    assert q3.loc["12-35", "churn_rate"] == pytest.approx(0.261, abs=0.001)
    assert q3.loc["36-72", "churn_rate"] == pytest.approx(0.121, abs=0.001)


def test_q4_services(full_df):
    """Q4 service features calculate churn rates properly."""
    res = compute_q4_services(full_df)
    assert "internet" in res and "payment" in res
    assert not res["internet"].empty
    assert not res["payment"].empty


def test_q5_net_gain(full_df):
    """Q5 equals 60,674 (within 1)."""
    net_gain = compute_q5_net_gain(full_df, r=0.5, d=0.2, months=12.0, c=0.0)
    assert net_gain == pytest.approx(60674, abs=1.0)


def test_overview_page_renders_all_dimensions():
    """AppTest loads overview.py without exception for every dimension option."""
    at = AppTest.from_file("app/views/overview.py", default_timeout=15)
    at.run()
    assert not at.exception, f"AppTest threw exception: {at.exception}"

    dim_radio = at.radio(key="overview_dim")
    assert dim_radio is not None

    for opt in ["Contract", "Internet service", "Payment method", "Tenure band"]:
        dim_radio.set_value(opt)
        at.run()
        assert not at.exception, f"AppTest failed for dimension {opt}: {at.exception}"
