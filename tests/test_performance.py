"""
tests/test_performance.py

Tests for Model Performance dashboard page and logic.
Validates:
- KPI values equal metrics.json
- Band table has four bands in order with observed churn monotonically increasing in both test and oof
- Model comparison chart lists all 8 pipelines and highlights the finalists
- Weakest segment calculation dynamically identifies lowest recall segment
- Fairness tables properly map gender and senior citizen values
- Top SHAP features format correctly
- Model card file exists and meets length requirements
- Streamlit AppTest renders without exception across toggle options
"""
import pytest
from streamlit.testing.v1 import AppTest

from app.data import (
    load_band_table,
    load_cv_results,
    load_metrics,
    load_segment_metrics,
    load_shap_importance,
)
from app.performance_logic import (
    check_band_ordering,
    find_weakest_segment,
    format_band_table,
    get_fairness_tables,
    get_model_comparison_data,
    get_performance_kpis,
    get_top_shap_features,
)
from churn_core import config


@pytest.fixture
def metrics_data():
    return load_metrics()


@pytest.fixture
def cv_data():
    return load_cv_results()


@pytest.fixture
def band_data():
    return load_band_table()


@pytest.fixture
def seg_data():
    return load_segment_metrics()


@pytest.fixture
def shap_data():
    return load_shap_importance()


def test_kpis_equal_metrics_json(metrics_data):
    """Headline KPIs must match metrics.json values exactly."""
    kpis = get_performance_kpis(metrics_data)
    assert kpis["auc_pr"]["value"] == pytest.approx(metrics_data["auc_pr"], abs=0.001)
    assert kpis["roc_auc"]["value"] == pytest.approx(metrics_data["roc_auc"], abs=0.001)
    assert kpis["brier"]["value"] == pytest.approx(metrics_data["brier_sigmoid"], abs=0.001)
    assert kpis["recall_precision"]["recall"] == pytest.approx(metrics_data["recall"], abs=0.001)
    assert kpis["recall_precision"]["precision"] == pytest.approx(metrics_data["precision"], abs=0.001)
    assert kpis["flagged"]["value"] == metrics_data["flagged"]
    assert kpis["flagged"]["n_test"] == metrics_data["n_test"]
    assert "0.655" in kpis["auc_pr"]["display"]
    assert "0.842" in kpis["roc_auc"]["display"]
    assert "496 of 1,409" in kpis["flagged"]["display"]


def test_band_table_ordered_and_churn_monotonically_increasing(band_data):
    """The band table must have 4 bands and observed churn must increase monotonically in both samples."""
    bands_test = format_band_table(band_data, sample="test")
    assert list(bands_test["band"]) == ["Low", "Medium", "High", "Very high"]

    bands_oof = format_band_table(band_data, sample="oof")
    assert list(bands_oof["band"]) == ["Low", "Medium", "High", "Very high"]

    monotonicity = check_band_ordering(band_data)
    assert monotonicity["test"] is True, "Observed churn rate does not strictly increase across test bands"
    assert monotonicity["oof"] is True, "Observed churn rate does not strictly increase across oof bands"


def test_model_comparison_contains_eight_pipelines(cv_data):
    """Comparison table must contain exactly 8 pipelines with finalists flagged."""
    comp_df = get_model_comparison_data(cv_data)
    assert len(comp_df) == 8
    finalists = comp_df[comp_df["is_finalist"]]
    assert len(finalists) == 2
    finalist_names = set(finalists["pipeline"])
    assert "LightGBM_ClassWeight" in finalist_names
    assert "LogisticRegression_ClassWeight" in finalist_names


def test_find_weakest_segment(seg_data):
    """Weakest segment calculation correctly identifies lowest recall segment."""
    weakest = find_weakest_segment(seg_data)
    assert weakest["group"] == "Two year"
    assert weakest["dimension"] == "contract"
    assert weakest["recall"] == 0.0
    assert weakest["caught"] == 0


def test_fairness_mapping(seg_data):
    """Fairness tables properly map gender to Female/Male and senior citizen status."""
    gender_table, senior_table = get_fairness_tables(seg_data)
    assert set(gender_table["Subgroup"]) == {"Female", "Male"}
    assert set(senior_table["Subgroup"]) == {"Non-senior", "Senior citizen"}

    # Female & Male have very close recall
    female_rec = gender_table[gender_table["Subgroup"] == "Female"]["Recall"].iloc[0]
    male_rec = gender_table[gender_table["Subgroup"] == "Male"]["Recall"].iloc[0]
    assert female_rec == pytest.approx(male_rec, abs=0.02)


def test_top_shap_features(shap_data):
    """Top SHAP features are sorted descending by importance and formatted cleanly."""
    top_shap = get_top_shap_features(shap_data, top_n=15)
    assert len(top_shap) == 15
    assert top_shap["importance"].is_monotonic_decreasing
    # Tenure should be top feature
    assert top_shap.iloc[0]["feature"] == "tenure"


def test_model_card_document():
    """Model card markdown document must exist and have adequate word count."""
    card_file = config.home() / "app" / "content" / "model_card.md"
    assert card_file.is_file(), "app/content/model_card.md does not exist"
    content = card_file.read_text(encoding="utf-8")
    words = content.split()
    assert 300 <= len(words) <= 600, f"Expected 300-600 words, got {len(words)}"
    assert "LightGBM" in content
    assert "Platt" in content or "Sigmoid" in content
    assert "AUC-PR" in content
    assert "0.655" in content


def test_performance_page_renders_all_metrics():
    """AppTest loads and renders performance.py without exception across toggle options."""
    at = AppTest.from_file("app/views/performance.py", default_timeout=15)
    at.run()
    assert not at.exception, f"AppTest threw exception: {at.exception}"

    metric_radio = at.radio(key="perf_comp_metric")
    assert metric_radio is not None

    for opt in ["AUC-PR (primary)", "ROC-AUC"]:
        metric_radio.set_value(opt)
        at.run()
        assert not at.exception, f"AppTest failed for metric {opt}: {at.exception}"
