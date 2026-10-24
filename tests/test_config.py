import csv

import pytest

from churn_core import config


def test_default_assumptions_are_the_documented_ones():
    a = config.default_assumptions()
    assert (a.r, a.d, a.months) == (0.5, 0.2, 12)


def test_stored_cutoff_matches_the_formula():
    # The cutoff was DERIVED (d / (d + r(1-d))), never tuned. The file and the formula must agree.
    a = config.default_assumptions()
    assert config.default_cutoff() == pytest.approx(a.d / (a.d + a.r * (1 - a.d)), abs=1e-12)
    assert config.default_cutoff() == pytest.approx(1 / 3, abs=1e-12)


def test_feature_order_is_sane():
    order = config.feature_order()
    assert len(order) == len(set(order)) > 10          # no duplicates, a real feature list
    for col in ["tenure", "monthly_charges", "total_charges", "senior_citizen"]:
        assert col in order


def test_model_version_matches_the_scores_file():
    path = config.reports_dir() / "scores.csv"
    with open(path, newline="", encoding="utf-8") as f:
        versions = {row["model_version"] for row in csv.DictReader(f)}
    assert versions == {config.MODEL_VERSION}


def test_churn_home_override(tmp_path, monkeypatch):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "model_config.json").write_text(
        '{"business_defaults": {"offer_success_rate_r": 0.4, "discount_d": 0.1,'
        ' "months_retained_M": 6, "cutoff": 0.2}, "feature_order": ["a", "b"]}')
    monkeypatch.setenv("CHURN_HOME", str(tmp_path))
    a = config.default_assumptions()
    assert (a.r, a.d, a.months) == (0.4, 0.1, 6)
    assert config.feature_order() == ["a", "b"]