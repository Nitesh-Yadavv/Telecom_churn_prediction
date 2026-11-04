import pandas as pd
import pytest

from churn_core import bands, config

PATH = config.reports_dir() / "dashboard_data.csv"
COLUMNS = ["customer_id", "split", "score_kind", "probability", "band", "flag", "actual_churn",
           "monthly_charges", "tenure", "tenure_band", "contract", "internet_service", "payment_method",
           "gender", "senior_citizen"]


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(PATH)


def test_columns_and_no_gaps(data):
    assert list(data.columns) == COLUMNS and not data.isna().any().any()
    assert data.customer_id.is_unique and data.probability.between(0, 1).all()


def test_bands_flags_and_kinds_are_consistent(data):
    assert list(data.band) == list(bands.assign_band(data.probability))
    assert (data.flag == (data.probability >= config.default_cutoff())).all()
    assert (data.score_kind == data.split.map({"train": "oof", "test": "test"})).all()


def test_it_matches_scores_csv_and_the_known_totals(data):
    scores = pd.read_csv(config.reports_dir() / "scores.csv").set_index("customer_id")
    assert (data.set_index("customer_id").probability - scores.p_cal).abs().max() < 1e-12
    test = data[data.split == "test"]
    assert len(data) == 7043 and len(test) == 1409 and test.flag.sum() == 496 and test.actual_churn.sum() == 374