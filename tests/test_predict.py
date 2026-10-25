import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn_core import bands, config, economics, model
from churn_core.predict import predict, predict_customers
from churn_core.schema import Customer, clean_raw

GOLDEN = json.loads((Path(__file__).parent / "golden_predictions.json").read_text())["rows"]


def _customers():
    return [Customer(**row["customer"]) for row in GOLDEN]


def test_the_models_were_trained_with_the_installed_library_versions():
    assert model.version_mismatches() == {}


def test_golden_customers_reproduce_the_notebook_probabilities():
    out = predict_customers(_customers())
    expected = np.array([row["expected_probability"] for row in GOLDEN])
    np.testing.assert_allclose(out.probability.to_numpy(), expected, atol=1e-6)


def test_gender_has_no_effect_on_the_prediction():
    base = predict_customers(_customers()).probability.to_numpy()
    flipped = []
    for row in GOLDEN:
        c = dict(row["customer"])
        c["gender"] = "Male" if c["gender"] == "Female" else "Female"
        flipped.append(Customer(**c))
    np.testing.assert_allclose(predict_customers(flipped).probability.to_numpy(), base, atol=1e-12)


def test_output_is_consistent_with_economics_and_bands():
    out = predict_customers(_customers())
    bill = [row["customer"]["monthly_charges"] for row in GOLDEN]
    assert list(out.columns) == ["probability", "band", "flag", "expected_gain", "model_version"]
    assert (out.model_version == config.MODEL_VERSION).all()
    assert (out.flag == (out.probability >= config.default_cutoff())).all()
    assert list(out.band) == list(bands.assign_band(out.probability))
    a = config.default_assumptions()
    np.testing.assert_allclose(
        out.expected_gain, economics.ev_per_customer(out.probability, bill, a.r, a.d, a.months))
    assert ((out.expected_gain > 0) == out.flag).all()      # contact exactly when the expected gain is positive


def test_custom_assumptions_move_the_cutoff_and_the_gain():
    base = predict_customers(_customers())
    easy = predict_customers(_customers(), r=1.0)             # every offer works: cutoff 0.2
    assert (easy.flag == (easy.probability >= 0.2)).all()
    assert easy.flag.sum() >= base.flag.sum()
    assert (easy.expected_gain >= base.expected_gain).all()


def test_the_input_index_is_kept():
    df = pd.DataFrame([row["customer"] for row in GOLDEN], index=[f"c{i}" for i in range(len(GOLDEN))])
    assert list(predict(df).index) == list(df.index)


# ---- the full skew test: runs only on your machine (needs the raw CSV and the exports) ----
RAW = config.data_dir() / "Telco-Customer-Churn.csv"
SPLIT = config.data_dir() / "split_assignment.csv"


@pytest.mark.skipif(not (RAW.exists() and SPLIT.exists()), reason="data files not available (CI)")
def test_reproduces_all_1409_saved_test_predictions():
    sa = pd.read_csv(SPLIT)
    ids = sa[sa.split == "test"].sort_values("position").customer_id
    raw = pd.read_csv(RAW).set_index("customerID").loc[ids].reset_index()
    out = predict(clean_raw(raw))
    saved = pd.read_csv(config.reports_dir() / "test_predictions.csv")
    assert list(saved.customer_id) == list(ids)
    np.testing.assert_allclose(out.probability.to_numpy(), saved.p_cal.to_numpy(), atol=1e-9)
    assert int(out.flag.sum()) == 496