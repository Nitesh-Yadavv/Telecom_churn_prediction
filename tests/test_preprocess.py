import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from churn_core import config
from churn_core.preprocess import build_features, encode_customers, load_bundle
from churn_core.schema import CLEAN_FEATURE_COLUMNS, Customer, clean_raw

GOOD = {
    "senior_citizen": False, "partner": True, "dependents": False, "tenure": 12,
    "phone_service": True, "multiple_lines": "No", "internet_service": "DSL",
    "online_security": "Yes", "online_backup": "No", "device_protection": "No",
    "tech_support": "No", "streaming_tv": "No", "streaming_movies": "No",
    "contract": "Month-to-month", "paperless_billing": True,
    "payment_method": "Electronic check", "monthly_charges": 55.0, "total_charges": 660.0,
}


def _customers():
    return [
        Customer(**GOOD),
        Customer(**{**GOOD, "contract": "Two year", "gender": "Male", "tenure": 60,
                    "internet_service": "Fiber optic", "senior_citizen": True}),
        Customer(**{**GOOD, "contract": "One year", "payment_method": "Mailed check",
                    "phone_service": False, "multiple_lines": "No phone service"}),
    ]


def test_encoder_and_config_agree_on_the_feature_order():
    assert list(load_bundle()["feature_order"]) == config.feature_order()


def test_output_shape_order_and_values():
    out = encode_customers(_customers())
    assert list(out.columns) == config.feature_order() and len(out) == 3
    assert np.isfinite(out.to_numpy(dtype=float)).all()


def test_the_baseline_categories_encode_to_zeros():
    """drop='first' makes Month-to-month and Female the reference categories (all zeros)."""
    out = encode_customers(_customers())
    assert out.loc[0, "contract_One year"] == 0 and out.loc[0, "contract_Two year"] == 0
    assert out.loc[0, "gender_Male"] == 0                  # the default gender
    assert out.loc[1, "contract_Two year"] == 1 and out.loc[1, "gender_Male"] == 1
    assert out.loc[2, "contract_One year"] == 1


def test_column_order_of_the_input_does_not_matter():
    df = pd.DataFrame([c.model_dump() for c in _customers()])
    shuffled = df[list(reversed(df.columns))]
    pd.testing.assert_frame_equal(build_features(df), build_features(shuffled))


def test_a_row_scores_the_same_alone_or_in_a_batch():
    df = pd.DataFrame([c.model_dump() for c in _customers()])
    batch = build_features(df)
    for i in range(len(df)):
        alone = build_features(df.iloc[[i]]).reset_index(drop=True)
        pd.testing.assert_frame_equal(alone, batch.iloc[[i]].reset_index(drop=True))


def test_unseen_category_and_missing_column_are_refused():
    df = pd.DataFrame([GOOD | {"gender": "Female"}])
    with pytest.raises(ValueError, match="unseen"):
        build_features(df.assign(contract="Weekly"))
    with pytest.raises(ValueError, match="missing column"):
        build_features(df.drop(columns=["tenure"]))


def test_light_modules_do_not_import_the_model_libraries():
    code = ("import sys, churn_core.config, churn_core.schema, churn_core.checks, "
            "churn_core.economics, churn_core.bands; "
            "bad = [m for m in ('lightgbm', 'sklearn', 'shap') if m in sys.modules]; "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)


# the real skew test
DATA = config.data_dir()
NEEDED = ["Telco-Customer-Churn.csv", "split_assignment.csv", "X_train_encoded.pkl", "X_test.pkl"]


@pytest.mark.skipif(not all((DATA / f).exists() for f in NEEDED), reason="data files not available (CI)")
@pytest.mark.parametrize("split, pickle", [("train", "X_train_encoded.pkl"), ("test", "X_test.pkl")])
def test_reproduces_the_saved_training_and_test_matrices(split, pickle):
    sa = pd.read_csv(DATA / "split_assignment.csv")
    ids = sa[sa.split == split].sort_values("position").customer_id
    raw = pd.read_csv(DATA / "Telco-Customer-Churn.csv").set_index("customerID").loc[ids].reset_index()
    rebuilt = build_features(clean_raw(raw))
    saved = joblib.load(DATA / pickle).reset_index(drop=True)
    assert list(rebuilt.columns) == list(saved.columns) == config.feature_order()
    assert len(rebuilt) == len(saved)
    pd.testing.assert_frame_equal(rebuilt, saved, check_dtype=False, atol=1e-9)
    assert set(CLEAN_FEATURE_COLUMNS) <= set(clean_raw(raw).columns)