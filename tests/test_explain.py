import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn_core import model
from churn_core.explain import (
    GROUP_INFO,
    LEVER_STATUS,
    LEVERS,
    contributions,
    explain,
    group_columns,
)
from churn_core.preprocess import build_features
from churn_core.schema import Customer, customer_frame

GOLDEN = json.loads((Path(__file__).parent / "golden_predictions.json").read_text())["rows"]


def _frame():
    return customer_frame([Customer(**row["customer"]) for row in GOLDEN])


def _logit(p):
    p = np.asarray(p)
    return np.log(p / (1 - p))


def test_every_feature_column_is_in_exactly_one_group():
    cols = list(build_features(_frame()).columns)
    groups = group_columns(cols)
    flat = [c for members in groups.values() for c in members]
    assert sorted(flat) == sorted(cols)
    assert len(groups["contract"]) == 2
    assert all(c in groups["internet"] for c in cols if c.endswith("No internet service"))


def test_every_group_has_a_label_and_every_lever_belongs_to_a_group():
    groups = group_columns(build_features(_frame()).columns)
    assert set(groups) == set(GROUP_INFO)
    assert set(LEVERS) <= set(GROUP_INFO)


def test_contributions_add_up_to_the_raw_models_prediction():
    """Base value + all contributions = the raw model's log-odds (SHAP's 'local accuracy')."""
    feats = build_features(_frame())
    contrib, base = contributions(feats)
    raw_p = model.predict_proba_raw(feats)
    np.testing.assert_allclose(base + contrib.sum(axis=1), _logit(raw_p), atol=1e-6)


@pytest.mark.filterwarnings("ignore:LightGBM binary classifier")
def test_contributions_equal_shap_tree_explainer():
    shap = pytest.importorskip("shap")
    feats = build_features(_frame())
    mine, _ = contributions(feats)
    values = shap.TreeExplainer(model.load_models()["raw"]).shap_values(feats)
    if isinstance(values, list):                      # older shap: one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:                              # newer shap: (rows, features, classes)
        values = values[:, :, 1]
    np.testing.assert_allclose(mine, values, atol=1e-6)


def test_gender_contributes_nothing():
    feats = build_features(_frame())
    contrib, _ = contributions(feats)
    assert np.abs(contrib[:, list(feats.columns).index("gender_Male")]).max() < 1e-12


def test_drivers_are_sorted_honest_and_labelled():
    for result in explain(_frame(), top_k=3):
        drivers = result["drivers"]
        assert 1 <= len(drivers) <= 3
        effects = [abs(d["effect"]) for d in drivers]
        assert effects == sorted(effects, reverse=True)
        for d in drivers:
            assert d["direction"] == ("raises" if d["effect"] > 0 else "lowers")
            assert d["label"] in d["sentence"] and d["direction"] in d["sentence"]
            if d["lever"] is not None:                # levers only next to risk-raising drivers
                assert d["effect"] > 0 and d["lever"]["status"] == LEVER_STATUS
        assert "not causes" in result["note"]


def test_waterfall_adds_up_and_drivers_come_from_it():
    feats = build_features(_frame())
    _, base = contributions(feats)
    for i, result in enumerate(explain(_frame())):
        assert result["base_logit"] == pytest.approx(base[i])
        total = result["base_logit"] + sum(w["effect"] for w in result["waterfall"])
        assert total == pytest.approx(result["raw_logit"], abs=1e-9)
        assert [d["feature"] for d in result["drivers"]] == [w["feature"] for w in result["waterfall"][:3]]


def test_works_on_a_single_customer_and_keeps_values_readable():
    one = explain(_frame().iloc[[0]], top_k=5)
    assert len(one) == 1
    values = {w["feature"]: w["value"] for w in one[0]["waterfall"]}
    if "tenure" in values:
        assert values["tenure"].endswith("months")
    assert isinstance(pd.DataFrame(one[0]["waterfall"]).effect.iloc[0], float)