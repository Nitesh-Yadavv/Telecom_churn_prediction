import numpy as np
import pandas as pd
import pytest

from churn_core import bands, config
from db.load_data import TABLE_COLUMNS, prepare, to_records


def test_each_frame_has_exactly_the_table_columns(loader_parts):
    frames = prepare(*loader_parts)
    assert set(frames) == set(TABLE_COLUMNS)
    for name, cols in TABLE_COLUMNS.items():
        assert list(frames[name].columns) == cols, name
        assert len(frames[name]) == 10


def test_split_scores_bands_and_flags_are_filled_consistently(loader_parts):
    raw, split, scores = loader_parts
    frames = prepare(raw, split, scores)
    expected_split = split.set_index("customer_id").loc[frames["customers"].customer_id, "split"]
    assert list(frames["customers"].split) == list(expected_split)
    s = frames["scores"]
    assert list(s.band) == list(bands.assign_band(s.probability))
    assert (s.flag == (s.probability >= config.default_cutoff())).all()
    assert set(s.score_kind) == {"oof", "test"}


def test_cleaning_matches_the_apps_rules(loader_parts):
    frames = prepare(*loader_parts)
    customers, billing, churn = frames["customers"], frames["billing"], frames["churn_status"]
    assert customers.senior_citizen.dtype == bool and billing.total_charges.iloc[0] == 0
    assert bool(customers.senior_citizen.iloc[1]) is True and bool(churn.churn.iloc[1]) is True
    assert churn.churn.dtype == bool


def _break(parts, how):
    raw, split, scores = (p.copy() for p in parts)
    if how == "flipped flag":
        scores.loc[0, "flag"] = 1 - int(scores.loc[0, "flag"])
    elif how == "unknown customer in scores":
        scores.loc[0, "customer_id"] = "NOT-IN-THE-CSV"
    elif how == "customer missing from the split file":
        split = split[split.customer_id != raw.customerID.iloc[0]]
    elif how == "wrong score kind":
        scores.loc[0, "score_kind"] = "test" if scores.loc[0, "score_kind"] == "oof" else "oof"
    elif how == "different monthly charge":
        scores.loc[0, "monthly_charges"] += 10
    elif how == "duplicate raw customer":
        raw = pd.concat([raw, raw.head(1)], ignore_index=True)
    return raw, split, scores


@pytest.mark.parametrize("how", ["flipped flag", "unknown customer in scores", "customer missing from the split file",
                                 "wrong score kind", "different monthly charge", "duplicate raw customer"])
def test_inconsistent_inputs_are_refused(loader_parts, how):
    with pytest.raises(ValueError):
        prepare(*_break(loader_parts, how))


def test_records_are_plain_python_values(loader_parts):
    frames = prepare(*loader_parts)
    rec = to_records(frames["customers"])[0]
    assert type(rec["senior_citizen"]) is bool and type(rec["tenure"]) is int and isinstance(rec["customer_id"], str)
    rec = to_records(frames["billing"])[0]
    assert type(rec["monthly_charges"]) is float
    assert to_records(pd.DataFrame({"a": [np.nan, 1.0]}))[0]["a"] is None


# ---- the real files: runs only on your machine (the raw CSV is not in the repo) ----
RAW = config.data_dir() / "Telco-Customer-Churn.csv"


@pytest.mark.skipif(not RAW.exists(), reason="raw CSV not available (CI)")
def test_the_real_files_give_the_known_totals():
    frames = prepare(pd.read_csv(RAW), pd.read_csv(config.data_dir() / "split_assignment.csv"),
                     pd.read_csv(config.reports_dir() / "scores.csv"))
    c, s = frames["customers"], frames["scores"]
    assert len(c) == len(s) == 7043 and (c.split == "train").sum() == 5634 and (c.split == "test").sum() == 1409
    flagged_test = s[s.customer_id.isin(c[c.split == "test"].customer_id)].flag.sum()
    assert flagged_test == 496