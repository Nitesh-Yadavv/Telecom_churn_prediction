import pandas as pd
import pytest

from churn_core import config
from churn_core.checks import MAX_ISSUES, check_upload, customer_warnings

BASE = {
    "customerID": "T-1", "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
    "OnlineSecurity": "Yes", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 55.0,
    "TotalCharges": 660.0,
}


def frame(*changes):
    """One row per change dict, each applied on top of a valid customer."""
    rows = [{**BASE, "customerID": f"T-{i}", **c} for i, c in enumerate(changes or [{}])]
    return pd.DataFrame(rows)


def find(report, level, column):
    return [i for i in report.issues if i.level == level and i.column == column]


def test_a_valid_file_has_no_issues():
    r = check_upload(frame({}, {"Contract": "Two year"}))
    assert r.issues == [] and r.valid_mask.all() and r.summary()["rows_with_errors"] == 0


def test_missing_required_column_is_a_file_level_error():
    r = check_upload(frame({}).drop(columns=["Contract"]))
    assert not r.valid_mask.any()
    assert find(r, "error", "Contract")[0].row is None


def test_gender_and_customer_id_are_optional():
    r = check_upload(frame({}).drop(columns=["gender", "customerID"]))
    assert r.issues == [] and r.valid_mask.all()


def test_unseen_category_names_the_row_and_value():
    r = check_upload(frame({}, {"Contract": "Weekly"}))
    issue = find(r, "error", "Contract")[0]
    assert issue.row == 3 and "Weekly" in issue.message          # header = row 1, first data row = 2
    assert list(r.valid_mask) == [True, False]


def test_numeric_problems():
    r = check_upload(frame({"MonthlyCharges": "abc"}, {"MonthlyCharges": -5}, {"tenure": None}))
    assert list(r.valid_mask) == [False, False, False]
    assert "not a number" in find(r, "error", "MonthlyCharges")[0].message
    assert "negative" in find(r, "error", "MonthlyCharges")[1].message
    assert find(r, "error", "tenure")


def test_blank_total_charges_is_ok_only_for_new_customers():
    r = check_upload(frame({"tenure": 0, "TotalCharges": " "}, {"tenure": 5, "TotalCharges": " "}))
    assert list(r.valid_mask) == [True, False]
    assert "tenure is 0" in find(r, "error", "TotalCharges")[0].message


def test_out_of_range_is_a_warning_and_the_row_is_still_scored():
    r = check_upload(frame({"MonthlyCharges": 5000}))
    assert r.valid_mask.all() and r.rows_with_warnings == 1
    assert "outside the training range" in find(r, "warning", "MonthlyCharges")[0].message


def test_contradicting_services_duplicates_and_extra_columns_warn():
    f = frame({"InternetService": "No"}, {"PhoneService": "No"}, {})
    f.loc[2, "customerID"] = f.loc[0, "customerID"]              # duplicate ID
    f["Notes"] = "x"
    r = check_upload(f)
    assert r.valid_mask.all()
    assert find(r, "warning", "OnlineSecurity") and find(r, "warning", "MultipleLines")
    assert find(r, "warning", "customerID") and any("Notes" in i.message for i in r.warnings)


def test_issue_list_is_capped_but_counts_stay_exact():
    n = MAX_ISSUES + 50
    r = check_upload(pd.DataFrame([{**BASE, "Contract": "Weekly"}] * n))
    assert r.truncated and len(r.issues) == MAX_ISSUES
    assert r.summary()["error_counts"]["Contract"] == n and r.n_valid == 0


def test_empty_file_is_an_error():
    r = check_upload(frame({}).iloc[0:0])
    assert r.n_rows == 0 and any("no rows" in i.message for i in r.errors)


def test_single_customer_range_warnings():
    assert customer_warnings({"tenure": 12, "monthly_charges": 55.0, "total_charges": 660.0}) == []
    assert customer_warnings({"tenure": 500, "monthly_charges": 55.0, "total_charges": 660.0})


# ---- runs only on your machine (the raw CSV is not in the repo) ----
RAW_CSV = config.data_dir() / "Telco-Customer-Churn.csv"


@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw CSV not available (CI)")
def test_the_real_dataset_passes_every_check():
    raw = pd.read_csv(RAW_CSV)
    r = check_upload(raw)
    assert r.summary()["error_counts"] == {} and r.n_valid == len(raw) == 7043