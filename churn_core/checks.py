"""

Data-health checks for uploaded files, run BEFORE anything reaches the model.

  ERROR   = the row cannot be scored (missing column, unseen category, not a number, negative value,
            blank TotalCharges for a customer who is not brand new). Error rows are left out.
  WARNING = the row is scored but looks unusual (value outside the training range, contradictory
            service answers, duplicate IDs, extra columns). The model is extrapolating there.

Limits come from models/reference_stats.json ("data_health"), learned from training rows only.
Input is the RAW (Kaggle / template) format. Row numbers count like a spreadsheet: header = row 1.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from churn_core import config
from churn_core.schema import CLEAN_TO_RAW, INTERNET_SERVICE_COLUMNS, OPTIONAL_RAW_COLUMNS

MAX_ISSUES = 500          # issues listed in detail; counts stay exact beyond this
NUMERIC_RAW = ["tenure", "MonthlyCharges", "TotalCharges"]
_INTERNET_RAW = [CLEAN_TO_RAW[c] for c in INTERNET_SERVICE_COLUMNS]


@dataclass(frozen=True)
class Issue:
    row: int | None        # spreadsheet row; None = the whole file
    column: str | None
    level: str             # "error" or "warning"
    message: str


@dataclass
class HealthReport:
    n_rows: int
    issues: list
    counts: dict           # {(level, column): exact count}
    valid_mask: np.ndarray  # True for rows that can be scored
    rows_with_warnings: int
    truncated: bool

    @property
    def errors(self) -> list:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def n_valid(self) -> int:
        return int(self.valid_mask.sum())

    def summary(self) -> dict:
        def by(level):
            return {col: n for (lvl, col), n in sorted(self.counts.items()) if lvl == level}
        return {"rows": self.n_rows, "rows_valid": self.n_valid,
                "rows_with_errors": self.n_rows - self.n_valid,
                "rows_with_warnings": self.rows_with_warnings,
                "error_counts": by("error"), "warning_counts": by("warning"),
                "issues_truncated": self.truncated}

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([i.__dict__ for i in self.issues],
                            columns=["row", "column", "level", "message"])


class _Collector:
    def __init__(self, n: int):
        self.n, self.issues, self.counts, self.truncated = n, [], {}, False
        self.error_rows = np.zeros(n, dtype=bool)
        self.warning_rows = np.zeros(n, dtype=bool)

    def _record(self, level, column, issue_or_none, k=1):
        key = (level, column or "file")
        self.counts[key] = self.counts.get(key, 0) + k
        if issue_or_none is not None:
            if len(self.issues) < MAX_ISSUES:
                self.issues.append(issue_or_none)
            else:
                self.truncated = True

    def file(self, level, column, message):
        self._record(level, column, Issue(None, column, level, message))

    def rows(self, level, column, mask, message):
        idx = np.flatnonzero(np.asarray(mask, dtype=bool))
        if idx.size == 0:
            return
        (self.error_rows if level == "error" else self.warning_rows)[idx] = True
        key = (level, column or "file")
        self.counts[key] = self.counts.get(key, 0) + int(idx.size)
        for i in idx:
            if len(self.issues) >= MAX_ISSUES:
                self.truncated = True
                break
            text = message(i) if callable(message) else message
            self.issues.append(Issue(int(i) + 2, column, level, text))

    def report(self, valid_mask) -> HealthReport:
        return HealthReport(self.n, self.issues, self.counts, valid_mask,
                            int(self.warning_rows.sum()), self.truncated)


def _text(series: pd.Series, column: str) -> pd.Series:
    t = series.astype(str).str.strip()
    return t.str.replace(r"\.0$", "", regex=True) if column == "SeniorCitizen" else t


def check_upload(df: pd.DataFrame) -> HealthReport:
    """Check an uploaded RAW-format frame. Never raises on bad data; reports it instead."""
    stats = config.reference_stats()["data_health"]
    n = len(df)
    c = _Collector(n)

    required = [col for col in stats["required_columns"] if col not in OPTIONAL_RAW_COLUMNS]
    missing = [col for col in required if col not in df.columns]
    for col in missing:
        c.file("error", col, f"Required column '{col}' is missing from the file")
    if n == 0:
        c.file("error", None, "The file has no rows")
    extra = [col for col in df.columns if col not in set(stats["required_columns"]) | {"customerID", "Churn"}]
    if extra:
        c.file("warning", None, f"These columns are not used and will be ignored: {', '.join(map(str, extra))}")
    if missing or n == 0:
        return c.report(np.zeros(n, dtype=bool))

    # --- categorical columns: value must be one the model was trained on
    for col, allowed in stats["categorical"].items():
        if col not in df.columns:
            continue
        text = _text(df[col], col)
        absent = df[col].isna().to_numpy()
        bad = (~absent) & (~text.isin(allowed)).to_numpy()
        c.rows("error", col, absent, f"{col} is missing a value")
        c.rows("error", col, bad,
               lambda i, col=col, text=text, allowed=allowed:
               f"'{text.iloc[i]}' is not an allowed value for {col} (allowed: {', '.join(allowed)})")

    # --- numeric columns: parse, blanks, negatives, training range
    num = {col: pd.to_numeric(df[col], errors="coerce") for col in NUMERIC_RAW}
    tenure_zero = (num["tenure"] == 0).to_numpy()
    for col in NUMERIC_RAW:
        raw = df[col]
        blank = (raw.isna() | raw.astype(str).str.strip().eq("")).to_numpy()
        not_number = num[col].isna().to_numpy() & ~blank
        c.rows("error", col, not_number, lambda i, col=col: f"'{df[col].iloc[i]}' is not a number")
        blank_bad = blank & ~tenure_zero if col == "TotalCharges" else blank
        c.rows("error", col, blank_bad,
               f"{col} is blank" + (" (allowed only when tenure is 0)" if col == "TotalCharges" else ""))
        negative = (num[col] < 0).to_numpy()
        c.rows("error", col, negative, lambda i, col=col: f"{col} is negative ({num[col].iloc[i]})")
        lim = stats["numeric"][col]
        outside = ((num[col] < lim["min"]) | (num[col] > lim["max"])).to_numpy() & ~negative
        c.rows("warning", col, outside,
               lambda i, col=col, lim=lim: f"{col} = {num[col].iloc[i]} is outside the training range "
                                           f"{lim['min']:g} to {lim['max']:g}; treat the score with care")

    # --- answers that contradict each other
    if "InternetService" in df.columns:
        no_internet = (df["InternetService"].astype(str).str.strip() == "No").to_numpy()
        for col in _INTERNET_RAW:
            says_none = (df[col].astype(str).str.strip() == "No internet service").to_numpy()
            c.rows("warning", col, no_internet != says_none,
                   f"InternetService and {col} contradict each other")
    if "PhoneService" in df.columns and "MultipleLines" in df.columns:
        no_phone = (df["PhoneService"].astype(str).str.strip() == "No").to_numpy()
        says_none = (df["MultipleLines"].astype(str).str.strip() == "No phone service").to_numpy()
        c.rows("warning", "MultipleLines", no_phone != says_none,
               "PhoneService and MultipleLines contradict each other")

    # --- duplicate customer IDs
    if "customerID" in df.columns:
        dup = df["customerID"].notna() & df["customerID"].duplicated(keep=False)
        c.rows("warning", "customerID", dup.to_numpy(),
               lambda i: f"customerID '{df['customerID'].iloc[i]}' appears more than once")

    return c.report(~c.error_rows)


def customer_warnings(values: dict) -> list:
    """Range warnings for ONE customer in CLEAN format (the single-customer API path).
    values needs tenure, monthly_charges, total_charges."""
    stats = config.reference_stats()["data_health"]["numeric"]
    out = []
    for clean_col in ["tenure", "monthly_charges", "total_charges"]:
        lim = stats[CLEAN_TO_RAW[clean_col]]
        v = values[clean_col]
        if v < lim["min"] or v > lim["max"]:
            out.append(f"{clean_col} = {v:g} is outside the training range "
                       f"{lim['min']:g} to {lim['max']:g}; treat the score with care")
    return out