"""
churn_core/config.py

The single place where paths, the model version and the saved business settings live, so no other
module hard-codes a number. Pure and light: standard library only (safe for the dashboard image).

Settings are READ from models/model_config.json, which notebook 04 wrote. Nothing here is guessed.
"""
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Must match the "model_version" column written to reports/scores.csv (a test checks this).
MODEL_VERSION = "churn-lgbm-sigmoid-v1"


def home() -> Path:
    """Project root. Set CHURN_HOME to override (useful in Docker or tests); otherwise the
    folder that contains the churn_core package."""
    env = os.environ.get("CHURN_HOME")
    return Path(env) if env else Path(__file__).resolve().parents[1]


def models_dir() -> Path:
    return home() / "models"


def reports_dir() -> Path:
    return home() / "reports"


def data_dir() -> Path:
    return home() / "data"


@lru_cache(maxsize=None)
def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def model_config() -> dict:
    """Contents of models/model_config.json."""
    return _read_json(str(models_dir() / "model_config.json"))


def reference_stats() -> dict:
    """Contents of models/reference_stats.json (distributions and data-health limits)."""
    return _read_json(str(models_dir() / "reference_stats.json"))


@dataclass(frozen=True)
class Assumptions:
    """The business assumptions behind the cutoff. These are assumptions, not measured facts."""
    r: float        # offer success rate: share of contacted churners who accept and stay
    d: float        # discount given, as a share of the monthly bill
    months: int     # months a retained customer keeps paying (the horizon)


def default_assumptions() -> Assumptions:
    b = model_config()["business_defaults"]
    return Assumptions(r=float(b["offer_success_rate_r"]),
                       d=float(b["discount_d"]),
                       months=int(b["months_retained_M"]))


def default_cutoff() -> float:
    """The cutoff stored with the model (derived from the defaults, not a tuned value)."""
    return float(model_config()["business_defaults"]["cutoff"])


def feature_order() -> list[str]:
    """Column order the model was trained on. LightGBM does not check column names at prediction
    time, so a reordered row would be scored wrongly without any error."""
    return list(model_config()["feature_order"])