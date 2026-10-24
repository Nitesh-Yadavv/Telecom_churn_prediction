import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from churn_core import bands, config

EDGES = (0.10, 1 / 3, 0.60)


def test_assignment_and_edge_inclusion():
    p = [0.0, 0.0999, 0.10, 0.20, 1 / 3, 0.45, 0.60, 1.0]
    assert list(bands.assign_band(p, EDGES)) == [
        "Low", "Low", "Medium", "Medium", "High", "High", "Very high", "Very high"]


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        bands.assign_band([0.5], (0.5, 0.4, 0.9))        # edges must increase
    with pytest.raises(ValueError):
        bands.assign_band([1.2], EDGES)
    with pytest.raises(ValueError):
        bands.assign_band([np.nan], EDGES)


def test_the_cutoff_edge_is_the_derived_cutoff_not_a_tuned_number():
    low_medium, cutoff, very_high = bands.default_edges()
    assert cutoff == config.default_cutoff()
    assert 0 < low_medium < cutoff < very_high < 1


def test_flagged_bands_match_the_campaign_flag():
    """'High' and above must mean exactly 'probability at or above the cutoff'."""
    p = pd.read_csv(config.reports_dir() / "test_predictions.csv").p_cal.to_numpy()
    flagged_by_band = bands.is_flagged_band(bands.assign_band(p))
    assert (flagged_by_band == (p >= config.default_cutoff())).all()


@pytest.mark.parametrize("sample, n", [("test", 1409), ("oof", 5634)])
def test_published_table_shows_the_bands_mean_something(sample, n):
    rows = bands.published_table(sample)
    assert [r["band"] for r in rows] == list(bands.LABELS)
    assert sum(r["customers"] for r in rows) == n
    assert all(r["customers"] >= 100 for r in rows)              # no tiny bands
    rates = [r["observed_churn_rate"] for r in rows]
    assert rates == sorted(rates) and rates[0] < rates[-1]       # risk rises with the band


def test_importing_bands_stays_light():
    code = ("import sys, churn_core.bands; "
            "bad = [m for m in ('lightgbm', 'sklearn', 'pandas', 'shap') if m in sys.modules]; "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)