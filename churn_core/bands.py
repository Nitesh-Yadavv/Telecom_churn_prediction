"""

Plain-language risk bands for a calibrated churn probability. Pure: numpy only (plus config).

  Low        below the first edge
  Medium     from the first edge up to the cost cutoff
  High       from the cost cutoff up to the last edge     <- worth contacting (flagged)
  Very high  at or above the last edge                    <- worth contacting (flagged)

The cutoff edge is NOT chosen from the data: it is the break-even probability derived from the
business assumptions (1/3 for r = 0.5, d = 0.2), so "High" and above means the same thing as the
campaign's flag. The other two edges come from the distribution of test-set scores (no outcomes
used) and are written by scripts/build_risk_bands.py together with the observed churn rate of each
band, so the bands are documented and can be shown to mean something.
Lower edges are inclusive: a probability exactly on an edge belongs to the higher band.
"""
import numpy as np

from churn_core import config

LABELS = ("Low", "Medium", "High", "Very high")
FLAGGED_LABELS = ("High", "Very high")

DESCRIPTIONS = {
    "Low": "Unlikely to leave",
    "Medium": "Some risk, but a discount would cost more than it saves on average",
    "High": "Likely to leave: a retention offer pays for itself on average",
    "Very high": "Very likely to leave: the strongest case for a retention offer",
}


def default_edges() -> tuple:
    """(low/medium edge, cost cutoff, high/very-high edge) saved with the model."""
    e = config.risk_bands()["edges"]
    return (float(e["low_medium"]), float(e["cutoff"]), float(e["high_very_high"]))


def assign_band(p, edges: tuple | None = None) -> np.ndarray:
    """Band label for each calibrated probability."""
    p = np.asarray(p, dtype=float)
    if p.size and (not np.all(np.isfinite(p)) or p.min() < 0 or p.max() > 1):
        raise ValueError("probabilities must be between 0 and 1")
    e = tuple(edges) if edges is not None else default_edges()
    if len(e) != 3 or not (e[0] < e[1] < e[2]):
        raise ValueError(f"edges must be three strictly increasing numbers, got {e}")
    return np.asarray(LABELS)[np.searchsorted(e, p, side="right")]


def is_flagged_band(band) -> np.ndarray:
    """True for the bands that mean 'contact this customer' at the default cutoff."""
    return np.isin(np.asarray(band), FLAGGED_LABELS)


def published_table(sample: str = "test") -> list:
    """The observed churn rate per band ('test' = headline, 'oof' = training customers' out-of-fold
    scores, a second check), as saved in models/risk_bands.json."""
    return config.risk_bands()["observed"][sample]