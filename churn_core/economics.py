"""

The money logic of the retention campaign, written once. It is a direct, tested copy of the
functions prototyped in notebooks/05_retention_policy.ipynb. Pure: numpy only (no model, no pandas),
so the dashboard image can import it without LightGBM or scikit-learn.

Model of one contact (everyone contacted is offered the discount):
  - a churner who accepts (probability r) keeps paying (1 - d) * bill for `months` months
  - a customer who would have stayed anyway takes the discount: costs d * bill for `months` months
  - every contact costs c

r (offer success rate), d (discount) and months are ASSUMPTIONS, not measured values.
"""
import numpy as np


def _check_assumptions(r: float, d: float, months: float) -> None:
    if not 0 < r <= 1:
        raise ValueError(f"r (offer success rate) must be in (0, 1], got {r}")
    if not 0 <= d < 1:
        raise ValueError(f"d (discount) must be in [0, 1), got {d}")
    if months <= 0:
        raise ValueError(f"months must be positive, got {months}")


def _as_float_array(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains missing or infinite values")
    return a


def cutoff_from(r: float, d: float) -> float:
    """Break-even churn probability with no contact cost: contact when p * r * (1 - d) > (1 - p) * d,
    i.e. p > d / (d + r * (1 - d)). With r = 0.5 and d = 0.2 this is 1/3."""
    _check_assumptions(r, d, 1)
    return d / (d + r * (1 - d))


def ev_per_customer(p, bill, r: float, d: float, months: float, c: float = 0.0) -> np.ndarray:
    """EXPECTED net value of contacting each customer, using calibrated probabilities p."""
    _check_assumptions(r, d, months)
    p, bill = _as_float_array(p, "p"), _as_float_array(bill, "bill")
    if p.shape != bill.shape:
        raise ValueError("p and bill must have the same length")
    if p.size and (p.min() < 0 or p.max() > 1):
        raise ValueError("p must be probabilities between 0 and 1")
    return months * bill * (p * r * (1 - d) - (1 - p) * d) - c


def realized_per_customer(y, bill, r: float, d: float, months: float, c: float = 0.0) -> np.ndarray:
    """Value of contacting each customer given the REAL outcome y (1 = churned), expected over
    whether a churner accepts the offer."""
    _check_assumptions(r, d, months)
    y, bill = np.asarray(y).astype(int), _as_float_array(bill, "bill")
    if y.shape != bill.shape:
        raise ValueError("y and bill must have the same length")
    return months * bill * np.where(y == 1, r * (1 - d), -d) - c


def realized_value(flag, y, bill, r: float, d: float, months: float, c: float = 0.0) -> float:
    """Total value of a flagged set, using real churn outcomes."""
    flag = np.asarray(flag, dtype=bool)
    return float(realized_per_customer(y, bill, r, d, months, c)[flag].sum())


def expected_value(flag, p, bill, r: float, d: float, months: float, c: float = 0.0) -> float:
    """Total value of a flagged set according to the model's own probabilities."""
    flag = np.asarray(flag, dtype=bool)
    return float(ev_per_customer(p, bill, r, d, months, c)[flag].sum())


def policy_threshold(p, cutoff: float) -> np.ndarray:
    """Plain rule: contact everyone whose probability is at or above the cutoff."""
    return _as_float_array(p, "p") >= cutoff


def policy_ranked(p, bill, r: float, d: float, months: float, c: float = 0.0,
                  fixed_cost: float = 0.0, capacity: int | None = None) -> np.ndarray:
    """Contact customers with POSITIVE expected value, best first, within a contact capacity.

    With a per-contact cost c the break-even probability depends on each customer's bill, so a
    single probability cutoff is no longer optimal. fixed_cost is a campaign-level cost: if the
    selected customers do not earn more than it, nobody is contacted."""
    ev = ev_per_customer(p, bill, r, d, months, c)
    n_positive = int((ev > 0).sum())
    k = n_positive if capacity is None else max(0, min(int(capacity), n_positive))
    flag = np.zeros(ev.shape[0], dtype=bool)
    flag[np.argsort(-ev, kind="stable")[:k]] = True
    if flag.any() and ev[flag].sum() <= fixed_cost:
        flag[:] = False
    return flag


def sensitivity(p, y, bill, r_values, d: float, months: float, base_cutoff: float) -> list[dict]:
    """How the real-outcome saving changes with the true offer success rate.
    For each r: the cutoff re-derived for that r ('adapted') and the cutoff kept at base_cutoff."""
    p = _as_float_array(p, "p")
    kept = p >= base_cutoff
    rows = []
    for r in r_values:
        cut = cutoff_from(r, d)
        adapted = p >= cut
        rows.append({
            "r": float(r), "cutoff": cut, "flagged": int(adapted.sum()),
            "saving_adapted": realized_value(adapted, y, bill, r, d, months),
            "saving_kept": realized_value(kept, y, bill, r, d, months),
        })
    return rows