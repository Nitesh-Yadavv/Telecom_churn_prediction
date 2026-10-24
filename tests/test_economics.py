import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from churn_core import config
from churn_core import economics as eco

R, D, M = 0.5, 0.2, 12


# ---------- hand-checkable examples ----------
def test_cutoff_formula():
    assert eco.cutoff_from(0.5, 0.2) == pytest.approx(1 / 3, abs=1e-12)
    assert eco.cutoff_from(1.0, 0.2) == pytest.approx(0.2, abs=1e-12)


def test_one_customer_by_hand():
    # p = 0.5, bill = 100:  12 * 100 * (0.5 * 0.5 * 0.8 - 0.5 * 0.2) = 120
    assert eco.ev_per_customer([0.5], [100], R, D, M)[0] == pytest.approx(120)
    # a churner who is contacted: 12 * 100 * 0.5 * 0.8 = 480; a loyal customer: -12 * 100 * 0.2 = -240
    assert eco.realized_per_customer([1], [100], R, D, M)[0] == pytest.approx(480)
    assert eco.realized_per_customer([0], [100], R, D, M)[0] == pytest.approx(-240)
    # contact cost is subtracted from each
    assert eco.realized_per_customer([1], [100], R, D, M, c=30)[0] == pytest.approx(450)


def test_expected_value_changes_sign_exactly_at_the_cutoff():
    cut = eco.cutoff_from(R, D)
    bills = np.array([20.0, 55.0, 119.0])            # the sign must not depend on the bill
    above = eco.ev_per_customer(np.full(3, cut + 0.01), bills, R, D, M)
    below = eco.ev_per_customer(np.full(3, cut - 0.01), bills, R, D, M)
    assert (above > 0).all() and (below < 0).all()


# ---------- policies ----------
def _random_customers(n=500, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.beta(2, 5, n)
    bill = rng.uniform(20, 120, n)
    y = (rng.random(n) < p).astype(int)
    return p, y, bill


def test_ranked_equals_threshold_without_contact_cost():
    p, _, bill = _random_customers()
    assert (eco.policy_ranked(p, bill, R, D, M) == eco.policy_threshold(p, eco.cutoff_from(R, D))).all()


def test_ranked_contacts_fewer_customers_as_contact_cost_grows():
    p, _, bill = _random_customers()
    counts = [eco.policy_ranked(p, bill, R, D, M, c=c).sum() for c in (0, 10, 40, 150)]
    assert counts == sorted(counts, reverse=True) and counts[0] > counts[-1]


def test_capacity_and_fixed_cost():
    p, _, bill = _random_customers()
    assert eco.policy_ranked(p, bill, R, D, M, capacity=50).sum() == 50
    assert eco.policy_ranked(p, bill, R, D, M, capacity=0).sum() == 0
    assert eco.policy_ranked(p, bill, R, D, M, fixed_cost=1e12).sum() == 0     # campaign not worth running


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        eco.cutoff_from(0, 0.2)
    with pytest.raises(ValueError):
        eco.cutoff_from(0.5, 1.0)
    with pytest.raises(ValueError):
        eco.ev_per_customer([1.2], [50], R, D, M)
    with pytest.raises(ValueError):
        eco.ev_per_customer([0.2, 0.3], [50], R, D, M)
    with pytest.raises(ValueError):
        eco.ev_per_customer([np.nan], [50], R, D, M)


def test_importing_economics_stays_light():
    code = ("import sys, churn_core.economics; "
            "bad = [m for m in ('lightgbm', 'sklearn', 'pandas', 'shap') if m in sys.modules]; "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)


# ---------- reproduces the saved results (the real data) ----------
@pytest.fixture(scope="module")
def preds():
    return pd.read_csv(config.reports_dir() / "test_predictions.csv")


@pytest.fixture(scope="module")
def metrics():
    return json.loads((config.reports_dir() / "metrics.json").read_text())


def test_reproduces_metrics_json(preds, metrics):
    p, y, bill = preds.p_cal, preds.actual_churn, preds.monthly_charges
    flag = eco.policy_threshold(p, config.default_cutoff())
    assert flag.sum() == metrics["flagged"]
    assert eco.realized_value(flag, y, bill, R, D, M) == pytest.approx(metrics["actual_saving_12m"], abs=0.01)
    assert eco.expected_value(flag, p, bill, R, D, M) == pytest.approx(metrics["expected_saving_12m"], abs=0.01)


def test_reproduces_notebook_numbers(preds):
    """The headline numbers from notebooks 04 and 05."""
    p, y, bill = preds.p_cal.to_numpy(), preds.actual_churn.to_numpy(), preds.monthly_charges.to_numpy()
    n = len(p)
    flag = eco.policy_threshold(p, config.default_cutoff())
    naive = ((preds.contract == "Month-to-month") & (preds.tenure < 12)).to_numpy()
    top_k = np.zeros(n, dtype=bool)
    top_k[np.argsort(-p)[:int(naive.sum())]] = True
    everyone = np.ones(n, dtype=bool)

    assert eco.realized_value(flag, y, bill, R, D, M) == pytest.approx(60674, abs=1)
    assert eco.expected_value(flag, p, bill, R, D, M) == pytest.approx(67009, abs=1)
    assert eco.realized_value(everyone, y, bill, R, D, M) == pytest.approx(-17132, abs=1)
    assert eco.realized_value(naive, y, bill, R, D, M) == pytest.approx(43079, abs=1)
    assert naive.sum() == 385
    assert eco.realized_value(top_k, y, bill, R, D, M) == pytest.approx(62382, abs=1)
    # random flags, same number as the model flags: expected value of a random draw
    random_expected = eco.realized_value(everyone, y, bill, R, D, M) / n * flag.sum()
    assert random_expected == pytest.approx(-6031, abs=1)


def test_reproduces_sensitivity_table(preds):
    rows = eco.sensitivity(preds.p_cal, preds.actual_churn, preds.monthly_charges,
                           [0.1, 0.3, 0.5, 0.7], D, M, base_cutoff=config.default_cutoff())
    adapted = {row["r"]: row["saving_adapted"] for row in rows}
    kept = {row["r"]: row["saving_kept"] for row in rows}
    assert adapted[0.1] == pytest.approx(2553, abs=1)
    assert adapted[0.3] == pytest.approx(28420, abs=1)
    assert adapted[0.5] == pytest.approx(60674, abs=1)
    assert adapted[0.7] == pytest.approx(104828, abs=1)
    assert kept[0.1] == pytest.approx(-21134, abs=1)
    assert kept[0.7] == pytest.approx(101578, abs=1)