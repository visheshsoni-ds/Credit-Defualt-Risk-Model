import numpy as np
from crgov.metrics import amex_metric, evaluate, ks_stat
from crgov.monitor import psi


def test_amex_perfect_and_random():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.25).astype(int)
    assert abs(amex_metric(y, y) - 1.0) < 1e-9
    assert abs(amex_metric(y, rng.random(5000))) < 0.1


def test_ks_bounds():
    y = np.array([0, 0, 1, 1]); assert ks_stat(y, np.array([.1, .2, .8, .9])) == 1.0


def test_psi_identical_and_shifted():
    rng = np.random.default_rng(1)
    a = rng.normal(size=20000)
    assert psi(a, rng.normal(size=20000)) < 0.02
    assert psi(a, rng.normal(loc=1.5, size=20000)) > 0.25


def test_psi_handles_missing():
    a = np.array([1.0, 2, 3, 4, 5, np.nan] * 500)
    assert psi(a, a) < 1e-6
    assert psi(a, np.full(3000, np.nan)) > 0.25


def test_evaluate_keys():
    y = np.array([0, 1] * 500); p = np.clip(y * 0.5 + 0.2, 0, 1)
    assert {"auc", "ks", "gini", "brier", "amex_metric", "calibration_max_gap", "calibration_excess_gap", "portfolio_pd_error"} <= set(evaluate(y, p))
