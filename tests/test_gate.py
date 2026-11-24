from crgov.validate import evaluate_gate

G = {"min_auc": 0.75, "min_ks": 0.40, "max_calibration_excess_gap": 0.02, "max_portfolio_pd_error": 0.02, "max_generalisation_gap": 0.05,
     "min_uplift_vs_champion": 0.002}
GOOD = {"auc": 0.80, "ks": 0.50, "calibration_excess_gap": 0.0, "portfolio_pd_error": 0.005}


def test_bootstrap_pass():
    r = evaluate_gate(GOOD, None, 0.82, G)
    assert r["passed"] and r["bootstrap"]


def test_each_failure_blocks():
    assert not evaluate_gate({**GOOD, "auc": 0.70}, None, 0.72, G)["passed"]
    assert not evaluate_gate({**GOOD, "ks": 0.3}, None, 0.82, G)["passed"]
    assert not evaluate_gate({**GOOD, "calibration_excess_gap": 0.08}, None, 0.82, G)["passed"]
    assert not evaluate_gate({**GOOD, "portfolio_pd_error": 0.06}, None, 0.82, G)["passed"]
    assert not evaluate_gate(GOOD, None, 0.95, G)["passed"]           # overfit


def test_must_beat_champion():
    champ = dict(GOOD)
    assert not evaluate_gate(GOOD, champ, 0.82, G)["passed"]           # no uplift
    assert evaluate_gate({**GOOD, "auc": 0.81}, champ, 0.82, G)["passed"]
