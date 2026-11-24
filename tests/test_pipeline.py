import pandas as pd
from crgov.governance import unverified_numbers
from crgov.monitor import decide
from crgov.pipeline import route

CFG = {"monitoring": {"psi_watch": .1, "psi_retrain": .25, "watch_amber_features": 5, "max_red_features": 10, "min_auc_floor": .72}}


def S(**k):
    return {"score_psi": .02, "n_red_features": 0, "n_amber_features": 0, "auc": None, **k}


def test_decide():
    assert decide(S(), CFG)[0] == "OK"
    assert decide(S(n_amber_features=6), CFG)[0] == "WATCH"
    assert decide(S(score_psi=.4), CFG)[0] == "RETRAIN"
    assert decide(S(n_red_features=12), CFG)[0] == "RETRAIN"
    assert decide(S(auc=.6), CFG)[0] == "RETRAIN"


def test_route():
    assert route("RETRAIN") == "retrain_and_validate"


def test_number_guard():
    facts = {"score_psi": 0.312, "n_rows": 1200, "date": "2026-01-04"}
    assert unverified_numbers("Score PSI is 0.312 on 1200 rows (2026-01-04).", facts) == []
    assert unverified_numbers("Score PSI is 0.512.", facts) == ["0.512"]
