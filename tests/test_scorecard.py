import numpy as np
import pandas as pd
from crgov.explain import humanise, top_k_reasons
from crgov.scorecard import fit_scorecard

SC = {"n_bins": 8, "iv_min": 0.02, "max_features": 10, "base_score": 600, "base_odds": 50, "pdo": 20, "C": 1.0}


def _data(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    X = pd.DataFrame({"a_last": z + rng.normal(size=n) * .5, "b_mean": -z + rng.normal(size=n) * .8,
                      "noise_last": rng.normal(size=n)})
    X.loc[rng.random(n) < .1, "a_last"] = np.nan
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * z - 1)))).astype(int)
    return X, y


def test_fit_and_score():
    X, y = _data()
    m, iv = fit_scorecard(X, y, SC)
    pdv = m.predict_pd(X)
    assert ((pdv > 0) & (pdv < 1)).all()
    assert (m.coef > 0).all()                                   # sign check enforced
    assert "noise_last" not in m.features                       # IV screen
    assert np.corrcoef(m.score(X), pdv)[0, 1] < -0.9            # higher score = lower PD
    assert len(m.reasons(X.head(3), 2)) == 3


def test_missing_column_is_null_not_crash():
    X, y = _data(); m, _ = fit_scorecard(X, y, SC)
    assert len(m.predict_pd(X[["a_last"]].head(5))) == 5


def test_humanise():
    assert "Delinquency" in humanise("D_39_last") and "Elevated risk" in humanise("ZZ_9_mean")


def test_reasons_one_per_variable():
    r = top_k_reasons(np.array([[5.0, 4.0, 3.0, 2.0]]), ["P_1_mean", "P_1_max", "D_2_last", "B_1_last"], 3)[0]
    assert [x["feature"] for x in r] == ["P_1_mean", "D_2_last", "B_1_last"]
