"""WoE + logistic-regression scorecard as an MLflow pyfunc model (the interpretable benchmark)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import mlflow.pyfunc

from .explain import top_k_reasons


# ---------------- binning / WoE ----------------
def _fit_edges(x: np.ndarray, n_bins: int):
    v = x[~np.isnan(x)]
    if v.size == 0 or np.unique(v).size < 2:
        return None
    return np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1]))


def _assign(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, x, side="right")
    idx[np.isnan(x)] = len(edges) + 1          # dedicated MISSING bin
    return idx


def _woe_iv(bin_idx: np.ndarray, y: np.ndarray, n_levels: int):
    bad = np.bincount(bin_idx[y == 1], minlength=n_levels).astype(float)
    good = np.bincount(bin_idx[y == 0], minlength=n_levels).astype(float)
    dg = (good + 0.5) / (good.sum() + 0.5 * n_levels)
    db = (bad + 0.5) / (bad.sum() + 0.5 * n_levels)
    woe = np.log(dg / db)                       # ln(good/bad): higher = safer
    return woe, float(((dg - db) * woe).sum()), good + bad


class ScorecardModel(mlflow.pyfunc.PythonModel):
    def __init__(self, features, edges, woe, coef, intercept, base_score, base_odds, pdo):
        self.features, self.edges, self.woe = list(features), edges, woe
        self.coef, self.intercept = np.asarray(coef), float(intercept)
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

    # ---- scoring ----
    def _woe_matrix(self, df: pd.DataFrame) -> np.ndarray:
        X = df.reindex(columns=self.features).astype(float)
        return np.column_stack([self.woe[f][_assign(X[f].to_numpy(), self.edges[f])] for f in self.features])

    def log_odds_good(self, df):
        return self._woe_matrix(df) @ self.coef + self.intercept

    def predict_pd(self, df) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(self.log_odds_good(df)))

    def score(self, df) -> np.ndarray:
        return self.offset + self.factor * self.log_odds_good(df)

    def points(self, df) -> np.ndarray:
        k = len(self.features)
        return (self._woe_matrix(df) * self.coef + self.intercept / k) * self.factor + self.offset / k

    def reasons(self, df, top_n: int = 4):
        """Standard adverse-action method: points lost vs the best bin of each feature."""
        pts = self.points(df)
        best = np.array([((self.woe[f] * c + self.intercept / len(self.features)) * self.factor
                          + self.offset / len(self.features)).max() for f, c in zip(self.features, self.coef)])
        deficit = best - pts
        return top_k_reasons(deficit, self.features, top_n)

    def points_table(self) -> pd.DataFrame:
        rows, k = [], len(self.features)
        for f, c in zip(self.features, self.coef):
            e = self.edges[f]
            labels = ([f"< {e[0]:.4g}"] + [f"[{e[i-1]:.4g}, {e[i]:.4g})" for i in range(1, len(e))]
                      + [f">= {e[-1]:.4g}", "MISSING"])
            for i, lab in enumerate(labels):
                w = self.woe[f][i]
                rows.append({"feature": f, "bin": lab, "woe": w, "coef": c,
                             "points": (w * c + self.intercept / k) * self.factor + self.offset / k})
        return pd.DataFrame(rows)

    def predict(self, context, model_input, params=None):
        return self.predict_pd(model_input)


# ---------------- fitting ----------------
def fit_scorecard(X: pd.DataFrame, y: np.ndarray, sc: dict) -> tuple[ScorecardModel, pd.DataFrame]:
    """Returns (model, iv_table). Steps: IV screen -> correlation filter -> LR on WoE -> sign check."""
    y = np.asarray(y).astype(int)
    edges, woe, iv = {}, {}, {}
    for f in X.columns:
        x = X[f].to_numpy(dtype=float)
        e = _fit_edges(x, sc["n_bins"])
        if e is None:
            continue
        w, i, _ = _woe_iv(_assign(x, e), y, len(e) + 2)
        edges[f], woe[f], iv[f] = e, w, i
    iv_tbl = pd.Series(iv).sort_values(ascending=False).rename("iv").to_frame()
    cand = iv_tbl[iv_tbl["iv"] >= sc["iv_min"]].index.tolist()[:200]
    if not cand:
        raise ValueError("No feature passes the IV threshold; lower scorecard.iv_min")

    W = pd.DataFrame({f: woe[f][_assign(X[f].to_numpy(dtype=float), edges[f])] for f in cand})
    chosen: list[str] = []
    for f in cand:                                     # greedy, highest IV first
        if len(chosen) >= sc["max_features"]:
            break
        if all(abs(np.corrcoef(W[f], W[g])[0, 1]) < 0.7 for g in chosen):
            chosen.append(f)

    y_good = 1 - y
    while True:                                        # drop wrong-sign coefficients and refit
        lr = LogisticRegression(C=sc["C"], max_iter=2000).fit(W[chosen], y_good)
        bad = [f for f, c in zip(chosen, lr.coef_[0]) if c <= 0]
        if not bad or len(chosen) <= 3:
            break
        chosen = [f for f in chosen if f not in bad]

    model = ScorecardModel(chosen, {f: edges[f] for f in chosen}, {f: woe[f] for f in chosen},
                           lr.coef_[0], lr.intercept_[0], sc["base_score"], sc["base_odds"], sc["pdo"])
    return model, iv_tbl
