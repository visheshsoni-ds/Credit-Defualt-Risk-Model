"""Credit-risk discrimination + calibration metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def ks_stat(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def gini(y, p) -> float:
    return 2 * roc_auc_score(y, p) - 1


def amex_metric(y_true, y_pred) -> float:
    """Kaggle Amex Default Prediction metric: mean of normalised weighted Gini and top-4% capture."""
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)

    def top4(y, p):
        o = np.argsort(-p, kind="stable"); ys = y[o]
        w = np.where(ys == 0, 20, 1); cum = np.cumsum(w)
        return ys[cum <= 0.04 * w.sum()].sum() / ys.sum()

    def wgini(y, p):
        o = np.argsort(-p, kind="stable"); ys = y[o]
        w = np.where(ys == 0, 20, 1)
        w_rand = np.cumsum(w / w.sum())
        lorentz = np.cumsum(ys * w) / (ys * w).sum()
        return ((lorentz - w_rand) * w).sum()

    return float(0.5 * (wgini(y, p) / wgini(y, y) + top4(y, p)))


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["decile"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    t = df.groupby("decile").agg(n=("y", "size"), mean_pd=("p", "mean"), observed=("y", "mean")).reset_index()
    t["gap"] = (t["mean_pd"] - t["observed"]).abs()
    t["se"] = np.sqrt(t["mean_pd"].clip(1e-6, 1 - 1e-6) * (1 - t["mean_pd"].clip(1e-6, 1 - 1e-6)) / t["n"])
    t["excess_gap"] = (t["gap"] - 3 * t["se"]).clip(lower=0)   # miscalibration beyond 3 standard errors of sampling noise
    return t


def evaluate(y, p, prefix: str = "") -> dict[str, float]:
    cal = calibration_table(y, p)
    m = {
        "auc": float(roc_auc_score(y, p)),
        "ks": ks_stat(y, p),
        "gini": gini(y, p),
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
        "amex_metric": amex_metric(y, p),
        "calibration_max_gap": float(cal["gap"].max()),            # raw, reported only (noisy on small deciles)
        "calibration_excess_gap": float(cal["excess_gap"].max()),  # noise-adjusted, used by the gate
        "portfolio_pd_error": float(abs(np.mean(p) - np.mean(y))),
        "mean_pd": float(np.mean(p)),
        "observed_rate": float(np.mean(y)),
    }
    return {f"{prefix}{k}": v for k, v in m.items()}
