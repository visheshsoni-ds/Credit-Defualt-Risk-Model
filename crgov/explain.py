"""Step 4 - explainability: global importance, per-customer reason codes, readable text."""
from __future__ import annotations

import argparse
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .config import ROOT

_SUFFIXES = ("last", "mean", "min", "max", "std", "nunique")
_AGG_TEXT = {"last": "latest value", "mean": "average", "min": "lowest value", "max": "highest value",
             "std": "volatility", "nunique": "number of distinct values"}


@lru_cache(maxsize=1)
def _reason_map() -> dict:
    p = ROOT / "configs" / "reason_text.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def humanise(feature: str) -> str:
    """Feature name -> adverse-action style reason text."""
    m = _reason_map()
    base, _, suffix = feature.rpartition("_")
    if suffix in _SUFFIXES and base:
        label = m.get(feature) or m.get(base)
        agg = _AGG_TEXT[suffix]
        return f"{label} ({agg})" if label else f"Elevated risk indicated by {base} ({agg})"
    return m.get(feature) or f"Elevated risk indicated by {feature}"


def base_name(feature: str) -> str:
    base, _, suffix = feature.rpartition("_")
    return base if suffix in _SUFFIXES and base else feature


def top_k_reasons(contrib: np.ndarray, names: list[str], top_n: int) -> list[list[dict]]:
    """contrib: (n_rows, n_features) risk-increasing contributions (positive = pushes PD up).
    One reason per underlying variable: P_1_mean and P_1_max must not both appear in an adverse-action list."""
    out = []
    for row in contrib:
        picked, seen = [], set()
        for i in np.argsort(-row):
            if row[i] <= 0 or len(picked) == top_n:
                break
            b = base_name(names[i])
            if b in seen:
                continue
            seen.add(b)
            picked.append({"feature": names[i], "reason": humanise(names[i]), "impact": float(row[i])})
        out.append(picked)
    return out


def lgbm_contributions(booster, X: pd.DataFrame) -> np.ndarray:
    """Per-feature log-odds contributions (TreeSHAP built into LightGBM); bias column dropped."""
    return booster.predict(X[booster.feature_name()].astype(float), pred_contrib=True)[:, :-1]


def global_importance(booster, X: pd.DataFrame, max_rows: int = 5000) -> pd.DataFrame:
    Xs = X.sample(min(len(X), max_rows), random_state=0)
    c = lgbm_contributions(booster, Xs)
    return (pd.DataFrame({"feature": booster.feature_name(), "mean_abs_shap": np.abs(c).mean(axis=0)})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def plot_importance(imp: pd.DataFrame, path, top: int = 20) -> None:
    t = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(t["feature"], t["mean_abs_shap"])
    ax.set_xlabel("mean |SHAP| (log-odds)")
    ax.set_title(f"Top {top} features")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_calibration(table: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(table["mean_pd"], table["observed"], "o-", label="deciles")
    lim = max(table["mean_pd"].max(), table["observed"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="grey", label="perfect")
    ax.set_xlabel("mean predicted PD"); ax.set_ylabel("observed default rate"); ax.legend()
    ax.set_title("Calibration by decile")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":  # python -m crgov.explain --n 5  -> reason codes for test customers from the champion
    from .config import init_mlflow, load_config
    from .data import load_features, make_splits
    from .scoring import load_scorer
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=5); ap.add_argument("--alias", default="champion")
    a = ap.parse_args()
    cfg = load_config(); init_mlflow(cfg)
    scorer = load_scorer(cfg, alias=a.alias)
    test = make_splits(load_features(cfg), cfg)["test"].head(a.n)
    pdv = scorer.predict_pd(test); rs = scorer.reasons(test, cfg["serving"]["top_reasons"])
    for cid, p, r in zip(test[cfg["data"]["id_col"]], pdv, rs):
        print(f"\n{cid[:12]}…  PD={p:.3f}")
        for x in r:
            print(f"   - {x['reason']}")
