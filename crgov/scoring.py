"""Uniform scorer interface over both model families + score scaling + registry loading."""
from __future__ import annotations

import numpy as np
import pandas as pd

import mlflow
import mlflow.lightgbm
import mlflow.pyfunc

from . import registry
from .explain import lgbm_contributions, top_k_reasons


def pd_to_score(pd_: np.ndarray, sc: dict) -> np.ndarray:
    """Same scaling as the scorecard: base_score at base_odds, pdo points to double the odds."""
    p = np.clip(pd_, 1e-6, 1 - 1e-6)
    factor = sc["pdo"] / np.log(2)
    offset = sc["base_score"] - factor * np.log(sc["base_odds"])
    return offset + factor * np.log((1 - p) / p)


class LgbmScorer:
    model_type = "lightgbm"

    def __init__(self, booster):
        self.booster = booster
        self.features = booster.feature_name()

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(df.reindex(columns=self.features).astype(float))

    def reasons(self, df: pd.DataFrame, top_n: int = 4):
        return top_k_reasons(lgbm_contributions(self.booster, df.reindex(columns=self.features)),
                             self.features, top_n)


class ScorecardScorer:
    model_type = "scorecard"

    def __init__(self, sc_model):
        self.m = sc_model
        self.features = sc_model.features

    def predict_pd(self, df):
        return self.m.predict_pd(df)

    def reasons(self, df, top_n: int = 4):
        return self.m.reasons(df, top_n)


def load_scorer(cfg: dict, version: int | str | None = None, alias: str | None = None):
    name = cfg["mlflow"]["model_name"]
    mv = registry.get_by_alias(name, alias) if alias else registry.client().get_model_version(name, str(version))
    if mv is None:
        raise LookupError(f"No model version for alias '{alias}'" if alias else f"{name} v{version} not found")
    uri = f"models:/{name}/{mv.version}"
    mtype = mv.tags.get("model_type")
    if mtype == "lightgbm":
        s = LgbmScorer(mlflow.lightgbm.load_model(uri))
    elif mtype == "scorecard":
        s = ScorecardScorer(mlflow.pyfunc.load_model(uri).unwrap_python_model())
    else:
        raise ValueError(f"Model version {mv.version} has no/unknown 'model_type' tag: {mtype!r}")
    s.version, s.run_id = int(mv.version), mv.run_id
    return s
