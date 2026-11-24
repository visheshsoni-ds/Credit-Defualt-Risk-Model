"""Pieces shared by both training scripts (step 1)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import mlflow
import pandas as pd

from .data import dataset_fingerprint
from .explain import plot_calibration
from .metrics import calibration_table, evaluate


def log_run_context(cfg: dict, df: pd.DataFrame, cols: list[str], splits: dict) -> None:
    d = cfg["data"]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        commit = "unknown"
    mlflow.set_tags({
        "data_fingerprint": dataset_fingerprint(df, cols),
        "git_commit": commit,
        "split_mode": d["split"]["mode"],
        "n_features": str(len(cols)),
        "dataset": "amex-default-prediction",
    })
    mlflow.log_params({f"split_{k}": v for k, v in d["split"].items()})
    mlflow.log_params({f"n_{k}": len(v) for k, v in splits.items()})
    mlflow.log_dict({"features": cols}, "inputs/feature_list.json")


def log_split_metrics(y_train, p_train, y_valid, p_valid, tmp: Path) -> dict:
    """Train + validation metrics. The TEST split is deliberately not touched here."""
    m = {**evaluate(y_train, p_train, "train_"), **evaluate(y_valid, p_valid, "valid_")}
    mlflow.log_metrics(m)
    cal = calibration_table(y_valid, p_valid)
    cal.to_csv(tmp / "calibration_valid.csv", index=False)
    plot_calibration(cal, tmp / "calibration_valid.png")
    mlflow.log_artifacts(str(tmp), "validation_split")
    return m


def save_reference(valid_df: pd.DataFrame, feats: list[str], n: int, tmp: Path, pd_valid) -> None:
    """Drift baseline (PSI) for monitoring: a sample of the model's key features and its PDs on the validation split."""
    ref = valid_df[feats].copy()
    ref["_pd"] = pd_valid
    ref = ref.sample(min(n, len(valid_df)), random_state=0)
    p = tmp / "reference.parquet"
    ref.to_parquet(p)
    mlflow.log_artifact(str(p), "monitoring")
