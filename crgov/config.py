"""Config + MLflow connection helpers shared by every step."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import mlflow
import yaml

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    p = Path(path or os.environ.get("CRGOV_CONFIG", ROOT / "configs" / "config.yaml"))
    cfg = yaml.safe_load(p.read_text())
    # allow env override of the tracking URI (docker, CI)
    cfg["mlflow"]["tracking_uri"] = os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
    return cfg


def path(cfg: dict, key: str) -> Path:
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p


def init_mlflow(cfg: dict, experiment: str | None = None) -> None:
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(experiment or cfg["mlflow"]["experiment"])
