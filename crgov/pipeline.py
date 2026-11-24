"""Plain-Python building blocks used by the Airflow DAG (kept out of the DAG file so they're testable)."""
from __future__ import annotations

from pathlib import Path

from .config import load_config, path


def find_batch(ds: str | None = None, cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    inc = path(cfg, "incoming")
    if ds and (inc / f"batch_{ds}.parquet").exists():
        return str(inc / f"batch_{ds}.parquet")
    files = sorted(inc.glob("batch_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No batches in {inc}")
    return str(files[-1])


def route(decision: str) -> str:
    return {"RETRAIN": "retrain_and_validate", "WATCH": "raise_alert", "OK": "no_action"}[decision]


def retrain_and_validate(cfg: dict | None = None) -> list[dict]:
    """Retrain both families, then run each new version through the SAME validation gate."""
    from . import train_lgbm, train_scorecard
    from .validate import validate
    cfg = cfg or load_config()
    out = []
    for trainer in (train_scorecard, train_lgbm):
        r = trainer.main(cfg)
        rep = validate(cfg, version=r["version"])
        out.append({"version": r["version"], "model_type": rep["model_type"], "passed": rep["passed"]})
    return out
