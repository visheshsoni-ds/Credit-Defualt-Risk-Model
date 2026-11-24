"""Step 5 - drift (PSI/CSI), score drift, back-test, and the retrain decision.
Run on one batch: python -m crgov.monitor --batch data/incoming/batch_2026-01-01.parquet"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from .config import init_mlflow, load_config
from .metrics import evaluate
from .scoring import load_scorer


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. Bins = deciles of the baseline, plus a MISSING bin."""
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    ev, av = e[~np.isnan(e)], a[~np.isnan(a)]
    if ev.size == 0 or np.unique(ev).size < 2:
        return 0.0
    edges = np.unique(np.quantile(ev, np.linspace(0, 1, bins + 1)[1:-1]))
    def share(v, n_total, n_nan):
        c = np.bincount(np.searchsorted(edges, v, side="right"), minlength=len(edges) + 1).astype(float)
        c = np.append(c, n_nan)
        return np.clip(c / n_total, 1e-4, None)
    pe = share(ev, len(e), len(e) - len(ev))
    pa = share(av, max(len(a), 1), len(a) - len(av))
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_status(v: float, cfg: dict) -> str:
    m = cfg["monitoring"]
    return "red" if v > m["psi_retrain"] else "amber" if v > m["psi_watch"] else "green"


def feature_drift(ref: pd.DataFrame, cur: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = [{"feature": f, "psi": psi(ref[f].to_numpy(), cur[f].to_numpy()) if f in cur else np.nan}
            for f in ref.columns]
    t = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    t["status"] = t["psi"].apply(lambda v: "missing" if np.isnan(v) else psi_status(v, cfg))
    return t


def decide(summary: dict, cfg: dict) -> tuple[str, list[str]]:
    m, why = cfg["monitoring"], []
    if summary["score_psi"] > m["psi_retrain"]:
        why.append(f"score PSI {summary['score_psi']:.3f} > {m['psi_retrain']}")
    if summary["n_red_features"] >= m["max_red_features"]:
        why.append(f"{summary['n_red_features']} features with PSI > {m['psi_retrain']}")
    if summary.get("auc") is not None and summary["auc"] < m["min_auc_floor"]:
        why.append(f"back-test AUC {summary['auc']:.3f} < floor {m['min_auc_floor']}")
    if why:
        return "RETRAIN", why
    if summary["score_psi"] > m["psi_watch"] or summary["n_amber_features"] >= m["watch_amber_features"]:
        return "WATCH", ["moderate drift on score or features"]
    return "OK", []


def run_monitoring(batch_path: str | Path, cfg: dict | None = None, batch_date: str | None = None) -> dict:
    """Scores a batch with the CHAMPION, compares to the training baseline, logs to MLflow, returns a summary."""
    cfg = cfg or load_config()
    init_mlflow(cfg, cfg["mlflow"]["monitoring_experiment"])
    batch = pd.read_parquet(batch_path)
    batch_date = batch_date or Path(batch_path).stem.replace("batch_", "")
    y_col = cfg["data"]["target_col"]

    scorer = load_scorer(cfg, alias="champion")
    local = mlflow.artifacts.download_artifacts(run_id=scorer.run_id, artifact_path="monitoring/reference.parquet")
    ref = pd.read_parquet(local)

    pd_ref = ref.pop("_pd").to_numpy()            # PDs the model produced on the baseline sample
    pd_cur = scorer.predict_pd(batch)
    drift = feature_drift(ref, batch, cfg)
    summary = {
        "batch_date": batch_date, "n_rows": int(len(batch)), "model_version": scorer.version,
        "model_type": scorer.model_type, "mean_pd": float(pd_cur.mean()),
        "score_psi": psi(pd_ref, pd_cur), "max_feature_psi": float(drift["psi"].max()),
        "n_red_features": int((drift["status"] == "red").sum()),
        "n_amber_features": int((drift["status"] == "amber").sum()), "auc": None,
    }
    if y_col in batch and batch[y_col].notna().any() and batch[y_col].nunique() > 1:
        bt = evaluate(batch[y_col], pd_cur)
        summary.update({"auc": bt["auc"], "ks": bt["ks"], "observed_rate": bt["observed_rate"],
                        "calibration_excess_gap": bt["calibration_excess_gap"]})
    summary["decision"], summary["reasons"] = decide(summary, cfg)

    with mlflow.start_run(run_name=f"monitor-{batch_date}"), tempfile.TemporaryDirectory() as td:
        mlflow.set_tags({"batch_date": batch_date, "decision": summary["decision"],
                         "model_version": str(scorer.version)})
        mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float)) and v is not None})
        drift.to_csv(Path(td) / "drift_table.csv", index=False)
        mlflow.log_artifact(str(Path(td) / "drift_table.csv"))
        mlflow.log_dict(summary, "summary.json")
    summary["drift_table"] = drift
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--batch", required=True)
    s = run_monitoring(ap.parse_args().batch)
    print({k: v for k, v in s.items() if k != "drift_table"})
    print(s["drift_table"].head(10).to_string(index=False))
