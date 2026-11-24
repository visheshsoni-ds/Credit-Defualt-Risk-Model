"""Step 3 - automated validation gate. The ONLY code path that promotes a model to champion.
Run: python -m crgov.validate [--version N]   (exit code 0 = passed, 1 = failed)"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import mlflow

from . import registry
from .config import init_mlflow, load_config
from .data import load_features, make_splits
from .explain import plot_calibration
from .metrics import calibration_table, evaluate
from .scoring import load_scorer


def evaluate_gate(cand: dict, champ: dict | None, train_auc: float, g: dict) -> dict:
    """Pure function (unit-tested). cand/champ are metric dicts on the SAME held-out test split."""
    checks = [
        ("min_auc", cand["auc"], f">= {g['min_auc']}", cand["auc"] >= g["min_auc"]),
        ("min_ks", cand["ks"], f">= {g['min_ks']}", cand["ks"] >= g["min_ks"]),
        ("calibration_excess_gap", cand["calibration_excess_gap"], f"<= {g['max_calibration_excess_gap']}",
         cand["calibration_excess_gap"] <= g["max_calibration_excess_gap"]),
        ("portfolio_pd_error", cand["portfolio_pd_error"], f"<= {g['max_portfolio_pd_error']}",
         cand["portfolio_pd_error"] <= g["max_portfolio_pd_error"]),
        ("generalisation_gap", train_auc - cand["auc"], f"<= {g['max_generalisation_gap']}",
         (train_auc - cand["auc"]) <= g["max_generalisation_gap"]),
    ]
    if champ is not None:
        uplift = cand["auc"] - champ["auc"]
        checks.append(("uplift_vs_champion", uplift, f">= {g['min_uplift_vs_champion']}",
                       uplift >= g["min_uplift_vs_champion"]))
    rows = [{"check": n, "value": float(v), "threshold": t, "passed": bool(p)} for n, v, t, p in checks]
    return {"checks": rows, "passed": all(r["passed"] for r in rows), "bootstrap": champ is None}


def validate(cfg: dict | None = None, version: int | None = None) -> dict:
    cfg = cfg or load_config()
    init_mlflow(cfg)
    name = cfg["mlflow"]["model_name"]
    if version is None:
        mv = registry.get_by_alias(name, registry.CHALLENGER)
        if mv is None:
            raise SystemExit("No challenger alias set - train a model first.")
        version = int(mv.version)
    cand_scorer = load_scorer(cfg, version=version)
    champ_mv = registry.get_by_alias(name, registry.CHAMPION)
    if champ_mv is not None and int(champ_mv.version) == version:
        raise SystemExit(f"v{version} is already the champion.")
    champ_scorer = load_scorer(cfg, version=int(champ_mv.version)) if champ_mv is not None else None

    y = cfg["data"]["target_col"]
    test = make_splits(load_features(cfg), cfg)["test"]
    cand = evaluate(test[y], cand_scorer.predict_pd(test))
    champ = evaluate(test[y], champ_scorer.predict_pd(test)) if champ_scorer else None

    run = mlflow.get_run(cand_scorer.run_id)
    train_auc = run.data.metrics["train_auc"]
    result = evaluate_gate(cand, champ, train_auc, cfg["gate"])
    report = {"model": name, "version": version, "model_type": cand_scorer.model_type,
              "champion_version": int(champ_mv.version) if champ_mv else None, "test_rows": len(test),
              "candidate_test_metrics": cand, "champion_test_metrics": champ,
              "train_auc": train_auc, "thresholds": cfg["gate"], **result, "validated_at": registry.now()}

    cal = calibration_table(test[y], cand_scorer.predict_pd(test))
    with mlflow.start_run(run_id=cand_scorer.run_id), tempfile.TemporaryDirectory() as td:  # attach to the model's own run
        mlflow.log_dict(report, "validation/validation_report.json")
        mlflow.log_metrics({f"test_{k}": v for k, v in cand.items()})
        cal.to_csv(Path(td) / "calibration_test.csv", index=False)
        plot_calibration(cal, Path(td) / "calibration_test.png")
        mlflow.log_artifacts(td, "validation")

    registry.tag(name, version, validation_status="passed" if result["passed"] else "failed",
                 validated_at=report["validated_at"], test_auc=f"{cand['auc']:.4f}", test_ks=f"{cand['ks']:.4f}")
    if result["passed"]:
        registry.promote_to_champion(name, version)
    return report


def _print(report: dict) -> None:
    print(f"\nValidation of {report['model']} v{report['version']} ({report['model_type']}) "
          f"on {report['test_rows']} held-out rows"
          + (" [bootstrap: no champion yet]" if report["bootstrap"] else f" vs champion v{report['champion_version']}"))
    for c in report["checks"]:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']:<22} {c['value']:>9.4f}  (need {c['threshold']})")
    print("=> PROMOTED to champion" if report["passed"] else "=> NOT promoted (stays challenger)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--version", type=int)
    rep = validate(version=ap.parse_args().version)
    _print(rep)
    sys.exit(0 if rep["passed"] else 1)
