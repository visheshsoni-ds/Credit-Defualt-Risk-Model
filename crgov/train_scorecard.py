"""Step 1 - WoE + logistic-regression scorecard (interpretable benchmark). Run: python -m crgov.train_scorecard"""
from __future__ import annotations

import tempfile
from pathlib import Path

import mlflow
import mlflow.pyfunc

from . import registry
from .config import ROOT, init_mlflow, load_config
from .data import feature_columns, load_features, make_splits
from .scorecard import fit_scorecard
from .train_common import log_run_context, log_split_metrics, save_reference


def main(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    init_mlflow(cfg)
    y_col = cfg["data"]["target_col"]
    df = load_features(cfg)
    cols = feature_columns(df, cfg)
    sp = make_splits(df, cfg)
    sc = cfg["scorecard"]

    with mlflow.start_run(run_name="scorecard") as run, tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        mlflow.log_params({f"sc_{k}": v for k, v in sc.items()})
        log_run_context(cfg, df, cols, sp)
        mlflow.set_tag("model_type", "scorecard")

        model, iv = fit_scorecard(sp["train"][cols], sp["train"][y_col].to_numpy(), sc)
        mlflow.log_metric("n_scorecard_features", len(model.features))

        p_train, p_valid = model.predict_pd(sp["train"]), model.predict_pd(sp["valid"])
        m = log_split_metrics(sp["train"][y_col], p_train, sp["valid"][y_col], p_valid, tmp)

        model.points_table().to_csv(tmp / "scorecard_points.csv", index=False)
        iv.to_csv(tmp / "information_value.csv")
        mlflow.log_artifact(str(tmp / "scorecard_points.csv"), "explain")
        mlflow.log_artifact(str(tmp / "information_value.csv"), "explain")
        save_reference(sp["valid"], model.features, cfg["monitoring"]["reference_rows"], tmp, p_valid)

        mlflow.pyfunc.log_model(name="model", python_model=model, code_paths=[str(ROOT / "crgov")],
                                registered_model_name=cfg["mlflow"]["model_name"],
                                pip_requirements=["mlflow", "numpy", "pandas", "scikit-learn", "pyyaml", "matplotlib"])

        version = registry.register_challenger(
            cfg["mlflow"]["model_name"], run.info.run_id, "scorecard",
            intended_use="Interpretable benchmark / candidate champion (portfolio demo, Amex data)",
            training_metrics={"auc": m["valid_auc"], "ks": m["valid_ks"], "amex_metric": m["valid_amex_metric"]})
    print(f"Scorecard registered as v{version} (challenger). {len(model.features)} features. "
          f"valid AUC={m['valid_auc']:.4f} KS={m['valid_ks']:.4f}")
    return {"version": version, "run_id": run.info.run_id, **m}


if __name__ == "__main__":
    main()
