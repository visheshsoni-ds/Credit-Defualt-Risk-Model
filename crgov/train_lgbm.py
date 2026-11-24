"""Step 1 - LightGBM challenger with full tracking. Run: python -m crgov.train_lgbm"""
from __future__ import annotations

import tempfile
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm

from . import registry
from .config import init_mlflow, load_config
from .data import feature_columns, load_features, make_splits
from .explain import global_importance, plot_importance
from .train_common import log_run_context, log_split_metrics, save_reference


def main(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    init_mlflow(cfg)
    y_col = cfg["data"]["target_col"]
    df = load_features(cfg)
    cols = feature_columns(df, cfg)
    sp = make_splits(df, cfg)
    L = cfg["lightgbm"]

    dtr = lgb.Dataset(sp["train"][cols], sp["train"][y_col], free_raw_data=False)
    dva = lgb.Dataset(sp["valid"][cols], sp["valid"][y_col], reference=dtr, free_raw_data=False)

    with mlflow.start_run(run_name="lightgbm") as run, tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        mlflow.log_params({**L["params"], "num_boost_round": L["num_boost_round"],
                           "early_stopping_rounds": L["early_stopping_rounds"]})
        log_run_context(cfg, df, cols, sp)
        mlflow.set_tag("model_type", "lightgbm")

        booster = lgb.train(L["params"], dtr, L["num_boost_round"], valid_sets=[dva], valid_names=["valid"],
                            callbacks=[lgb.early_stopping(L["early_stopping_rounds"], verbose=False)])
        mlflow.log_metric("best_iteration", booster.best_iteration)

        p_train, p_valid = booster.predict(sp["train"][cols]), booster.predict(sp["valid"][cols])
        m = log_split_metrics(sp["train"][y_col], p_train, sp["valid"][y_col], p_valid, tmp)

        imp = global_importance(booster, sp["valid"][cols])
        imp.to_csv(tmp / "importance.csv", index=False)
        plot_importance(imp, tmp / "importance.png")
        mlflow.log_artifact(str(tmp / "importance.csv"), "explain")
        mlflow.log_artifact(str(tmp / "importance.png"), "explain")

        top = imp["feature"].head(cfg["monitoring"]["reference_features"]).tolist()
        save_reference(sp["valid"], top, cfg["monitoring"]["reference_rows"], tmp, p_valid)

        mlflow.lightgbm.log_model(booster, name="model", registered_model_name=cfg["mlflow"]["model_name"])

        version = registry.register_challenger(
            cfg["mlflow"]["model_name"], run.info.run_id, "lightgbm",
            intended_use="Application/behavioural PD scoring (portfolio demo, Amex data)",
            training_metrics={"auc": m["valid_auc"], "ks": m["valid_ks"], "amex_metric": m["valid_amex_metric"]})
    print(f"LightGBM registered as v{version} (challenger). valid AUC={m['valid_auc']:.4f} "
          f"KS={m['valid_ks']:.4f} amex={m['valid_amex_metric']:.4f}")
    return {"version": version, "run_id": run.info.run_id, **m}


if __name__ == "__main__":
    main()
