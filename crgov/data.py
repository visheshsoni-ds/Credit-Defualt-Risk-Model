"""Feature building (statement-level -> one row per customer) and train/valid/test splits."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split

from .config import load_config, path


def _scan(p):
    return pl.scan_parquet(p) if str(p).endswith(".parquet") else pl.scan_csv(p, infer_schema_length=10000)


def build_features(cfg: dict | None = None) -> None:
    """Aggregate statement history to customer level: last/mean/min/max/std for numerics,
    last/nunique for (label-encoded) categoricals. Writes data/processed/features.parquet."""
    cfg = cfg or load_config()
    d = cfg["data"]
    id_col, date_col, target = d["id_col"], d["date_col"], d["target_col"]

    lf = _scan(path(cfg, "raw_data"))
    cols = lf.collect_schema().names()
    cats = [c for c in d["categorical"] if c in cols]
    nums = [c for c in cols if c not in cats + [id_col, date_col]]

    # label-encode categoricals with a fixed, saved mapping
    encoders = {}
    exprs = [pl.col(date_col).cast(pl.Date, strict=False)]
    for c in cats:
        uniq = sorted(x for x in lf.select(pl.col(c).cast(pl.Utf8).unique()).collect()[c].to_list() if x is not None)
        encoders[c] = {v: i for i, v in enumerate(uniq)}
        exprs.append(pl.col(c).cast(pl.Utf8).replace_strict(encoders[c], default=None).cast(pl.Float32).alias(c))
    exprs += [pl.col(c).cast(pl.Float32) for c in nums]
    lf = lf.with_columns(exprs).sort([id_col, date_col])

    aggs = [pl.len().alias("n_statements"), pl.col(date_col).max().alias(f"{date_col}_last")]
    for c in nums:
        col = pl.col(c)
        aggs += [col.last().alias(f"{c}_last"), col.mean().alias(f"{c}_mean"),
                 col.min().alias(f"{c}_min"), col.max().alias(f"{c}_max"), col.std().alias(f"{c}_std")]
    for c in cats:
        aggs += [pl.col(c).last().alias(f"{c}_last"), pl.col(c).n_unique().cast(pl.Float32).alias(f"{c}_nunique")]

    feats = lf.group_by(id_col).agg(aggs)
    lp = path(cfg, "raw_labels")
    labels = (pl.read_parquet(lp) if str(lp).endswith(".parquet") else pl.read_csv(lp)).select([id_col, target])
    out = feats.join(labels.lazy(), on=id_col, how="inner")
    out = out.collect(engine="streaming")

    dest = path(cfg, "features")
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(dest)
    (dest.parent / "encoders.json").write_text(json.dumps(encoders, indent=2))
    print(f"features: {out.shape} -> {dest}  | default rate {out[target].mean():.4f}")


def non_feature_cols(cfg: dict) -> list[str]:
    d = cfg["data"]
    return [d["id_col"], d["target_col"], f"{d['date_col']}_last"]


def load_features(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_parquet(path(cfg, "features"))


def feature_columns(df: pd.DataFrame, cfg: dict) -> list[str]:
    drop = set(non_feature_cols(cfg))
    return [c for c in df.columns if c not in drop]


def make_splits(df: pd.DataFrame, cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    """train / valid / test. The test split is never used for fitting or early stopping:
    the validation gate (step 3) is the only consumer."""
    cfg = cfg or load_config()
    s, d = cfg["data"]["split"], cfg["data"]
    y = d["target_col"]
    if s["mode"] == "time":
        df = df.sort_values(f"{d['date_col']}_last").reset_index(drop=True)
        n = len(df)
        n_test, n_valid = int(n * s["test_size"]), int(n * s["valid_size"])
        return {"train": df.iloc[: n - n_test - n_valid], "valid": df.iloc[n - n_test - n_valid : n - n_test],
                "test": df.iloc[n - n_test :]}
    rest, test = train_test_split(df, test_size=s["test_size"], stratify=df[y], random_state=s["seed"])
    train, valid = train_test_split(rest, test_size=s["valid_size"] / (1 - s["test_size"]),
                                    stratify=rest[y], random_state=s["seed"])
    return {"train": train, "valid": valid, "test": test}


def dataset_fingerprint(df: pd.DataFrame, cols: list[str]) -> str:
    """Cheap, stable data version tag for MLflow."""
    import hashlib
    h = hashlib.sha256()
    h.update(str((df.shape, list(cols))).encode())
    h.update(pd.util.hash_pandas_object(df[[c for c in df.columns[:3]]], index=False).values.tobytes())
    return h.hexdigest()[:16]
