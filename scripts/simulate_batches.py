"""Create daily monitoring batches from the held-out TEST split (never used to train).
Batches from --drift-from onward get a shift on the key features so you can watch the DAG flip to RETRAIN.
   python scripts/simulate_batches.py --days 6 --drift-from 4"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crgov.config import load_config, path  # noqa: E402
from crgov.data import feature_columns, load_features, make_splits  # noqa: E402


def main(days: int, drift_from: int, shift: float, prefixes: list[str], start: str):
    cfg = load_config()
    test = make_splits(load_features(cfg), cfg)["test"].sample(frac=1, random_state=1).reset_index(drop=True)
    cols = [c for c in feature_columns(test, cfg) if c.split("_")[0] in prefixes]
    out = path(cfg, "incoming"); out.mkdir(parents=True, exist_ok=True)
    dates = [d.date().isoformat() for d in __import__("pandas").date_range(start, periods=days)]
    for k, (idx, ds) in enumerate(zip(np.array_split(np.arange(len(test)), days), dates), start=1):
        chunk = test.iloc[idx].copy()
        if k >= drift_from:
            for c in cols:
                chunk[c] = chunk[c] + shift * test[c].std()
        chunk.to_parquet(out / f"batch_{ds}.parquet")
        print(f"batch_{ds}.parquet  rows={len(chunk)}  {'DRIFTED' if k >= drift_from else 'clean'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=6); ap.add_argument("--drift-from", type=int, default=99)
    ap.add_argument("--shift", type=float, default=1.0); ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--prefixes", nargs="+", default=["P", "D", "B"])
    a = ap.parse_args()
    main(a.days, a.drift_from, a.shift, a.prefixes, a.start)
