"""Generate Amex-shaped synthetic data so you can run the whole pipeline before touching the 16 GB Kaggle files.
   python scripts/make_synthetic_data.py --customers 8000"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    num_cols = [f"{p}_{i}" for p, k in [("P", 3), ("D", 20), ("B", 12), ("S", 8), ("R", 8)] for i in range(1, k + 1)]
    num_cols = [c for c in num_cols if c != "S_2"]   # S_2 is the statement-date column
    cat_cols = {"B_30": [0, 1, 2], "B_38": [1, 2, 3, 4, 5], "D_63": ["CR", "CO", "CL", "XZ", "XM", "XL"],
                "D_64": ["O", "-1", "R", "U"]}
    latent = rng.normal(size=n)                                   # unobserved creditworthiness
    n_stmt = rng.integers(3, 14, size=n)
    w = rng.normal(size=len(num_cols)) * 0.5
    w[:8] = [1.4, -1.0, 0.9, 0.8, -0.7, 0.6, 0.5, -0.5]           # a few genuinely predictive features
    rows, ids = [], [f"{i:016x}" for i in rng.integers(0, 2**62, size=n)]
    end = pd.Timestamp("2018-03-15")
    for i in range(n):
        k = n_stmt[i]
        base = latent[i] * w + rng.normal(size=len(num_cols)) * 0.7
        X = base + rng.normal(size=(k, len(num_cols))) * 0.4 + np.linspace(0, 0.3 * latent[i], k)[:, None]
        X[rng.random(X.shape) < 0.08] = np.nan
        d = pd.DataFrame(X, columns=num_cols)
        d.insert(0, "S_2", [(end - pd.DateOffset(months=int(k - 1 - j))).date() for j in range(k)])
        d.insert(0, "customer_ID", ids[i])
        for c, vals in cat_cols.items():
            d[c] = rng.choice(vals, size=k, p=None)
            d.loc[rng.random(k) < 0.05, c] = None
        rows.append(d)
    raw = pd.concat(rows, ignore_index=True)
    prob = 1 / (1 + np.exp(-(1.6 * latent - 1.3 + rng.normal(size=n) * 0.5)))   # ~25-30% default rate
    labels = pd.DataFrame({"customer_ID": ids, "target": (rng.random(n) < prob).astype(int)})
    (ROOT / "data/raw").mkdir(parents=True, exist_ok=True)
    raw.to_parquet(ROOT / "data/raw/train_data.parquet")
    labels.to_csv(ROOT / "data/raw/train_labels.csv", index=False)
    print(f"raw {raw.shape}, customers {n}, default rate {labels.target.mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--customers", type=int, default=8000)
    main(ap.parse_args().customers)
