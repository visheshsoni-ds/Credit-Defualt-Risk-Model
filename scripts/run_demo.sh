#!/usr/bin/env bash
# Full demo on synthetic data (about 2 minutes). Assumes: pip install -r requirements.txt and the MLflow server is running.
set -euo pipefail
export PYTHONPATH="$PWD"
python scripts/make_synthetic_data.py --customers 8000
python -c "from crgov.data import build_features; build_features()"
python -m crgov.train_scorecard          # step 1
python -m crgov.validate || true         # step 3 -> champion
python -m crgov.train_lgbm
python -m crgov.validate || true         # challenger must beat the champion
python -m crgov.registry                 # step 2: see aliases
python -m crgov.explain --n 3            # step 4
python scripts/simulate_batches.py --days 6 --drift-from 4
python -m crgov.monitor --batch data/incoming/batch_2026-01-05.parquet   # step 5
python -c "from crgov.governance import build_model_card; print(build_model_card())"   # step 6
echo "Now: uvicorn crgov.serve:app --port 8000   (step 7)"
