# Credit-risk PD governance stack (MLflow + Airflow + FastAPI)

Builds on a LightGBM / MLflow credit-default project (Amex Default Prediction data) and adds the **model-risk-management
layer** that credit-risk teams expect around a model: tracked experiments, a champion/challenger registry, an automated
validation gate, reason codes, drift monitoring, a model card with governed LLM commentary, and a scoring API.

```
raw statements ─► features (customer level) ─► train: WoE scorecard + LightGBM ─► MLflow tracking
                                                                  │
                                         registry: version tagged, alias `challenger`
                                                                  │
                          validate (gate: AUC, KS, calibration, overfit, uplift vs champion)
                                       pass ─► alias `champion`    fail ─► stays challenger
                                                                  │
        ┌─────────────────────────────────────────────────────────┴───────────────┐
   FastAPI /score (champion, hot reload)                       Airflow daily DAG
   PD + score + band + reason codes                score batch → PSI/CSI + back-test → OK / WATCH / RETRAIN
                                                       → report + model card + LLM commentary (+ audit log)
```

## Step → file map
| # | Step | Code |
|---|------|------|
| 1 | Training with full tracking | `crgov/train_lgbm.py`, `crgov/train_scorecard.py`, `crgov/train_common.py`, `crgov/data.py`, `crgov/metrics.py` |
| 2 | Registry aliases + tags | `crgov/registry.py` |
| 3 | Validation gate | `crgov/validate.py` (+ `tests/test_gate.py`) |
| 4 | Explainability + reason codes | `crgov/explain.py`, `crgov/scorecard.py`, `configs/reason_text.yaml` |
| 5 | Monitoring DAG | `crgov/monitor.py`, `crgov/pipeline.py`, `dags/daily_monitoring.py` |
| 6 | Model card + LLM commentary | `crgov/governance.py` |
| 7 | Serving API | `crgov/serve.py`, `Dockerfile`, `docker-compose.yml` |

All thresholds, paths and model settings live in `configs/config.yaml`.

## Quick start (synthetic data, ~2 min)
```bash
pip install -r requirements.txt

# 1. start MLflow with a portable artifact store (HTTP proxy -> no absolute D:\ paths stored in the DB)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root mlflow-artifacts:/ \
              --artifacts-destination ./mlartifacts --serve-artifacts --host 127.0.0.1 --port 5000

# 2. in another terminal (Windows: run the commands in scripts/run_demo.sh one by one, or use Git Bash/WSL)
bash scripts/run_demo.sh
uvicorn crgov.serve:app --port 8000
pytest -q
```
Score a customer: `POST /score` with `{"customer_id": "x", "features": {"P_2_last": 0.31, "D_39_mean": 1.2, ...}}`.

## Using the real Amex data
1. Download `train_data.csv` (or a parquet conversion) and `train_labels.csv` from Kaggle into `data/raw/`
   and set `paths.raw_data` / `paths.raw_labels` in the config. The raw file is ~16 GB; `build_features` streams it with
   Polars, but convert to parquet first if you can.
2. `python -c "from crgov.data import build_features; build_features()"`
3. Continue from `python -m crgov.train_scorecard`.
Your old `mlflow.db` is not needed: this stack starts a clean registry (your old runs/paths are Windows-local).

## Airflow
```bash
pip install "apache-airflow==<version>" --constraint <official constraints URL for your Python>
export PYTHONPATH=$PWD AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags
airflow standalone
```
Drop daily batches into `data/incoming/batch_YYYY-MM-DD.parquet` (`scripts/simulate_batches.py` makes some, with optional drift).
**The DAG file is syntax-checked and its logic is unit-tested in `crgov/pipeline.py`, but it has not been run under a live Airflow.**

## LLM commentary
Set `ANTHROPIC_API_KEY` (optionally `CRGOV_LLM_MODEL`) to have Claude write the note. Without a key a deterministic template
is used. The model only receives computed numbers; any figure in its output that is not in those numbers is flagged, and
every prompt/response is appended to `reports/audit_log.jsonl`.

## Known limitations / honest notes
- **Data split:** Amex statements end in a tight date window, so the default is a customer-level stratified split (one row per
  customer after aggregation = no customer leakage). True out-of-time validation is not possible on this dataset; set
  `data.split.mode: time` on data with a real time spread.
- **Fairness:** the dataset is anonymised; the model card says disparate-impact testing cannot be done.
- **Gate calibration check** is noise-adjusted (3 standard errors per decile) because raw decile gaps on ~100 customers are noise.
- Serving expects the already-aggregated customer features, not raw statement history (no feature store).
- `docker-compose.yml` is untested. The monitoring back-test assumes labels arrive with the batch; in real life they lag.
- Synthetic data is easy and the true model is logistic, so the scorecard beats LightGBM there; on real Amex data expect the opposite.
