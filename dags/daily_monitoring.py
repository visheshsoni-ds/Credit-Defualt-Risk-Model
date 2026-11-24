"""Step 5 - daily Airflow DAG: score batch with champion -> drift/back-test -> branch -> report (+ LLM commentary).
Put the repo root on PYTHONPATH for the Airflow scheduler/worker (see README)."""
from __future__ import annotations

from datetime import datetime

import pandas as pd

try:                                         # Airflow 3
    from airflow.sdk import dag, task, get_current_context
except ImportError:                          # Airflow 2.x
    from airflow.decorators import dag, task
    from airflow.operators.python import get_current_context


@dag(dag_id="credit_risk_daily_monitoring", schedule="@daily", start_date=datetime(2026, 1, 1),
     catchup=False, tags=["credit-risk", "monitoring"])
def credit_risk_daily_monitoring():
    @task
    def find_batch() -> str:
        from crgov.pipeline import find_batch as _find
        ds = get_current_context().get("ds")
        return _find(ds)

    @task
    def monitor(batch_path: str) -> dict:
        from crgov.config import load_config, path
        from crgov.monitor import run_monitoring
        cfg = load_config()
        s = run_monitoring(batch_path, cfg)
        drift = s.pop("drift_table")
        csv = path(cfg, "reports") / f"drift_{s['batch_date']}.csv"
        csv.parent.mkdir(parents=True, exist_ok=True)
        drift.to_csv(csv, index=False)
        s["drift_csv"] = str(csv)
        return s

    @task.branch
    def route(summary: dict) -> str:
        from crgov.pipeline import route as _route
        return _route(summary["decision"])

    @task
    def retrain_and_validate() -> list:
        from crgov.pipeline import retrain_and_validate as _rv
        return _rv()

    @task
    def raise_alert(summary: dict) -> None:
        # Replace with Slack/email. Kept as a log line so the demo has no external dependency.
        print(f"ALERT: moderate drift on {summary['batch_date']}: {summary['reasons']}")

    @task
    def no_action() -> None:
        print("No action required.")

    @task(trigger_rule="none_failed_min_one_success")
    def report(summary: dict) -> str:
        from crgov.governance import daily_report
        return str(daily_report(summary, pd.read_csv(summary["drift_csv"])))

    s = monitor(find_batch())
    branch = route(s)
    actions = [retrain_and_validate(), raise_alert(s), no_action()]
    branch >> actions
    rep = report(s)
    actions >> rep


credit_risk_daily_monitoring()
