"""Step 2 - registry helpers: aliases (challenger/champion), tags, safe promotion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

CHAMPION, CHALLENGER = "champion", "challenger"


def client() -> MlflowClient:
    return MlflowClient()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_by_alias(name: str, alias: str):
    try:
        return client().get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def version_for_run(name: str, run_id: str):
    mvs = [m for m in client().search_model_versions(f"name='{name}'") if m.run_id == run_id]
    if not mvs:
        raise LookupError(f"No registered version of {name} for run {run_id}")
    return max(mvs, key=lambda m: int(m.version))


def tag(name: str, version, **tags) -> None:
    for k, v in tags.items():
        client().set_model_version_tag(name, str(version), k, str(v))


def register_challenger(name: str, run_id: str, model_type: str, intended_use: str, training_metrics: dict) -> int:
    """Called by training scripts right after log_model. Version starts as validation_status=pending."""
    mv = version_for_run(name, run_id)
    tag(name, mv.version, model_type=model_type, validation_status="pending", intended_use=intended_use,
        training_run_id=run_id, registered_at=now(), **{f"train_{k}": f"{v:.4f}" for k, v in training_metrics.items()})
    client().set_registered_model_alias(name, CHALLENGER, mv.version)
    return int(mv.version)


def promote_to_champion(name: str, version, approved_by: str = "validation_gate") -> None:
    """Only the validation gate should call this."""
    c = client()
    old = get_by_alias(name, CHAMPION)
    if old is not None and str(old.version) != str(version):
        tag(name, old.version, status="retired", retired_at=now())
    c.set_registered_model_alias(name, CHAMPION, str(version))
    chal = get_by_alias(name, CHALLENGER)
    if chal is not None and str(chal.version) == str(version):
        c.delete_registered_model_alias(name, CHALLENGER)
    tag(name, version, status="champion", approved_by=approved_by, promoted_at=now())


def _show(name: str) -> None:
    amap = {}
    for a in (CHAMPION, CHALLENGER):
        mv = get_by_alias(name, a)
        if mv is not None:
            amap.setdefault(int(mv.version), []).append(a)
    for m in sorted(client().search_model_versions(f"name='{name}'"), key=lambda m: int(m.version)):
        t = m.tags
        print(f"v{m.version:>3} aliases={amap.get(int(m.version), [])!s:<16} type={t.get('model_type','?'):<10} "
              f"validation={t.get('validation_status','?'):<8} status={t.get('status','-')}")


if __name__ == "__main__":  # python -m crgov.registry   (list versions)
    from .config import init_mlflow, load_config
    cfg = load_config(); init_mlflow(cfg)
    argparse.ArgumentParser().parse_args()
    _show(cfg["mlflow"]["model_name"])
