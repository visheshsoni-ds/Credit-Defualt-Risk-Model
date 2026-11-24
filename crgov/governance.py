"""Step 6 - model card (built from MLflow) + LLM governance commentary (with number verification + audit log)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

import mlflow
import pandas as pd

from . import registry
from .config import init_mlflow, load_config, path


# ------------------------------------------------------------------ model card
def _md_table(rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "\n".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows)


def build_model_card(cfg: dict | None = None, version: int | None = None, alias: str = "champion") -> Path:
    cfg = cfg or load_config()
    init_mlflow(cfg)
    name = cfg["mlflow"]["model_name"]
    mv = registry.client().get_model_version(name, str(version)) if version else registry.get_by_alias(name, alias)
    if mv is None:
        raise SystemExit(f"No model version found for alias '{alias}'")
    run = mlflow.get_run(mv.run_id)
    m, t, p = run.data.metrics, mv.tags, run.data.params
    try:
        rep = mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/validation/validation_report.json")
    except Exception:
        rep = None

    perf = [{"Split": s, "AUC": f"{m.get(f'{k}auc', float('nan')):.4f}", "KS": f"{m.get(f'{k}ks', float('nan')):.4f}",
             "Gini": f"{m.get(f'{k}gini', float('nan')):.4f}", "Brier": f"{m.get(f'{k}brier', float('nan')):.4f}",
             "Amex metric": f"{m.get(f'{k}amex_metric', float('nan')):.4f}"}
            for s, k in [("Train", "train_"), ("Validation", "valid_"), ("Test (held-out, gate only)", "test_")] if f"{k}auc" in m]
    checks = ([{"Check": c["check"], "Value": f"{c['value']:.4f}", "Threshold": c["threshold"],
                "Result": "PASS" if c["passed"] else "FAIL"} for c in rep["checks"]] if rep else [])
    mon = cfg["monitoring"]
    top = ""
    try:
        f = "explain/importance.csv" if t.get("model_type") == "lightgbm" else "explain/information_value.csv"
        df = pd.read_csv(mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path=f)).head(10)
        top = df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False)
    except Exception:
        top = "_not available_"

    md = f"""# Model Card - {name} v{mv.version}

_Generated {registry.now()} from MLflow run `{mv.run_id}`._

## 1. Purpose and scope
- **Intended use:** {t.get('intended_use', 'n/a')}
- **Target:** probability of default (PD) within the Amex outcome window; score scaled to {cfg['scorecard']['base_score']} points at
  {cfg['scorecard']['base_odds']}:1 good:bad odds, {cfg['scorecard']['pdo']} points to double the odds.
- **Not intended for:** real lending decisions. Portfolio/demo model on a public, anonymised dataset.

## 2. Model details
- **Type:** {t.get('model_type')}   |   **Registry status:** {t.get('status', 'candidate')}   |   **Validation:** {t.get('validation_status', 'n/a')}
- **Approved by:** {t.get('approved_by', 'n/a')} at {t.get('promoted_at', 'n/a')}
- **Features:** {run.data.tags.get('n_features', '?')} candidate customer-level features{(f", {int(m['n_scorecard_features'])} retained in the scorecard" if 'n_scorecard_features' in m else '')}
  (last / mean / min / max / std over statement history); data fingerprint `{run.data.tags.get('data_fingerprint', '?')}`;
  git commit `{run.data.tags.get('git_commit', '?')}`.
- **Split:** {p.get('split_mode', run.data.tags.get('split_mode', '?'))}, seed {p.get('split_seed', '?')};
  rows train/valid/test = {p.get('n_train', '?')}/{p.get('n_valid', '?')}/{p.get('n_test', '?')}.

## 3. Performance
{_md_table(perf, ['Split', 'AUC', 'KS', 'Gini', 'Brier', 'Amex metric'])}

## 4. Validation gate result
{_md_table(checks, ['Check', 'Value', 'Threshold', 'Result']) if checks else '_not yet validated_'}

Calibration plots and tables: run artifacts under `validation/` and `validation_split/`.

## 5. Explainability
Per-customer reason codes are produced at scoring time (TreeSHAP contributions for LightGBM; points-lost-vs-best-bin for the
scorecard). Top drivers of this model:

{top}

## 6. Fairness and bias
The Amex dataset is anonymised and contains no protected attributes, so disparate-impact testing **cannot be performed** here.
A production model would require proxy analysis and group-level performance/approval checks before deployment.

## 7. Limitations
- Trained on one public dataset; no out-of-time validation is possible if statement dates cluster tightly.
- Calibration is checked on a random held-out split, not on a shifted population or through a stress scenario.
- Reject inference, through-the-cycle adjustment and IFRS 9 staging/LGD/EAD are out of scope.

## 8. Monitoring plan
- Daily PSI per feature and on the score; amber > {mon['psi_watch']}, red > {mon['psi_retrain']}.
- Retrain trigger: score PSI red, >= {mon['max_red_features']} red features, or back-test AUC < {mon['min_auc_floor']}.
- Every retrain goes through the same validation gate; promotion is automated only on pass.
"""
    out = path(cfg, "reports") / f"model_card_{name}_v{mv.version}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    with mlflow.start_run(run_id=mv.run_id):
        mlflow.log_artifact(str(out), "governance")
    return out


# ------------------------------------------------------------------ LLM commentary
SYSTEM = ("You are a model-risk analyst writing a short daily monitoring note for a credit PD model. "
          "Use ONLY the numbers in the JSON facts provided. Never invent, estimate or round figures into new values. "
          "Structure: 1) Headline decision, 2) What moved (score, features, back-test if present), "
          "3) Recommended action. Max 150 words. Plain text.")


def _numbers(obj) -> set[float]:
    out: set[float] = set()
    if isinstance(obj, dict):
        for v in obj.values():
            out |= _numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out |= _numbers(v)
    elif isinstance(obj, bool) or obj is None:
        pass
    elif isinstance(obj, (int, float)):
        out |= {float(obj), float(obj) * 100}
    elif isinstance(obj, str):
        out |= {float(x) for x in re.findall(r"\d+(?:\.\d+)?", obj)}
    return out


def unverified_numbers(text: str, facts: dict) -> list[str]:
    """Numbers in the commentary that don't appear in the computed facts (guard against invented figures)."""
    allowed = _numbers(facts)
    bad = []
    for tok in re.findall(r"\d+(?:\.\d+)?", text):
        x = float(tok)
        if x.is_integer() and x <= 10:
            continue
        if not any(math.isclose(x, a, rel_tol=0.01, abs_tol=0.006) for a in allowed):
            bad.append(tok)
    return bad


def _facts(summary: dict, drift: pd.DataFrame, top_n: int) -> dict:
    keep = ["batch_date", "n_rows", "model_version", "model_type", "mean_pd", "score_psi", "max_feature_psi",
            "n_red_features", "n_amber_features", "auc", "ks", "observed_rate", "calibration_excess_gap",
            "decision", "reasons"]
    f = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items() if k in keep and v is not None}
    f["top_drifted_features"] = [{"feature": r.feature, "psi": round(float(r.psi), 3), "status": r.status}
                                 for r in drift.head(top_n).itertuples()]
    return f


def _template(f: dict) -> str:
    s = (f"Decision: {f['decision']}. Champion v{f['model_version']} ({f['model_type']}) scored {f['n_rows']} accounts "
         f"on {f['batch_date']} with mean PD {f['mean_pd']}. Score PSI is {f['score_psi']}; "
         f"{f['n_red_features']} features are red and {f['n_amber_features']} amber (max feature PSI {f['max_feature_psi']}).")
    if "auc" in f:
        s += f" Back-test AUC is {f['auc']} with observed default rate {f['observed_rate']}."
    if f["top_drifted_features"]:
        t = f["top_drifted_features"][0]
        s += f" Largest mover: {t['feature']} (PSI {t['psi']})."
    action = {"RETRAIN": "Action: retrain and run the validation gate.", "WATCH": "Action: monitor closely; no retrain yet.",
              "OK": "Action: none."}[f["decision"]]
    return s + " " + action


def generate_commentary(summary: dict, drift: pd.DataFrame, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    facts = _facts(summary, drift, cfg["monitoring"]["top_n_features"] // 3 or 5)
    source, text = "template", _template(facts)
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            model = os.environ.get("CRGOV_LLM_MODEL", "claude-sonnet-5-5")
            r = anthropic.Anthropic().messages.create(
                model=model, max_tokens=500, system=SYSTEM,
                messages=[{"role": "user", "content": "Facts:\n" + json.dumps(facts, indent=2)}])
            text, source = r.content[0].text.strip(), f"llm:{model}"
        except Exception as e:  # never let commentary break the pipeline
            source = f"template (llm failed: {type(e).__name__})"
    flagged = unverified_numbers(text, facts)
    if flagged:
        text += f"\n\n[WARNING: figures not found in computed facts: {', '.join(flagged)} - verify before use]"
    rec = {"ts": registry.now(), "source": source, "model_version": summary.get("model_version"),
           "facts_sha256": hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest(),
           "facts": facts, "output": text, "unverified_numbers": flagged}
    log = path(cfg, "reports") / "audit_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def daily_report(summary: dict, drift: pd.DataFrame, cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    c = generate_commentary(summary, drift, cfg)
    n = cfg["monitoring"]["top_n_features"]
    tbl = drift.head(n).copy(); tbl["psi"] = tbl["psi"].round(4)
    md = f"""# Daily model monitoring - {summary['batch_date']}

**Decision: {summary['decision']}** {('- ' + '; '.join(summary['reasons'])) if summary['reasons'] else ''}

| Metric | Value |
|---|---|
| Champion | v{summary['model_version']} ({summary['model_type']}) |
| Rows scored | {summary['n_rows']} |
| Mean PD | {summary['mean_pd']:.4f} |
| Score PSI | {summary['score_psi']:.4f} |
| Red / amber features | {summary['n_red_features']} / {summary['n_amber_features']} |
| Back-test AUC | {f"{summary['auc']:.4f}" if summary.get('auc') is not None else 'n/a (no labels)'} |

## Commentary ({c['source']})
{c['output']}

## Top drifted features
{tbl.to_markdown(index=False) if hasattr(tbl, 'to_markdown') else tbl.to_string(index=False)}
"""
    out = path(cfg, "reports") / f"monitoring_{summary['batch_date']}.md"
    out.write_text(md)
    return out


if __name__ == "__main__":  # python -m crgov.governance  -> model card for the champion
    print(build_model_card())
