"""Step 7 - scoring API. Always serves models:/<name>@champion and hot-reloads when the alias moves.
Run: uvicorn crgov.serve:app --port 8000"""
from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import registry
from .config import init_mlflow, load_config
from .scoring import load_scorer, pd_to_score

cfg = load_config()
app = FastAPI(title="Credit PD scoring API", version="0.1.0")
_state = {"scorer": None, "checked": 0.0}
_lock = threading.Lock()


def _scorer():
    """Lazy load; every `reload_seconds` check if the champion alias now points to another version."""
    now, s = time.time(), cfg["serving"]
    with _lock:
        cur = _state["scorer"]
        if cur is None or now - _state["checked"] > s["reload_seconds"]:
            init_mlflow(cfg)
            mv = registry.get_by_alias(cfg["mlflow"]["model_name"], s["alias"])
            if mv is None:
                if cur is None:
                    raise HTTPException(503, "No champion model registered yet")
            elif cur is None or int(mv.version) != cur.version:
                _state["scorer"] = load_scorer(cfg, alias=s["alias"])
            _state["checked"] = now
        return _state["scorer"]


class ScoreRequest(BaseModel):
    customer_id: str | None = None
    features: dict[str, float | None]      # customer-level aggregated features (same names as training)


class Reason(BaseModel):
    feature: str
    reason: str


class ScoreResponse(BaseModel):
    customer_id: str | None
    pd: float
    score: int
    risk_band: str
    reasons: list[Reason]
    model_version: int
    model_type: str
    warnings: list[str] = []


def _band(score: float) -> str:
    for b in cfg["serving"]["risk_bands"]:
        if score >= b["min_score"]:
            return b["name"]
    return cfg["serving"]["risk_bands"][-1]["name"]


@app.get("/health")
def health():
    s = _scorer()
    return {"status": "ok", "model_version": s.version, "model_type": s.model_type}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    s = _scorer()
    df = pd.DataFrame([req.features]).astype(float)
    missing = [f for f in s.features if f not in df.columns]
    warns = []
    if len(missing) / max(len(s.features), 1) > 0.5:
        raise HTTPException(422, f"{len(missing)}/{len(s.features)} required features missing, e.g. {missing[:5]}")
    if missing:
        warns.append(f"{len(missing)} features missing (treated as null), e.g. {missing[:3]}")
    p = float(s.predict_pd(df)[0])
    sc = float(pd_to_score(np.array([p]), cfg["scorecard"])[0])
    reasons = s.reasons(df, cfg["serving"]["top_reasons"])[0]
    return ScoreResponse(customer_id=req.customer_id, pd=p, score=int(round(sc)), risk_band=_band(sc),
                         reasons=[Reason(feature=r["feature"], reason=r["reason"]) for r in reasons],
                         model_version=s.version, model_type=s.model_type, warnings=warns)
