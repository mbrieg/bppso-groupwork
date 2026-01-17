"""processing_times.sampling

One place to sample durations for the simulator.

Modes (DurationSampler.mode)
---------------------------
- total_parametric: sample total_seconds from parametric JSON
- proc_qr_plus_wait_ref: sample proc_seconds via quantile bundle + wait_seconds empirically, then sum
- proc_param_plus_wait_ref: sample proc_seconds from parametric + wait_seconds empirically, then sum

This file also contains small helpers to:
- load JSON artefacts
- sample from compressed reference counts
- build an X_row with the exact columns expected by a QuantileModelBundle
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import json
import math

import numpy as np
import pandas as pd


# ----------------------------
# JSON model I/O (cached)
# ----------------------------

_JSON_CACHE: Dict[str, Dict[str, Any]] = {}


def load_json(path: str | Path) -> Dict[str, Any]:
    p = str(Path(path).resolve())
    if p not in _JSON_CACHE:
        _JSON_CACHE[p] = json.loads(Path(p).read_text())
    return _JSON_CACHE[p]


# ----------------------------
# Compressed empirical references
# ----------------------------


def sample_from_counts(counts: Sequence[Sequence[float]], rng: np.random.Generator) -> float:
    """Sample one value from a compressed list [(count, value), ...]."""
    if not counts:
        return 0.0
    weights = np.asarray([c for c, _ in counts], dtype=float)
    values = np.asarray([v for _, v in counts], dtype=float)

    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    total = float(weights.sum())
    if total <= 0:
        return float(values[0])

    u = rng.random() * total
    cum = 0.0
    for w, v in zip(weights, values):
        cum += float(w)
        if u <= cum:
            return float(v)
    return float(values[-1])


# ----------------------------
# Parametric models (Gamma / Lognorm / Const)
# ----------------------------


def sample_parametric(spec: Dict[str, Any], rng: np.random.Generator) -> float:
    dist = spec.get("dist")
    if dist == "const":
        return float(spec.get("value", 0.0))

    if dist == "gamma":
        a = float(spec["params"]["a"])
        scale = float(spec["params"]["scale"])
        return max(0.0, float(rng.gamma(shape=a, scale=scale)))

    if dist == "lognorm":
        s = float(spec["params"]["s"])
        scale = float(spec["params"]["scale"])
        mu = math.log(scale) if scale > 0 else 0.0
        return max(0.0, float(rng.lognormal(mean=mu, sigma=s)))

    return 0.0


def sample_activity_parametric(models: Dict[str, Any], activity: str, rng: np.random.Generator,
                              fallback_seconds: Tuple[float, float] = (300.0, 900.0)) -> float:
    spec = models.get(activity)
    if spec is None:
        lo, hi = fallback_seconds
        return float(rng.uniform(lo, hi))
    return sample_parametric(spec, rng)


# ----------------------------
# Quantile Regression helpers
# ----------------------------


def ensure_feature_frame(bundle: Any, row: Union[Mapping[str, Any], pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """Create a 1-row DataFrame with exactly bundle.cat_cols + bundle.num_cols.

    This prevents the "num cols lost" issue when you build X_row manually.
    - missing categorical -> filled with ''
    - missing numeric -> filled with 0.0
    """
    if isinstance(row, pd.DataFrame):
        X = row.copy()
        if len(X) != 1:
            raise ValueError("X_row must be a single row")
    elif isinstance(row, pd.Series):
        X = row.to_frame().T
    else:
        X = pd.DataFrame([dict(row)])

    cat_cols = list(getattr(bundle, "cat_cols", []))
    num_cols = list(getattr(bundle, "num_cols", []))
    cols = cat_cols + num_cols

    for c in cat_cols:
        if c not in X.columns:
            X[c] = ""

    for c in num_cols:
        if c not in X.columns:
            X[c] = 0.0

    # order + type safety
    X = X[cols].copy()
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    return X


def predict_quantiles_seconds(bundle: Any, X_row: Union[Mapping[str, Any], pd.Series, pd.DataFrame]) -> Dict[float, float]:
    """Predict quantiles for a single row, returning seconds.

    If bundle.y_transform == 'log1p', the model predicts log1p(seconds) and we undo via expm1.
    """
    X = ensure_feature_frame(bundle, X_row)
    preds: Dict[float, float] = {}
    for q, model in bundle.models.items():
        preds[float(q)] = float(model.predict(X)[0])

    if getattr(bundle, "y_transform", None) == "log1p":
        preds = {q: float(np.expm1(v)) for q, v in preds.items()}

    return preds


def sample_from_quantiles(q_preds: Dict[float, float], rng: np.random.Generator) -> float:
    """Piecewise-linear sampling from three quantiles.

    Works for any (qL,qM,qH), e.g. (0.1,0.5,0.9) or (0.2,0.5,0.8).
    """
    qs = sorted(q_preds.keys())
    if len(qs) < 3:
        return float(q_preds[qs[len(qs)//2]])

    qL, qM, qH = qs[0], qs[1], qs[2]
    vL, vM, vH = float(q_preds[qL]), float(q_preds[qM]), float(q_preds[qH])

    vL = min(vL, vM)
    vH = max(vH, vM)

    u = float(rng.random())

    if u <= qL:
        return max(0.0, vL)
    if u >= qH:
        return max(0.0, vH)

    if u <= qM:
        t = (u - qL) / (qM - qL)
        return max(0.0, vL + t * (vM - vL))

    t = (u - qM) / (qH - qM)
    return max(0.0, vM + t * (vH - vM))


# ----------------------------
# High-level sampler (plug into simulator)
# ----------------------------


@dataclass
class DurationSampler:
    mode: str
    rng: np.random.Generator

    total_models: Optional[Dict[str, Any]] = None
    proc_models: Optional[Dict[str, Any]] = None
    wait_reference: Optional[Dict[str, Any]] = None
    proc_qr: Optional[Any] = None

    def sample_total_seconds(self, activity: str, *, X_row=None) -> float:
        if self.mode == "total_parametric":
            if self.total_models is None:
                raise ValueError("total_models not set")
            return sample_activity_parametric(self.total_models, activity, self.rng)

        if self.mode == "proc_qr_plus_wait_ref":
            if self.proc_qr is None or self.wait_reference is None:
                raise ValueError("proc_qr and wait_reference must be set")
            if X_row is None:
                raise ValueError("X_row is required for proc_qr_plus_wait_ref")
            q_preds = predict_quantiles_seconds(self.proc_qr, X_row)
            proc_sec = sample_from_quantiles(q_preds, self.rng)
            wait_counts = self.wait_reference.get(activity, [])
            wait_sec = sample_from_counts(wait_counts, self.rng) if wait_counts else 0.0
            return max(0.0, proc_sec + wait_sec)

        if self.mode == "proc_param_plus_wait_ref":
            if self.proc_models is None or self.wait_reference is None:
                raise ValueError("proc_models and wait_reference must be set")
            proc_sec = sample_activity_parametric(self.proc_models, activity, self.rng)
            wait_counts = self.wait_reference.get(activity, [])
            wait_sec = sample_from_counts(wait_counts, self.rng) if wait_counts else 0.0
            return max(0.0, proc_sec + wait_sec)

        raise ValueError(f"Unknown mode: {self.mode}")

    def sample_total_timedelta(self, activity: str, *, X_row=None) -> timedelta:
        return timedelta(seconds=float(self.sample_total_seconds(activity, X_row=X_row)))
