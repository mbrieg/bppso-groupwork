"""processing_times.basic_analysis

Reusable helpers for:
- building segments and instance durations
- adding modelling features (time-of-day, weekday, occurrence)
- exporting empirical references and fitting simple parametric distributions

The core modelling table is the instance dataframe `inst` with columns:
- case:concept:name
- concept:name
- instance
- proc_seconds
- total_seconds
- wait_seconds
- inst_start
- inst_end

This module is designed so you can:
- do analysis in a notebook
- export JSON artefacts
- reuse the same artefacts in a simulator
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import json
import math

import numpy as np
import pandas as pd
import scipy.stats as stats

# Lifecycle sets (align with your original Functions.py)
BEGIN = {"start", "resume"}
END = {"suspend", "complete", "ate_abort", "withdraw"}
TERMINAL = {"complete", "ate_abort", "withdraw"}


def build_segments_and_instances(df: pd.DataFrame, workflow_only: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build segment-level and instance-level processing time tables.

    Required columns in df:
    - case:concept:name
    - concept:name
    - time:timestamp
    - lifecycle:transition
    Optional:
    - org:resource

    Returns
    -------
    seg:
        segment table with seg_start/seg_end and segment proc_seconds
    proc_inst:
        instance table with proc_seconds (sum of segments)
    """
    x = df.dropna(subset=["time:timestamp"]).copy()
    x["time:timestamp"] = pd.to_datetime(x["time:timestamp"], utc=True)

    if workflow_only:
        x = x[x["concept:name"].astype(str).str.startswith("W_")]

    x = x.sort_values(["case:concept:name", "concept:name", "time:timestamp"])  # stable order

    seg_rows: List[dict] = []
    inst_rows: List[dict] = []

    for (case, act), g in x.groupby(["case:concept:name", "concept:name"], sort=False):
        inst = 0
        open_ts = None
        open_res = None
        proc_sum = 0.0
        worked = False

        for _, r in g.iterrows():
            tr = r.get("lifecycle:transition", None)
            ts = r["time:timestamp"]
            res = r.get("org:resource", np.nan)

            if tr in BEGIN:
                open_ts = ts
                open_res = res
                worked = True

            elif tr in END:
                if open_ts is not None:
                    dur = (ts - open_ts).total_seconds()
                    if dur >= 0:
                        seg_rows.append(
                            {
                                "case:concept:name": case,
                                "concept:name": act,
                                "instance": inst,
                                "org:resource": open_res,
                                "seg_start": open_ts,
                                "seg_end": ts,
                                "end_transition": tr,
                                "proc_seconds": float(dur),
                            }
                        )
                        proc_sum += float(dur)

                    open_ts = None
                    open_res = None

                if tr in TERMINAL:
                    if worked:
                        inst_rows.append(
                            {
                                "case:concept:name": case,
                                "concept:name": act,
                                "instance": inst,
                                "proc_seconds": float(proc_sum),
                            }
                        )

                    inst += 1
                    open_ts = None
                    open_res = None
                    proc_sum = 0.0
                    worked = False

    seg = pd.DataFrame(seg_rows)
    proc_inst = pd.DataFrame(inst_rows)
    return seg, proc_inst


def build_instance_times(
    seg: pd.DataFrame,
    proc_inst: pd.DataFrame,
    keys: Sequence[str] = ("case:concept:name", "concept:name", "instance"),
    clip_wait_lower: float = 0.0,
) -> pd.DataFrame:
    """Derive instance start/end, total_seconds and wait_seconds.

    wait_seconds = max(clip_wait_lower, total_seconds - proc_seconds)
    """
    if seg.empty:
        raise ValueError("seg is empty; cannot derive inst_start/inst_end")

    seg2 = seg.copy()
    seg2["seg_start"] = pd.to_datetime(seg2["seg_start"], utc=True)
    seg2["seg_end"] = pd.to_datetime(seg2["seg_end"], utc=True)

    inst_bounds = (
        seg2.groupby(list(keys), as_index=False)
        .agg(inst_start=("seg_start", "min"), inst_end=("seg_end", "max"))
    )
    inst_bounds["total_seconds"] = (inst_bounds["inst_end"] - inst_bounds["inst_start"]).dt.total_seconds().astype(float)

    out = proc_inst.merge(inst_bounds, on=list(keys), how="left")

    out["wait_seconds"] = out["total_seconds"].astype(float) - out["proc_seconds"].astype(float)
    out["wait_seconds"] = out["wait_seconds"].clip(lower=float(clip_wait_lower))

    return out


def add_occurrence_features(inst: pd.DataFrame, instance_col: str = "instance") -> pd.DataFrame:
    """Add occurrence index features.

    In your logs, `instance` already represents the occurrence per (case, activity).
    We keep it AND add explicit numeric features:
    - occ_idx (int)
    - occ_log (log1p)
    """
    out = inst.copy()
    out["occ_idx"] = out[instance_col].astype(int)
    out["occ_log"] = np.log1p(out["occ_idx"].astype(float))
    return out


def add_time_features(
    inst: pd.DataFrame,
    ts_col: str = "inst_start",
    *,
    add_weekday: bool = True,
    add_tod_sin_cos: bool = True,
    drop_ts_na: bool = False,
) -> pd.DataFrame:
    """Add time-of-day features based on inst_start (or other timestamp column).

    Adds:
    - minute_of_day (0..1439)
    - tod_sin, tod_cos (circular time encoding)
    - weekday (0=Mon..6=Sun)
    """
    out = inst.copy()
    ts = pd.to_datetime(out[ts_col], utc=True, errors="coerce")

    if drop_ts_na:
        out = out.loc[ts.notna()].copy()
        ts = pd.to_datetime(out[ts_col], utc=True, errors="coerce")

    if add_tod_sin_cos:
        minute_of_day = ts.dt.hour * 60 + ts.dt.minute
        angle = 2 * np.pi * (minute_of_day / (24 * 60))
        out["minute_of_day"] = minute_of_day.astype(float)
        out["tod_sin"] = np.sin(angle)
        out["tod_cos"] = np.cos(angle)

    if add_weekday:
        out["weekday"] = ts.dt.weekday.astype(float)

    return out


def _round_values(values: np.ndarray, round_to: Optional[float]) -> np.ndarray:
    if round_to is None or round_to == 0:
        return values
    return np.round(values / float(round_to)) * float(round_to)


def build_duration_counts(
    inst: pd.DataFrame,
    value_col: str,
    *,
    act_col: str = "concept:name",
    round_to: Optional[float] = 1.0,
    drop_zeros: bool = False,
    drop_na: bool = True,
) -> Dict[str, List[Tuple[int, float]]]:
    """Build JSON-friendly compressed duration lists per activity.

    Returns dict: activity -> [(count, value), ...] sorted by value.
    """
    x = inst[[act_col, value_col]].copy()
    if drop_na:
        x = x.dropna(subset=[value_col])

    x[value_col] = x[value_col].astype(float)
    if drop_zeros:
        x = x.loc[x[value_col] > 0].copy()

    out: Dict[str, List[Tuple[int, float]]] = {}
    for act, g in x.groupby(act_col, sort=False):
        vals = g[value_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        vals = _round_values(vals, round_to)
        if len(vals) == 0:
            continue
        uniq, cnt = np.unique(vals, return_counts=True)
        pairs = sorted([(int(c), float(v)) for v, c in zip(uniq, cnt)], key=lambda t: t[1])
        out[str(act)] = pairs
    return out


def fit_parametric_models(
    counts: Dict[str, List[Tuple[int, float]]],
    *,
    min_n: int = 30,
    allow_lognorm: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Fit a simple per-activity parametric distribution (gamma vs lognorm).

    Input is compressed counts: act -> [(count, value), ...]
    Output is JSON-serializable: act -> {dist: ..., params: ...}
    """
    models: Dict[str, Dict[str, Any]] = {}

    for act, pairs in counts.items():
        # expand minimally for fit
        vals = np.repeat([v for _, v in pairs], [c for c, _ in pairs]).astype(float)
        vals = vals[np.isfinite(vals)]
        vals = vals[vals > 0]

        if len(vals) < max(3, min_n):
            # fallback: constant at median
            models[act] = {"dist": "const", "value": float(np.median(vals)) if len(vals) else 0.0}
            continue

        # Fit gamma (loc fixed at 0)
        try:
            a_g, loc_g, scale_g = stats.gamma.fit(vals, floc=0)
            ll_g = float(np.sum(stats.gamma.logpdf(vals, a=a_g, loc=0, scale=scale_g)))
            k_g = 2
            aic_g = 2 * k_g - 2 * ll_g
        except Exception:
            aic_g = float("inf")

        # Fit lognorm (loc fixed at 0)
        aic_l = float("inf")
        if allow_lognorm:
            try:
                s_l, loc_l, scale_l = stats.lognorm.fit(vals, floc=0)
                ll_l = float(np.sum(stats.lognorm.logpdf(vals, s=s_l, loc=0, scale=scale_l)))
                k_l = 2
                aic_l = 2 * k_l - 2 * ll_l
            except Exception:
                aic_l = float("inf")

        if aic_l < aic_g:
            models[act] = {"dist": "lognorm", "params": {"s": float(s_l), "scale": float(scale_l)}}
        else:
            models[act] = {"dist": "gamma", "params": {"a": float(a_g), "scale": float(scale_g)}}

    return models


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True))


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())

def add_case_context(
    inst: pd.DataFrame,
    df: pd.DataFrame,
    *,
    case_col: str = "case:concept:name",
    ctx_cols: Sequence[str] = ("case:ApplicationType", "case:RequestedAmount"),
    strict: bool = False,
    numeric_cols: Sequence[str] = ("case:RequestedAmount",),
    add_log1p: bool = True,
    log1p_col_map: Optional[Dict[str, str]] = None,
    agg: str = "first",
) -> pd.DataFrame:
    """Attach case-level context columns to the instance table."""
    if case_col not in inst.columns:
        raise KeyError(f"inst is missing case column '{case_col}'")

    missing = [c for c in ctx_cols if c not in df.columns]
    if missing:
        raise KeyError(f"df is missing context columns: {missing}")

    if agg not in {"first", "last"}:
        raise ValueError("agg must be 'first' or 'last'")

    cols = [case_col] + list(ctx_cols)
    case_df = df[cols].copy().dropna(subset=[case_col])

    # Const-within-case check (only non-null values)
    for c in ctx_cols:
        nun = case_df.loc[case_df[c].notna()].groupby(case_col)[c].nunique(dropna=True)
        n_bad = int((nun > 1).sum())
        if n_bad > 0:
            msg = f"Context column '{c}' is not constant within {n_bad} cases"
            if strict:
                raise ValueError(msg)
            print(f"[basic_analysis] WARN: {msg}; using agg='{agg}'.")

    agg_fn = "first" if agg == "first" else "last"
    case_tbl = case_df.groupby(case_col, as_index=False).agg({c: agg_fn for c in ctx_cols})

    out = inst.merge(case_tbl, on=case_col, how="left")

    # Numeric conversion + optional log1p
    numeric_cols = tuple(numeric_cols)
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if add_log1p and numeric_cols:
        m = dict(log1p_col_map or {})
        for c in numeric_cols:
            if c not in out.columns:
                continue
            if c not in m:
                m[c] = "req_log" if c == "case:RequestedAmount" else f"{c}_log1p"
            out[m[c]] = np.log1p(out[c].fillna(0.0).astype(float))

    return out
