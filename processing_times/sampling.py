from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union
import json, math
import numpy as np

try:
    import joblib
except Exception:
    joblib = None


class ModelSpecError(ValueError):
    pass


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _as_rng(rng: Optional[np.random.Generator], seed: Optional[int]) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng(seed)


def _time_features(now: Optional[datetime]) -> Dict[str, float]:
    if now is None:
        return {"minute_of_day": 0.0, "tod_sin": 0.0, "tod_cos": 0.0, "weekday": 0.0}
    minute_of_day = now.hour * 60 + now.minute + now.second / 60.0
    angle = 2 * math.pi * (minute_of_day / (24 * 60))
    return {
        "minute_of_day": float(minute_of_day),
        "tod_sin": float(math.sin(angle)),
        "tod_cos": float(math.cos(angle)),
        "weekday": float(now.weekday()),
    }


@dataclass(frozen=True)
class _Emp:
    values: np.ndarray
    probs: np.ndarray


class SpecSampler:
    """JSON: const/gamma/lognorm/empirical(counts)."""

    def __init__(self, specs: Mapping[str, Mapping[str, Any]], *, default_value: float = 0.0, seed: Optional[int] = None):
        self.specs = {str(k): dict(v) for k, v in specs.items()}
        self.default_value = float(default_value)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._emp: Dict[str, _Emp] = {}
        self._prep()

    @classmethod
    def from_json(cls, path: Union[str, Path], **kw) -> "SpecSampler":
        return cls(load_json(path), **kw)

    def _prep(self) -> None:
        for act, spec in self.specs.items():
            dist = spec.get("dist")
            if dist not in {"const", "gamma", "lognorm", "empirical"}:
                raise ModelSpecError(f"Unknown dist={dist!r} for {act}")

            if dist == "empirical":
                pairs = spec.get("counts", [])
                cnts, vals = [], []
                for c, v in pairs:
                    c = int(c); v = float(v)
                    if c > 0 and np.isfinite(v):
                        cnts.append(c); vals.append(v)
                if not cnts:
                    continue
                probs = np.asarray(cnts, float)
                probs /= probs.sum()
                self._emp[act] = _Emp(values=np.asarray(vals, float), probs=probs)

    def sample(self, activity: str, *, rng=None) -> float:
        rng = rng if rng is not None else self.rng
        spec = self.specs.get(str(activity))
        if not spec:
            return float(self.default_value)

        dist = spec.get("dist")
        if dist == "const":
            return float(max(0.0, spec.get("value", self.default_value)))

        if dist == "gamma":
            p = spec.get("params", {})
            return float(max(0.0, rng.gamma(shape=float(p["a"]), scale=float(p["scale"]), size=1)[0]))

        if dist == "lognorm":
            p = spec.get("params", {})
            scale = float(p["scale"])
            mu = math.log(scale) if scale > 0 else 0.0
            sigma = float(p["s"])
            return float(max(0.0, rng.lognormal(mean=mu, sigma=sigma, size=1)[0]))

        if dist == "empirical":
            emp = self._emp.get(str(activity))
            if emp is None:
                return float(self.default_value)
            i = rng.choice(len(emp.values), p=emp.probs)
            return float(max(0.0, emp.values[i]))

        return float(self.default_value)


class QuantileBundleSampler:
    """Joblib bundle -> predicts q_low/q_mid/q_high -> sample triangular."""

    def __init__(self, bundle: Mapping[str, Any], *, seed: Optional[int] = None):
        self.seed = seed
        self.models = dict(bundle["models"])
        self.rng = np.random.default_rng(seed)
        self.qs = tuple(bundle["qs"])  # (low, mid, high)
        meta = dict(bundle["meta"])
        self.cat_cols = list(meta.get("cat_cols", []))
        self.num_cols = list(meta.get("num_cols", []))

    @classmethod
    def from_joblib(cls, path: Union[str, Path], **kw) -> "QuantileBundleSampler":
        if joblib is None:
            raise RuntimeError("joblib required")
        return cls(joblib.load(path), **kw)

    def _row(self, activity: str, now: Optional[datetime], instance: int, ctx: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        base = {"concept:name": str(activity), "instance": float(instance)}
        base.update(_time_features(now))
        if ctx:
            base.update(dict(ctx))
            if "case:ApplicationType" not in base and "application_type" in base:
                base["case:ApplicationType"] = base["application_type"]
            if "case:RequestedAmount" not in base and "requested_amount" in base:
                base["case:RequestedAmount"] = base["requested_amount"]
        row = {}
        for c in self.cat_cols:
            row[c] = str(base.get(c, "UNK"))
        for c in self.num_cols:
            try:
                row[c] = float(base.get(c, 0.0))
            except Exception:
                row[c] = 0.0
        return row

    def sample(self, activity: str, *, now: Optional[datetime], instance: int, ctx: Optional[Mapping[str, Any]], rng= None) -> float:
        rng = rng if rng is not None else self.rng
        import pandas as pd
        X = pd.DataFrame([self._row(activity, now, instance, ctx)], columns=self.cat_cols + self.num_cols)

        qL, qM, qH = self.qs
        a = float(self.models[qL].predict(X)[0])
        m = float(self.models[qM].predict(X)[0])
        b = float(self.models[qH].predict(X)[0])

        # basic sanitizing
        m = max(0.0, m)
        a = max(0.0, min(a, m))
        b = max(m, b)

        if b <= a:
            return m
        return float(rng.triangular(left=a, mode=m, right=b, size=1)[0])


class ProcessingTimeSampler:
    def __init__(self, *, proc=None, wait=None, total=None, qr=None, seed=None):
        self.proc = proc
        self.wait = wait
        self.total = total
        self.qr = dict(qr) if qr else {}
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_paths(cls, *, proc_json=None, wait_json=None, total_json=None, qr_joblib=None, seed=None, default_value=0.0):
        proc = SpecSampler.from_json(proc_json, default_value=default_value, seed=seed) if proc_json else None
        wait = SpecSampler.from_json(wait_json, default_value=default_value, seed=seed) if wait_json else None
        total = SpecSampler.from_json(total_json, default_value=default_value, seed=seed) if total_json else None

        qr = {}
        if qr_joblib:
            for kind, path in qr_joblib.items():
                qr[str(kind)] = QuantileBundleSampler.from_joblib(path, seed=seed)

        return cls(proc=proc, wait=wait, total=total, qr=qr, seed=seed)

    def sample(self, activity: str, *, kind="proc", now=None, instance=0, ctx=None, rng=None, use_qr=False) -> float:
        rng = rng if rng is not None else self.rng
        kind = str(kind)

        if use_qr and kind in self.qr:
            return float(self.qr[kind].sample(activity, now=now, instance=instance, ctx=ctx, rng=rng))

        if kind == "proc":
            if self.proc is None: raise ModelSpecError("No proc sampler")
            return float(self.proc.sample(activity, rng=rng))

        if kind == "wait":
            if self.wait is None: raise ModelSpecError("No wait sampler")
            return float(self.wait.sample(activity, rng=rng))

        if kind == "total":
            if self.total is not None:
                return float(self.total.sample(activity, rng=rng))
            if self.proc is None or self.wait is None:
                raise ModelSpecError("No total and no proc+wait")
            return float(self.proc.sample(activity, rng=rng) + self.wait.sample(activity, rng=rng))

        raise ModelSpecError(f"Unknown kind={kind}")
