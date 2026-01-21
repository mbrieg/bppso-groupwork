
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import json
import math
import numpy as np

try:
    import joblib
except Exception:
    joblib = None

from scipy.stats import norm


class ModelSpecError(ValueError):
    """Wird geworfen wenn ein Artefakt / Spec kaputt oder unerwartet ist."""


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _as_rng(rng: Optional[np.random.Generator], seed: Optional[int]) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng(seed)


def _clip_nonneg(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x[~np.isfinite(x)] = 0.0
    return np.clip(x, 0.0, None)


def _apply_p0(spec: Mapping[str, Any], out: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    opt: zero.inflation.
    """
    p0 = spec.get("p0", None)
    if p0 is None:
        return out
    try:
        p0f = float(p0)
    except Exception:
        return out
    if p0f <= 0.0:
        return out
    p0f = min(1.0, max(0.0, p0f))
    mask = rng.random(out.shape) < p0f
    if not mask.any():
        return out
    out2 = out.copy()
    out2[mask] = 0.0
    return out2


def _time_features(now: Optional[datetime]) -> Tuple[float, float, float, float]:
    """Return (minute_of_day, tod_sin, tod_cos, weekday). Falls now=None -> (0,0,0,0)."""
    if now is None:
        return 0.0, 0.0, 0.0, 0.0
    # include seconds/microseconds for smoother minute_of_day
    minute_of_day = (
        now.hour * 60
        + now.minute
        + now.second / 60.0
        + now.microsecond / 60.0 / 1_000_000.0
    )
    angle = 2 * math.pi * (minute_of_day / (24 * 60))
    return float(minute_of_day), float(math.sin(angle)), float(math.cos(angle)), float(now.weekday())


@dataclass(frozen=True)
class _EmpCache:
    values: np.ndarray
    probs: np.ndarray


class SpecSampler:
    """
    Sampler for JSON-Specs:
      - const
      - gamma
      - lognorm
      - empirical (counts)
    plus optional p0 (Zero-Inflation)
    """

    def __init__(
        self,
        specs: Mapping[str, Mapping[str, Any]],
        *,
        default_value: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self.specs: Dict[str, Dict[str, Any]] = {str(k): dict(v) for k, v in specs.items()}
        self.default_value = float(default_value)
        self.seed = seed
        self._empirical: Dict[str, _EmpCache] = {}
        self._validate_and_prepare()

    @classmethod
    def from_json(cls, path: Union[str, Path], **kwargs: Any) -> "SpecSampler":
        return cls(load_json(path), **kwargs)

    def _validate_and_prepare(self) -> None:
        for act, spec in self.specs.items():
            if not isinstance(spec, Mapping):
                raise ModelSpecError(f"Spec for {act!r} must be a dict, got: {type(spec)}")

            dist = spec.get("dist")
            if dist not in {"const", "gamma", "lognorm", "empirical"}:
                raise ModelSpecError(f"Unknown dist={dist!r} for activity {act!r}")

            # Optional p0
            if spec.get("p0", None) is not None:
                try:
                    p0 = float(spec.get("p0"))
                except Exception:
                    raise ModelSpecError(f"p0 must be numeric for activity {act!r}")
                if not (0.0 <= p0 <= 1.0):
                    raise ModelSpecError(f"p0 must be in [0,1] for activity {act!r}, got {p0}")
                spec["p0"] = float(p0)
            else:
                spec.pop("p0", None)

            if dist == "const":
                if "value" not in spec:
                    raise ModelSpecError(f"Missing 'value' for const model of {act!r}")
                spec["value"] = float(spec["value"])

            elif dist in {"gamma", "lognorm"}:
                params = spec.get("params")
                if not isinstance(params, Mapping):
                    raise ModelSpecError(f"Missing/invalid 'params' for {dist} model of {act!r}")
                if dist == "gamma":
                    if "a" not in params or "scale" not in params:
                        raise ModelSpecError(f"Gamma params must include 'a' and 'scale' for {act!r}")
                    spec["params"] = {"a": float(params["a"]), "scale": float(params["scale"])}
                else:
                    if "s" not in params or "scale" not in params:
                        raise ModelSpecError(f"Lognorm params must include 's' and 'scale' for {act!r}")
                    spec["params"] = {"s": float(params["s"]), "scale": float(params["scale"])}

            elif dist == "empirical":
                counts = spec.get("counts")
                if not isinstance(counts, Sequence) or len(counts) == 0:
                    raise ModelSpecError(f"Empirical spec needs non-empty 'counts' for {act!r}")

                cnts, vals = [], []
                for pair in counts:
                    if not isinstance(pair, Sequence) or len(pair) != 2:
                        raise ModelSpecError(f"Empirical counts for {act!r} must be pairs, got {pair!r}")
                    c = int(pair[0])
                    v = float(pair[1])
                    if c <= 0 or not np.isfinite(v):
                        continue
                    cnts.append(c)
                    vals.append(v)

                if len(cnts) == 0:
                    raise ModelSpecError(f"Empirical counts for {act!r} are empty after cleaning")

                values = np.asarray(vals, dtype=float)
                probs = np.asarray(cnts, dtype=float)
                probs = probs / probs.sum()
                self._empirical[act] = _EmpCache(values=values, probs=probs)

    def get_spec(self, activity: str) -> Dict[str, Any]:
        return self.specs.get(str(activity), {"dist": "const", "value": self.default_value})

    def sample(self, activity: str, *, n: int = 1, rng: Optional[np.random.Generator] = None) -> Union[float, np.ndarray]:
        rng = _as_rng(rng, self.seed)
        spec = self.get_spec(activity)
        dist = spec.get("dist")

        if dist == "const":
            v = max(0.0, float(spec.get("value", self.default_value)))
            out = np.full(n, v, dtype=float)
            out = _apply_p0(spec, out, rng)
            return float(out[0]) if n == 1 else out

        if dist == "gamma":
            p = spec["params"]
            out = rng.gamma(shape=float(p["a"]), scale=float(p["scale"]), size=n)
            out = _clip_nonneg(out)
            out = _apply_p0(spec, out, rng)
            return float(out[0]) if n == 1 else out

        if dist == "lognorm":
            p = spec["params"]
            scale = float(p["scale"])
            mu = math.log(scale) if scale > 0 else 0.0
            sigma = float(p["s"])
            out = rng.lognormal(mean=mu, sigma=sigma, size=n)
            out = _clip_nonneg(out)
            out = _apply_p0(spec, out, rng)
            return float(out[0]) if n == 1 else out

        if dist == "empirical":
            cache = self._empirical.get(str(activity))
            if cache is None:
                v = max(0.0, self.default_value)
                out = np.full(n, v, dtype=float)
                out = _apply_p0(spec, out, rng)
                return float(out[0]) if n == 1 else out

            idx = rng.choice(len(cache.values), size=n, replace=True, p=cache.probs)
            out = cache.values[idx]
            out = _clip_nonneg(out)
            out = _apply_p0(spec, out, rng)
            return float(out[0]) if n == 1 else out

        raise ModelSpecError(f"Unhandled dist={dist!r} for activity {activity!r}")


class QuantileBundleSampler:
    """
    Sampler für saved Quantile-Regression Bundles (joblib).
    Wir nehmen die vorhergesagten Quantile und machen daraus eine einfache Verteilung.

    Default: Fit einer Lognormal-Verteilung auf (q_low, q_high) und sampeln daraus.
    Fallback: Triangular(low, median, high).
    """

    def __init__(self, bundle: Mapping[str, Any], *, seed: Optional[int] = None) -> None:
        self.bundle = dict(bundle)
        self.seed = seed

        if "models" not in self.bundle or "meta" not in self.bundle or "qs" not in self.bundle:
            raise ModelSpecError("Quantile bundle must contain keys: 'models', 'meta', 'qs'")

        self.models: Dict[float, Any] = dict(self.bundle["models"])
        self.qs: Tuple[float, float, float] = tuple(self.bundle["qs"])  # (low, mid, high)

        meta = dict(self.bundle["meta"])
        self.cat_cols: Sequence[str] = list(meta.get("cat_cols", []))
        self.num_cols: Sequence[str] = list(meta.get("num_cols", []))

        # Flexible: allow additional case context columns (e.g., case:ApplicationType)
        if 'concept:name' not in self.cat_cols:
            raise ModelSpecError(f"Expected meta.cat_cols to include 'concept:name', got {self.cat_cols!r}")

        # Required numeric context
        needed_base = {'instance', 'weekday'}
        if not needed_base.issubset(set(self.num_cols)):
            raise ModelSpecError(f"Expected num_cols to include {sorted(needed_base)}, got {self.num_cols!r}")

        # Time-of-day encoding: allow either minute_of_day or (tod_sin,tod_cos) or both
        has_minute = 'minute_of_day' in set(self.num_cols)
        has_sincos = {'tod_sin', 'tod_cos'}.issubset(set(self.num_cols))
        if not (has_minute or has_sincos):
            raise ModelSpecError(
                f"Expected num_cols to include either 'minute_of_day' or both 'tod_sin'/'tod_cos', got {self.num_cols!r}"
            )

    @classmethod
    def from_joblib(cls, path: Union[str, Path], **kwargs: Any) -> "QuantileBundleSampler":
        if joblib is None:
            raise RuntimeError("joblib is required to load quantile bundles")
        bundle = joblib.load(path)
        return cls(bundle, **kwargs)

    def predict_quantiles(
        self,
        activity: str,
        *,
        now: Optional[datetime] = None,
        instance: int = 0,
        ctx: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[float, float, float]:
        minute_of_day, tod_sin, tod_cos, weekday = _time_features(now)

        import pandas as pd  # local import

        ctx = dict(ctx) if ctx is not None else {}

        # Base feature dict (can be extended/overridden by ctx)
        base: Dict[str, Any] = {
            'concept:name': str(activity),
            'minute_of_day': float(minute_of_day),
            'tod_sin': float(tod_sin),
            'tod_cos': float(tod_cos),
            'instance': float(instance),
            'weekday': float(weekday),
        }
        base.update(ctx)

        row: Dict[str, Any] = {}
        for c in self.cat_cols:
            v = base.get(c, 'UNK')
            row[c] = 'UNK' if v is None else str(v)
        for c in self.num_cols:
            v = base.get(c, 0.0)
            try:
                row[c] = float(v)
            except Exception:
                row[c] = 0.0

        X = pd.DataFrame([row], columns=list(self.cat_cols) + list(self.num_cols))

        qL, qM, qH = self.qs
        pL = float(self.models[qL].predict(X)[0])
        pM = float(self.models[qM].predict(X)[0])
        pH = float(self.models[qH].predict(X)[0])

        # Ordnung + Nonnegativity
        pM = max(0.0, pM)
        pL = max(0.0, min(pL, pM))
        pH = max(pM, pH)
        return pL, pM, pH

    def sample(
        self,
        activity: str,
        *,
        now: Optional[datetime] = None,
        instance: int = 0,
        ctx: Optional[Mapping[str, Any]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        rng = _as_rng(rng, self.seed)
        qL, qM, qH = self.predict_quantiles(activity, now=now, instance=instance, ctx=ctx)

        if not (qH > qL):
            return float(max(0.0, qM))

        pL, _, pH = self.qs
        zL = float(norm.ppf(pL))
        zH = float(norm.ppf(pH))

        if qL > 0 and qH > 0 and zH != zL:
            logL, logH = math.log(qL), math.log(qH)
            sigma = (logH - logL) / (zH - zL)
            if sigma < 1e-9:
                return float(max(0.0, qM))
            mu = logL - sigma * zL
            return float(max(0.0, rng.lognormal(mean=mu, sigma=sigma, size=1)[0]))

        # Fallback: triangular
        return float(max(0.0, rng.triangular(left=qL, mode=qM, right=qH, size=1)[0]))


class ProcessingTimeSampler:
    """
    High-Level Sampler: proc / wait / total.
    Optional: für bestimmte kinds kannst du QR nutzen (use_qr=True).
    """

    def __init__(
        self,
        *,
        proc: Optional[SpecSampler] = None,
        wait: Optional[SpecSampler] = None,
        total: Optional[SpecSampler] = None,
        qr: Optional[Mapping[str, QuantileBundleSampler]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.proc = proc
        self.wait = wait
        self.total = total
        self.qr = dict(qr) if qr is not None else {}
        self.seed = seed

    @classmethod
    def from_paths(
        cls,
        *,
        proc_json: Optional[Union[str, Path]] = None,
        wait_json: Optional[Union[str, Path]] = None,
        total_json: Optional[Union[str, Path]] = None,
        qr_joblib: Optional[Mapping[str, Union[str, Path]]] = None,
        seed: Optional[int] = None,
        default_value: float = 0.0,
    ) -> "ProcessingTimeSampler":
        proc = SpecSampler.from_json(proc_json, default_value=default_value, seed=seed) if proc_json else None
        wait = SpecSampler.from_json(wait_json, default_value=default_value, seed=seed) if wait_json else None
        total = SpecSampler.from_json(total_json, default_value=default_value, seed=seed) if total_json else None

        qr: Dict[str, QuantileBundleSampler] = {}
        if qr_joblib:
            for kind, path in qr_joblib.items():
                qr[str(kind)] = QuantileBundleSampler.from_joblib(path, seed=seed)

        return cls(proc=proc, wait=wait, total=total, qr=qr, seed=seed)

    def sample(
        self,
        activity: str,
        *,
        kind: str = "proc",
        now: Optional[datetime] = None,
        instance: int = 0,
        ctx: Optional[Mapping[str, Any]] = None,
        rng: Optional[np.random.Generator] = None,
        use_qr: bool = False,
    ) -> float:
        kind = str(kind)
        rng = _as_rng(rng, self.seed)

        if use_qr and kind in self.qr:
            return float(self.qr[kind].sample(activity, now=now, instance=instance, ctx=ctx, rng=rng))

        if kind == "proc":
            if self.proc is None:
                raise ModelSpecError("No proc sampler configured")
            return float(self.proc.sample(activity, n=1, rng=rng))

        if kind == "wait":
            if self.wait is None:
                raise ModelSpecError("No wait sampler configured")
            return float(self.wait.sample(activity, n=1, rng=rng))

        if kind == "total":
            if self.total is not None:
                return float(self.total.sample(activity, n=1, rng=rng))
            if self.proc is None or self.wait is None:
                raise ModelSpecError("No total sampler and proc/wait not both configured")
            return float(self.proc.sample(activity, n=1, rng=rng) + self.wait.sample(activity, n=1, rng=rng))

        raise ModelSpecError(f"Unknown kind={kind!r}")


__all__ = [
    "ModelSpecError",
    "SpecSampler",
    "QuantileBundleSampler",
    "ProcessingTimeSampler",
]
