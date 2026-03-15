import os
import pickle
from datetime import date
from typing import Dict, Tuple, List, Optional, Set

import pandas as pd
import pm4py

RateTable = Dict[Tuple[int, int, bool], float]
TZ_NAME = "Europe/Amsterdam"

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

DEFAULT_XES_PATH = os.path.join(project_root, "data", "BPI Challenge 2017.xes.gz")

ARTIFACTS_DIR = os.path.join(project_root, "spawn_rates", "artifacts")
DEFAULT_HOLIDAYS_PATH = os.path.join(ARTIFACTS_DIR, "holidays_nl.pkl")
DEFAULT_RATE_TABLE_PATH = os.path.join(ARTIFACTS_DIR, "rate_table_nl_hourly.pkl")
DEFAULT_RECENCY_HALF_LIFE_DAYS = 90


def _generate_nl_holidays(years: List[int]) -> List[date]:
    try:
        import holidays as holidays_lib
    except ImportError as e:
        raise ImportError("Install with: pip install holidays") from e

    nl_data = holidays_lib.Netherlands(years=years)
    return sorted(list(nl_data.keys()))


def get_holidays(
        force_rebuild: bool = False,
        years: Optional[List[int]] = None,
        path: str = DEFAULT_HOLIDAYS_PATH,
) -> List[date]:
    if years is None:
        years = [2016, 2017, 2018]

    if (not force_rebuild) and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    hols = _generate_nl_holidays(years)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(hols, f)

    return hols


def _build_rate_table_from_df(
    df: pd.DataFrame,
    holidays_set: Set[date],
    recency_half_life_days: Optional[int] = None,
) -> RateTable:
    required = {"case:concept:name", "time:timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    df = df[["case:concept:name", "time:timestamp"]].copy()
    df["timestamp"] = (
        pd.to_datetime(df["time:timestamp"], utc=True, errors="coerce")
        .dt.tz_convert(TZ_NAME)
    )

    arrivals = df.groupby("case:concept:name")["timestamp"].min().reset_index()
    if arrivals.empty:
        return {}

    arrivals["date"] = arrivals["timestamp"].dt.date
    arrivals["weekday"] = arrivals["timestamp"].dt.weekday
    arrivals["hour"] = arrivals["timestamp"].dt.hour
    arrivals["is_holiday"] = arrivals["date"].isin(holidays_set) if holidays_set else False

    if recency_half_life_days is not None and recency_half_life_days > 0:
        arrivals["date_dt"] = pd.to_datetime(arrivals["date"])
        ref_date = arrivals["date_dt"].max()
        age_days = (ref_date - arrivals["date_dt"]).dt.days.astype(float)
        arrivals["weight"] = 0.5 ** (age_days / float(recency_half_life_days))

        counts = (
            arrivals.groupby(["weekday", "hour", "is_holiday"])["weight"]
            .sum()
            .rename("weighted_count")
            .reset_index()
        )

        unique_days = arrivals[["date", "date_dt"]].drop_duplicates().copy()
        unique_days["weekday"] = unique_days["date_dt"].dt.weekday
        unique_days["is_holiday"] = unique_days["date"].isin(holidays_set) if holidays_set else False
        day_age = (ref_date - unique_days["date_dt"]).dt.days.astype(float)
        unique_days["weight"] = 0.5 ** (day_age / float(recency_half_life_days))

        day_exposure = (
            unique_days.groupby(["weekday", "is_holiday"])["weight"]
            .sum()
            .rename("weighted_exposure")
            .reset_index()
        )

        rates = counts.merge(day_exposure, on=["weekday", "is_holiday"], how="left")
        rates["rate"] = rates["weighted_count"] / rates["weighted_exposure"]
    else:
        counts = (
            arrivals.groupby(["weekday", "hour", "is_holiday"])
            .size()
            .rename("count")
            .reset_index()
        )

        calendar = pd.DataFrame({
            "date": pd.date_range(arrivals["date"].min(), arrivals["date"].max(), freq="D")
        })
        calendar["date"] = calendar["date"].dt.date
        calendar["weekday"] = pd.to_datetime(calendar["date"]).dt.weekday
        calendar["is_holiday"] = calendar["date"].isin(holidays_set) if holidays_set else False

        day_exposure = (
            calendar.groupby(["weekday", "is_holiday"])
            .size()
            .rename("day_exposure")
            .reset_index()
        )

        rates = counts.merge(day_exposure, on=["weekday", "is_holiday"], how="left")
        rates["rate"] = rates["count"] / rates["day_exposure"]

    rate_table = {
        (int(r.weekday), int(r.hour), bool(r.is_holiday)): float(r.rate)
        for r in rates.itertuples(index=False)
    }

    for wd in range(7):
        for hr in range(24):
            k_false = (wd, hr, False)
            k_true = (wd, hr, True)
            if k_true not in rate_table and k_false in rate_table:
                rate_table[k_true] = rate_table[k_false]

    return rate_table


def get_rate_table(
        holidays_list: List[date],
        force_rebuild: bool = False,
        xes_path: str = DEFAULT_XES_PATH,
        cache_path: str = DEFAULT_RATE_TABLE_PATH,
        recency_half_life_days: Optional[int] = DEFAULT_RECENCY_HALF_LIFE_DAYS,
) -> RateTable:
    effective_cache_path = cache_path
    if recency_half_life_days is not None and recency_half_life_days > 0:
        base, ext = os.path.splitext(cache_path)
        effective_cache_path = f"{base}_hl{recency_half_life_days}{ext}"

    if (not force_rebuild) and os.path.exists(effective_cache_path):
        with open(effective_cache_path, "rb") as f:
            return pickle.load(f)

    df = pm4py.convert_to_dataframe(pm4py.read_xes(xes_path))

    h_set = {h.date() if hasattr(h, "date") else h for h in holidays_list}
    rate_table = _build_rate_table_from_df(
        df,
        h_set,
        recency_half_life_days=recency_half_life_days,
    )

    os.makedirs(os.path.dirname(effective_cache_path), exist_ok=True)
    with open(effective_cache_path, "wb") as f:
        pickle.dump(rate_table, f)

    return rate_table
