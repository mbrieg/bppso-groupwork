import os
import pickle
from datetime import date
from typing import Dict, Tuple, List, Optional, Set

import pandas as pd
import pm4py

RateTable = Dict[Tuple[int, int, bool], float]
TZ_NAME = "Europe/Amsterdam"

DEFAULT_XES_PATH = "data/BPI Challenge 2017.xes.gz"
DEFAULT_HOLIDAYS_PATH = "spawn_rates/artifacts/holidays_nl.pkl"
DEFAULT_RATE_TABLE_PATH = "spawn_rates/artifacts/rate_table_nl_hourly.pkl"


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


def _build_rate_table_from_df(df: pd.DataFrame, holidays_set: Set[date]) -> RateTable:
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
    arrivals["date"] = arrivals["timestamp"].dt.date
    arrivals["weekday"] = arrivals["timestamp"].dt.weekday
    arrivals["hour"] = arrivals["timestamp"].dt.hour
    arrivals["is_holiday"] = arrivals["date"].isin(holidays_set) if holidays_set else False

    counts = arrivals.groupby(["weekday", "hour", "is_holiday"]).size()
    days = arrivals.groupby(["weekday", "hour", "is_holiday"])["date"].nunique()

    # lambda = average arrivals per observed day in that slot
    return (counts / days).to_dict()


def get_rate_table(
        holidays_list: List[date],
        force_rebuild: bool = False,
        xes_path: str = DEFAULT_XES_PATH,
        cache_path: str = DEFAULT_RATE_TABLE_PATH,
) -> RateTable:
    if (not force_rebuild) and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    df = pm4py.convert_to_dataframe(pm4py.read_xes(xes_path))

    h_set = {h.date() if hasattr(h, "date") else h for h in holidays_list}
    rate_table = _build_rate_table_from_df(df, h_set)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(rate_table, f)

    return rate_table
