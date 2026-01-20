import os
import pickle
import pandas as pd
import pm4py
from typing import Optional, List, Dict, Tuple

def _holidays_to_set(holidays: Optional[List]) -> set:
    return {h.date() if hasattr(h, "date") else h for h in (holidays or [])}


def _build_rate_table_from_df(df: pd.DataFrame, holidays: Optional[List]) -> Dict[Tuple[int, int, bool], float]:
    # Expect: case id + timestamp
    if "case:concept:name" not in df.columns:
        raise KeyError("Missing column 'case:concept:name' in dataframe.")
    if "time:timestamp" not in df.columns:
        raise KeyError("Missing column 'time:timestamp' in dataframe.")

    # Parse UTC and convert to NL local time for weekday/hour/holiday correctness
    df["timestamp"] = (
        pd.to_datetime(df["time:timestamp"], utc=True, errors="coerce")
        .dt.tz_convert("Europe/Amsterdam")
    )

    # Arrival time per case = first event
    arrivals = (
        df.groupby("case:concept:name")["timestamp"]
        .min()
        .reset_index()
    )

    arrivals["date"] = arrivals["timestamp"].dt.date
    arrivals["hour"] = arrivals["timestamp"].dt.hour
    arrivals["weekday"] = arrivals["timestamp"].dt.weekday

    holidays_set = _holidays_to_set(holidays)
    arrivals["is_holiday"] = arrivals["date"].isin(holidays_set) if holidays_set else False

    # Count arrivals per slot
    counts = (
        arrivals
        .groupby(["weekday", "hour", "is_holiday"])
        .size()
        .reset_index(name="arrival_count")
    )

    # Number of observed days per slot (robust normalization)
    slot_days = (
        arrivals
        .groupby(["weekday", "hour", "is_holiday"])["date"]
        .nunique()
        .reset_index(name="num_days")
    )

    merged = counts.merge(
        slot_days,
        on=["weekday", "hour", "is_holiday"],
        how="inner"
    )

    merged["lambda"] = merged["arrival_count"] / merged["num_days"]

    return merged.set_index(["weekday", "hour", "is_holiday"])["lambda"].to_dict()


def generate_rate_table(path: str, holidays: Optional[List] = None) -> Dict[Tuple[int, int, bool], float]:
    """
    Builds a rate table {(weekday, hour, is_holiday): lambda} from either:
    - CSV file containing 'case:concept:name' and 'time:timestamp'
    - XES or XES.GZ event log (loaded via pm4py)
    """

    ext = os.path.splitext(path.lower())[1]

    # Handle .xes.gz explicitly
    if path.lower().endswith(".xes.gz") or ext == ".xes":
        log = pm4py.read_xes(path)
        df = pm4py.convert_to_dataframe(log)
        return _build_rate_table_from_df(df, holidays)

    # Default: CSV
    df = pd.read_csv(path)
    return _build_rate_table_from_df(df, holidays)

def save_rate_table(rate_table: Dict[Tuple[int, int, bool], float], path: str) -> None:
    """Speichert die Rate Table einmalig auf die Platte."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(rate_table, f)


def load_rate_table(path: str) -> Dict[Tuple[int, int, bool], float]:
    """Laedt eine zuvor gespeicherte Rate Table."""
    with open(path, "rb") as f:
        return pickle.load(f)

def get_or_build_rate_table(
        xes_path: str,
        cache_path: str,
        holidays=None,
        force_rebuild: bool = False,
):
    """
    Baut die Rate Table genau einmal und cached sie.
    Bei weiteren Runs wird sie nur geladen.
    """
    if not force_rebuild and os.path.exists(cache_path):
        return load_rate_table(cache_path)

    rate_table = generate_rate_table(xes_path, holidays=holidays)
    save_rate_table(rate_table, cache_path)
    return rate_table
