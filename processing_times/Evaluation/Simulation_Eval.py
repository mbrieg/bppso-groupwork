import pandas as pd
import numpy as np
from pathlib import Path
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
BASE_DIR = Path(__file__).resolve().parent
SIM_DIR = BASE_DIR / "simulation_results"

print("BASE_DIR:", BASE_DIR)
print("SIM_DIR:", SIM_DIR)


log = xes_importer.apply("/Users/simonimmler/PycharmProjects/Praktikum/data/BPI Challenge 2017.xes")
emp_df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

print(emp_df.columns)
print(emp_df.head())

RUNS = {
    "QR_Advanced": SIM_DIR / "QR_proc_ADVANCED.csv",
    "Basic_Advanced": SIM_DIR / "Basic_Total_ADVANCED.csv",
    "QR_Random": SIM_DIR / "QR_proc_Random.csv",
    "Basic_Random": SIM_DIR / "Basic_Total_Random.csv",
    "QR_ShortestQueue": SIM_DIR / "QR_proc_ShortestQueue.csv",
    "Basic_ShortestQueue": SIM_DIR / "Basic_Total_ShortestQueue.csv",
}

# =========================================
# LOADING
# =========================================
def load_log(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
    return df


# =========================================
# CASE-LEVEL STATS
# =========================================
def get_case_stats(df: pd.DataFrame) -> pd.DataFrame:
    case_col = "case:concept:name"

    case_times = (
        df.groupby(case_col)["time:timestamp"]
        .agg(case_start="min", case_end="max")
        .reset_index()
    )
    case_times["case_duration_sec"] = (
        case_times["case_end"] - case_times["case_start"]
    ).dt.total_seconds()

    events_per_case = (
        df.groupby(case_col)
        .size()
        .rename("events_per_case")
        .reset_index()
    )

    out = case_times.merge(events_per_case, on=case_col, how="left")
    return out


def summarize_log(df: pd.DataFrame, label: str) -> dict:
    case_stats = get_case_stats(df)

    lifecycle_counts = df["lifecycle:transition"].value_counts(dropna=False).to_dict()

    return {
        "setup": label,
        "n_events": len(df),
        "n_cases": df["case:concept:name"].nunique(),
        "n_activities": df["concept:name"].nunique(),
        "n_start": lifecycle_counts.get("start", 0),
        "n_complete": lifecycle_counts.get("complete", 0),
        "avg_events_per_case": case_stats["events_per_case"].mean(),
        "mean_case_duration_sec": case_stats["case_duration_sec"].mean(),
        "median_case_duration_sec": case_stats["case_duration_sec"].median(),
        "p90_case_duration_sec": case_stats["case_duration_sec"].quantile(0.90),
        "p95_case_duration_sec": case_stats["case_duration_sec"].quantile(0.95),
        "max_case_duration_sec": case_stats["case_duration_sec"].max(),
        "share_cases_le_1s": (case_stats["case_duration_sec"] <= 1).mean(),
        "share_cases_gt_1day": (case_stats["case_duration_sec"] > 86400).mean(),
        "time_min": df["time:timestamp"].min(),
        "time_max": df["time:timestamp"].max(),
    }


# =========================================
# COMPARISON AGAINST EMPIRICAL BASELINE
# =========================================
def compare_to_empirical(emp_summary: dict, sim_summary: dict) -> dict:
    metrics = [
        "avg_events_per_case",
        "mean_case_duration_sec",
        "median_case_duration_sec",
        "p90_case_duration_sec",
        "p95_case_duration_sec",
        "share_cases_le_1s",
        "share_cases_gt_1day",
    ]

    row = {"setup": sim_summary["setup"]}
    for m in metrics:
        emp = emp_summary[m]
        sim = sim_summary[m]
        row[f"{m}_emp"] = emp
        row[f"{m}_sim"] = sim
        row[f"{m}_abs_diff"] = sim - emp
        row[f"{m}_rel_diff"] = np.nan if emp == 0 else (sim - emp) / emp
    return row


# =========================================
# ACTIVITY SHARE COMPARISON
# =========================================
def activity_share_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    shares = (
        df["concept:name"]
        .value_counts(normalize=True)
        .rename(label)
        .reset_index()
    )
    shares.columns = ["concept:name", label]
    return shares


def compare_activity_shares(emp_df: pd.DataFrame, sim_df: pd.DataFrame, sim_label: str) -> pd.DataFrame:
    emp = activity_share_table(emp_df, "empirical_share")
    sim = activity_share_table(sim_df, f"{sim_label}_share")

    out = emp.merge(sim, on="concept:name", how="outer").fillna(0.0)
    out["abs_diff"] = (out[f"{sim_label}_share"] - out["empirical_share"]).abs()
    out = out.sort_values("abs_diff", ascending=False)
    return out


# =========================================
# OVERALL DISTANCE SCORE
# =========================================
def standardized_distance(emp_summary: dict, sim_summary: dict) -> float:
    """
    Smaller = closer to empirical log.
    """
    metrics = [
        "avg_events_per_case",
        "mean_case_duration_sec",
        "median_case_duration_sec",
        "p90_case_duration_sec",
        "p95_case_duration_sec",
        "share_cases_le_1s",
        "share_cases_gt_1day",
    ]

    dist = 0.0
    count = 0
    for m in metrics:
        emp = emp_summary[m]
        sim = sim_summary[m]
        if emp == 0:
            continue
        dist += abs(sim - emp) / abs(emp)
        count += 1
    return dist / count if count > 0 else np.nan


# =========================================
# RUN EVERYTHING
# =========================================

emp_summary = summarize_log(emp_df, "Empirical")

all_summaries = []
comparison_rows = []
distance_rows = []

for label, path in RUNS.items():
    sim_df = load_log(path)
    sim_summary = summarize_log(sim_df, label)
    all_summaries.append(sim_summary)

    comparison_rows.append(compare_to_empirical(emp_summary, sim_summary))

    distance_rows.append({
        "setup": label,
        "distance_to_empirical": standardized_distance(emp_summary, sim_summary)
    })

summary_df = pd.DataFrame(all_summaries)
comparison_df = pd.DataFrame(comparison_rows)
distance_df = pd.DataFrame(distance_rows).sort_values("distance_to_empirical")

print("\n=== EMPIRICAL SUMMARY ===")
print(pd.Series(emp_summary))

print("\n=== SIMULATION SUMMARIES ===")
print(summary_df.to_string(index=False))

print("\n=== COMPARISON TO EMPIRICAL ===")
print(comparison_df.to_string(index=False))

print("\n=== DISTANCE TO EMPIRICAL (SMALLER = BETTER) ===")
print(distance_df.to_string(index=False))

summary_df.to_csv(BASE_DIR/"all_simulation_summaries.csv", index=False)
comparison_df.to_csv(BASE_DIR/"comparison_to_empirical.csv", index=False)
distance_df.to_csv(BASE_DIR/"distance_to_empirical.csv", index=False)