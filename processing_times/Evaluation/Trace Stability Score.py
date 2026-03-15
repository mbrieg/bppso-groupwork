import pandas as pd
import numpy as np

# --------------------------------------------------
# INPUT
# --------------------------------------------------
df = pd.read_csv("all_simulation_summaries.csv")

# empirical summary from your earlier code
empirical = {
    "avg_events_per_case": 38.16,
    "median_case_duration_sec": 1649181.0,
    "p90_case_duration_sec": 2784843.0,   # replace if needed
    "share_cases_le_1s": 0.0,
    "share_cases_gt_1day": 0.9924
}

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def similarity_ratio(sim, emp):
    if emp == 0:
        return 1.0 if sim == 0 else 0.0
    return max(0.0, 1.0 - abs(sim - emp) / abs(emp))

def similarity_share(sim, emp):
    return max(0.0, 1.0 - abs(sim - emp))

# --------------------------------------------------
# ARS: Allocator Realism Score
# --------------------------------------------------
def compute_ars(row, emp):
    s_events = similarity_ratio(row["avg_events_per_case"], emp["avg_events_per_case"])
    s_median = similarity_ratio(row["median_case_duration_sec"], emp["median_case_duration_sec"])
    s_p90 = similarity_ratio(row["p90_case_duration_sec"], emp["p90_case_duration_sec"])
    s_short = similarity_share(row["share_cases_le_1s"], emp["share_cases_le_1s"])
    s_long = similarity_share(row["share_cases_gt_1day"], emp["share_cases_gt_1day"])

    ars = (
        0.25 * s_events +
        0.25 * s_median +
        0.20 * s_p90 +
        0.15 * s_short +
        0.15 * s_long
    )

    return pd.Series({
        "S_events_emp": s_events,
        "S_median_emp": s_median,
        "S_p90_emp": s_p90,
        "S_short_emp": s_short,
        "S_long_emp": s_long,
        "ARS": ars
    })

# --------------------------------------------------
# TSS: Trace Stability Score
# --------------------------------------------------
def compute_tss(row, target_events=38.16, tail_target=15.0):
    # 1) enough events per case
    s_events = min(1.0, row["avg_events_per_case"] / target_events)

    # 2) avoid instant cases
    s_short = 1.0 - row["share_cases_le_1s"]

    # 3) avoid too many >1 day cases
    s_long = 1.0 - row["share_cases_gt_1day"]

    # 4) moderate upper tail
    median = max(row["median_case_duration_sec"], 1.0)
    tail_ratio = row["p95_case_duration_sec"] / median
    s_tail = min(1.0, tail_target / tail_ratio)

    tss = (
        0.30 * s_events +
        0.25 * s_short +
        0.25 * s_long +
        0.20 * s_tail
    )

    return pd.Series({
        "S_events_stab": s_events,
        "S_short_stab": s_short,
        "S_long_stab": s_long,
        "tail_ratio": tail_ratio,
        "S_tail_stab": s_tail,
        "TSS": tss
    })

# --------------------------------------------------
# RUN
# --------------------------------------------------
ars_scores = df.apply(lambda row: compute_ars(row, empirical), axis=1)
tss_scores = df.apply(lambda row: compute_tss(row, target_events=empirical["avg_events_per_case"], tail_target=15.0), axis=1)

result = pd.concat([df[["setup"]], ars_scores, tss_scores], axis=1)

# Optional combined score
result["CombinedScore"] = 0.5 * result["ARS"] + 0.5 * result["TSS"]

result = result.sort_values("CombinedScore", ascending=False)

print(result[[
    "setup", "ARS", "TSS", "CombinedScore",
    "tail_ratio", "S_events_emp", "S_short_stab", "S_long_stab"
]])

result.to_csv("allocator_scores.csv", index=False)