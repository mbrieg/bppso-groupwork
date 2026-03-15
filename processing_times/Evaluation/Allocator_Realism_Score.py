import pandas as pd
import numpy as np

# Beispiel: empirical summary als dict
empirical = {
    "avg_events_per_case": 38.16,
    "median_case_duration_sec": 1649181.0,
    "p90_case_duration_sec": 2784843.0,   # Beispielwert einsetzen
    "share_cases_le_1s": 0.0,             # falls im empirical log praktisch keine instant cases
    "share_cases_gt_1day": 0.9924
}

# CSV mit deinen Simulationssummaries laden
df = pd.read_csv("all_simulation_summaries.csv")

def similarity_ratio(sim, emp):
    if emp == 0:
        return 1.0 if sim == 0 else 0.0
    return max(0.0, 1.0 - abs(sim - emp) / abs(emp))

def similarity_share(sim, emp):
    return max(0.0, 1.0 - abs(sim - emp))

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
        "S_events": s_events,
        "S_median": s_median,
        "S_p90": s_p90,
        "S_short": s_short,
        "S_long": s_long,
        "ARS": ars
    })

scores = df.apply(lambda row: compute_ars(row, empirical), axis=1)
result = pd.concat([df[["setup"]], scores], axis=1)
result = result.sort_values("ARS", ascending=False)

print(result)
result.to_csv("allocator_realism_score.csv", index=False)