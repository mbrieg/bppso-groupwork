from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

# Falls die CSVs direkt im gleichen Ordner liegen:
INPUT_DIR = BASE_DIR

# Falls sie in einem Unterordner liegen, stattdessen z.B.:
# INPUT_DIR = BASE_DIR / "simulation_results"

OUTPUT_DIR = BASE_DIR / "report_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

SUMMARIES_PATH = INPUT_DIR / "all_simulation_summaries.csv"
SCORES_PATH = INPUT_DIR / "allocator_scores.csv"
DIST_PATH = INPUT_DIR / "distance_to_empirical.csv"

# =========================================================
# LOAD DATA
# =========================================================
summ = pd.read_csv(SUMMARIES_PATH)
scores = pd.read_csv(SCORES_PATH)
dist = pd.read_csv(DIST_PATH)

# =========================================================
# PREP
# =========================================================
def split_setup(setup: str):
    if setup.startswith("QR_"):
        return "QR proc", setup.replace("QR_", "")
    elif setup.startswith("Basic_"):
        return "Basic total", setup.replace("Basic_", "")
    return "Other", setup

summ[["model", "allocator"]] = summ["setup"].apply(lambda x: pd.Series(split_setup(x)))
scores[["model", "allocator"]] = scores["setup"].apply(lambda x: pd.Series(split_setup(x)))
dist[["model", "allocator"]] = dist["setup"].apply(lambda x: pd.Series(split_setup(x)))

allocator_order = ["Advanced", "Random", "ShortestQueue"]

summ["allocator"] = pd.Categorical(summ["allocator"], categories=allocator_order, ordered=True)
scores["allocator"] = pd.Categorical(scores["allocator"], categories=allocator_order, ordered=True)
dist["allocator"] = pd.Categorical(dist["allocator"], categories=allocator_order, ordered=True)

summ = summ.sort_values(["allocator", "model"])
scores = scores.sort_values(["allocator", "model"])
dist = dist.sort_values(["allocator", "model"])

# helper columns
summ["mean_hours"] = summ["mean_case_duration_sec"] / 3600.0
summ["median_hours"] = summ["median_case_duration_sec"] / 3600.0
summ["p95_hours"] = summ["p95_case_duration_sec"] / 3600.0
summ["share_cases_le_1s_pct"] = 100 * summ["share_cases_le_1s"]
summ["share_cases_gt_1day_pct"] = 100 * summ["share_cases_gt_1day"]

# =========================================================
# PLOTTING HELPER
# =========================================================
def grouped_bar_plot(df, value_col, ylabel, title, filename, use_log=False):
    pivot = df.pivot(index="allocator", columns="model", values=value_col).reindex(allocator_order)
    ax = pivot.plot(kind="bar", figsize=(8, 5))
    ax.set_title(title)
    ax.set_xlabel("Resource allocator")
    ax.set_ylabel(ylabel)
    if use_log:
        ax.set_yscale("log")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

# =========================================================
# 1) EVENTS PER CASE
# =========================================================
grouped_bar_plot(
    summ,
    "avg_events_per_case",
    "Average events per case",
    "Average events per case by allocator and processing-time setup",
    "events_per_case.png"
)

# =========================================================
# 2) MEDIAN CASE DURATION
# =========================================================
grouped_bar_plot(
    summ,
    "median_hours",
    "Median case duration (hours)",
    "Median case duration by allocator and processing-time setup",
    "median_case_duration_hours.png"
)

# =========================================================
# 3) P95 CASE DURATION (LOG SCALE)
# =========================================================
grouped_bar_plot(
    summ,
    "p95_hours",
    "P95 case duration (hours, log scale)",
    "P95 case duration by allocator and processing-time setup",
    "p95_case_duration_hours_log.png",
    use_log=True
)

# =========================================================
# 4) SHARE OF INSTANT CASES
# =========================================================
grouped_bar_plot(
    summ,
    "share_cases_le_1s_pct",
    "Share of cases ≤ 1 second (%)",
    "Share of near-instantaneous cases by allocator and setup",
    "share_cases_le_1s.png"
)

# =========================================================
# 5) SHARE OF LONG CASES
# =========================================================
grouped_bar_plot(
    summ,
    "share_cases_gt_1day_pct",
    "Share of cases > 1 day (%)",
    "Share of long-running cases by allocator and setup",
    "share_cases_gt_1day.png"
)

# =========================================================
# 6) COMBINED SCORE
# =========================================================
score_plot = scores.copy()
score_plot["label"] = score_plot["allocator"].astype(str) + " | " + score_plot["model"]

score_plot = score_plot.sort_values("CombinedScore", ascending=False)

plt.figure(figsize=(9, 5))
plt.bar(score_plot["label"], score_plot["CombinedScore"])
plt.title("Combined score across all setups")
plt.ylabel("Combined score")
plt.xlabel("Setup")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "combined_score.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 7) ARS AND TSS SEPARATELY
# =========================================================
plt.figure(figsize=(9, 5))
plt.bar(score_plot["label"], score_plot["ARS"])
plt.title("Allocator Realism Score (ARS)")
plt.ylabel("ARS")
plt.xlabel("Setup")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ars_score.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(9, 5))
plt.bar(score_plot["label"], score_plot["TSS"])
plt.title("Trace Stability Score (TSS)")
plt.ylabel("TSS")
plt.xlabel("Setup")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tss_score.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 8) DISTANCE TO EMPIRICAL
# =========================================================
dist_plot = dist.copy()
dist_plot["label"] = dist_plot["allocator"].astype(str) + " | " + dist_plot["model"]
dist_plot = dist_plot.sort_values("distance_to_empirical", ascending=True)

plt.figure(figsize=(9, 5))
plt.bar(dist_plot["label"], dist_plot["distance_to_empirical"])
plt.title("Distance to empirical log")
plt.ylabel("Distance to empirical (lower is better)")
plt.xlabel("Setup")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "distance_to_empirical.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 9) HEATMAP OF KEY METRICS
# =========================================================
import numpy as np

# Merge all relevant metrics into one table
heat = summ.merge(
    scores[["setup", "ARS", "TSS", "CombinedScore"]],
    on="setup",
    how="left"
).merge(
    dist[["setup", "distance_to_empirical"]],
    on="setup",
    how="left"
)

heat["label"] = heat["allocator"].astype(str) + " | " + heat["model"]

# Select metrics for heatmap
heat_metrics = heat[[
    "label",
    "avg_events_per_case",
    "median_hours",
    "p95_hours",
    "share_cases_le_1s_pct",
    "share_cases_gt_1day_pct",
    "ARS",
    "TSS",
    "CombinedScore",
    "distance_to_empirical"
]].copy()

# Set label as index
heat_metrics = heat_metrics.set_index("label")

# Metrics where higher is better
higher_better = [
    "avg_events_per_case",
    "ARS",
    "TSS",
    "CombinedScore"
]

# Metrics where lower is better
lower_better = [
    "median_hours",              # optional: if you want shorter typical cases to be better
    "p95_hours",
    "share_cases_le_1s_pct",
    "share_cases_gt_1day_pct",
    "distance_to_empirical"
]

# Min-max normalization to [0,1]
heat_norm = heat_metrics.copy()

for col in heat_norm.columns:
    col_min = heat_norm[col].min()
    col_max = heat_norm[col].max()

    if col_max == col_min:
        heat_norm[col] = 1.0
    else:
        heat_norm[col] = (heat_norm[col] - col_min) / (col_max - col_min)

# Invert metrics where lower is better
for col in lower_better:
    if col in heat_norm.columns:
        heat_norm[col] = 1.0 - heat_norm[col]

# Plot heatmap
plt.figure(figsize=(11, 5.5))
plt.imshow(heat_norm.values, aspect="auto")
plt.colorbar(label="Normalized score")
plt.xticks(range(len(heat_norm.columns)), heat_norm.columns, rotation=35, ha="right")
plt.yticks(range(len(heat_norm.index)), heat_norm.index)
plt.title("Heatmap of key evaluation metrics across all setups")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "evaluation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

# Also save raw and normalized versions
heat_metrics.to_csv(OUTPUT_DIR / "evaluation_heatmap_raw.csv")
heat_norm.to_csv(OUTPUT_DIR / "evaluation_heatmap_normalized.csv")

print("Saved heatmap:", OUTPUT_DIR / "evaluation_heatmap.png")
# =========================================================
# 10) ANNOTATED HEATMAP
# =========================================================
fig, ax = plt.subplots(figsize=(11, 5.5))
im = ax.imshow(heat_norm.values, aspect="auto")

# ticks
ax.set_xticks(range(len(heat_norm.columns)))
ax.set_xticklabels(heat_norm.columns, rotation=35, ha="right")
ax.set_yticks(range(len(heat_norm.index)))
ax.set_yticklabels(heat_norm.index)

# annotate normalized values
for i in range(heat_norm.shape[0]):
    for j in range(heat_norm.shape[1]):
        val = heat_norm.iloc[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Normalized score (higher = better)")

ax.set_title("Annotated heatmap of key evaluation metrics")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "evaluation_heatmap_annotated.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved annotated heatmap:", OUTPUT_DIR / "evaluation_heatmap_annotated.png")
# =========================================================
# OPTIONAL: EXPORT A SMALL PLOT SUMMARY TABLE
# =========================================================
plot_summary = summ[[
    "setup", "avg_events_per_case", "mean_hours", "median_hours",
    "p95_hours", "share_cases_le_1s_pct", "share_cases_gt_1day_pct"
]].copy()

plot_summary.to_csv(OUTPUT_DIR / "plot_summary.csv", index=False)

print("Done. Figures saved in:", OUTPUT_DIR)
for p in sorted(OUTPUT_DIR.iterdir()):
    print(" -", p.name)