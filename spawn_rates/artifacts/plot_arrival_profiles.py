import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

BLUE = "#0065BD"
ORANGE = "#ff8800"
# File locations (relative to this script)
ARTIFACTS_DIR = os.path.dirname(os.path.abspath(__file__))
RATE_TABLE_PATH = os.path.join(ARTIFACTS_DIR, "rate_table_nl_hourly.pkl")

OUT_WEEKDAY_BAR = os.path.join(ARTIFACTS_DIR, "arrival_by_weekday.png")
OUT_HOURLY_HOLIDAY = os.path.join(ARTIFACTS_DIR, "hourly_arrivals_weekday_vs_holiday.png")


def main() -> None:
    if not os.path.exists(RATE_TABLE_PATH):
        raise FileNotFoundError(
            f"Rate table not found at {RATE_TABLE_PATH}. "
            "Generate it first (get_rate_table(..., force_rebuild=True)) or copy it into artifacts."
        )

    with open(RATE_TABLE_PATH, "rb") as f:
        rate_table = pickle.load(f)

    rows = [
        {"weekday": wd, "hour": hr, "holiday": is_h, "rate": float(rate)}
        for (wd, hr, is_h), rate in rate_table.items()
    ]
    df = pd.DataFrame(rows)

    # -------- Plot 1: Average arrival rate by weekday (non-holiday) --------
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    wd_profile = df[df["holiday"] == False].groupby("weekday")["rate"].mean().reindex(range(7))

    plt.figure()
    plt.bar(weekday_names, wd_profile.values, color=BLUE)
    plt.xlabel("Weekday")
    plt.ylabel("Average Arrivals per Hour")
    plt.title("Average Arrival Rates by Weekday (Non-Holiday)")
    plt.tight_layout()
    plt.savefig(OUT_WEEKDAY_BAR, dpi=300, bbox_inches="tight")
    plt.close()

    # -------- Plot 2: Hourly profile weekday vs holiday (Mon-Fri) --------
    weekday_df = df[(df["weekday"] < 5) & (df["holiday"] == False)]
    holiday_df = df[(df["weekday"] < 5) & (df["holiday"] == True)]

    weekday_profile = weekday_df.groupby("hour")["rate"].mean().reindex(range(24))
    holiday_profile = holiday_df.groupby("hour")["rate"].mean().reindex(range(24))

    plt.figure()
    plt.plot(weekday_profile.index, weekday_profile.values, label="Weekday", color=BLUE)
    plt.plot(
        holiday_profile.index,
        holiday_profile.values,
        label="Holiday",
        color=ORANGE,
    )
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Arrivals per Hour")
    plt.title("Hourly Arrival Rates: Weekday vs Holiday (Mon-Fri)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_HOURLY_HOLIDAY, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved plots:")
    print(" -", OUT_WEEKDAY_BAR)
    print(" -", OUT_HOURLY_HOLIDAY)


if __name__ == "__main__":
    main()
