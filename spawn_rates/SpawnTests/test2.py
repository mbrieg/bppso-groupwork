from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from spawn_rates.advanced_spawner import AdvancedSpawner
from spawn_rates.rate_table import get_or_build_rate_table


def main():
    print("------------------------------")
    print("RUNNING SPAWN RATE TEST")
    print("------------------------------")

    tz = ZoneInfo("Europe/Amsterdam")

    log_path = "/Users/lucashi/TUM/Praktikum/Group Exercise/bppso-groupwork/data/BPI Challenge 2017.xes.gz"
    cache_path = "/Users/lucashi/TUM/Praktikum/Group Exercise/bppso-groupwork/spawn_rates/artifacts/rate_table_nl_hourly.pkl"

    print("Loading rate table...")
    rate_table = get_or_build_rate_table(
        xes_path=log_path,
        cache_path=cache_path,
        holidays=[]
    )

    print(f"Rate table size: {len(rate_table)}")

    spawner = AdvancedSpawner(
        rate_table=rate_table,
        holidays=[],
        seed=43
    )

    start_time = datetime(2017, 1, 2, 8, 15, tzinfo=tz)
    end_time = start_time + timedelta(days=2)

    print("\nGenerating spawn times...\n")

    t = start_time
    spawns = []

    while t < end_time and len(spawns) < 200:
        t = spawner.calculate_next_spawn(t)
        spawns.append(t)

    print(f"Generated {len(spawns)} spawns\n")

    print("First 20 spawn times:")
    for s in spawns[:20]:
        print(s)

    print("\nLast spawn time:")
    print(spawns[-1] if spawns else "None")

    print("\nDONE.")


if __name__ == "__main__":
    main()


