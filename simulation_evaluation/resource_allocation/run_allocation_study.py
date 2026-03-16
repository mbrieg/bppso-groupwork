"""
run_allocation_study.py
-----------------------
Runs the simulation for each resource allocation method over N replications
and saves each run to:
    simulation_evaluation/resource_allocation/data/<method>/run_NN.csv

Usage:
    python simulation_evaluation/resource_allocation/run_allocation_study.py [--runs N] [--max-events M]

Defaults: 5 replications, 50 000 events per run.
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STUDY_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

import resources.ResourceAllocator as ra
from decision_analysis.DPManager import DPManager
from resources.ResourceManager import ResourceManager
from sim_core.engine import Engine
from sim_core.pn_model import wrap_net
from sim_core.bpmn_io import read_bpmn
from spawn_rates import AdvancedSpawner, get_rate_table, get_holidays
from processing_times.sampling import ProcessingTimeSampler

METHODS = {
    "random":         ra.Methods.RANDOM,
    "round_robin":    ra.Methods.ROUND_ROBIN,
    "shortest_queue": ra.Methods.SHORTEST_QUEUE,
    "batch_k5":       ra.Methods.BATCHING,
    "advanced_local":  ra.Methods.ADVANCED_LOCAL,
    "advanced_global": ra.Methods.ADVANCED_GLOBAL,
}

METHOD_DELTAS = {
    "advanced_local": 5,
    "advanced_global": 10,
}

OUT_ROOT = STUDY_DIR / "data"

# Fixed paths (same for all runs)
BPMN_PATH  = PROJECT_ROOT / "data" / "process_model.bpmn"
RULES_PATH = PROJECT_ROOT / "decision_analysis" / "decision_prob_rules.json"
PROC_JSON  = PROJECT_ROOT / "processing_times" / "Basic_Models" / "processing_models_proc.json"
TOTAL_JSON = PROJECT_ROOT / "processing_times" / "Basic_Models" / "processing_models_full_dur.json"
WAIT_JSON  = PROJECT_ROOT / "processing_times" / "Basic_Models" / "wait_reference.json"
AVAIL_FILE = "availabilities_advanced.csv"
START_TIME = datetime(2016, 5, 17, 9, 15, 0)
SIM_DAYS   = 14  # simulated calendar days — same horizon for all methods


def build_engine(method: ra.Methods, seed: int, delta: int = 1) -> Engine:
    bpmn_net, initial_marking, final_marking = read_bpmn(str(BPMN_PATH))
    pn_model = wrap_net(bpmn_net, initial_marking, final_marking)

    holidays   = get_holidays()
    rate_table = get_rate_table(holidays)
    spawner    = AdvancedSpawner(rate_table=rate_table, holidays=holidays, seed=seed)

    res_manager = ResourceManager(
        permissions=str("role_permissions.csv"),
        availabilities=AVAIL_FILE,
        method=method,
        delta=delta,
        batch_k=5,
    )
    dp_manager = DPManager(
        pn=pn_model,
        mode="basic",
        rules_path=str(RULES_PATH),
    )
    pt = ProcessingTimeSampler.from_paths(
        proc_json=str(PROC_JSON),
        total_json=str(TOTAL_JSON),
        wait_json=str(WAIT_JSON),
        seed=seed,
        default_value=60.0,
    )

    return Engine(
        pn=pn_model,
        spawner=spawner,
        resource_manager=res_manager,
        decision_manager=dp_manager,
        pt_sampler=pt,
        start_time=START_TIME,
        pt_use_qr=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",       type=int, default=5,   help="Replications per method")
    parser.add_argument(
        "--run-indices",
        nargs="+",
        type=int,
        help="Explicit run indices to execute (e.g. --run-indices 1 2 3 4)",
    )
    parser.add_argument("--days",        type=int, default=SIM_DAYS, help="Simulated calendar days per run (same horizon for all methods)")
    parser.add_argument("--max-events", type=int, default=2_000_000, help="Hard safety cap; should never be hit under normal use")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip run_NN.csv files that already exist",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS.keys()),
        help="Run only selected methods (default: all methods)",
    )
    args = parser.parse_args()

    selected_methods = {name: METHODS[name] for name in (args.methods or METHODS.keys())}
    run_indices = args.run_indices if args.run_indices is not None else list(range(args.runs))

    total = len(selected_methods) * len(run_indices)
    done  = 0

    for method_name, method_enum in selected_methods.items():
        out_dir = OUT_ROOT / method_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for run_idx in run_indices:
            seed = 1000 * (list(METHODS).index(method_name) + 1) + run_idx
            out_path = out_dir / f"run_{run_idx:02d}.csv"
            if args.skip_existing and out_path.exists():
                print(f"\n[skip] method={method_name}  run={run_idx:02d}  existing={out_path.relative_to(PROJECT_ROOT)}", flush=True)
                continue

            done += 1
            print(f"\n[{done}/{total}] method={method_name}  run={run_idx:02d}  seed={seed}", flush=True)

            delta = METHOD_DELTAS.get(method_name, 1)
            engine = build_engine(method_enum, seed, delta=delta)
            engine.spawn()
            sim_end = START_TIME + timedelta(days=args.days)
            engine.run(max_events=args.max_events, end_time=sim_end)

            df = pd.DataFrame(engine.log)
            df.to_csv(out_path, index=False)
            print(f"  → saved {len(df):,} events to {out_path.relative_to(PROJECT_ROOT)}", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
