import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from decision_analysis.DPManager import DPManager
from resources.ResourceManager import ResourceManager
from sim_core.engineOG import EngineOG
from sim_core.pn_model import wrap_net
from sim_core.bpmn_io import read_bpmn
from spawn_rates import AdvancedSpawner, get_rate_table, get_holidays
from processing_times.sampling import ProcessingTimeSampler


def run_test():
    print("Preparing Dataframes...")
    permissions_path = ''
    availabilities_path = ''

    print("Loading BPMN Model...")
    bpmn_net, initial_marking, final_marking = read_bpmn('data/process_model.bpmn')
    pn_model = wrap_net(bpmn_net, initial_marking, final_marking)
    
    print("Initializing Manager and Engine...")
    res_manager = ResourceManager()


    print("Initializing Decision Manager")
    rules_path = Path(__file__).resolve().parents[1] / "decision_analysis" / "decision_prob_rules.json"
    model_path = Path(__file__).resolve().parents[1] / "decision_analysis" / "simulation_brain.pkl"
    dp_manager = DPManager(pn=pn_model, mode="advanced",model_path= str(model_path),rules_path= str(rules_path))

    print("SpawnRates...")
    holidays = get_holidays()                  # loads or generates NL holidays and caches them
    rate_table = get_rate_table(holidays)      # loads cached rate table or builds it once
    spawner = AdvancedSpawner(
        rate_table=rate_table,
        holidays=holidays,
        seed=42,
    )
    root = Path(__file__).resolve().parents[1]  # ggf. anpassen (0/1/2)
    pt_dir = root / "processing_times"

    pt = ProcessingTimeSampler.from_paths(
        proc_json=pt_dir / "processing_models_proc.json",
        total_json=pt_dir / "processing_models_full_dur.json",
        wait_json=pt_dir / "wait_reference.json",
        seed=42,
        default_value=60.0
    )

    engine = EngineOG(pn_model, spawner, res_manager,dp_manager, pt_sampler=pt)

    print("Running Simulation...")
    engine.spawn()
    engine.run(max_events=50000)

    sim_log = pd.DataFrame(engine.log)
    print("\n--- Simulation Output ---")
    print(sim_log.head(30))

    #didnt wanna break the previous output
    #sim_log.to_csv("test_output.csv", index=False)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"test_output_advanced_{timestamp}.csv"

    sim_log.to_csv(output_filename, index=False)
    print("\nResults saved to 'test_output.csv'")


if __name__ == "__main__":
    run_test()
