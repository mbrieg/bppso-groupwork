import os
import pandas as pd
from datetime import datetime
from pathlib import Path

import resources.ResourceAllocator
from decision_analysis.DPManager import DPManager
from resources.ResourceManager import ResourceManager
from sim_core.engine import Engine
from sim_core.pn_model import wrap_net
from sim_core.bpmn_io import read_bpmn
from spawn_rates import AdvancedSpawner, get_rate_table, get_holidays
from processing_times.sampling import ProcessingTimeSampler

project_root = Path(__file__).resolve().parents[0]


def main():

    # Define paths
    bpmn_path = os.path.join(project_root, "data", "process_model.bpmn")
    # output_csv = "sim_output/final_simulation_log.csv"
    rules_path = os.path.join(project_root, "decision_analysis", "decision_prob_rules.json")
    model_path = os.path.join(project_root, "decision_analysis", "simulation_brain.pkl")
    proc_json = os.path.join(project_root, "processing_times", "Basic_Models/processing_models_proc.json")
    total_json = os.path.join(project_root, "processing_times", "Basic_Models/processing_models_full_dur.json")
    wait_json = os.path.join(project_root, "processing_times", "Basic_Models/wait_reference.json")
    qr_joblib = os.path.join(project_root, "processing_times", "Advanced_Model/proc_qr_bundle.joblib")

    print("Starting Setup...")

    # Load bpmn model
    bpmn_net, initial_marking, final_marking = read_bpmn(bpmn_path)
    pn_model = wrap_net(bpmn_net, initial_marking, final_marking)

    # Load managers
    holidays = get_holidays()  # loads or generates NL holidays and caches them
    rate_table = get_rate_table(holidays)  # loads cached rate table or builds it once
    spawner = AdvancedSpawner(
        rate_table=rate_table,
        holidays=holidays,
        seed=42,
    )
    avail_file = 'availabilities_advanced.csv'
    #avail_file = 'availabilities_9to5.csv'
    allocation_method = resources.ResourceAllocator.Methods.ADVANCED_GLOBAL
    res_manager = ResourceManager(permissions='role_permissions.csv', availabilities=avail_file, method=allocation_method, delta=10 , batch_k=5)
    dp_manager = DPManager(pn=pn_model, mode="advanced", model_path=str(model_path), rules_path=str(rules_path))
    pt = ProcessingTimeSampler.from_paths(
        proc_json=proc_json,
        total_json=total_json,
        qr_joblib={"proc": qr_joblib},
        wait_json=wait_json,
        seed=42,
        default_value=60.0
    )

    # Initialise engine
    start_time = datetime(2016, 5, 17, 9, 15, 0)
    engine = Engine(pn=pn_model,
                    spawner=spawner,
                    resource_manager=res_manager,
                    decision_manager=dp_manager,
                    pt_sampler=pt,
                    start_time=start_time,
                    pt_use_qr=True)
    print("...Setup complete")

    # Run simulation
    print("\n" + "=" * 60)
    print("STARTING SIMULATION")
    print("=" * 60)
    engine.spawn()
    #In order to mock the event log, I increased the num of events to 475 306, you can change again as you want
    engine.run(max_events=100000)
    sim_log = pd.DataFrame(engine.log)
    dp_manager.print_routing_stats()

    #output_csv = "decision_analysis/sim_output_advanced.csv"
    #output_csv = "simulation_evaluation/results/sim_output_9to5.csv"
    output_csv = ("processing_times/Evaluation/simulation_results/QR_proc_ADVANCED.csv")
    print("\n--- Simulation Output ---")
    print(sim_log.head(300))
    sim_log.to_csv(output_csv, index=False, chunksize=300)
    print(f"\nsimulation_results saved to '{output_csv}'")


if __name__ == "__main__":
    main()
