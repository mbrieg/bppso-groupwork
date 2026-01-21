import sys
import os
import pm4py

from sim_core import engineOLD, pn_model, bpmn_io
from resources.ResourceManager import ResourceManager
from decision_analysis.decision_point_manager import DecisionPointManager
from spawn_rates import StaticSpawner, AdvancedSpawner, get_rate_table, get_holidays

sys.stdout.reconfigure(line_buffering=True)
current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(current_dir, "sim_core"))
sys.path.append(os.path.join(current_dir, "resources"))


def main():
    print("\n" + "="*60)
    print("STARTING SIMULATION")
    print("="*60)

    xes_path = os.path.join("data", "BPI Challenge 2017.xes.gz")
    bpmn_path = os.path.join("data", "process_model.bpmn")
    output_csv = "sim_output/final_simulation_log.csv"

    if not os.path.exists(xes_path):
        print(f"CRITICAL ERROR: Log file not found at {xes_path}")
        return

    print(" Initializing Simulation Engine...")
    dp_manager = DecisionPointManager(bpmn_path=bpmn_path, mode='basic')
    resource_manager = ResourceManager()
    # TODO spawner refactoring
    holidays = get_holidays()                  # loads or generates NL holidays and caches them
    rate_table = get_rate_table(holidays)      # loads cached rate table or builds it once
    spawner = AdvancedSpawner(
        rate_table=rate_table,
        holidays=holidays,
        seed=42,
    )

    bpmn_net, initial_marking, final_marking = bpmn_io.read_bpmn(bpmn_path)
    pn = pn_model.wrap_net(bpmn_net, initial_marking, final_marking)
    eng = engine.Engine(
        pn=pn,
        spawner=spawner,
        resource_manager=resource_manager,
        decision_manager=dp_manager,  
        max_cases=5              
    )

    print("\n" + "-"*30)
    print("RUNNING SIMULATION...")
    print("-"*30)

    try:
        # schedule first case using the spawner
        first_spawn_time = spawner.calculate_next_spawn(eng.now)
        eng.spawn(at_time=first_spawn_time)
        eng.run(max_events=2000)
        print("\nSuccess: Simulation finished.")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("-" * 30)
    print("EXPORTING RESULTS...")
    eng.export_log(output_csv)    # Generates csv
    eng.print_statistics()        # shows summary in console
    print("-" * 30)
    print(f"Results saved to: {output_csv}")
    print("-" * 30)


if __name__ == "__main__":
    main()
