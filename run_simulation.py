import sys
import os
import pm4py
sys.stdout.reconfigure(line_buffering=True)
current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(current_dir, "sim_core"))
sys.path.append(os.path.join(current_dir, "resources"))

from sim_core.engine import Engine
from resources.ResourceManager import ResourceManager
from decision_analysis.decision_point_manager import DecisionPointManager
from spawn_rates import StaticSpawner, AdvancedSpawner, get_rate_table, get_holidays


def run_system_test():
    print("\n" + "="*60)
    print("STARTING FULL SIMULATION")
    print("="*60)

    xes_path = os.path.join("data", "BPI Challenge 2017.xes.gz")
    bpmn_path = os.path.join("data", "process_model.bpmn")
    output_csv = "sim_output/final_simulation_log.csv"

    if not os.path.exists(xes_path):
        print(f"CRITICAL ERROR: Log file not found at {xes_path}")
        return

    log = pm4py.read_xes(xes_path) 
    if not isinstance(log, pm4py.objects.log.obj.EventLog):
        log = pm4py.convert_to_event_log(log) 
    print(f"      -> Loaded {len(log)} traces.")

    dp_manager = DecisionPointManager(log, 
                                      bpmn_path=bpmn_path,
                                      mode='basic') 

    pn_structure = dp_manager.get_pn_model()
    print(f"      -> Structure Ready: {len(pn_structure.place_ids)} places.")

    resource_manager = ResourceManager()

    #Spawn rates part
    holidays = get_holidays()                  # loads or generates NL holidays and caches them
    rate_table = get_rate_table(holidays)      # loads cached rate table or builds it once
    spawner = AdvancedSpawner(
        rate_table=rate_table,
        holidays=holidays,
        seed=42,
    )

    print(" Initializing Simulation Engine...")


    engine = Engine(
        pn=pn_structure,
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
        first_spawn_time = spawner.calculate_next_spawn(engine.now)
        engine.spawn(at_time=first_spawn_time)
        engine.run(max_events=2000)
        print("\nSuccess: Simulation finished.")

    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("-" * 30)
    print("EXPORTING RESULTS...")
    engine.export_log(output_csv)    # Generates csv
    engine.print_statistics()        # shows summary in console
    print("-" * 30)
    print(f"Results saved to: {output_csv}")
    
    print("-" * 30)


if __name__ == "__main__":
    run_system_test()
