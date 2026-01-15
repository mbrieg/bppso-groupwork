import sys
import os
import pm4py

current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(current_dir, "sim_core"))
sys.path.append(os.path.join(current_dir, "resources"))

from sim_core.engine import Engine
from sim_core.pn_model import wrap_net
from resources.ResourceManager import ResourceManager
from decision_analysis.decision_point_manager import DecisionPointManager

def run_system_test():
    print("\n" + "="*60)
    print("STARTING FULL SIMULATION (NEW STRUCTURE)")
    print("="*60)

    xes_path = os.path.join("data", "BPI Challenge 2017.xes.gz")
    output_csv = "final_simulation_log.csv"

    if not os.path.exists(xes_path):
        print(f"CRITICAL ERROR: Log file not found at {xes_path}")
        return

    log = pm4py.read_xes(xes_path)
    print(f"      -> Loaded {len(log)} traces.")

    dp_manager = DecisionPointManager(log, mode='advanced') 

    pn_structure = wrap_net(dp_manager.net, dp_manager.im, dp_manager.fm)
    print(f"      -> Structure Ready: {len(pn_structure.place_ids)} places.")

    resource_manager = ResourceManager()

    print(" Initializing Simulation Engine...")
    engine = Engine(
        pn=pn_structure,
        resource_manager=resource_manager,
        decision_manager=dp_manager,  
        max_cases=100              
    )

    print("\n" + "-"*30)
    print("RUNNING SIMULATION...")
    print("-"*30)

    try:
        engine.spawn() 
        engine.run(max_events=5000)
        print("\nSuccess: Simulation finished.")

    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("-" * 30)