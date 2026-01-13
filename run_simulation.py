import sys
import os
import pickle
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sim_core_dir = os.path.join(current_dir, "sim_core")
basic_dir = os.path.join(current_dir, "basic")
advanced_dir = os.path.join(current_dir, "advanced")

sys.path.append(sim_core_dir)
sys.path.append(basic_dir)
sys.path.append(advanced_dir)

try:
    import c45_tree
except ImportError:
    try:
        from advanced import c45_tree
        sys.modules['c45_tree'] = c45_tree
    except ImportError:
        print("Error: c45_tree.py not found in advanced")

from sim_core.engine import Engine
from sim_core.pn_model import wrap_net
from basic_decision_point_analysis import load_petri_from_bpmn

# Integration Test

def run_system_test():
    print("Start Full Simulation (Advanced Mode)")

    bpmn_path = os.path.join("data", "BPI Challenge 2017 Loan Application Process-6-4.bpmn")
    basic_model_path = "basic_routing_model.pkl"
    adv_model_path = "decision_models.pkl"
    output_csv = "final_simulation_log.csv"

    if not os.path.exists(bpmn_path):
        print(f"Error: Bpmn file not found at {bpmn_path}")
        return
    
    print("Loading bpmn model..")
    net, im, fm = load_petri_from_bpmn(bpmn_path)


    pn_structure = wrap_net(net, im, fm)
    print(f" Model loaded: {len(pn_structure.place_ids)} places, {len(pn_structure.trans_ids)} transitions.")

    print("Loading Decision models..")

    basic_model = None
    if os.path.exists(basic_model_path):
        with open(basic_model_path, "rb") as f:
            basic_model = pickle.load(f)
        print(" Basic routing model loaded.")
    else:
        print(" Basic Routing model not found.")

    adv_model = None
    if os.path.exists(adv_model_path):
        with open(adv_model_path, "rb") as f:
            adv_model = pickle.load(f)
        print(f" Advanced Decision Trees loaded.")
    else:
        print(" Advanced Decision Trees not loaded. Engine will use Basic/Random)")

    print("Simulation Engine..")

    engine = Engine(
        pn=pn_structure,
        mode="advanced",
        basic_model=basic_model,
        advanced_model=adv_model,
        max_cases=100
    )

    print("Run Simulation")

    try:
        engine.spawn()
        engine.run(max_events=5000)
        print("Simulation finished successfully.")

    except Exception as e:
        print(f" Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("Export Results..")
    engine.export_log(output_csv)
    engine.print_statistics()

    print(f"Done '{output_csv}' for results.")

if __name__ == "__main__":
    run_system_test()