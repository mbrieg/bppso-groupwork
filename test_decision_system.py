import os
import pm4py
from decision_analysis.decision_point_manager import DecisionPointManager

def test_system():
    # 1. setup paths
    log_path = os.path.join("data", "BPI Challenge 2017.xes.gz")
    output_folder = "data_test_output"

    if not os.path.exists(log_path):
        print(f" Error: Log file not found at {log_path}")
        return

    print("--- 1. LOADING LOG ---")
    log = pm4py.read_xes(log_path)
    print(f"Log loaded: {len(log)} traces.")

    # 2. initialize the manager
    # trigger Discovery -> Structure Analysis -> Basic Training
    print("\n--- 2. INITIALIZING MANAGER ---")
    manager = DecisionPointManager(log, mode='basic', output_folder=output_folder)

    # 3. inspect decision points
    print("\n--- 3. INSPECTING DECISION POINTS ---")
    dps = manager.decision_points
    print(f"Total Decision Points Found: {len(dps)}")
    
    # Let's peek at the first 3 decision points to see if they learned anything
    count = 0
    for place_name, dp in dps.items():
        print(f"\n Decision Point: {place_name}")
        print(f"   Real Inputs (Context): {dp.incoming_activities}")
        print(f"   Possible Outputs: {dp.get_possible_activities()}")
        
        # Check if probabilities were learned
        if dp.probs_conditioned:
            print("    Conditioned Probabilities learned (First-Hit)")
            # Show one example
            example_trigger = list(dp.probs_conditioned.keys())[0]
            print(f"   Example: If input is '{example_trigger}' -> {dp.probs_conditioned[example_trigger]}")
        elif dp.probs_marginal:
            print("    Only Marginal Probabilities learned.")
        else:
            print("    No probabilities learned (Dead decision point?)")
        
        count += 1
        if count >= 3: break

    # 4. simulation a prediction
    print("\n--- 4. SIMULATING A PREDICTION ---")
    # Pick a random decision point to test
    if dps:
        test_dp_name = list(dps.keys())[0]
        test_dp = dps[test_dp_name]
        
        # Fake a "Previous Activity"
        if test_dp.incoming_activities:
            fake_prev_act = list(test_dp.incoming_activities)[0]
        else:
            fake_prev_act = "Unknown Activity"
            
        print(f"Testing Prediction at {test_dp_name} coming from '{fake_prev_act}'...")
        
        # we need the actual Place object for the manager function
        test_place_obj = test_dp.petri_place
        
        # ask the DPManager
        next_transition = manager.get_next_transition(test_place_obj, {'prev_activity': fake_prev_act})
        
        if next_transition:
            print(f" Manager Decision: Fire transition '{next_transition.label}'")
        else:
            print(" Manager returned none (Dead end or error)")

    print("\n System test completed.")

if __name__ == "__main__":
    test_system()