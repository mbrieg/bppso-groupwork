import os
import pm4py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from decision_analysis.decision_point_manager import DecisionPointManager

def test_system():
    # 1. setup paths
    log_path = os.path.join("data", "BPI Challenge 2017.xes.gz")
    output_folder = "data_test_output"
    bpmn_path = os.path.join("data","process_model.bpmn")
    if not os.path.exists(bpmn_path):
        print(f" Error: BPMN file not found at {bpmn_path}")
        return
    if not os.path.exists(log_path):
        print(f" Error: Log file not found at {log_path}")
        return

    print("--- 1. LOADING LOG ---")
    log = pm4py.read_xes(log_path)

    if not isinstance(log, pm4py.objects.log.obj.EventLog):
        print("  Converting DataFrame to EventLog...")
        log = pm4py.convert_to_event_log(log)
        
    print(f"Log loaded: {len(log)} traces.")

    # 2. initialize the manager
    # trigger Discovery -> Structure Analysis -> Basic Training
    print("\n--- 2. INITIALIZING MANAGER ---")
    manager = DecisionPointManager(log, 
                                   bpmn_path=bpmn_path,
                                   mode='advanced', 
                                   output_folder=output_folder)

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
        
        if manager.mode == 'advanced' and hasattr(manager.router, 'classifiers'):
            # Check if a Tree exists for this specific decision point
            if place_name in manager.router.classifiers:
                print("    [Advanced] C4.5 Decision Tree successfully trained.")
                # Show which features (columns) the tree is using
                feats = manager.router.feature_names.get(place_name, [])
                print(f"    [Advanced] Features used: {feats}")
            else:
                print("    [Advanced] No specific ML model (Likely deterministic or not visited).")

        # Check if probabilities were learned
        if dp.probs_conditioned:
            print("    [B] Conditioned Probabilities learned (First-Hit)")
            # Show one example
            example_trigger = list(dp.probs_conditioned.keys())[0]
            print(f"   [B] Example: If input is '{example_trigger}' -> {dp.probs_conditioned[example_trigger]}")
        elif dp.probs_marginal:
            print("   [B] Only Marginal Probabilities learned.")
        else:
            print("   [B] No probabilities learned (Dead decision point?)")
        
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
            # 1. Get the technical name (Label or ID)
            trans_display_name = next_transition.label if next_transition.label else next_transition.name
            
            # 2. Find the Target Activity (Reverse Lookup)
            # we look through the dict: { "Activity Name": TransitionObj }
            target_activity = "Unknown Target"
            
            for act_name, trans_obj in test_dp.outgoing_transitions.items():
                if trans_obj == next_transition:
                    target_activity = act_name
                    break
          
            print(f" Manager Decision: Target Activity is '{target_activity}' --> Fire transition '{trans_display_name}'")
            
        else:
            print(" Manager returned None (Dead end or error)")

    print("\n System test completed.")

if __name__ == "__main__":
    test_system()