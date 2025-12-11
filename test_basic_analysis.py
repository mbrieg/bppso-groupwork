from basic_decision_point_analysis import build_basic_branching_model

def main():
    
    bpmn_path = "/Users/zeynepcetin/Decision Point Analysis/data folder/BPI Challenge 2017 Loan Application Process-6-4.bpmn"
    xes_path = "/Users/zeynepcetin/Decision Point Analysis/data folder/BPI Challenge 2017.xes.gz"       

    print("Loading model and log...")
    model_data = build_basic_branching_model(bpmn_path, xes_path)

    print("\n=== Decision Places ===")
    for p in model_data["decision_places"]:
        # place object may not have a nice name, so use repr as fallback
        name = getattr(p, "name", repr(p))
        print(f"- Place: {name}")

    print("\n=== Branch Probabilities ===")
    branch_probs = model_data["branch_probabilities"]
    if not branch_probs:
        print("No branch probabilities were computed. Either the log does not contain any decision points, or mapping between model and log failed.")
    else:
        for p, probs in branch_probs.items():
            name = getattr(p, "name", repr(p))
            print(f"\nDecision place: {name}")
            for act, prob in probs.items():
                print(f"  -> {act}: {prob:.3f}")

if __name__ == "__main__":
    main()
