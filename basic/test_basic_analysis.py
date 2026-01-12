import math
from collections import defaultdict
from pm4py.objects.log.importer.xes import importer as xes_importer

from basic_decision_point_analysis import (
    build_basic_branching_model, 
    compute_branch_counts_first_hit, 
    compute_branch_probabilities
)

def evaluate_logloss_conditioned_vs_marginal(
    log,
    decision_places,
    place_preset_labels,
    place_postset_labels,
    cond_probs,      # P(out | place, prev)
    marginal_probs,  # P(out | place)
    horizon=60,
    min_p=1e-12
):
    """
    Calculates the Log-Loss metric to quantify how much better the 
    Conditioned model is compared to the Marginal model.
    """
    # Index triggers
    preset_index = defaultdict(list)
    for p in decision_places:
        for a in place_preset_labels.get(p, set()):
            preset_index[a].append(p)

    postset_map = {p: set(place_postset_labels.get(p, set())) for p in decision_places}

    n = 0
    ll_marg = 0.0
    ll_cond = 0.0

    for trace in log:
        acts = [str(ev["concept:name"]).strip() for ev in trace]
        
        # Track state for multiple potential parallel episodes
        open_since = {p: None for p in decision_places}
        prev_act_for_open = {p: None for p in decision_places} 

        for i, act in enumerate(acts):
            # 1. Open episodes (Triggering)
            for p in preset_index.get(act, []):
                if open_since[p] is None:
                    open_since[p] = i
                    prev_act_for_open[p] = act

            # 2. Check outcomes
            for p in decision_places:
                start_i = open_since[p]
                if start_i is None:
                    continue

                # Horizon Check
                if (i - start_i) > horizon:
                    open_since[p] = None
                    prev_act_for_open[p] = None
                    continue

                if i == start_i:
                    continue

                # Outcome Hit
                if act in postset_map[p]:
                    outcome = act
                    prev_key = prev_act_for_open[p]

                    # A) Baseline Probability
                    pm = marginal_probs.get(p, {}).get(outcome, 0.0)
                    pm = max(pm, min_p) # Avoid log(0)

                    # B) Conditioned Probability
                    # Use fallback to marginal if this specific path (trigger -> place) was never seen in training
                    pc = cond_probs.get(p, {}).get(prev_key, {}).get(outcome, 0.0)
                    if pc <= 0.0:
                        pc = pm
                    pc = max(pc, min_p)

                    ll_marg += -math.log(pm)
                    ll_cond += -math.log(pc)
                    n += 1

                    # Close episode
                    open_since[p] = None
                    prev_act_for_open[p] = None

    if n == 0:
        print("No episodes found for evaluation.")
        return None

    print(f"Episodes evaluated: {n}")
    print(f"Baseline (Marginal) avg log-loss : {ll_marg / n:.6f}")
    print(f"Smart (Conditioned) avg log-loss : {ll_cond / n:.6f}")
    
    improvement = (ll_marg - ll_cond) / n
    print(f"Improvement (Lower is better)    : {improvement:.6f}")
    
    if improvement > 0:
        print("SUCCESS: The conditioned model is more accurate than the basic model.")
    else:
        print("WARNING: Context does not seem to add value/ data is sparse.")

    return (ll_marg / n, ll_cond / n)

def debug_first_hit_any_preset(log, preset_labels_set, postset_labels_set, horizon=50):
    """
    Independent verification logic (Raw Counter) to ensure the Model isn't hallucinating.
    """
    preset = set(preset_labels_set)
    postset = set(postset_labels_set)
    
    opened = 0
    skipped = 0
    counts = defaultdict(int)

    for trace in log:
        acts = [str(ev["concept:name"]).strip() for ev in trace]
        n = len(acts)
        open_since = None
        i = 0

        while i < n:
            act = acts[i]
            
            # Open
            if open_since is None:
                if act in preset:
                    open_since = i
                    opened += 1
                i += 1
                continue

            # Skip self-loop check on same index
            if i == open_since:
                i += 1
                continue

            # Horizon check
            if (i - open_since) > horizon:
                skipped += 1
                open_since = None
                continue 

            # Outcome check
            if act in postset:
                counts[act] += 1
                open_since = None
                continue 

            i += 1

        if open_since is not None:
            skipped += 1

    return opened, dict(counts), skipped

def print_first_hit_check_for_place_any_preset(log, place, presets_map, postsets_map, model_probs, horizon=50):
    name = getattr(place, "name", repr(place))
    preset_labels = presets_map.get(place, set())
    postset_labels = postsets_map.get(place, set())

    print(f"\n--- Validation for Place: {name} ---")
    if not preset_labels or not postset_labels:
        print("  (Skipping: Missing preset or postset labels)")
        return

    opened, counts, skipped = debug_first_hit_any_preset(
        log, preset_labels, postset_labels, horizon=horizon
    )

    hits = sum(counts.values())
    if hits == 0:
        print("  (No outcomes found within horizon)")
        return

    print(f"  Raw Data Distribution (Horizon={horizon}):")
    for out in sorted(postset_labels):
        c = counts.get(out, 0)
        raw_p = c / hits
        # Get model probability (Marginal) for comparison
        mod_p = model_probs.get(place, {}).get(out, 0.0)
        
        # Check alignment
        match_status = "[OK]" if abs(raw_p - mod_p) < 0.05 else "[WARN]"
        print(f"    {match_status} {out:<30} | Raw: {raw_p:.4f} vs Model: {mod_p:.4f}")

def horizon_sensitivity_check(log, decision_places, place_preset_labels, place_postset_labels, horizons=(20, 50, 80, 120)):
    """
    Checks if the probabilities stabilize as we increase the horizon.
    """
    probs_by_horizon = {}
    print("\n=== HORIZON SENSITIVITY CHECK ===")

    for h in horizons:
        branch_counts = compute_branch_counts_first_hit(
            log, decision_places, place_preset_labels, place_postset_labels, horizon=h
        )
        
        _, marginal_probs = compute_branch_probabilities(branch_counts)
        probs_by_horizon[h] = marginal_probs

    # Calculate L1 distance between consecutive horizons
    print("\n=== Stability (L1 Distance) ===")
    base_h = horizons[0]
    
    for h in horizons[1:]:
        print(f"\nComparing Horizon {base_h} vs {h}")
        total_dist = 0.0
        
        for p in decision_places:
            name = getattr(p, "name", repr(p))
            branches = place_postset_labels[p]
            l1 = 0.0
            for b in branches:
                p1 = probs_by_horizon[base_h].get(p, {}).get(b, 0.0)
                p2 = probs_by_horizon[h].get(p, {}).get(b, 0.0)
                l1 += abs(p1 - p2)
            
            total_dist += l1
            if l1 > 0.1: # Threshold for "significant change"
                print(f"  [UNSTABLE] {name}: L1 distance = {l1:.4f}")
            else:
                print(f"  [STABLE]   {name}: L1 distance = {l1:.4f}")

def main():
    bpmn_path = "/Users/zeynepcetin/Decision Point Analysis/data folder/BPI Challenge 2017 Loan Application Process-6-4.bpmn"
    xes_path = "/Users/zeynepcetin/Decision Point Analysis/data folder/BPI Challenge 2017.xes.gz"

    print("Loading model and log...")
    log = xes_importer.apply(xes_path)
    
    # Building the model
    print("Building Decision Model (Horizon=60)...")
    model_data = build_basic_branching_model(bpmn_path, xes_path, horizon=60, verbose=False)
    
    cond_probs = model_data["branch_probabilities_conditioned"]
    marginal_probs = model_data["branch_probabilities_marginal"]

    # Evaluating Predictive Power (Log-Loss)
    print("\n=== 1. PREDICTIVE POWER CHECK ===")
    evaluate_logloss_conditioned_vs_marginal(
        log=log,
        decision_places=model_data["decision_places"],
        place_preset_labels=model_data["place_preset_labels"],
        place_postset_labels=model_data["place_postset_labels"],
        cond_probs=cond_probs,
        marginal_probs=marginal_probs,
        horizon=60
    )

    # Validation against Raw Counts
    print("\n=== 2. REALISM CHECK (Raw vs Model) ===")
    for p in model_data["decision_places"]:
        print_first_hit_check_for_place_any_preset(
            log=log,
            place=p,
            presets_map=model_data["place_preset_labels"],
            postsets_map=model_data["place_postset_labels"],
            model_probs=marginal_probs,
            horizon=60
        )

    # Sensitivity Analysis for Horizon
    print("\n=== 3. STABILITY CHECK (Horizon Sensitivity) ===")
    horizon_sensitivity_check(
        log=log,
        decision_places=model_data["decision_places"],
        place_preset_labels=model_data["place_preset_labels"],
        place_postset_labels=model_data["place_postset_labels"],
        horizons=(20, 50, 80, 120)
    )

if __name__ == "__main__":
    main()