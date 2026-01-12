from collections import defaultdict, deque
import random

from pm4py.objects.bpmn.importer import importer as bpmn_importer
from pm4py.objects.conversion.bpmn import converter as bpmn_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.utils import petri_utils

# PART 1: BUILDING THE MODEL
def load_petri_from_bpmn(bpmn_path):
    bpmn_graph = bpmn_importer.apply(bpmn_path)
    net, im, fm = bpmn_converter.apply(bpmn_graph)
    return net, im, fm

def load_event_log(xes_path):
    return xes_importer.apply(xes_path)

def get_preset_labels_with_backtracking(place, max_back_depth=2):
    """
    Identifies which activities (labels) lead into this place.
    Backtracks through invisible transitions if necessary.
    """
    # 1) Direct preset transitions
    direct_preset = {arc.source for arc in place.in_arcs}
    labels = {t.label.strip() for t in direct_preset if t.label is not None}
    
    if labels:
        return labels

    # 2) Backtracking for invisible transitions
    visible_labels = set()
    visited_transitions = set()
    visited_places = {place}
    queue = deque()

    for t in direct_preset:
        visited_transitions.add(t)
        if t.label is None:
            for arc in t.in_arcs:
                prev_place = arc.source
                if prev_place not in visited_places:
                    visited_places.add(prev_place)
                    queue.append((prev_place, 1))

    while queue:
        curr_place, depth = queue.popleft()
        if depth > max_back_depth:
            continue

        curr_preset = {arc.source for arc in curr_place.in_arcs}
        for t in curr_preset:
            if t in visited_transitions:
                continue
            visited_transitions.add(t)

            if t.label is not None:
                visible_labels.add(t.label.strip())
            else:
                if depth < max_back_depth:
                    for arc in t.in_arcs:
                        prev_place = arc.source
                        if prev_place not in visited_places:
                            visited_places.add(prev_place)
                            queue.append((prev_place, depth + 1))

    return visible_labels

def build_place_structures(net, verbose=False):
    decision_places = [] 
    place_preset_labels = {}
    place_postset_labels = {}

    for p in net.places:
        preset_transitions = {arc.source for arc in p.in_arcs}
        postset_transitions = {arc.target for arc in p.out_arcs}
        visible_postset = {t for t in postset_transitions if t.label is not None}

        # Definition of a Decision Point (XOR): Place with >= 2 visible outcomes
        if len(visible_postset) >= 2:
            decision_places.append(p)
        
        place_preset_labels[p] = get_preset_labels_with_backtracking(p, max_back_depth=2)
        place_postset_labels[p] = {t.label.strip() for t in visible_postset if t.label is not None}

    if verbose:
        print(f"\n=== Found {len(decision_places)} Decision Places ===")
            
    return decision_places, place_preset_labels, place_postset_labels

def compute_branch_counts_first_hit(log, decision_places, place_preset_labels, place_postset_labels, horizon=60):
    """
    Conditioned First-Hit Analysis.
    Returns: counts[place][trigger_activity][outcome_activity]
    """
    branch_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # Fast lookup index: Activity -> List of Places it triggers
    preset_index = defaultdict(list)
    for p in decision_places:
        for a in place_preset_labels.get(p, set()):
            preset_index[a].append(p)

    postset_map = {p: set(place_postset_labels.get(p, set())) for p in decision_places}

    for trace in log:
        acts = [str(ev["concept:name"]).strip() for ev in trace]
        
        # State tracking for this trace
        open_since = {p: None for p in decision_places}
        open_trigger = {p: None for p in decision_places} 

        for i, act in enumerate(acts):
            # 1) Open episodes (Triggering)
            for p in preset_index.get(act, []):
                if open_since[p] is None:
                    open_since[p] = i
                    open_trigger[p] = act

            # 2) Check outcomes for open episodes
            for p in decision_places:
                start_i = open_since[p]
                if start_i is None:
                    continue

                # Horizon check
                if (i - start_i) > horizon:
                    open_since[p] = None
                    open_trigger[p] = None
                    continue

                if i == start_i:
                    continue

                # First valid outcome hit !!
                if act in postset_map[p]:
                    trigger = open_trigger[p]
                    branch_counts[p][trigger][act] += 1
                    
                    # -> Close the episode
                    open_since[p] = None
                    open_trigger[p] = None

    return branch_counts

def compute_branch_probabilities(branch_counts_conditioned):
    """
    Calculates both Conditioned and Marginal probabilities.
    """
    cond_probs = {}
    marginal_probs = {}

    for p, trig_map in branch_counts_conditioned.items():
        cond_probs[p] = {}
        marginal_counts = defaultdict(int)
        marginal_total = 0

        for trig, out_map in trig_map.items():
            trig_total = sum(out_map.values())
            if trig_total == 0: continue

            # Conditioned: P(Next | Place, Trigger Activity)
            cond_probs[p][trig] = {out: c / trig_total for out, c in out_map.items()}

            # Accumulate for Marginal
            for out, c in out_map.items():
                marginal_counts[out] += c
                marginal_total += c

        # Marginal: P(Next | Place)
        if marginal_total > 0:
            marginal_probs[p] = {out: c / marginal_total for out, c in marginal_counts.items()}

    return cond_probs, marginal_probs

def build_transition_activity_mappings(net):
    transition_to_activity = {}
    activity_to_transitions = defaultdict(set)

    for t in net.transitions:
        if t.label is None:
            continue  
        activity = t.label.strip()
        transition_to_activity[t] = activity
        activity_to_transitions[activity].add(t)

    return transition_to_activity, activity_to_transitions

def build_basic_branching_model(bpmn_path, xes_path, horizon=60, verbose=False):
    """
    Main function to build the complete routing model.
    """
    if verbose: print("Loading model and log...")
    net, im, fm = load_petri_from_bpmn(bpmn_path)
    log = load_event_log(xes_path)
    
    transition_to_activity, activity_to_transitions = build_transition_activity_mappings(net)

    if verbose: print("Analyzing structures...")
    decision_places, place_preset, place_postset = build_place_structures(net, verbose=verbose)

    if verbose: print(f"Counting branches with Horizon={horizon}...")
    branch_counts = compute_branch_counts_first_hit(log, decision_places, place_preset, place_postset, horizon=horizon)
    
    cond_probs, marginal_probs = compute_branch_probabilities(branch_counts)

    return {
        "net": net, "im": im, "fm": fm,
        "decision_places": decision_places,
        "transition_to_activity": transition_to_activity,
        "branch_probabilities_conditioned": cond_probs,
        "branch_probabilities_marginal": marginal_probs,
        "place_preset_labels": place_preset,
        "place_postset_labels": place_postset
    }

# PART 2: EXECUTION PHASE (for Simulation Engine)
def route_at_decision_point(place, 
                            previous_activity, 
                            enabled_transitions, 
                            branch_probabilities_conditioned, 
                            branch_probabilities_marginal, 
                            transition_to_activity):
    """
    Determines the next transition to fire based on *history*.
    """
    if not enabled_transitions:
        return None

    # 1. Try Conditioned Probability
    probs_for_place = None
    if place in branch_probabilities_conditioned:
        if previous_activity in branch_probabilities_conditioned[place]:
            probs_for_place = branch_probabilities_conditioned[place][previous_activity]
            
    # 2. Fallback to Marginal Probability
    if probs_for_place is None:
        probs_for_place = branch_probabilities_marginal.get(place, {})

    # 3. Filter enabled transitions against the probability map
    candidates = {}
    for t in enabled_transitions:
        act = transition_to_activity.get(t, None)
        
        # Note: Invisible transitions (None) are skipped here. 
        # In a Basic approach, we cannot easily assign probabilities to them without Alignments.
        if act is None: 
            continue
            
        p = probs_for_place.get(act, 0.0)
        if p > 0:
            candidates[t] = p

    # 4. If no valid candidates (e.g., all enabled are invisible or new paths), Random Fallback
    if not candidates:
        return random.choice(list(enabled_transitions))
    
    # 5. Roulette Wheel Selection
    total = sum(candidates.values())
    if total <= 0:
        return random.choice(list(enabled_transitions))

    r = random.uniform(0, total)
    upto = 0.0
    for t, val in candidates.items():
        if upto + val >= r:
            return t
        upto += val
        
    return list(candidates.keys())[-1]