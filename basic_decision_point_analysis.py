from collections import defaultdict
import random

from pm4py.objects.bpmn.importer import importer as bpmn_importer
from pm4py.objects.conversion.bpmn import converter as bpmn_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.utils import petri_utils

#First things first loading the bpmn model
def load_petri_from_bpmn(bpmn_path):
    bpmn_graph = bpmn_importer.apply(bpmn_path)
    net, im, fm = bpmn_converter.apply(bpmn_graph)
    return net, im, fm

"""
    From the Petri net, extract:
      decision places (XOR points)
      each place's preset/postset transitions
      activity label sets for those transitions
    A decision place (XOR) is defined as a place with at least 2 outgoing arcs
"""

def load_event_log(xes_path):
    log = xes_importer.apply(xes_path)
    return log

        
def get_preset_labels_with_backtracking(place, max_back_depth=2):
    """
    Docstring for get_preset_labels_with_backtracking
    
    for one place:
    1) first try the labels of transitions which can be seen
    2) if all of labels are None, back tracking to the 1/2
      availalble labels of transitions
    """
    from collections import deque

    # 1) Doğrudan preset transitions
    direct_preset = {arc.source for arc in place.in_arcs}
    labels = {
        t.label.strip()
        for t in direct_preset
        if t.label is not None
    }
    if labels:
        return labels  # Eğer buradan bir şey bulduysak, yeterli

    # 2) Geriye doğru görünür transition arama (sadece gerekirse)
    visible_labels = set()
    visited_transitions = set()
    visited_places = set([place])

    # Kuyruğa, bu place'den başlayan görünmez transition'ları atalım
    queue = deque()

    for t in direct_preset:
        visited_transitions.add(t)
        if t.label is None:
            # invisible ise, bir adım geri gidilecek places
            for arc in t.in_arcs:
                prev_place = arc.source
                if prev_place not in visited_places:
                    visited_places.add(prev_place)
                    queue.append((prev_place, 1))  # depth = 1

    # BFS ile max_back_depth kadar geriye yürü
    while queue:
        curr_place, depth = queue.popleft()
        if depth > max_back_depth:
            continue

        # curr_place'e giren tüm transition'lar
        curr_preset = {arc.source for arc in curr_place.in_arcs}
        for t in curr_preset:
            if t in visited_transitions:
                continue
            visited_transitions.add(t)

            if t.label is not None:
                visible_labels.add(t.label.strip())
            else:
                # Hâlâ invisible ise ve depth sınırına gelmediysek, bir place daha geri git
                if depth < max_back_depth:
                    for arc in t.in_arcs:
                        prev_place = arc.source
                        if prev_place not in visited_places:
                            visited_places.add(prev_place)
                            queue.append((prev_place, depth + 1))

    return visible_labels



def build_place_structures(net):
    decision_places = [] #list
    place_preset_transitions = {} #dictionary
    place_postset_transitions = {}
    place_preset_labels = {}
    place_postset_labels = {}

    for p in net.places:
        #preset and postset transitions
        #print("Place:", p.name or p)
        post = petri_utils.post_set(p)
        #print("Outgoing transitions and labels:")
        #for t in post:
        #    print("  -", t, "label:", t.label)

        #Place P1 (arc.source) ----> (arc.target) Transition T1
        preset_transitions = {arc.source for arc in p.in_arcs}
        postset_transitions = {arc.target for arc in p.out_arcs}

        visible_postset = {t for t in postset_transitions if t.label is not None}

        #XOR criterion
        if len(postset_transitions) >= 2 and len(visible_postset) >= 1:
            decision_places.append(p)
        
        #activity labels excluding invisible transitions
        preset_labels = get_preset_labels_with_backtracking(p, max_back_depth=2)

        postset_labels = {
            t.label.strip()
            for t in visible_postset
            if t.label is not None
        }
         #transition dict
        place_preset_transitions[p] = preset_transitions
        place_postset_transitions[p] = postset_transitions

        place_preset_labels[p] = preset_labels
        place_postset_labels[p] = postset_labels

    print("\n=== Decision Places ===")
    for p in decision_places:
        print(
            "- Place:", p,
            "\n    preset labels:", place_preset_labels[p],
            "\n    postset labels:", place_postset_labels[p],
        )
    
            
    return (decision_places, 
                place_preset_transitions, 
                place_postset_transitions,
                place_preset_labels,
                place_postset_labels)
    


def build_transition_activity_mappings(net):
    """
    Docstring for build_transition_activity_mappings
    
    transition_to_activity : mapping transition -> activity label
    activity_to_transitions. mapping activity label -> set of transitions

    label=None ignored
    """

    transition_to_activity = {}
    activity_to_transitions = defaultdict(set)

    for t in net.transitions:
        if t.label is None:
            continue  
        activity = t.label
        transition_to_activity[t] = activity
        activity_to_transitions[activity].add(t)

    return transition_to_activity, activity_to_transitions

#we must identify the decision place for event pair
def find_decision_place_for_pair(prev_act, 
                                 next_act, 
                                 decision_places, 
                                 place_preset_labels, 
                                 place_postset_labels):
    
    if prev_act is not None:
        prev_act = str(prev_act).strip()
    if next_act is not None:
        next_act = str(next_act).strip()

    strict_candidates = []
    loose_candidates = []

    for p in decision_places:
        preset_labels = place_preset_labels[p]
        postset_labels = place_postset_labels[p]

        # 1) Sıkı koşul: hem prev hem next tutuyor
        if prev_act in preset_labels and next_act in postset_labels:
            strict_candidates.append(p)

        # 2) Gevşek koşul: sadece next tutuyorsa
        if next_act in postset_labels:
            loose_candidates.append(p)

    # Öncelik: sıkı eşleşme
    if len(strict_candidates) == 1:
        return strict_candidates[0]
    elif len(strict_candidates) > 1:
        # burada istersen ek bir seçim mantığı uygulayabilirsin
        return strict_candidates[0]

    # Sıkı eşleşme yoksa, gevşek tek aday varsa onu al
    if len(loose_candidates) == 1:
        return loose_candidates[0]

    # Aksi halde belirsiz → None
    return None

        

#branch frequencies
def compute_branch_counts(log, 
                          decision_places,
                            place_preset_labels, 
                            place_postset_labels):
    """
    Docstring for compute_branch_counts
    it iterates over the real event log and counts, for each decision point p, how ofen each outgoing activity (branch) is taken

    branch_counts[p][activity] = count --> nested default dict needed
    """

    """
 --> log = [
   trace1 = [
      {"concept:name": "A_Submitted", "time:timestamp": ...},
      {"concept:name": "A_Precheck", ...},
      ...
   ],

   trace2 = [
      ...
   ],
    """
    branch_counts = defaultdict(lambda: defaultdict(int))

    for trace in log:
        if len(trace) < 2:
            continue

        for i in range(len(trace) - 1):
            prev_event = trace[i]
            next_event = trace[i +1]

            # normalize activity names from log (strip)

            prev_act = str(prev_event["concept:name"]).strip()
            next_act = str(next_event["concept:name"]).strip()

            # identify associated decision place
            p = find_decision_place_for_pair(
                prev_act, next_act,
                decision_places,
                place_preset_labels,
                place_postset_labels
            )

            if p is None:
                continue

            branch_counts[p][next_act] += 1

    return branch_counts

#probabilities
def compute_branch_probabilities(branch_counts):
    """
    Docstring for compute_branch_probabilities
    
    prob = count / sum(counts at one decision point) 
    """

    branch_probabilities = {}

    for p, counts in branch_counts.items():
        total = sum(counts.values())
        if total == 0:
            continue
        probs = {act : c / total for act, c in counts.items()} # key: act value: c / total
        branch_probabilities[p] = probs

    return branch_probabilities

# directly use this function in the simulation to run the decision point analysis
def route_at_decision_point(place, 
                            enabled_transitions, 
                            branch_probabilities, 
                            transition_to_activity):
    """
    Docstring for route_at_decision_point
    now, selects the next transition to fire at a decision point accoriding to probabilities
    during simulation

    returns:
    chosen_transition : the transition selected for execution 
    """
    probs_for_place = branch_probabilities.get(place, None)

    # If no prob info available --> fallback to uniform random
    if probs_for_place is None or len(probs_for_place) == 0:
        return random.choice(list(enabled_transitions))
    

    #Filter probs for currently enabled transitions
    filtered = {}
    for t in enabled_transitions:
        act = transition_to_activity.get(t, None)
        if act is None:
            continue
        p_act = probs_for_place.get(act, 0.0)
        if p_act > 0.0:
            filtered[t] = p_act
    
    if len(filtered) == 0:
        return random.choice(list(enabled_transitions))
    
    #Normalize
    total = sum(filtered.values())
    if total <= 0.0:
        return random.choice(list(enabled_transitions))
    
    normalized = {t: v / total for t, v in filtered.items()}

    #Roulette-wheel sampling
    r = random.random()
    cumulative = 0.0
    for t, p in normalized.items():
        cumulative += p
        if r <= cumulative:
            return t

    # numerical fallback
    return list(normalized.keys())[-1]

def build_basic_branching_model(bpmn_path, xes_path):
    """
    Full pipeline for the Basic branch decision approach:

      BPMN -> Petri Net
      Extract decision places
      Count branch frequencies from event log
      Convert counts into branch probabilities

    Returns a dictionary containing:
        - net, im, fm
        - decision places
        - transition/activity mappings
        - computed branch probabilities
    """
    # Model
    net, im, fm = load_petri_from_bpmn(bpmn_path)

    # Decision points and label sets
    (decision_places,
     place_preset_transitions,
     place_postset_transitions,
     place_preset_labels,
     place_postset_labels) = build_place_structures(net)

    # Transition <-> activity mapping
    transition_to_activity, activity_to_transitions = build_transition_activity_mappings(net)

    # Event log
    log = load_event_log(xes_path)

    # Counts
    branch_counts = compute_branch_counts(
        log,
        decision_places,
        place_preset_labels,
        place_postset_labels
    )

    # Probabilities
    branch_probabilities = compute_branch_probabilities(branch_counts)

    return {
        "net": net,
        "im": im,
        "fm": fm,
        "decision_places": decision_places,
        "transition_to_activity": transition_to_activity,
        "activity_to_transitions": activity_to_transitions,
        "branch_probabilities": branch_probabilities
    }