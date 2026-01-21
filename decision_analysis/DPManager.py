import json
import random
from collections import deque, defaultdict

class DPManager:
    def __init__(self, pn, rules_path="decision_prob_rules.json", max_lookahead_depth=6):
        self.pn = pn
        self.rules_path = rules_path
        self.probabilities = {}
        self.max_lookahead_depth = max_lookahead_depth
        self._load_rules()

    def _load_rules(self):
        try:
            with open(self.rules_path, 'r') as f:
                self.probabilities = json.load(f)
            print(f"DecisionManager: Loaded rules from {self.rules_path}")
        except FileNotFoundError:
            print(f"DecisionManager: WARNING - {self.rules_path} not found. Defaulting to Random.")

    def _enabled_from_marking(self, marking):
        return [
            t for t in self.pn.trans_ids
            if all(marking.get(p, 0) > 0 for p in self.pn.inputs.get(t, []))
        ]
    
    def _fire_copy(self, marking, tid):
        """Fire tid on a COPY of marking and return the new marking."""
        m2 = dict(marking)
        for p in self.pn.inputs.get(tid, []):
            m2[p] = m2.get(p, 0) - 1
        for p in self.pn.outputs.get(tid, []):
            m2[p] = m2.get(p, 0) + 1
        return m2

    def _marking_key(self, marking):
        """Hashable key for visited set (ignore zeros for stability)."""
        return tuple(sorted((p, c) for p, c in marking.items() if c != 0))

    def _label(self, tid):
        return self.pn.labels.get(tid, "")

    def _is_silent(self, tid):
        return self._label(tid) == ""

    def _reachable_visible_labels_after_silent(self, marking, silent_tid, max_depth):
        """
        Counterfactual BFS:
        - Fire silent_tid once (on copy)
        - Then traverse further SILENT transitions up to max_depth
        - Collect visible labels that become enabled along the way
        Returns: dict(label -> count) (count is a weak proxy for "how often encountered")
        """
        start = self._fire_copy(marking, silent_tid)

        q = deque([(start, 0)])
        visited = {self._marking_key(start)}
        counts = defaultdict(int)

        while q:
            m, d = q.popleft()
            enabled = self._enabled_from_marking(m)

            # collect visible enabled labels
            for t in enabled:
                lbl = self._label(t)
                if lbl != "":
                    counts[lbl] += 1

            if d >= max_depth:
                continue

            # expand only through silent transitions
            for t in enabled:
                if self._is_silent(t):
                    m2 = self._fire_copy(m, t)
                    k = self._marking_key(m2)
                    if k not in visited:
                        visited.add(k)
                        q.append((m2, d + 1))

        return dict(counts)

    def _choose_silent_via_rules(self, marking, silent_candidates, learned_dist):
        """
        Goal: approximate the learned_dist over *visible outcomes* by routing through silent transitions.
        Steps:
          1) For each silent, compute reachable visible labels (look-ahead)
          2) Build union of reachable labels; restrict learned_dist to that set and renormalize
          3) Sample a TARGET visible label according to restricted learned_dist
          4) Pick a silent that can reach that label (tie-break random)
        """
        reachable_map = {}   # silent_tid -> dict(label -> count)
        union_labels = set()

        for stid in silent_candidates:
            reach = self._reachable_visible_labels_after_silent(
                marking, stid, self.max_lookahead_depth
            )
            if reach:
                reachable_map[stid] = reach
                union_labels.update(reach.keys())

        if not union_labels:
            return None  # no info; caller will fallback random

        # Restrict learned_dist to reachable labels and renormalize
        weights = {}
        total = 0.0
        for lbl in union_labels:
            w = float(learned_dist.get(lbl, 0.0))
            if w > 0:
                weights[lbl] = w
                total += w

        if total <= 0:
            return None  # learned_dist gives no mass to reachable labels

        labels = list(weights.keys())
        probs = [weights[lbl] / total for lbl in labels]

        # Sample target visible label
        target = random.choices(labels, weights=probs, k=1)[0]

        # Choose among silent transitions that can reach target
        feasible = [stid for stid, reach in reachable_map.items() if target in reach]
        if feasible:
            return random.choice(feasible)

        return None
    

    def get_next_transition(self, case_id, enabled_transitions, last_activity, marking=None):
        """
        For engine it returns a single transition ID based on the last activity and learned probabilities
        """
        if len(enabled_transitions) == 1:
            return enabled_transitions[0]
        
        candidates_map = {}

        for tid in enabled_transitions:
            # pn.labels is a dict: {'trans_123': 'A_Accepted'}
            label = self.pn.labels.get(tid, "")
            candidates_map.setdefault(label, []).append(tid)
        # Now we have the labels of enabled transitions and look into the probability csv 
        # whether there is a probability helps to decide where to go

        learned_dist = self.probabilities.get(last_activity, None)

        if learned_dist is not None:
            visible_labels = [lbl for lbl in candidates_map.keys() if lbl != ""]
            if visible_labels:
                weights = [float(learned_dist.get(lbl, 0.0)) for lbl in visible_labels]
                if sum(weights) > 0:
                    #after founding matching probabilty, take the first one from the list
                    chosen_label = random.choices(visible_labels, weights=weights, k=1)[0]
                    return random.choice(candidates_map[chosen_label])
            
            if marking is not None:
                silent_candidates = [tid for tid in enabled_transitions if self.pn.labels.get(tid, "") == ""]
                if silent_candidates:
                    chosen = self._choose_silent_via_rules(marking, silent_candidates, learned_dist)
                    if chosen is not None:
                        return chosen
                
            
        # Do what Luca as random macht, tid = random.choice(enabled)
        # e.g. when last_activity is None at the start of the process   
        return random.choice(enabled_transitions)    

    """
        if last_activity in self.probabilities:
            learned_dist= self.probabilities[last_activity]
            weights = []
            valid_candidates = []

            for label in candidate_labels:
                
                Now get the probability out of json. If label is "" --> silent transition and weight 0
                
                w = learned_dist.get(label, 0.0)
                weights.append(w)
                valid_candidates.append(candidates_map[label])

            if sum(weights) > 0:
                #after founding matching probabilty, take the first one from the list
                return random.choices(valid_candidates, weights=weights, k=1)[0]
            
    """

        
    