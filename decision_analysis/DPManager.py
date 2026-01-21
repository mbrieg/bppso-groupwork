import json
import random
from collections import defaultdict

class DPManager:
    def __init__(self, pn, rules_path="decision_prob_rules.json"):
        self.pn = pn
        self.rules_path = rules_path
        self.probabilities = {}

        self._load_rules()

    def _load_rules(self):
        try:
            with open(self.rules_path, 'r') as f:
                self.probabilities = json.load(f)
            print(f"DecisionManager: Loaded rules from {self.rules_path}")
        except FileNotFoundError:
            print(f"DecisionManager: WARNING - {self.rules_path} not found. Defaulting to Random.")

    def get_next_transition(self, case_id, enabled_transitions, last_activity):
        if len(enabled_transitions) == 1:
            return enabled_transitions[0]

        # label -> list[tids]
        candidates_map = defaultdict(list)
        for tid in enabled_transitions:
            label = self.pn.labels.get(tid, "")
            if isinstance(label, str):
                label = label.strip()
            candidates_map[label].append(tid)

        learned_dist = self.probabilities.get(last_activity, None)

        if learned_dist:
            visible_labels = [lbl for lbl in candidates_map.keys() if lbl != ""]
            if visible_labels:
                weights = [float(learned_dist.get(lbl, 0.0)) for lbl in visible_labels]
                if sum(weights) > 0:
                    chosen_label = random.choices(visible_labels, weights=weights, k=1)[0]
                    return random.choice(candidates_map[chosen_label])

        # fallback: if there is visible select a random among visibles, not full random
        visible_tids = [tid for lbl, tids in candidates_map.items() if lbl != "" for tid in tids]
        return random.choice(visible_tids) if visible_tids else random.choice(enabled_transitions)
    
        # Do what Luca as random macht, tid = random.choice(enabled)
        # e.g. when last_activity is None at the start of the process   
        #return random.choice(enabled_transitions)   
        
        
    