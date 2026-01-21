import random
from collections import defaultdict

class BasicRouter:
    def __init__(self, probabilities, pn):
        self.probabilities = probabilities
        self.pn = pn

    def route(self, enabled, last_activity):
        # label -> list[tids]
        candidates_map = defaultdict(list)

        for tid in enabled:
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
        return random.choice(visible_tids) if visible_tids else random.choice(enabled)
    
        # Do what Luca as random macht, tid = random.choice(enabled)
        # e.g. when last_activity is None at the start of the process   
        #return random.choice(enabled_transitions)   
        