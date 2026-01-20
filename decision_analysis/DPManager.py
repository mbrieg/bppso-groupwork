import json
import random

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
        """
        For engine it returns a single transition ID based on the last activity and learned probabilities
        """
        if len(enabled_transitions) == 1:
            return enabled_transitions[0]
        
        candidates_map = {}
        for tid in enabled_transitions:
            # pn.labels is a dict: {'trans_123': 'A_Accepted'}
            label = self.pn.labels.get(tid, "")
            candidates_map[label] = tid
        # Now we have the labels of enabled transitions and look into the probability csv 
        # whether there is a probability helps to decide where to go
        candidate_labels = list(candidates_map.keys())

        if last_activity in self.probabilities:
            learned_dist= self.probabilities[last_activity]

            weights = []
            valid_candidates = []

            for label in candidate_labels:
                """
                Now get the probability out of json. If label is "" --> silent transition and weight 0
                """
                w = learned_dist.get(label, 0.0)
                weights.append(w)
                valid_candidates.append(candidates_map[label])

            if sum(weights) > 0:
                #after founding matching probabilty, take the first one from the list
                return random.choices(valid_candidates, weights=weights, k=1)[0]
            
        # Do what Luca as random macht, tid = random.choice(enabled)
        # e.g. when last_activity is None at the start of the process   
        return random.choice(enabled_transitions)    



        
    