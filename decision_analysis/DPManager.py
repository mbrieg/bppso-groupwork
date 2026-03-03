import sys
import json
import random
import pickle
from collections import defaultdict
from .BasicRouter import BasicRouter
from .AdvancedRouter import AdvancedRouter
try:
    from decision_analysis import c45_tree
    sys.modules['c45_tree'] = c45_tree
except ImportError:
    import c45_tree
    sys.modules['c45_tree'] = c45_tree

class DPManager:
    def __init__(self, pn, mode="basic", model_path="simulation_brain.pkl", rules_path="decision_prob_rules.json"):
        self.pn = pn
        self.mode = mode.lower()
        self.router = None

        if self.mode == "advanced":
            self._setup_advanced(model_path, rules_path)
        else:
            self._setup_basic(rules_path)

    def _setup_basic(self, rules_path):
        """Probabilitiy based BasicRouter"""
        try:
            with open(rules_path, 'r') as f:
                probabilities = json.load(f)
            self.router = BasicRouter(probabilities, self.pn)
            print(f"DPManager: BasicRouter activited (Rules: {rules_path})")
        except FileNotFoundError:
            print("DPManager: Not found , working random")
            self.router = BasicRouter({}, self.pn)

    def _setup_advanced(self, model_path, rules_path):
        """C4.5 tree + AdvancedRouter"""
        try:
            with open(model_path, "rb") as f:
                pkg = pickle.load(f)
            self.router = AdvancedRouter(pkg["model"], pkg["bins"], self.pn)
            print(f"DPManager: AdvancedRouter activated. (Model: {model_path})")
        except Exception as e:
            print(f"DPManager: Advanced not found ({e}), Back to basic.")
            self.mode = "basic"
            self._setup_basic(rules_path)   

    def get_next_transition(self, case_id, enabled_transitions, last_activity,
                            last_duration_sec=0, case_start_time=None, current_now=None,
                            case_context=None):
        """
        EngineOG calls it.
        """
        # A) If there is only one choice no need to think actually
        if len(enabled_transitions) == 1:
            return enabled_transitions[0]

        # B) Take the decision from router
        tid = None
        if self.router:
            if self.mode == "advanced":
                # AdvancedRouter wants these parameters
                tid = self.router.route(
                    enabled=enabled_transitions,
                    last_activity=last_activity,
                    last_duration =last_duration_sec,
                    case_start_time=case_start_time,
                    current_now=current_now,
                    case_context=case_context
                )
            else:
                # BasicRouter
                tid = self.router.route(enabled=enabled_transitions, last_activity=last_activity)

        # C) Fallback: Random
        if tid is None:
            return self._fallback_random(enabled_transitions)
        
        return tid

    def _fallback_random(self, enabled):
        """Wenn nothing works, macht was Luca macht vorher"""
        visible_tids = [t for t in enabled if self.pn.labels.get(t, "") != ""]
        return random.choice(visible_tids) if visible_tids else random.choice(enabled)