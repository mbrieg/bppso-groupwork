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
    BASIC_PREFERRED = {
        # C4.5 learned from real log paths not present in BPMN — Basic is more accurate
        "W_Call incomplete files",   # real log: A_Validating 99% → path missing in BPMN
        "A_Incomplete",
        "W_Call after offers",
        "W_Complete application",
        "O_Returned",
        "A_Accepted",
        "A_Denied",
        "O_Accepted",
        "O_Create Offer",
        "A_Complete",
        "W_Handle leads",
    }

    # Activities that signal a rejection in the case history
    _REJECTION_ACTIVITIES = {"O_Refused", "O_Cancelled", "A_Denied", "A_Cancelled"}
    # Activities that signal an accepted offer
    _ACCEPT_ACTIVITIES = {"O_Accepted"}
    # Activities that increment the offer counter
    _OFFER_ACTIVITIES = {"O_Created", "O_Create Offer"}

    def __init__(self, pn, mode="basic", model_path="simulation_brain.pkl", rules_path="decision_prob_rules.json"):
        self.pn = pn
        self.mode = mode.lower()
        self.router = None
        self.fallback_router = None

        # Per-case dynamic state (mirrors xor_identification.ipynb training logic)
        self._case_rejection = {}        # case_id → bool
        self._case_accepted_offer = {}   # case_id → bool
        self._case_offer_count = {}      # case_id → int
        self._case_activity_counts = {}  # case_id → {activity: int}

        # Routing decision counters (diagnostic)
        self._count_advanced = 0
        self._count_fallback_basic = 0
        self._count_fallback_random = 0

        # Decision log: records every multi-choice decision made
        # Each entry: (last_activity, enabled_labels, chosen_label)
        self.decision_log = []

        if self.mode == "advanced":
            self._setup_advanced(model_path, rules_path)
        elif self.mode == "random":
            print("DPManager: Random mode — all decisions uniform random.")
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
        """C4.5 tree + AdvancedRouter, with BasicRouter as smarter fallback"""
        try:
            with open(model_path, "rb") as f:
                pkg = pickle.load(f)
            model = pkg.get("models") or pkg.get("model")
            self.router = AdvancedRouter(
                model, pkg["bins"], self.pn,
                encoder=pkg.get("encoder"),
                feature_names=pkg.get("features")
            )
            print(f"DPManager: AdvancedRouter activated. (Model: {model_path})")
        except Exception as e:
            print(f"DPManager: Advanced not found ({e}), Back to basic.")
            self.mode = "basic"
        # Always load BasicRouter as fallback (used when AdvancedRouter can't match)
        try:
            with open(rules_path, 'r') as f:
                probabilities = json.load(f)
            self.fallback_router = BasicRouter(probabilities, self.pn)
        except FileNotFoundError:
            self.fallback_router = None
        if self.mode == "basic":
            self.router = self.fallback_router

    def _update_case_state(self, case_id, last_activity):
        """Update per-case dynamic features based on the just-completed activity.
        Mirrors the state-tracking logic used during training in xor_identification.ipynb."""
        if case_id not in self._case_activity_counts:
            self._case_activity_counts[case_id] = {}
            self._case_rejection[case_id] = False
            self._case_accepted_offer[case_id] = False
            self._case_offer_count[case_id] = 0

        if last_activity and last_activity != "START":
            counts = self._case_activity_counts[case_id]
            counts[last_activity] = counts.get(last_activity, 0) + 1

            if last_activity in self._REJECTION_ACTIVITIES:
                self._case_rejection[case_id] = True
            if last_activity in self._ACCEPT_ACTIVITIES:
                self._case_accepted_offer[case_id] = True
            if last_activity in self._OFFER_ACTIVITIES:
                self._case_offer_count[case_id] += 1

    def _inject_dynamic_features(self, case_id, last_activity, case_context):
        """Overwrite stale spawn-time values with live tracked state."""
        if case_context is None:
            case_context = {}

        offer_count = self._case_offer_count.get(case_id, 0)
        case_context["has_rejection"]      = 1 if self._case_rejection.get(case_id, False) else 0
        case_context["has_accepted_offer"] = 1 if self._case_accepted_offer.get(case_id, False) else 0
        case_context["offer_category"]     = (
            "none"   if offer_count == 0 else
            "single" if offer_count == 1 else
            "two"    if offer_count == 2 else
            "three"  if offer_count == 3 else
            "many"
        )
        activity_counts = self._case_activity_counts.get(case_id, {})
        case_context["is_repeated"] = str(activity_counts.get(last_activity, 0) > 1)

        return case_context

    def get_next_transition(self, case_id, enabled_transitions, last_activity,
                            last_duration_sec=0, case_start_time=None, current_now=None,
                            case_context=None):
        """
        EngineOG calls it.
        """
        # A) If there is only one choice no need to think
        if len(enabled_transitions) == 1:
            return enabled_transitions[0]

        # Update dynamic case state and inject into context before routing
        self._update_case_state(case_id, last_activity)
        case_context = self._inject_dynamic_features(case_id, last_activity, case_context)

        # B) Take the decision from router
        tid = None
        if self.mode == "random":
            tid = self._fallback_random(enabled_transitions)
        use_basic_directly = (self.mode == "advanced" and
                               last_activity in self.BASIC_PREFERRED and
                               self.fallback_router is not None)
        if use_basic_directly:
            tid = self.fallback_router.route(enabled=enabled_transitions, last_activity=last_activity)
        elif self.router:
            if self.mode == "advanced":
                tid = self.router.route(
                    enabled=enabled_transitions,
                    last_activity=last_activity,
                    last_duration=last_duration_sec,
                    case_start_time=case_start_time,
                    current_now=current_now,
                    case_context=case_context
                )
            else:
                tid = self.router.route(enabled=enabled_transitions, last_activity=last_activity)

        # C) Fallback: BasicRouter (probability-based, better than random)
        if self.mode == "advanced":
            if use_basic_directly:
                self._count_fallback_basic += 1   # BASIC_PREFERRED → BasicRouter directly
            elif tid is None and self.fallback_router:
                tid = self.fallback_router.route(enabled=enabled_transitions, last_activity=last_activity)
                if tid is not None:
                    self._count_fallback_basic += 1
            elif tid is not None:
                self._count_advanced += 1

        # D) Last resort: random
        if tid is None:
            self._count_fallback_random += 1
            tid = self._fallback_random(enabled_transitions)

        # Log the decision (last_activity, visible enabled labels, chosen label)
        chosen_label = self.pn.labels.get(tid, "")
        enabled_labels = [self.pn.labels.get(t, "") for t in enabled_transitions
                          if self.pn.labels.get(t, "") != ""]
        if enabled_labels:
            self.decision_log.append((last_activity, tuple(sorted(enabled_labels)), chosen_label))

        return tid

    def print_routing_stats(self):
        total = self._count_advanced + self._count_fallback_basic + self._count_fallback_random
        if total == 0:
            print("No routing decisions recorded.")
            return
        print(f"Routing stats (multi-choice decisions only):")
        print(f"  C4.5 Advanced:      {self._count_advanced:5d}  ({100*self._count_advanced/total:.1f}%)")
        print(f"  Basic (preferred or fallback): {self._count_fallback_basic:5d}  ({100*self._count_fallback_basic/total:.1f}%)")
        print(f"  Random (last resort):{self._count_fallback_random:5d}  ({100*self._count_fallback_random/total:.1f}%)")

    def _fallback_random(self, enabled):
        """Last resort when nothing works"""
        visible_tids = [t for t in enabled if self.pn.labels.get(t, "") != ""]
        return random.choice(visible_tids) if visible_tids else random.choice(enabled)