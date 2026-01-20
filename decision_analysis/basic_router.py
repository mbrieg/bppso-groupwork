from collections import defaultdict
import random
import pandas as pd
from .utils import get_traces_from_log

class BasicRouter:

    """
    The core of the Basic Simulation.
    
    It analyzes the event Log to learn routing probabilities.
    Unlike simple frequency counting, this class uses a "Conditioned First-Hit" algorithm.
    
    Concepts:
    1. Conditioned: The probability of going to 'B' depends on where we came from ('A').
    2. First-Hit: We look ahead in the trace. The "first" valid outcome we see closes the decision.
    3. Horizon: A limit on how far we look ahead. If we don't see an outcome within x steps, 
       we assume the connection is lost (noise/long loops).
    """

    def __init__(self, log, decision_points):
        """
        log : event log
        decision_points = dict place_name = DecisionPointObj from structues.py
        """

        self.log = log
        self.decision_points = decision_points

    def train(self, horizon = 60, debug = True):
        """
        Analyses the log and populates the prob tables in the DecisionPoint objects.
        horizon = max num of steps to wait for an outcome after a decision point is triggered.
        """

        # for debug
        print(f"Training Basic Router (First-Hit Logic, Horizon={horizon})...")

        trace_iterator = get_traces_from_log(self.log)
        if len(trace_iterator) == 0:
            print("Basic Router Warning: No traces found to train on.")
            return

        if debug:
            print(f" DEBUG: First trace sample (first 10 acts): {trace_iterator[0][:10]}")

        preset_index = defaultdict(list)
        for dp in self.decision_points.values():
            for act in dp.incoming_activities:
                preset_index[act].append(dp)

        # if I am at DecisionPoint x, what are the valid next steps, outcomes?
        postset_map = {dp.place_name: set(dp.get_possible_activities()) for dp in self.decision_points.values()}

        if debug and self.decision_points:
            test_dp = list(self.decision_points.values())[0]
            print(f" DEBUG: Checking setup for ONE decision point ('{test_dp.place_name}')")
            print(f"    -> Expecting Inputs (Triggers): {test_dp.incoming_activities}")
            print(f"    -> Expecting Outputs (Targets): {test_dp.get_possible_activities()}")

        # dict to store counts: counts[place][trigger_activity][outcome_activity] = N
        branch_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        match_counter = 0

        # Trace is a string list ( ['Start', 'Approve', ...])
        for trace_idx, trace in enumerate(trace_iterator):
            
            # extract activity names (list of strings)
            try: 
                acts = [str(act).strip() for act in trace]
            except: 
                continue # Skip malformed traces

            # state tracking:
            open_since = {name: None for name in self.decision_points}   # when did it start? (index)
            open_trigger = {name: None for name in self.decision_points} # what triggered it? (Activity Name)

            for i, act in enumerate(acts):
                
                # check triggers (Start of a decision)
                for dp in preset_index.get(act, []):
                    # only start if we aren't already waiting for an outcome for this specific DP
                    if open_since[dp.place_name] is None:
                        open_since[dp.place_name] = i
                        open_trigger[dp.place_name] = act
                        
                        # DEBUG: Triggered?
                        if debug and match_counter < 3 and trace_idx < 3:
                            print(f" Triggered '{dp.place_name}' by '{act}' at step {i}")
                
                # check outcomes (End of a decision)
                for dp_name in list(self.decision_points.keys()):
                    start_i = open_since[dp_name]
                    
                    # if this DP is not waiting for an outcome, skip
                    if start_i is None: continue

                    # horizon check (Timeout)
                    if (i - start_i) > horizon:
                        open_since[dp_name] = None 
                        continue
                    
                    # skip the trigger itself
                    if i == start_i: continue 

                    # match check
                    if act in postset_map[dp_name]:
                        # we found the First Hit
                        trigger = open_trigger[dp_name]
                        
                        # record the statistic
                        branch_counts[dp_name][trigger][act] += 1
                        
                        # close the episode
                        open_since[dp_name] = None
                        
                        match_counter += 1
                        # DEBUG: Is a match?
                        if debug and match_counter <= 5:
                            print(f" Match: {dp_name}: {trigger} -> {act}")

        print(f" DEBUG: Total matches found: {match_counter}")
        
        # calculate probs 
        self._calculate_probabilities(branch_counts)

    def _calculate_probabilities(self, counts):
        """
        Converts raw counts into percentages and stores them inside the DP objects.
        """

        for dp_name, trig_map in counts.items():
            dp = self.decision_points[dp_name]
            
            # conditioned Probabilities: P(Next | Trigger)
            # if the trigger was 'Approve', 90% go to 'Notify', 10% 'Archive'.
            for trig, out_map in trig_map.items():
                total = sum(out_map.values())
                if total > 0:
                    dp.probs_conditioned[trig] = {k: v/total for k, v in out_map.items()}
            
            # marginal Probabilities: P(Next)
            # this is the aggregate average, used as a fallback if we see a 
            # new/unknown trigger during simulation.
            marginal_counts = defaultdict(int)
            for out_map in trig_map.values():
                for k, v in out_map.items():
                    marginal_counts[k] += v
            
            total_m = sum(marginal_counts.values())
            if total_m > 0:
                dp.probs_marginal = {k: v/total_m for k, v in marginal_counts.items()}
        
        print("Basic Router training completed.")

    def predict(self, place_name, context):
        """
        Called by the Engine
        asks I am at the place_name and the last thing I did was context (history list or prev_act string). 
        where should I go now?
        """
        dp = self.decision_points.get(place_name)

        # if this place isnt knpw as a dp, return none (engine handles default)
        if not dp: return None

        # Parse context
        prev_act = None
        history = []
        
        if isinstance(context, list):
            history = context
            if len(history) > 0:
                prev_act = history[-1].strip() # Clean it just in case
        elif isinstance(context, str):
            prev_act = context
            history = [context]

        # Stratagey 1: Conditioned Prob (best accuracy)
        # do we have specific stats for this prev act
        probs = dp.probs_conditioned.get(prev_act)

        # Strategy 2: Marginal Prob fallback
        if not probs: 
            probs = dp.probs_marginal

        # Strategy 3: Random
        # if we have really no data, pick a random valid output
        if not probs:
            candidates = dp.get_possible_activities()
            if not candidates : return None
            return random.choice(candidates)

        # Apply Loop Decay / Penalty
        weighted_probs = probs.copy()
        
        if prev_act and history:
            # Count how many times we've done this recently
            consecutive_count = 0
            for act in reversed(history):
                # Ensure we compare stripped strings
                if str(act).strip() == prev_act:
                    consecutive_count += 1
                else:
                    break
            
            # The 'prev_act' is what we JUST did.
            # If the router suggests doing 'prev_act' AGAIN, checks how many times we already did it.
            # But wait, 'prev_act' is the INCOMING trigger.
            # The 'probs' keys are the OUTGOING next steps.
            # If we want to prevent loops (doing A -> A -> A), we check if the candidate next step is 'prev_act'?
            # Actually, the loop usually is A -> B -> A -> B.
            # But the user logic was: "If we try to repeat the same activity...".
            
            # Let's apply the penalty to ANY predicted activity that matches the immediate history
            # But the user logic specifically checked `if prev_act in weighted_probs`.
            # If `prev_act` (the trigger) is also a possible OUTCOME, then it's a self-loop (A->A).
            
            # User Code adapted:
            # P_new = P_old / (3 ^ count) -> Drastic reduction
            if prev_act in weighted_probs and consecutive_count > 0:
                penalty = 3 ** consecutive_count 
                original_p = weighted_probs[prev_act]
                weighted_probs[prev_act] = original_p / penalty
                # print(f"Decayed '{prev_act}' chance from {original_p:.2f} to {weighted_probs[prev_act]:.4f} (Count: {consecutive_count})")

        # Strategy 4: Roulette Wheel Selection (Weighted random)
        # randomly choose based on the calculated probs
        choices = list(weighted_probs.keys())
        weights = list(weighted_probs.values())

        return random.choices(choices, weights= weights, k=1)[0]
