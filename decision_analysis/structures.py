from collections import deque
from .utils import is_invisible_label

class DecisionPoint:
    """
    This class represents a decision point (XOR Split Place) in the petri net.
    There can be tau transitions used for routing logic that do not represent the real work.
    If we try to decide where to go next based on solely on the immediate previous node, we might land on an invisible transition, which actually give us no context.
    So, we need backtracking to look through the tau transitions to find last 'actual' activity that occured.
    This provides us the necessary context for making decisions.
    """
    def __init__(self, place_name, petri_net_obj):
        # e.g. place name p12
        self.place_name = place_name

        #the actual place obj from petri net
        self.petri_place = petri_net_obj

        #set to store the names of 'actual' activities that lead us to this place
        #gathered by the backtraking logic
        self.incoming_activities = set()

        #dict mapping activity names to their corresponding transition objects
        # "reject" : <Transition t2>...
        self.outgoing_transitions = {}

        #stores the probabilities based on history
        # "previous act A" : ("next act B", 0.8), ("next act c", 0.2)
        self.probs_conditioned = {}

        #general probabilities if we dont know the history
        #("next act B", 0.7), ("next act c", 0.3)
        self.probs_marginal = {}


    def analyse_preset(self, max_depth = 2):
        """"
        Public method to trigger the analysis of what comes before this place
        Calls the internal backtracking
        """
        self.incoming_activities = self._get_preset_labels_with_backtraking(self.petri_place, max_depth)

    def detect_outgoing_transitions(self):
        """
        Logic to trigger the analysis of what comes after this place.
        Scans outgoing arcs. If it finds a silent transition, it looks 
        forward with BFS to find the next real activity name
        """
        for arc in self.petri_place.out_arcs:
            trans = arc.target
            
            activity_name = None
            is_hidden = is_invisible_label(trans.label)

            if is_hidden:
                # Silent transition found. Look forward to find the next real activity.
                activity_name = self._resolve_downstream_label(trans)
                
                # Fallback if BFS returns nothing, use internal name
                if not activity_name:
                    activity_name = trans.name
            elif trans.label:
                # Normal visible transition
                activity_name = trans.label.strip()
            else:
                # Visible but no label (rare), use ID
                activity_name = trans.name
            
            # Register the result
            if activity_name:
                self.add_outgoing(trans, activity_name)


    def add_outgoing(self, trans, act_name):
        """
        Registers a possible path forward from this decision point
        """
        if act_name:
            # name -> object mapping. The engine decide on a name and we neeed to give 
            # it back to the transition object to execute.
            self.outgoing_transitions[act_name] = trans

    def get_possible_activities(self):
        """
        returns a simple list of names of all activities that can happen NEXT.
        """
        return list(self.outgoing_transitions.keys())
    
    def get_safe_activity(self, predicted_act, trace_history, limit=5):
        """
        Infinite Loop Breaker for activities such as W_. In final_output, it is seen 
        that there is always a lot of W_ activites but not even one A_Accepted activity.
        If predicted_act creates a loop > limit (e.g. W_Complete app. happening 20 times),
        it returns an alternative activity to break the loop.
        """
        # Validation: Ensure the prediction is actually a valid option
        if predicted_act not in self.outgoing_transitions:
            # Fallback just return the first valid option available
            if not self.outgoing_transitions:
                return None
            return list(self.outgoing_transitions.keys())[0]

        # Check for Infinite Loop
        if isinstance(trace_history, list) and trace_history:
            loop_count = 0
            # Walk backwards through history to count consecutive repetitions
            for item in reversed(trace_history):
                # Extract string name if item is a dictionary (Event object)
                if isinstance(item, dict):
                    act_name = item.get("concept:name")
                elif hasattr(item, "label"): # Handle object with label attribute
                    act_name = item.label
                else:
                    act_name = str(item) # Fallback for strings

                # Compare Sanitized String names
                if act_name == predicted_act:
                    loop_count += 1
                else:
                    break
            
            # 3. If we hit the limit, FORCE a change
            if loop_count >= limit:
                print(f"DEBUG: Loop limit ({limit}) hit for {predicted_act}. Switching path.")
                print(f"DEBUG: Breaking loop. Options available: {list(self.outgoing_transitions.keys())}")
                
                # Get all options except the one causing the loop
                alternatives = [
                    act for act in self.outgoing_transitions.keys() 
                    if act != predicted_act
                ]
                
                if alternatives:
                    # Return the first safe alternative found
                    return alternatives[0]

        # 4. If safe, return the original prediction
        return predicted_act

    def _get_preset_labels_with_backtraking(self, place, max_back_depth = 2):
        """
        Identifies which activities (labels) lead into this place.
        If the immediate parent is a 'silent' (tau) transition, it digs deeper 
        backwards into the graph to find the real activity.
        Backtracks through invisible transitions if necessary.

        place: The Petri Net place to start looking from.
        max_back_depth: How many steps back we are allowed (to look to prevent infinite loops).
        """
        # Look at the arc coming into this place
        direct_preset = {arc.source for arc in place.in_arcs}

        #Filter for transitions that actually have a name(Label is not none)
        labels = {t.label.strip() for t in direct_preset if not is_invisible_label(t.label)}

        # If we found real activities then return them directly
        if labels:
            return labels

        #If we are here it means the immediate parents are invisible
        # The search backwards begins
        visible_labels = set()
        visited_transitions = set()
        visited_places = {place} # keeping track of places we've seen to avoid circles

        #Use a queue for BFS (petri place, current depth)
        queue = deque()

        #Look at the places before the immediate invisible transitions
        for t in direct_preset:
            visited_transitions.add(t)
            if is_invisible_label(t.label): 
                for arc in t.in_arcs:
                    prev_place = arc.source
                    if prev_place not in visited_places:
                        # If its not visited add to the queue
                        visited_places.add(prev_place)
                        queue.append((prev_place, 1)) # depth = 1

        # Process the queue until its empty
        while queue:
            curr_place, depth = queue.popleft()

            #Stop if we've gone too far back
            if depth > max_back_depth:
                continue
            
            #Look at inputs to this current place in the search
            curr_preset = {arc.source for arc in curr_place.in_arcs}
            for t in curr_preset:
                if t in visited_transitions:
                    continue
                visited_transitions.add(t)

                if not is_invisible_label(t.label):
                    # found one, bc it has a label now real activity
                    visible_labels.add(t.label.strip())
                else:
                    # still invisible? keep digging if we have still max depth to go
                    if depth < max_back_depth:
                        for arc in t.in_arcs:
                            prev_place = arc.source
                            if prev_place not in visited_places:
                                visited_places.add(prev_place)
                                queue.append((prev_place, depth + 1))

        return visible_labels
    
    def _resolve_downstream_label(self, start_trans):
        """
        Looks past silent transitions (now Forward BFS) to find the next visible activity name.
        Used when the immediate outgoing transition is invisible.
        """
        queue = deque([start_trans])
        visited = {start_trans}
        
        # Limit search depth/steps to prevent infinite hangs in bad models
        steps = 0
        max_steps = 100 

        while queue and steps < max_steps:
            curr_trans = queue.popleft()
            steps += 1
            
            # If we found a visible label, return it immediately
            if not is_invisible_label(curr_trans.label):
                return curr_trans.label.strip()
            
            # Continue searching downstream
            for out_arc in curr_trans.out_arcs:
                next_place = out_arc.target
                for next_arc in next_place.out_arcs:
                    next_trans = next_arc.target
                    if next_trans not in visited:
                        visited.add(next_trans)
                        queue.append(next_trans)
        
        return None
    
    def __repr__(self):
        # for debug
        return f"<DecisionPoint '{self.place_name}' | In: {list(self.incoming_activities)} | Out: {list(self.outgoing_transitions.keys())}>"






