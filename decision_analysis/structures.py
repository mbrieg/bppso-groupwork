from collections import deque

class DecisionPoint:
    """
    This class represents a decision point (XOR Split Place) in the petri net.
    There can be tau transitions used for routing loic that do not represent the real work.
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
        labels = {t.label.strip() for t in direct_preset if t.label is not None}

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
            if t.label is None: # confirming it is invisible
                for arc in t.in_arcs:
                    prev_place = arc.source
                    if prev_place not in visited_places:
                        #If its not visited add to the queue
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

                if t.label is not None:
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
    
    def __repr__(self):
        # for debug
        return f"<DecisionPoint '{self.place_name}' | In: {list(self.incoming_activities)} | Out: {list(self.outgoing_transitions.keys())}>"






