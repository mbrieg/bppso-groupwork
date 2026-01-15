import pm4py
import os
from .discovery import HeuristicProcessDiscovery
from .structures import DecisionPoint
from .basic_router import BasicRouter
# add advanced _router 

class DecisionPointManager:
    """
    The bridge between Engine and the Decision analysis modules.
    1. Discovers the process model from the log (Process Discovery)
    2. Analyses the petri net structure to find decision points with backtraking
    3. Trains the router (Basic: Probabilistic / Advanced: Deciison Tree)
    4. Answers the question : "what happens next?" during simulation 
    """

    def __init__(self, log, mode='basic', output_folder="data", config=None):
        """
        log: The event log object.
        mode: 'basic' (Probabilistic) or 'advanced' (Data-aware/ML).
        output_folder: Where to save the discovered pnml file.
        config: Optional dictionary for parameters (e.g., horizon).
        
        """
        self.log = log
        self.mode = mode
        self.config = config if config else {}
        
        # model discovery
        print("Manager: Starting Process Discovery...")
        self.miner = HeuristicProcessDiscovery(
            dependency_threshold=0.8, 
            and_threshold=0.65
        )
        self.net, self.im, self.fm = self.miner.discover(self.log)
        
        # save the model --> engine needs to read this file
        self._save_models(output_folder)
        
        # Structural analysis
        # DecisionPoint class from structures.py
        print("Manager: Analyzing Decision Points...")
        self.decision_points = self._analyse_structure()
        
        # Train the router (BASIC / ADVANCED)
        print(f"Manager: Training Router (Mode: {mode})...")
        if mode == 'advanced':
            # fall back to basic safely
            # self.router = AdvancedRouter(self.log, self.decision_points)
            self.router = BasicRouter(self.log, self.decision_points)
        else:
            # BasicRouter class from basic.py
            self.router = BasicRouter(self.log, self.decision_points)
        
        # get horizon from config, default is 60 steps
        horizon = self.config.get('horizon', 60)
        self.router.train(horizon=horizon)

        print("DecisionPointManager is ready.")

    def _save_models(self, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        pnml_path = os.path.join(output_folder, "discovered_model.pnml")
        pm4py.write_pnml(self.net, self.im, self.fm, pnml_path)
        print(f" --> Model saved to: {pnml_path}")

    def _analyse_structure(self):
        """
        Scans the petri net and converts XOR places into DecisionPoint objects
        """

        dps = {}
        for place in self.net.places:
            # If a place has 2 or more outgoing arcs, it is a DecisionPoint (XOR Split).
            if len(place.out_arcs) >= 2:
                dp = DecisionPoint(place.name, place)
                
                # 1- Use Backtracking to find the REAL incoming activities (structures.py)
                dp.analyse_preset(max_depth=2)
                
                # 2- Register outgoing paths
                for arc in place.out_arcs:
                    trans = arc.target
                    if trans.label: # Only visible activities can be selected as a 'choice'
                        dp.add_outgoing(trans, trans.label.strip())
                
                # Only add if it has valid outgoing options
                if dp.get_possible_activities():
                    dps[place.name] = dp
        
        print(f"  -> Found {len(dps)} decision points in the model.")
        return dps
    
    def get_next_transition(self, current_place, trace_history):
        """
        The main function called by the simulation engine
        current_place: the petri net object token is currently here
        trace_histroy: a list of acts that happened so far, or a dict context like (prev_act : A)

        returns: 
        the selected transition object to fire
        """
        # is this a known Decision Point?
        if current_place.name not in self.decision_points:
            # if not (only 1 path exists), return the default path.
            if current_place.out_arcs:
                return list(current_place.out_arcs)[0].target
            return None # dead end

        # retrieve the Decision Point Object
        dp = self.decision_points[current_place.name]
        
        # extract Context (Previous Activity)
        # We need the last activity from the history to use conditioned probability.
        prev_act = None
        if isinstance(trace_history, dict):
            prev_act = trace_history.get('prev_activity')
        elif isinstance(trace_history, list) and trace_history:
            prev_act = trace_history[-1]
        elif isinstance(trace_history, str):
            prev_act = trace_history

        # ask the Router: "Where should I go?"
        predicted_activity_name = self.router.predict(current_place.name, prev_act)

        # convert Name back to Transition Object
        if predicted_activity_name and predicted_activity_name in dp.outgoing_transitions:
            return dp.outgoing_transitions[predicted_activity_name]
        
        # fallback 
        # if the Router returns None (unknown path) or something went wrong, 
        # pick the first valid option to prevent simulation crash.
        valid_transitions = list(dp.outgoing_transitions.values())
        if valid_transitions:
            return valid_transitions[0]
        
        # absolute fallback (should probably not reach here)
        return list(current_place.out_arcs)[0].target
        

