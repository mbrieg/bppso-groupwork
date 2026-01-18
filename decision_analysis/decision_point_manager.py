import pm4py
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'sim_core'))
from .utils import is_invisible_label

from sim_core.bpmn_io import read_bpmn
from sim_core.pn_model import wrap_net

from collections import deque
from .structures import DecisionPoint
from .basic_router import BasicRouter
from .advanced_router import AdvancedRouter

class DecisionPointManager:
    """
    The bridge between Engine and the Decision analysis modules.
    """

    def __init__(self, log, bpmn_path=None, net=None, im=None, fm=None, mode='basic', output_folder="data", config=None):
        self.log = log
        self.mode = mode
        self.config = config if config else {}

        if net is None or im is None or fm is None:
            if bpmn_path is None:
                raise ValueError(
                    "Must provide either:\n"
                    "bpmn_path (to load from file using sim_core), OR\n"
                    "(net, im, fm) if already loaded"
                )
            
            print(f"Manager: Loading BPMN model using sim_core from: {bpmn_path}")
            
            # Use sim_core's BPMN loader
            try:
                self.net, self.im, self.fm = read_bpmn(bpmn_path)
                print(f" Successfully loaded Petri net with sim_core")
            except Exception as e:
                print(f" Error loading BPMN with sim_core: {e}")
                raise
            
            # wrap it in PNModel for consistency with sim_core
            self.pn_model = wrap_net(self.net, self.im, self.fm)
            print(f" Wrapped as PNModel: {len(self.pn_model.place_ids)} places, "
                  f"{len(self.pn_model.trans_ids)} transitions")
            
            self.model_filename = os.path.basename(bpmn_path).replace('.bpmn', '.pnml')
        else:
            print(f"Manager: Using provided Petri net model...")
            self.net = net
            self.im = im
            self.fm = fm
            
            # Wrap provided net in PNModel
            self.pn_model = wrap_net(self.net, self.im, self.fm)
            self.model_filename = "provided_model.pnml"
        
        # Save the model
        self._save_models(output_folder)
        
        print("Manager: Analyzing Decision Points...")
        self.decision_points = self._analyse_structure()
        
        print(f"Manager: Training Router (Mode: {mode})...")
        if mode == 'advanced':
            self.router = AdvancedRouter(
                self.log, 
                self.decision_points, 
                self.net, 
                self.im, 
                self.fm
            )
        else:
            print("Manager: Pre-processing log for Basic Router...")
            simple_log = self._preprocess_log_for_basic_router()
            self.router = BasicRouter(simple_log, self.decision_points)
        
        horizon = self.config.get('horizon', 60)
        
        # Basic router train()
        try:
            self.router.train(horizon=horizon, debug=(mode=='basic'))
        except TypeError:
            self.router.train(horizon=horizon)

        print("DecisionPointManager is ready.")
        

    def _preprocess_log_for_basic_router(self):
        """
        Convert event log to simple list of activity name lists for Basic router
        """
        simple_log = []
        for trace in self.log:
            simple_trace = []
            for event in trace:
                name = None
                if hasattr(event, 'get'): 
                    name = event.get('concept:name')
                elif hasattr(event, '_dict'): 
                    name = event._dict.get('concept:name')
                else: 
                    name = str(event)
                
                if name: 
                    simple_trace.append(name.strip())
            simple_log.append(simple_trace)
        return simple_log

    def _save_models(self, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        pnml_path = os.path.join(output_folder, self.model_filename)
        pm4py.write_pnml(self.net, self.im, self.fm, pnml_path)
        print(f" --> Model saved to: {os.path.abspath(pnml_path)}")

    def _analyse_structure(self):
        """
        Scans the petri net and converts XOR places into DecisionPoint objects
        """
        dps = {}
        for place in self.net.places:
            # Only add if it has valid outgoing options
            if len(place.out_arcs) >= 2:
                dp = DecisionPoint(place.name, place)
                
                # 1. Backtracking
                dp.analyse_preset(max_depth=2)
               
                for arc in place.out_arcs:
                    trans = arc.target
                    label = trans.label
                    name = trans.name

                    is_hidden = is_invisible_label(trans.label)
                    
                    activity_name = None

                    if is_hidden:
                        # Basic Router + Advanced için görünmez yolları çöz
                        activity_name = self._resolve_downstream_activity(trans)
                        if not activity_name:
                            activity_name = trans.name
                    elif trans.label:
                        activity_name = trans.label.strip()
                    else:
                        # Advanced mode veya görünür ama etiketsiz (nadiren olur)
                        activity_name = trans.name
                    
                    if activity_name:
                        dp.add_outgoing(trans, activity_name)
                
                if dp.get_possible_activities():
                    dps[place.name] = dp
        
        print(f"  -> Found {len(dps)} decision points in the model.")
        return dps

    def _resolve_downstream_activity(self, start_trans):
        """
        BASIC MODE ONLY for now:
        Looks past silent transitions to find the next visible activity name.
        Uses BFS to find the nearest real activity.
        """
        queue = deque([start_trans])
        visited = {start_trans}
        
        # Maksimum 100 adım ileri git (Güvenlik)
        steps = 0
        while queue and steps < 100:
            curr_trans = queue.popleft()
            steps += 1
            
            # 1. Bu geçişin etiketi geçerli bir aktivite mi?
            if not is_invisible_label(curr_trans.label):
                return curr_trans.label.strip()
            
            # 2. Değilse, bir sonraki adımları kuyruğa ekle
            for out_arc in curr_trans.out_arcs:
                next_place = out_arc.target
                for next_arc in next_place.out_arcs:
                    next_trans = next_arc.target
                    if next_trans not in visited:
                        visited.add(next_trans)
                        queue.append(next_trans)
            
        return None

    def get_next_transition(self, current_place, trace_history):

        if current_place.name not in self.decision_points:
            if current_place.out_arcs:
                return list(current_place.out_arcs)[0].target
            return None 
        
        # retrieve the Decision Point Object
        dp = self.decision_points[current_place.name]

        prediction_input = None
        
        if self.mode == 'advanced':
            if isinstance(trace_history, dict):
                prediction_input = trace_history
            else:
                prediction_input = {'history': trace_history if isinstance(trace_history, list) else []}
        else:
            # BasicRouter
            if isinstance(trace_history, dict):
                prediction_input = trace_history.get('prev_activity') or trace_history.get('history', [])[-1]
            elif isinstance(trace_history, list) and trace_history:
                prediction_input = trace_history[-1]
            elif isinstance(trace_history, str):
                prediction_input = trace_history
        
        # ask the Router: "Where should I go?"
        predicted_activity_name = self.router.predict(current_place.name, prediction_input)

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
    
    def get_pn_model(self):
        """Returns the PNModel representation for use with other sim_core components"""
        return self.pn_model