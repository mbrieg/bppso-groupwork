import random
import pandas as pd
from datetime import datetime

class Router:
    def __init__(self, mode="random", basic_model = None, advanced_model = None ):
        """
        Docstring for __init__
        
        :param self: 
        :param mode: random / basic / advanced
        :param basic_model: model dict from basic_decision_analysis.py
        :param advanced_model: model dict form advanced_deicison_analysis.py
        """

        valid_modes = ["random","basic","advanced"]

        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, but is '{mode}'")
        
        self.mode = mode
        self.basic_model = basic_model
        self.advanced_model = advanced_model

        self.basic_probs = {}
        if self.mode in ["basic", "advanced"] and self.basic_model:
            self._prepare_basic_model()

        if mode == "basic" and not self.basic_probs:
            print(" basic mode selected but model prep failed. ")

        if mode == "advanced" and not self.advanced_model:
            print(" advanced mode selected but no advanced model loaded. ")

    def _prepare_basic_model(self):
        "place objects --> strings"

        try:
            cond = self.basic_model.get("branch_probabilities_conditioned", {})
            marg = self.basic_model.get("branch_probabilities_marginal", {})
            
            if not cond:
                print(" Basic model has no conditioned probabilities")
                return
            
            # Place mapping (obj -> string)
            place_map = {}
            for p in list(cond.keys()) + list(marg.keys()):
                if hasattr(p, "name"):
                    place_map[p] = p.name
                else:
                    place_map[p] = str(p)
            
            # Organize according to string IDs
            for p_obj, probs in cond.items():
                p_name = place_map.get(p_obj, str(p_obj))
                self.basic_probs[p_name] = {
                    "cond": probs,
                    "marg": marg.get(p_obj, {})
                }
            
            print(f" Basic model prepared: {len(self.basic_probs)} decision places")
            
        except Exception as e:
            print(f" Error preparing basic model: {e}")
            import traceback
            traceback.print_exc()
    
    def get_decision_place(self, enabled_ids, pn_inputs, marking):
        """
        Find the common input placec of enabled transitions
        Return only the places which have tokens
        """
        if not enabled_ids:
            return None
        
        common_inputs = set(pn_inputs.get(enabled_ids[0], []))
        
        # intrsection
        for tid in enabled_ids[1:]:
            inputs = set(pn_inputs.get(tid, []))
            common_inputs.intersection_update(inputs)
        
        # return the first place with token
        for place in common_inputs:
            if marking.get(place, 0) > 0:
                return place
        
        return None
    
    def decide(self, enabled_ids, pn, case_meta, marking, current_time=None):
        """
        Decide which transition to be fired
        enabled_ids: ids of enabled transition
        pn: pn model 
        case_meta: {'history': [...], 'attributes': {...}, 'start_time': datetime}
        marking: current marking (place -> token count)
        current_time: current simulation time (for advanced)

        return 
            str: choosen transition id
        """
        if len(enabled_ids) == 1:
            return enabled_ids[0]
        
        # Find decision place
        place_id = self.get_decision_place(enabled_ids, pn.inputs, marking)

        if not place_id:
            # e.g. parallel --> random
            return random.choice(enabled_ids)
        
        # Decide according to the model
        if self.mode == "basic":
            return self._decide_basic(place_id, enabled_ids, case_meta, pn)
        elif self.mode == "advanced":
            return self._decide_advanced(place_id, enabled_ids, case_meta, pn, current_time)
        # as default for safety
        return random.choice(enabled_ids)
    

    def _decide_basic(self, place_id, enabled_ids, case_meta, pn):
        """
        Decide according to basic probability table.
        """
        if place_id not in self.basic_probs:
            # if no model available --> random
            return random.choice(enabled_ids)
            
        probs = self.basic_probs[place_id]
        history = case_meta.get("history", [])
        last_activity = history[-1] if history else None
        
        # Conditioned / Marginal probability
        target_dist = None
        
        if last_activity and last_activity in probs["cond"]:
            # Conditioned: P(next | place, previous_activity)
            target_dist = probs["cond"][last_activity]
        else:
            # Marginal: P(next | place)
            target_dist = probs["marg"]
            
        if not target_dist:
            return random.choice(enabled_ids)

        # Enabled transition < -- > probability mapping
        candidates = {}
        for tid in enabled_ids:
            label = pn.labels.get(tid, "").strip()
            if not label:  # Silent transition'ları atla
                continue
            
            prob = target_dist.get(label, 0.0)
            if prob > 0:
                candidates[tid] = prob
        
        if not candidates:
            # no enabled transition model --> random
            return random.choice(enabled_ids)
        
        # Roulette wheel selection
        total = sum(candidates.values())
        if total <= 0:
            return random.choice(enabled_ids)
        
        r = random.uniform(0, total)
        cumulative = 0.0
        
        for tid, prob in candidates.items():
            cumulative += prob
            if cumulative >= r:
                return tid
        
        # Fallback
        return list(candidates.keys())[-1]

    def _decide_advanced(self, place_id, enabled_ids, case_meta, pn, current_time):
        """
        Uses ML model, if fails back to the basic. 
        """
        if not self.advanced_model or place_id not in self.advanced_model:
            # If there is no place for the dvanced model -->  basic
            return self._decide_basic(place_id, enabled_ids, case_meta, pn)
        
        try:
            model_data = self.advanced_model[place_id]
            tree = model_data.get("tree") # c4.5 decision tree object
            features_needed = model_data.get("features", [])
            
            if not tree:
                return self._decide_basic(place_id, enabled_ids, case_meta, pn)
            
            # Feature vector for ML
            input_row = self._prepare_features(case_meta, current_time, features_needed)
            
            # ML prediction
            df_input = pd.DataFrame([input_row])
            prediction = tree.predict(df_input)[0]  # Activity name 
            
            print(f" [Advanced Router] Place: {place_id} | Input: {input_row['case:RequestedAmount']} | Prediction: {prediction}")
            
            # Predicted activity < -- > enabled transitions
            for tid in enabled_ids:
                label = pn.labels.get(tid, "").strip()
                if label == prediction:
                    return tid
            
            # If prediction is not enabled --> basic
            return self._decide_basic(place_id, enabled_ids, case_meta, pn)
            
        except Exception as e:
            print(f" Advanced decision failed for place {place_id}: {e}")
            # Fall back to basic
            return self._decide_basic(place_id, enabled_ids, case_meta, pn)

    def _prepare_features(self, case_meta, current_time, features_needed):
        """
        Feature vector for advanced 
        """
        features = {}
        
        # History
        history = case_meta.get("history", [])
        features["prev_activity"] = history[-1] if history else "__START__"
        
        # Temporal features
        if current_time:
            features["hour_of_day"] = current_time.hour
            features["day_of_week"] = current_time.weekday()
            
            # Case duration
            start_time = case_meta.get("start_time")
            if start_time:
                duration_seconds = (current_time - start_time).total_seconds()
                features["case_duration_hours"] = duration_seconds / 3600.0
            else:
                features["case_duration_hours"] = 0.0
        else:
            features["hour_of_day"] = 0
            features["day_of_week"] = 0
            features["case_duration_hours"] = 0.0
        
        # Case attributes
        attrs = case_meta.get("attributes", {})
        for key, value in attrs.items():
            features[key] = value
        
        # Only the features model expects
        result = {}
        for feat in features_needed:
            result[feat] = features.get(feat, None)
        
        return result    