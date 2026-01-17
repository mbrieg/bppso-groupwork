import os
import sys
import pickle
import random
import pandas as pd
import numpy as np
from collections import defaultdict
from pm4py.algo.conformance.alignments.petri_net import algorithm as align_algo
from pm4py.objects.log.obj import EventLog
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.dummy import DummyClassifier

from .c45_tree import C45DecisionTree

class AdvancedRouter:
    """
    Uses Alignments to replay history and C4.5 Trees to learn from:
      1. Control Flow (Previous Activity)
      2. Case Attributes (e.g. Amount, Goal)
      3. Temporal Features (Hour, Day, Duration)
    """

    def __init__(self, log, decision_points, net, im, fm):
        self.log = log
        self.decision_points = decision_points
        self.net = net
        self.im = im
        self.fm = fm
        
        self.classifiers = {} 
        self.feature_names = {} # stores which features each tree uses
        
        # configuration for data extraction
        self.case_features = [
            "case:LoanGoal",
            "case:ApplicationType", 
            "case:RequestedAmount"
        ]
        # features that C4.5 should treat as nominal (categorical)
        self.nominal_cols = [
            "prev_activity", 
            "case:LoanGoal", 
            "case:ApplicationType", 
            "day_of_week"
        ]

    def train(self, horizon=60):
        print(f"Training Advanced Router (Alignments + Data-Aware C4.5)...")
        
        # 1. extract data (using alignments)
        place_to_df = self._extract_training_data()

        # 2. defining attribute types
        # We need to tell the tree which columns are numeric vs nominal
        # (This is dynamic based on what columns actually exist in the df)
        base_attr_types = {
            "prev_activity": "nominal",
            "hour_of_day": "numeric",
            "day_of_week": "nominal",
            "case_duration_hours": "numeric",
            "case:RequestedAmount": "numeric",
            "case:LoanGoal": "nominal",
            "case:ApplicationType": "nominal"
        }

        # 3. train trees loop
        trained_count = 0
        
        for dp_name, df in place_to_df.items():
            if df.empty: continue
            
            print(f"\n--- Training Decision Point: {dp_name} ---")
            
            # Pre-processing and subsampling
            df = self._preprocess_data(df)
            if df.empty: continue

            X = df.drop(columns=["y"])
            y = df["y"]

            # If there is only 1 possible outcome (e.g. "skip_60"), don't train a tree.
            # Just verify it and create a dummy predictor.
            if len(y.unique()) < 2:
                print(f"    > Deterministic point (only '{y.unique()[0]}'). Creating dummy model.")
                # Create a simple tree that always predicts the one available class
                dummy_tree = C45DecisionTree(attribute_types={})
                dummy_tree.fit(X, y)
                self.classifiers[dp_name] = dummy_tree
                self.feature_names[dp_name] = list(X.columns)
                trained_count += 1
                continue

            # construct attr_types for this specific dataframe
            current_attr_types = {}
            for col in X.columns:
                # default to numeric if not specified
                dtype = base_attr_types.get(col, "numeric")
                # override if it's in our nominal list
                if col in self.nominal_cols: 
                    dtype = "nominal"
                current_attr_types[col] = dtype

            #  train / validate
            try:
                final_tree, score = self._train_single_tree(X, y, current_attr_types)
                
                #  Save Model
                self.classifiers[dp_name] = final_tree
                self.feature_names[dp_name] = list(X.columns)
                trained_count += 1
                
                print(f"    > Accuracy: {score:.4f}")
                
            except Exception as e:
                print(f"    > Failed to train: {e}")

        print(f"\nAdvanced Router trained. Models built for {trained_count}/{len(self.decision_points)} decision points.")

    def predict(self, place_name, context):
        """
        Predicts next transition.
        place_name: Name of the current place (Decision Point)
        context: Dictionary containing:
            - 'history': list of prev activities
            - 'attributes': dict of case data {'case:Amount': 5000}
            - 'time': dict {'hour': 10, 'day': 2, 'duration': 3600} (Optional)
        """
        clf = self.classifiers.get(place_name)
        if not clf: return None

        # prepare input vector
        # we must construct a dictionary matching the training features
        features_needed = self.feature_names.get(place_name, [])
        input_vector = {}

        # 1. history
        history = context.get('history', [])
        prev_act = history[-1] if history else "__START__"
        input_vector['prev_activity'] = prev_act

        # 2. case attributes
        attributes = context.get('attributes', {})
        for k, v in attributes.items():
            input_vector[k] = v
            
        # 3. if simulator provides it -> otherwise use defaults
        # Include processing times and resources !!
        time_ctx = context.get('time', {})
        input_vector['hour_of_day'] = time_ctx.get('hour', 12) # Default noon
        input_vector['day_of_week'] = time_ctx.get('day', 0)   # Default Monday
        input_vector['case_duration_hours'] = time_ctx.get('duration', 0.0)

        # filter only needed features and handle missing ones
        final_input = {}
        for feat in features_needed:
            final_input[feat] = input_vector.get(feat, 0) # 0 or safe default

        try:
            return clf.predict_one(final_input)
        except:
            return None


    def _extract_training_data(self):
        """
        Uses pm4py alignments to extract precise context for every decision.
        """
        cache_file = "alignment_cache.pkl"
        
        # try to load from cache
        if os.path.exists(cache_file):
            print(f"  Found '{cache_file}'. Loading pre-calculated alignments...")
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"  Failed to load cache: {e}. Recalculating")


        SAMPLE_SIZE = 2000
        
        if len(self.log) > SAMPLE_SIZE:
            print(f"  Log is huge ({len(self.log)} traces). Sampling {SAMPLE_SIZE} traces for alignment...")
            # Sample from the list version of the log to ensure random selection
            log_for_training = EventLog(random.sample(list(self.log), SAMPLE_SIZE))
        else:
            log_for_training = self.log

        print(f"  Calculating alignments for {len(log_for_training)} traces (this may take a while)...")
        try:
            alignments = align_algo.apply_log(log_for_training, self.net, self.im, self.fm)
        except Exception as e:
            print(f"  Error: alignment failed ({e}). Returning empty data.")
            return {}

        # 1. map Transitions to Activities
        trans_map = {}
        for t in self.net.transitions:
            if t.label: 
                trans_map[t] = t.label.strip()
            else: # added for advanced router invisible change test
                #Give a name to invisible transitions so the router sees them
                trans_map[t] = t.name
        
        # 2. map Transitions to their Preset Places
        trans_preset = {}
        for t in self.net.transitions:
            trans_preset[t] = {arc.source.name for arc in t.in_arcs}

        place_to_rows = defaultdict(list)
        decision_place_names = set(self.decision_points.keys())

        # 3. replay alignments
        for case_idx, res in enumerate(alignments):
            aln = res['alignment']
            trace = log_for_training[case_idx]
            
            # case attributes
            # ensure keys match self.case_features (e.g. "case:Amount")
            c_attrs = {f"case:{k}": v for k,v in trace.attributes.items()}
            
            # time setup
            start_time = None
            if len(trace) > 0 and "time:timestamp" in trace[0]:
                start_time = trace[0]["time:timestamp"]

            log_pos = -1
            last_activity = "__START__"

            for model_move, log_move in aln:
                if log_move != ">>":
                    log_pos += 1
                
                if model_move == ">>": continue # Skip invisible model steps if any

                # Find transition object in net
                # Alignments return (name, label), we need to match 
                # For simplicity, we assume model_move is the Transition Label or Name
                curr_trans_obj = None
                for t in self.net.transitions:
                    if t.name == model_move[0] or t.label == model_move[1]:
                        curr_trans_obj = t
                        break
                    
                if not curr_trans_obj and model_move[1] is not None:
                     for t in self.net.transitions:
                        if t.label == model_move[1]:
                            curr_trans_obj = t
                            break
                
                if not curr_trans_obj: continue

                # is this transition triggered from a DP?
                if curr_trans_obj in trans_preset:
                    sources = trans_preset[curr_trans_obj]
                    # Find intersection with known decision places
                    active_dps = sources.intersection(decision_place_names)
                    
                    if len(active_dps) == 1:
                        dp_name = list(active_dps)[0]
                        outcome = trans_map.get(curr_trans_obj)

                        if outcome:
                            # build row
                            row = {}
                            row['prev_activity'] = last_activity
                            row['y'] = outcome
                            
                            # Add case attrs
                            for feat in self.case_features:
                                row[feat] = c_attrs.get(feat, None)

                            # add time
                            if log_pos >= 0 and log_pos < len(trace) and "time:timestamp" in trace[log_pos]:
                                curr_t = trace[log_pos]["time:timestamp"]
                                row['hour_of_day'] = curr_t.hour
                                row['day_of_week'] = curr_t.weekday()
                                if start_time:
                                    row['case_duration_hours'] = (curr_t - start_time).total_seconds() / 3600.0
                            else:
                                row['hour_of_day'] = 12
                                row['day_of_week'] = 0
                                row['case_duration_hours'] = 0

                            place_to_rows[dp_name].append(row)

                if curr_trans_obj.label:
                    last_activity = curr_trans_obj.label

        final_data = {k: pd.DataFrame(v) for k,v in place_to_rows.items()}
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(final_data, f)
            print(f"  Saved extracted data to '{cache_file}'.")
        except:
            print("   Warning: Could not save cache file.")

        return final_data
    

    def _preprocess_data(self, df, max_samples=10000):
        # 1. subsampling if too large
        if len(df) > max_samples:
            try:
                df, _ = train_test_split(df, train_size=max_samples, stratify=df['y'], random_state=42)
            except:
                df = df.sample(n=max_samples, random_state=42)
        
        # 2. clean Numeric Columns coerce errors)
        # 3. drop columns with all NaNs or single value
        cols_to_drop = []
        for col in df.columns:
            if col == 'y': continue
            if df[col].nunique(dropna=True) <= 1:
                cols_to_drop.append(col)
        
        return df.drop(columns=cols_to_drop)

    def _train_single_tree(self, X, y, attr_types):
        """
        Train-Test Split 
        Baseline comparison 
        Final Fit
        """
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        # Train
        tree = C45DecisionTree(
            attribute_types=attr_types,
            min_samples_split=30,  # prevent overfitting
            max_depth=7
        )
        tree.fit(X_train, y_train)
        
        # evaluation
        preds = tree.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # re-train on full data for production
        final_tree = C45DecisionTree(
            attribute_types=attr_types,
            min_samples_split=30,
            max_depth=7
        )
        final_tree.fit(X, y) # Full fit
        
        return final_tree, acc