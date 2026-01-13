# advanced_decision_point_analysis.py

from collections import defaultdict
from typing import Dict, List, Any
import os
import pickle

import pandas as pd
import numpy as np
from pm4py.algo.conformance.alignments.petri_net import algorithm as align_algo
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.dummy import DummyClassifier

# Existing modules
from basic_decision_point_analysis import (
    load_petri_from_bpmn,
    load_event_log,
    build_place_structures,
    build_transition_activity_mappings,
)
from c45_tree import C45DecisionTree

def extract_decision_training_data(
    log,
    net,
    im,
    fm,
    decision_places,
    transition_to_activity,
    feature_columns: List[str],
) -> Dict[Any, pd.DataFrame]:
    """
    Extracts historical data and Case Attributes using Alignment replay.
    """
    
    print(f"Alignments calculation ({len(log)} traces)...")
    alignments = align_algo.apply_log(log, net, im, fm)

    # Precompute structures for performance
    trans_preset_places = {}
    for t in net.transitions:
        preset = {arc.source for arc in t.in_arcs}
        trans_preset_places[t] = preset

    name_to_transition = {}
    for t in net.transitions:
        if t.label: 
            name_to_transition[t.label] = t
        elif t.name: 
            name_to_transition[t.name] = t

    place_to_rows = defaultdict(list)

    for case_idx, alignment_res in enumerate(alignments):
        aln = alignment_res["alignment"]
        trace = log[case_idx]
        
        # Get case start time
        case_start_time = None
        if len(trace) > 0 and "time:timestamp" in trace[0]:
            case_start_time = trace[0]["time:timestamp"]
        
        # Case attribute extraction
        current_case_attrs = {f"case:{k}": v for k, v in trace.attributes.items()}
        
        log_pos = -1 
        last_activity = "__START__"
        
        for model_move, log_move in aln:
            if log_move != ">>":
                log_pos += 1
            
            if model_move == ">>":
                continue

            t_obj = model_move if model_move in transition_to_activity else name_to_transition.get(model_move)
            
            if t_obj:
                preset_places = trans_preset_places.get(t_obj, set())
                dp_candidates = [p for p in preset_places if p in decision_places]
                
                if len(dp_candidates) == 1:
                    dp = dp_candidates[0]
                    act = transition_to_activity.get(t_obj, t_obj.label) 

                    row_data = {}
                    
                    # Temporal features
                    if log_pos >= 0 and log_pos < len(trace) and "time:timestamp" in trace[log_pos]:
                        curr_time = trace[log_pos]["time:timestamp"]
                        row_data["hour_of_day"] = curr_time.hour
                        row_data["day_of_week"] = curr_time.weekday()
                        
                        if case_start_time:
                            duration_seconds = (curr_time - case_start_time).total_seconds()
                            row_data["case_duration_hours"] = duration_seconds / 3600.0
                    else:
                        row_data["hour_of_day"] = -1
                        row_data["day_of_week"] = -1
                        row_data["case_duration_hours"] = 0

                    # Extract case attributes
                    for col in feature_columns:
                        if col.startswith("case:"):
                            val = current_case_attrs.get(col, None)
                            row_data[col] = val
                    
                    # History
                    row_data["prev_activity"] = last_activity
                    
                    # Target
                    row_data["y"] = act
                    
                    place_to_rows[dp].append(row_data)

            if t_obj and t_obj.label:
                last_activity = t_obj.label

    return {p: pd.DataFrame(rows) for p, rows in place_to_rows.items()}


def train_trees_per_place(
    place_to_df: Dict[Any, pd.DataFrame],
    attribute_types: Dict[str, str],
    random_state: int = 0,
    min_samples_per_place: int = 30,
    max_samples_for_training: int = 10000,
):
    """
    Training loop with cleaning, Cross-Validation, and Baseline comparison.
    Includes subsampling to handle large datasets efficiently.
    """
    place_to_model = {}

    for p, df in place_to_df.items():
        if df.empty:
            continue
        
        if len(df) < min_samples_per_place:
            print(f"Skipping place {p.name}: Not enough samples ({len(df)} < {min_samples_per_place})")
            continue

        df = df.copy()
        
        #SUBSAMPLING FOR SPEED
        original_size = len(df)
        if original_size > max_samples_for_training:
            print(f"  [Optimization] Dataset has {original_size:,} rows. Subsampling to {max_samples_for_training:,}...")
            try:
                df, _ = train_test_split(
                    df, 
                    train_size=max_samples_for_training,
                    stratify=df["y"],
                    random_state=random_state
                )
                print(f"  [Success] Subsampled while maintaining class balance.")
            except ValueError:
                print(f"  [Warning] Stratification failed. Using random sampling.")
                df = df.sample(n=max_samples_for_training, random_state=random_state)

        # Drop technical columns
        for col in ["case_index", "event_index", "prefix_len"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        if "y" not in df.columns:
            continue

        y = df["y"]
        X = df.drop(columns=["y"])

        # Numeric column cleaning
        for col, dtype in attribute_types.items():
            if col in X.columns and dtype == "numeric":
                X[col] = pd.to_numeric(X[col], errors='coerce')

        # Drop useless columns
        usable_cols = []
        for col in X.columns:
            if X[col].notna().sum() == 0:
                continue
            if X[col].nunique(dropna=True) <= 1:
                continue
            usable_cols.append(col)

        X = X[usable_cols]
        if X.shape[1] == 0:
            print(f"Skipping place {p.name}: No usable features left.")
            continue

        # Target check
        if y.nunique() < 2:
            print(f"Skipping place {p.name}: Only one target class found ({y.unique()[0]}).")
            continue

        print(f"Training for place {p.name} (Shape: {X.shape})...")

        # Spliting data
        try:
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=random_state
            )
            try:
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
                )
            except ValueError:
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=0.5, random_state=random_state
                )
        except ValueError as e:
            print(f"Skipping place {p.name} due to split error: {e}")
            continue

        # Cross-validation
        cv_scores = []
        if y_train.nunique() >= 2:
            min_class_count = y_train.value_counts().min()
            n_splits = min(5, min_class_count)
            
            if n_splits >= 2:
                kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                try:
                    print(f"  Running {n_splits}-fold Cross-Validation..")
                    for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train), 1):
                        X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[valid_idx]
                        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[valid_idx]

                        attr_types_sub = {a: t for a, t in attribute_types.items() if a in X_tr.columns}
                        
                        tree = C45DecisionTree(
                            attribute_types=attr_types_sub,
                            min_samples_split=50,
                            max_depth=10
                        )
                        tree.fit(X_tr, y_tr)
                        preds = tree.predict(X_va)
                        acc = (pd.Series(preds, index=y_va.index) == y_va).mean()
                        cv_scores.append(acc)
                        print(f"    Fold {fold_idx}: {acc:.4f}")
                    
                    print(f"  CV Mean Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
                    
                except Exception as e:
                    print(f"  CV warning for {p.name}: {e}")

        # Final model training
        X_train_full = pd.concat([X_train, X_val], axis=0)
        y_train_full = pd.concat([y_train, y_val], axis=0)
        
        attr_types_sub = {a: t for a, t in attribute_types.items() if a in X_train_full.columns}
        
        final_tree = C45DecisionTree(
            attribute_types=attr_types_sub,
            min_samples_split=50,
            max_depth=10
        ) 
        
        try:
            print(f"  Training final model on {len(X_train_full):,} samples...")
            final_tree.fit(X_train_full, y_train_full)
            test_preds = final_tree.predict(X_test)
            
            # Calculate metrics
            acc = accuracy_score(y_test, test_preds)
            report = classification_report(y_test, test_preds, output_dict=True, zero_division=0)
            f1 = report['weighted avg']['f1-score']
            
            # Baseline comparison
            dummy = DummyClassifier(strategy="most_frequent")
            dummy.fit(X_train_full, y_train_full)
            baseline_acc = dummy.score(X_test, y_test)
            
            improvement = acc - baseline_acc
            
            # Print results
            print(f"\n{'='*60}")
            print(f"RESULTS: {p.name}")
            print(f"{'='*60}")
            print(f"  Original Dataset Size : {original_size:,} episodes")
            print(f"  Training Set Size     : {len(X_train_full):,} samples")
            print(f"  Test Set Size         : {len(X_test):,} samples")
            print(f"  ---")
            print(f"  Tree Accuracy         : {acc:.4f}")
            print(f"  Baseline Acc          : {baseline_acc:.4f} (Most Frequent)")
            print(f"  Improvement           : {improvement:+.4f} ({improvement/baseline_acc*100:+.1f}%)")
            print(f"  F1-Score (Weighted)   : {f1:.4f}")
            if cv_scores:
                print(f"  CV Mean ± Std         : {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            print(f"  ---")
            print(f"  Per-Class Performance:")
            print(classification_report(y_test, test_preds, zero_division=0))
            print(f"{'='*60}\n")

            # Store model
            place_to_model[p] = {
                "tree": final_tree,
                "cv_scores": cv_scores,
                "test_acc": acc,
                "f1_score": f1,
                "baseline_acc": baseline_acc,
                "improvement": improvement,
                "features": list(X_train_full.columns),
                "original_samples": original_size,
                "training_samples": len(X_train_full),
            }

        except Exception as e:
            print(f"Training failed for {p.name}: {e}")
            import traceback
            traceback.print_exc()

    return place_to_model


def main():
    """
    End-to-end function with optimization, caching, export for simulator.
    """

    bpmn_path = "/Users/zeynepcetin/Decision Point Analysis/data folder/BPI Challenge 2017 Loan Application Process-6-4.bpmn"
    xes_path = "/Users/zeynepcetin/Decision Point Analysis/data folder/BPI Challenge 2017.xes.gz"
    output_dir = "/decision point analysis/advanced/decision_trees_output"
    
    # Configuration flags
    SAVE_EXTRACTED_DATA = True    # True at first
    LOAD_FROM_SAVED_DATA = False  # Later true
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("--- LOADING MODEL AND LOG ---")
    net, im, fm = load_petri_from_bpmn(bpmn_path)
    log = load_event_log(xes_path)

    print("--- BUILDING DECISION PLACES ---")
    structures = build_place_structures(net)
    
    if len(structures) == 3:
        decision_places, place_preset_transitions, place_postset_transitions = structures
    elif len(structures) >= 5:
        decision_places, place_preset_transitions, place_postset_transitions, _, _ = structures
    else:
        decision_places = structures[0]

    transition_to_activity, _ = build_transition_activity_mappings(net)

    feature_columns = [
        "case:LoanGoal",
        "case:ApplicationType",
        "case:RequestedAmount",
        "case:CreditScore", 
    ]

    print("--- EXTRACTING TRAINING DATA ---")
    
    # Load from saved data if available
    if LOAD_FROM_SAVED_DATA:
        print("[LOADING] Loading data from 'decision_data_full.pkl'...")
        try:
            with open("decision_data_full.pkl", "rb") as f:
                place_to_df = pickle.load(f)
            print("[SUCCESS] Data loaded from cache")
        except FileNotFoundError:
            print("[WARNING] Saved data not found. Extracting from scratch...")
            LOAD_FROM_SAVED_DATA = False
    
    # Extract from log
    if not LOAD_FROM_SAVED_DATA:
        place_to_df = extract_decision_training_data(
            log=log,
            net=net,
            im=im,
            fm=fm,
            decision_places=decision_places,
            transition_to_activity=transition_to_activity,
            feature_columns=feature_columns,
        )
        
        if SAVE_EXTRACTED_DATA:
            print("[SAVING] Data saved to 'decision_data_full.pkl'...")
            with open("decision_data_full.pkl", "wb") as f:
                pickle.dump(place_to_df, f)
            print("[SUCCESS] Data cached! Next time set LOAD_FROM_SAVED_DATA=True")

    print("\n--- Data Summary ---")
    for p, df in place_to_df.items():
        name = getattr(p, "name", repr(p))
        if not df.empty:
            print(f"Place {name}: {len(df)} rows. Cols: {list(df.columns)}")
        else:
            print(f"Place {name}: No data found.")

    attribute_types = {
        "prev_activity": "nominal",
        "case:LoanGoal": "nominal",
        "case:ApplicationType": "nominal",
        "case:RequestedAmount": "numeric",
        "case:CreditScore": "numeric",
        "hour_of_day": "numeric",      
        "case_duration_hours": "numeric"
    }

    print("\n--- TRAINING TREES ---")
    place_to_model = train_trees_per_place(
        place_to_df=place_to_df,
        attribute_types=attribute_types,
        min_samples_per_place=30,
        max_samples_for_training=10000,
    )

    if not place_to_model:
        print("No models trained.")
        return

    print("\n--- RESULTS & EXPORT ---")
    summary_data = []
    
    simulator_export_data = {}

    for p, info in place_to_model.items():
        name = getattr(p, "name", repr(p)).replace(" ", "_").replace(":", "")
        print(f"\nDecision Place: {name}")
        print(f"  Features Used: {info['features']}")
        print(f"  Test Accuracy: {info['test_acc']:.3f}")
        print(f"  Improvement over Baseline: {info['improvement']:+.3f}")
        
        place_id_str = getattr(p, "name", str(p))
        simulator_export_data[place_id_str] = {
            "tree": info["tree"],
            "features": info["features"]
        }

        # Feature importance analysis
        try:
            importances = info["tree"].get_feature_importance()
            sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            print(" Top Features Influencing Decision:")
            for feature, score in sorted_imp[:5]:
                print(f"    - {feature}: {score:.4f}")
            
            top_features = ", ".join([f[0] for f in sorted_imp[:3]])
        except Exception as e:
            print(f"  Could not calculate importance: {e}")
            top_features = "N/A"

        # Export visualization
        tree = info["tree"]
        dot_data = tree.export_graphviz()
        
        filename = os.path.join(output_dir, f"tree_{name}.dot")
        with open(filename, "w") as f:
            f.write(dot_data)
        print(f"  Graphviz tree saved to: {filename}")
        
        # Collect summary data
        summary_data.append({
            "Decision Place": name,
            "Episodes": info.get('original_samples', 'N/A'),
            "Training Samples": info.get('training_samples', 'N/A'),
            "Test Accuracy": f"{info['test_acc']:.4f}",
            "Baseline Acc": f"{info['baseline_acc']:.4f}",
            "Improvement": f"{info['improvement']:+.4f}",
            "F1-Score": f"{info['f1_score']:.4f}",
            "Top Features": top_features
        })
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    summary_path = os.path.join(output_dir, "results_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[SAVED] Summary table saved to: {summary_path}")

    
    print("\n" + "="*80)
    print("SAVING MODELS FOR SIMULATOR")
    print("="*80)
    sim_model_path = "decision_models.pkl"
    with open(sim_model_path, "wb") as f:
        pickle.dump(simulator_export_data, f)
    
    print(f"[SUCCESS] Advanced Decision Models saved to '{sim_model_path}'")
    print(f"Ready for integration to the simulator engine")
    
    print(f"\nDone. Check '{output_dir}' folder.")


if __name__ == "__main__":
    main()
