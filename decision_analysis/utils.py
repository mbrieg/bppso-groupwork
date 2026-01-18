import pandas as pd
import pm4py
import re

def is_invisible_label(label):
    """
    Checks if a transition label represents an invisible/routing step.
    Covers: None, 'tau', 'hid', 'skip', 'init_loop', 'sfl_sid', and raw UUIDs.
    """
    if label is None:
        return True
    
    label = str(label).strip()
    
    if not label:
        return True
    
    lower_label = label.lower()
    prefixes = ["tau", "hid", "skip", "init_loop", "sfl_sid"]
    if any(lower_label.startswith(p) for p in prefixes):
        return True
    
    # 2. UUID (32+ characters and containing "-")
    # "5f054da4-5097-4d02-99e6-68f5a25b9f1c"
    uuid_pattern = re.compile(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}', re.IGNORECASE)
    
    if uuid_pattern.search(label):
        if " " not in label:
            return True

    return False


def get_traces_from_log(log):
    """
    Analyses the given log (DataFrame veya EventLog)
    and returns a list of list of only activity names
    e.g.:
    [
      ['Application', 'Accepted', 'Pending'],
      ['Application', 'Refused', 'Pending'],
      ...
    ]
    """
    # 1. (EventLog obj or list)
    if isinstance(log, list) or isinstance(log, pm4py.objects.log.obj.EventLog):
        return log

    # 2. Pandas DataFrame --> groupBy
    if isinstance(log, pd.DataFrame):
        print("Log Utility: DataFrame detected. Converting using fast groupBy...")
        
        case_col = 'case:concept:name'
        act_col = 'concept:name'
        
        # not standard look at others
        if case_col not in log.columns:
            candidates = [c for c in log.columns if 'case' in c.lower() or 'id' in c.lower()]
            if candidates: case_col = candidates[0]
        
        if act_col not in log.columns:
            candidates = [c for c in log.columns if 'concept' in c.lower() or 'activ' in c.lower()]
            if candidates: act_col = candidates[0]

        print(f"Log Utility: Grouping by '{case_col}', Activity column: '{act_col}'")
        
        try:
            # Pandas GroupBy
            # apply(list) collecting the activites for a case in a list
            grouped = log.groupby(case_col)[act_col].apply(list)
            traces = grouped.values
            print(f"Log Utility: Successfully extracted {len(traces)} traces.")
            return traces
            
        except Exception as e:
            print(f"Log Utility Error: Grouping failed -> {e}")
            return []

    return []