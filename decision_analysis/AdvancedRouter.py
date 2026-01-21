
import pandas as pd
from collections import defaultdict
class AdvancedRouter:
    """Decide according to the pkl created as a output of the xor_d.ipynb"""

    def __init__(self, model, bins, pn):
        self.model = model
        self.bins = bins
        self.pn = pn

    def _get_bin_label(self, value, bin_edges, labels):
        """sec/hours -> categories"""
        if value <= bin_edges[0]: return labels[0]
        if value >= bin_edges[-1]: return labels[-1]
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] < value <= bin_edges[i+1]:
                return labels[i]
        return labels[-1]

    def route(self, enabled, last_activity, last_duration, case_start_time, current_now):
        try:
            dur_labels = ['VeryShort', 'Short', 'Medium', 'Long', 'VeryLong']
            age_labels = ['Very_New', 'New', 'Old', 'Delayed']

            # Numerical --> Categorical
            dur_bin = self._get_bin_label(last_duration, self.bins["duration_bin"], dur_labels)
            age_hours = (current_now - case_start_time).total_seconds() / 3600
            age_cat = self._get_bin_label(age_hours, self.bins["case_age_category"], age_labels)

            input_data = pd.Series({
                "prev_activity": str(last_activity),
                "duration_bin": dur_bin,
                "case_age_category": age_cat
            })

            predicted_label = self.model.predict_one(input_data)

            for tid in enabled:
                if tid == predicted_label or self.pn.labels.get(tid, "") == predicted_label:
                    return tid
        except Exception as e:
            print(f"Advanced Router Error: {e}")
            return None