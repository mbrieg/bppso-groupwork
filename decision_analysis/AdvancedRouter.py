
import pandas as pd
from collections import defaultdict
class AdvancedRouter:
    """Decide according to the pkl created as a output."""

    def __init__(self, model, bins, pn, encoder=None, feature_names=None):
        self.model = model
        self.bins = bins
        self.pn = pn
        self.encoder = encoder            # None → C4.5 mode; set → RF mode
        self.feature_names = feature_names

    def _get_bin_label(self, value, bin_edges, labels):
        """sec/hours -> categories"""
        if value <= bin_edges[0]: return labels[0]
        if value >= bin_edges[-1]: return labels[-1]
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] < value <= bin_edges[i+1]:
                return labels[i]
        return labels[-1]

    def route(self, enabled, last_activity, last_duration, case_start_time, current_now,case_context=None):
        predicted_tid = None

        try:
            dur_labels = ['VeryShort', 'Short', 'Medium', 'Long', 'VeryLong']
            age_labels = ['Very_New', 'New', 'Old', 'Delayed']

            # Numerical --> Categorical
            dur_bin = self._get_bin_label(last_duration, self.bins["duration_bin"], dur_labels)
            age_hours = (current_now - case_start_time).total_seconds() / 3600
            age_cat = self._get_bin_label(age_hours, self.bins["case_age_category"], age_labels)

            _OFFER_LIFECYCLE = {'O_Create Offer','O_Created','O_Sent (mail and online)','O_Sent (online only)',
                                'O_Accepted','O_Cancelled','O_Returned','O_Refused'}
            _APP_DECISION    = {'A_Accepted','A_Concept','A_Complete','A_Validating',
                                'A_Incomplete','A_Cancelled','A_Denied','A_Pending'}
            _WORKFLOW        = {'W_Complete application','W_Validate application','W_Call after offers',
                                'W_Call incomplete files','W_Handle leads','W_Assess potential fraud',
                                'W_Shortened completion ','W_Personal Loan collection'}
            _CASE_START      = {'A_Create Application','A_Submitted'}
            def _grp(a):
                if a == 'START':          return 'start'
                if a in _OFFER_LIFECYCLE: return 'offer'
                if a in _APP_DECISION:    return 'app_decision'
                if a in _WORKFLOW:        return 'workflow'
                if a in _CASE_START:      return 'case_start'
                return 'other'

            raw_prev2 = str((case_context or {}).get("prev_activity_2", "START"))
            input_data = pd.Series({
                "prev_activity": str(last_activity),
                "prev_activity_2_group": _grp(raw_prev2),
                "duration_bin": dur_bin,
                "case_age_category": age_cat
            })

            if case_context:
                input_data["loan_goal"]           = str(case_context.get("loan_goal", "Unknown"))
                input_data["application_type"]    = str(case_context.get("application_type", "Unknown"))
                input_data["amount_category"]     = str(case_context.get("amount_category", "medium"))
                input_data["credit_score_bin"]    = str(case_context.get("credit_score_bin", "unknown"))
                input_data["offer_category"]      = str(case_context.get("offer_category", "none"))
                input_data["has_rejection"]       = str(case_context.get("has_rejection", 0))
                input_data["has_accepted_offer"]  = str(case_context.get("has_accepted_offer", 0))
                input_data["is_repeated"]         = str(case_context.get("is_repeated", "False"))
            enabled_labels = {self.pn.labels.get(tid, ""): tid for tid in enabled}

            if self.encoder is not None and self.feature_names is not None:
                # RF path: encode then predict
                input_df = pd.DataFrame([input_data[self.feature_names].to_dict()])
                X_enc = self.encoder.transform(input_df)
                predicted_label = self.model.predict(X_enc)[0]
                if predicted_label in enabled_labels:
                    predicted_tid = enabled_labels[predicted_label]
            else:
                # C4.5 path — stochastic sample from leaf distribution; None if label not in enabled
                if isinstance(self.model, dict):
                    tree = self.model.get(str(last_activity))
                    if tree is None:
                        raise KeyError(f"No tree for activity: {last_activity}")
                else:
                    tree = self.model

                predicted_label = tree.predict_one_stochastic(input_data)
                if predicted_label in enabled_labels:
                    predicted_tid = enabled_labels[predicted_label]
        except Exception as e:
            print(f"Advanced Router Error: {e}")
            return None  # Let DPManager handle fallback via BasicRouter

        return predicted_tid  # None if no matching transition found
        