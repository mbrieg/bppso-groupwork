import random
from datetime import datetime, timedelta
import pandas as pd

# -----------------------------
# 1) HARDCODED CONTROL-FLOW
#    (aus deinem process_model.bpmn extrahiert)
# -----------------------------

# Für jeden Task: mögliche nächste Tasks (vereinfachter Kontrollfluss, Gateways "aufgelöst")
SUCCESSORS = {
    "A_Create Application": {"A_Concept", "A_Submitted", "W_Complete application"},
    "A_Submitted": {"W_Handle leads"},
    "W_Handle leads": {"A_Concept", "W_Complete application", "W_Handle leads"},
    "A_Concept": {"A_Accepted"},
    "W_Complete application": {"A_Accepted", "W_Complete application"},
    "A_Accepted": {"O_Create Offer"},
    "O_Create Offer": {"O_Created"},
    "O_Created": {
        "A_Complete",
        "O_Cancelled",
        "O_Sent (mail and online)",
        "W_Call after offers",
    },
    "O_Sent (mail and online)": {
        "A_Cancelled",
        "A_Validating",
        "O_Cancelled",
        "O_Create Offer",
        "W_Validate application",
    },
    "W_Call after offers": {
        "A_Cancelled",
        "A_Validating",
        "O_Cancelled",
        "O_Create Offer",
        "W_Call after offers",
        "W_Validate application",
    },
    "A_Complete": {
        "A_Cancelled",
        "A_Validating",
        "O_Cancelled",
        "O_Create Offer",
        "W_Validate application",
    },
    "A_Validating": {
        "A_Denied",
        "A_Incomplete",
        "O_Accepted",
        "O_Returned",
        "W_Call incomplete files",
    },
    "W_Validate application": {
        "A_Denied",
        "A_Incomplete",
        "O_Accepted",
        "O_Returned",
        "W_Call incomplete files",
        "W_Validate application",
    },
    "W_Call incomplete files": {
        "A_Cancelled",
        "A_Validating",
        "O_Cancelled",
        "O_Create Offer",
        "W_Call incomplete files",
        "W_Validate application",
    },
    "A_Incomplete": {
        "A_Cancelled",
        "A_Validating",
        "O_Cancelled",
        "O_Create Offer",
        "W_Validate application",
    },
    "O_Returned": {
        "A_Denied",
        "A_Incomplete",
        "O_Accepted",
        "W_Call incomplete files",
    },
    "O_Accepted": {"A_Pending"},
    "A_Pending": {"O_Cancelled"},
    "O_Cancelled": {"O_Cancelled", "O_Create Offer"},
    "A_Denied": {"O_Refused"},
    "O_Refused": set(),          # danach Ende
    "A_Cancelled": set(),        # Ende-Pfade
}

# Startaktivität laut BPMN (nach Start-Event)
START_ACTIVITY = "A_Create Application"

# Optional: Aktivitäten, bei denen der Prozess "natürlich" endet,
# auch wenn SUCCESSORS noch etwas zulassen würden.
END_ACTIVITIES = {
    "O_Refused",
    "A_Cancelled",
    "O_Cancelled",
}


# -----------------------------
# 2) DAUER-MODELL FÜR AKTIVITÄTEN
#    (Platzhalter, kannst du mit Log-Werten ersetzen)
# -----------------------------

MEAN_DURATION_MINUTES = {
    "A_Create Application": 5,
    "A_Submitted": 1,
    "W_Handle leads": 20,
    "A_Concept": 60,
    "W_Complete application": 30,
    "A_Accepted": 5,
    "O_Create Offer": 10,
    "O_Created": 5,
    "O_Sent (mail and online)": 5,
    "W_Call after offers": 10,
    "A_Complete": 10,
    "A_Validating": 15,
    "W_Validate application": 15,
    "W_Call incomplete files": 10,
    "A_Incomplete": 5,
    "O_Returned": 5,
    "O_Accepted": 5,
    "A_Pending": 5,
    "O_Cancelled": 3,
    "A_Denied": 3,
    "O_Refused": 2,
    "A_Cancelled": 2,
}


def sample_duration(activity_name: str) -> timedelta:
    """
    Simple duration model: exponential around a mean duration per activity.
    """
    mean = MEAN_DURATION_MINUTES.get(activity_name, 10)
    value = random.expovariate(1.0 / mean)
    return timedelta(minutes=value)


# -----------------------------
# 3) CONTROL FLOW ENGINE
# -----------------------------

def next_activities(activity_name: str):
    """
    Returns the set of possible next activities for the given activity.
    """
    return list(SUCCESSORS.get(activity_name, set()))


def simulate_case(case_id: str,
                  start_time: datetime,
                  max_steps: int = 100) -> list[dict]:
    """
    Simulate one process instance (case) as a linear trace.
    """
    events = []
    current_time = start_time
    current_activity = START_ACTIVITY
    steps = 0

    while True:
        steps += 1
        if steps > max_steps:
            # safety break to avoid infinite loops
            break

        # Activity duration
        dur = sample_duration(current_activity)
        start = current_time
        end = current_time + dur

        # Event logging (similar to BPIC17 structure)
        events.append({
            "case:concept:name": case_id,
            "concept:name": current_activity,
            "time:timestamp": end,   # oder start, je nach Konvention
            # optional: start/end getrennt speichern
            "time:start_timestamp": start,
            "time:end_timestamp": end,
        })

        current_time = end

        # Check natural end
        if current_activity in END_ACTIVITIES:
            break

        # Choose next activity
        candidates = next_activities(current_activity)
        if not candidates:
            break

        # simple: uniform random choice
        current_activity = random.choice(candidates)

    return events


def simulate_log(n_cases: int = 100,
                 interarrival_minutes: float = 5.0,
                 base_start: datetime | None = None) -> pd.DataFrame:
    """
    Simulate an event log with many cases.
    """
    if base_start is None:
        base_start = datetime(2020, 1, 1, 8, 0, 0)

    all_events: list[dict] = []

    for i in range(n_cases):
        case_id = str(i)
        case_start = base_start + timedelta(minutes=i * interarrival_minutes)
        case_events = simulate_case(case_id, case_start)
        all_events.extend(case_events)

    df = pd.DataFrame(all_events)
    return df


if __name__ == "__main__":
    # kleines Beispiel: 10 Cases simulieren und als CSV speichern
    random.seed(42)
    df_sim = simulate_log(n_cases=10)
    print(df_sim.head())
    df_sim.to_csv("simulated_log.csv", index=False)
    print("Simulated log written to simulated_log.csv")
