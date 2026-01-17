import heapq
import pandas as pd
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from processing_times import Functions as fk

try:
    from .router import Router
except ImportError:
    try:
        from router import Router
    except ImportError:
        # basic dummy router for safety
        class Router:
            def __init__(self, *args, **kwargs):
                self.mode = "random"

            def decide(self, enabled, pn, meta, marking, current_time):
                return random.choice(enabled)


@dataclass(order=True)
class Event:
    time: datetime
    type: str = field(compare=False)
    case_id: int = field(compare=False)
    transition_id: str = field(compare=False, default=None)
    resource: str = field(compare=False, default=None)
    duration: timedelta = field(compare=False, default=None)


class Engine:
    def __init__(self, pn, resource_manager, start_time=None, mode="random",
                 basic_model=None, advanced_model=None, max_cases=50):
        self.pn = pn
        self.resource_manager = resource_manager
        self.now = start_time or datetime(2016, 1, 1, 9, 15, 0)
        self.queue = []
        self.cases = {}
        self.cases_meta = {}
        self.log = []
        self.next_case_id = 0
        self.max_cases = max_cases
        self.router = Router(
            mode=mode,
            basic_model=basic_model,
            advanced_model=advanced_model
        )

        print(f" Simulation Engine initialized")
        print(f"  - Decision mode: {mode}")
        print(f"  - Max cases: {max_cases}")
        print(f"  - Resources: {len(self.resource_manager.get_resources())}")

    def spawn(self, at_time=None):
        heapq.heappush(self.queue, Event(at_time or self.now, "SPAWN", self.next_case_id + 1))

    def run(self, max_events=1000):
        count = 0
        while self.queue and count < max_events:
            e = heapq.heappop(self.queue)
            self.now = e.time

            if e.type == "SPAWN":
                self._handle_spawn(e)
            elif e.type == "START":
                self._handle_start(e)
            elif e.type == "RETRY":  # Used when no resource was available yet
                self._process_flow(e.case_id)
            elif e.type == "COMPLETE":
                self._handle_complete(e)
            count += 1

    def _process_flow(self, case_id):
        m = self.cases[case_id]
        enabled = [t for t in self.pn.trans_ids if all(m.get(p, 0) > 0 for p in self.pn.inputs.get(t, []))]
        if not enabled:
            return

        while enabled:
            case_meta = self.cases_meta.get(case_id, {})
            tid = self.router.decide(
                enabled_ids=enabled,
                pn=self.pn,
                case_meta=case_meta,
                marking=m,
                current_time=self.now
            )  # 1.4 XOR logic added
            label = self.pn.labels.get(tid, "")

            if label == "":  # Silent Gateway, instant consume and produce
                self._consume(m, tid)
                self._produce(m, tid)
                enabled = [t for t in self.pn.trans_ids if all(m.get(p, 0) > 0 for p in self.pn.inputs.get(t, []))]
            else:  # Real Transition
                task_duration = fk.sample_duration(label, path="./processing_times/processing_models.json")  # TODO: Insert duration of event
                res = self.resource_manager.assign_resource(label, self.now, task_duration)
                if res:  # Resource is assigned NOW
                    heapq.heappush(self.queue, Event(self.now, "START", case_id, tid, res, task_duration))
                else:
                    # Find next possible starting time
                    # print("DEBUG: No available resource right now!")
                    next_avail_time = self.resource_manager.get_earliest_availability(label, self.now)
                    # print("DEBUG: Earliest availability: ", self.resource_manager.get_earliest_availability(label, self.now))
                    retry_time = self.now + timedelta(minutes=15)  # See if any resource has been released until then
                    if next_avail_time and next_avail_time > self.now:
                        retry_time = max(retry_time, next_avail_time)
                    heapq.heappush(self.queue, Event(retry_time, "RETRY", case_id))
                break

    def _handle_spawn(self, e):
        self.next_case_id += 1
        self.cases[e.case_id] = dict(self.pn.im)

        # 1.4 Decision Point Analysis case history metadata
        # initial marking
        self.cases[e.case_id] = dict(self.pn.im)
        # case metadata
        self.cases_meta[e.case_id] = {
            "history": [],
            "attributes": self._generate_case_attributes(),
            "start_time": self.now
        }

        # 1.2 Basic: Static parametric distribution (e.g.: Exponential), only 10 for testing
        if self.next_case_id < self.max_cases:
            inter_arrival_time = random.expovariate(1 / 30)  # Average every 30 mins
            next_arrival = self.now + timedelta(minutes=inter_arrival_time)
            heapq.heappush(self.queue, Event(next_arrival, "SPAWN", self.next_case_id + 1))

        self._process_flow(e.case_id)

    def _handle_start(self, e):
        self._consume(self.cases[e.case_id], e.transition_id)
        self._record(e, "start")
        duration = e.duration  # TODO 1.3 Processing times
        heapq.heappush(self.queue, Event(self.now + duration, "COMPLETE", e.case_id, e.transition_id, e.resource))

    def _handle_complete(self, e):
        self._produce(self.cases[e.case_id], e.transition_id)
        self._record(e, "complete")

        # 1.4 Decision Analysis Router 
        # Add to history
        label = self.pn.labels.get(e.transition_id, "").strip()
        if label:
            self.cases_meta[e.case_id]["history"].append(label)

        self._process_flow(e.case_id)

    def _consume(self, m, tid):
        for p in self.pn.inputs.get(tid, []):
            m[p] -= 1

    def _produce(self, m, tid):
        for p in self.pn.outputs.get(tid, []):
            m[p] = m.get(p, 0) + 1

    def _is_case_complete(self, case_id):
        """
        Check whether the case comes to the final marking
        """
        m = self.cases[case_id]
        fm = self.pn.fm

        # Comparing the num of tokens for each place
        for place_id, expected_tokens in fm.items():
            if m.get(place_id, 0) != expected_tokens:
                return False

        # Check if there is remaining tokens
        for place_id, tokens in m.items():
            if tokens > 0 and place_id not in fm:
                return False

        return True

    def _record(self, e, phase):
        self.log.append({
            "case:concept:name": e.case_id,
            "concept:name": self.pn.labels.get(e.transition_id, e.transition_id),
            "time:timestamp": self.now,
            "lifecycle:transition": phase,
            "org:resource": e.resource
        })

    def _generate_case_attributes(self):
        """
        Generates case attributes based on BPI 2017 --> according to output of data_validation.py
        CreditScore is excluded as it is missing in the source dataset.
        """
        app_type = random.choices(
            ["New credit", "Limit raise"],
            weights=[28120, 3389],
            k=1
        )[0]

        goal_options = [
            "Car", "Home improvement", "Existing loan takeover",
            "Other, see explanation", "Unknown", "Not speficied",
            "Remaining debt home", "Extra spending limit", "Caravan / Camper",
            "Motorcycle", "Boat", "Tax payments", "Business goal", "Debt restructuring"
        ]

        goal_weights = [
            9328, 7669, 5601,
            2985, 2365, 1065,
            842, 625, 369,
            275, 201, 152, 30, 2
        ]

        loan_goal = random.choices(goal_options, weights=goal_weights, k=1)[0]

        # Requested amount
        amount = round(random.triangular(100, 60000, 12500), 2)

        return {
            "case:ApplicationType": app_type,
            "case:LoanGoal": loan_goal,
            "case:RequestedAmount": amount
            # "case:CreditScore": removed due to missing data in source log
        }

    def export_log(self, path="simulation_log.csv"):
        if not self.log:
            return
        df = pd.DataFrame(self.log)
        df.to_csv(path, index=False)
        print(f"\n Event log exported to: {path}")
        print(f"  - {len(df)} events")
        print(f"  - {df['case:concept:name'].nunique()} unique cases")
        print(f"  - {df['concept:name'].nunique()} unique activities")

    def print_statistics(self):
        """
        Simulation statistics 
        """
        if not self.log:
            print("No events recorded")
            return

        df = pd.DataFrame(self.log)

        print(f"\n{'=' * 60}")
        print("SIMULATION STATISTICS")
        print('=' * 60)

        # Case statistics
        print(f"\nCases:")
        print(f"  Total: {df['case:concept:name'].nunique()}")

        # Activity statistics
        print(f"\nActivities:")
        activity_counts = df['concept:name'].value_counts()
        for act, count in activity_counts.head(10).items():
            print(f"  {act}: {count}")

        # Resource statistics
        print(f"\nResources:")
        resource_counts = df['org:resource'].value_counts()
        for res, count in resource_counts.items():
            print(f"  {res}: {count} tasks")

        print('=' * 60)
