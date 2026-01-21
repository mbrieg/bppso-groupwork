import heapq
import os
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from decision_analysis.generators import CaseGenerator
from processing_times.sampling import ProcessingTimeSampler
import numpy as np


@dataclass(order=True)
class Event:
    time: datetime
    type: str = field(compare=False)
    case_id: int = field(compare=False)
    transition_id: str = field(compare=False, default=None)
    resource: str = field(compare=False, default=None)
    duration: timedelta = field(compare=False, default=None)


class Engine:
    def __init__(self, pn, spawner, resource_manager, decision_manager, start_time=None, max_cases=50):
        self.pn = pn
        self.spawner = spawner  # Spawner Object Strategy
        self.resource_manager = resource_manager
        self.decision_manager = decision_manager
        self.case_generator = CaseGenerator()
        self.now = start_time or datetime(2016, 1, 4, 9, 15, 0)
        self.queue = []
        self.cases = {}  # Token state per case: {case_id: {place_id: count}}
        self.cases_meta = {}  # Context per case: {case_id: {history: [], attributes: {}}}
        self.log = []
        self.next_case_id = 0
        self.max_cases = max_cases
        self.place_map = {p.name: p for p in self.decision_manager.net.places}

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.time_model_path = os.path.join(base_dir, "processing_times", "processing_models.json")
        self.rng = np.random.default_rng(42)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pt_dir = os.path.join(base_dir, "processing_times")

        self.proc_sampler = ProcessingTimeSampler.from_paths(
            proc_json=os.path.join(pt_dir, "processing_models_proc.json"),
            qr_joblib={"proc": os.path.join(pt_dir, "proc_qr_bundle.joblib")},  # <—
            seed=None,
            default_value=60.0
        )

        self.total_sampler = ProcessingTimeSampler.from_paths(
            total_json=os.path.join(pt_dir, "processing_models_full_dur.json"),
            seed=None,
            default_value=0.0
        )
        print(f" Simulation Engine initialized")
        print(f"  - Decision mode: {self.decision_manager.mode}")
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

        # Have to debug the engine
        if case_id == 1:
            print(f"\n[DEBUG ENGINE] Processing Case {case_id} at {self.now}")
            print(f"  -> Current Tokens (Marking): {m}")
            # Check what is theoretically enabled
            all_trans = self.pn.trans_ids
            potential = []
            for t in all_trans:
                inputs = self.pn.inputs.get(t, [])
                if all(m.get(p, 0) > 0 for p in inputs):
                    potential.append(t)
            print(f"  -> Enabled Transitions found: {potential}")
            potential_labels = [self.pn.labels.get(t, t).strip() for t in potential]
            print(f"  -> Enabled Labels: {potential_labels}")
            if not potential:
                print("  -> CRITICAL: No transitions enabled. Check Initial Marking (im).")

        if not enabled: return

        loop_prevention_counter = 0
        MAX_SILENT_STEPS = 100

        while enabled:
            if loop_prevention_counter > MAX_SILENT_STEPS:
                break

            tid = None
            # check for conflicts (XOR Splits)
            decision_found = False
            for p_id, tokens in m.items():
                if tokens > 0:
                    place_obj = self.place_map.get(p_id)

                    # Check if this place is a known Decision Point (registered in DPManager)
                    if place_obj and place_obj.name in self.decision_manager.decision_points:

                        # Prepare context
                        ctx = self.cases_meta.get(case_id, {}).get("history", [])

                        # Ask Manager: "Where should I go from here?"
                        t_obj = self.decision_manager.get_next_transition(place_obj, ctx)

                        if t_obj:
                            # Verify the suggested transition is actually enabled right now
                            if t_obj.name in enabled:
                                tid = t_obj.name
                                decision_found = True
                                break
                            else:
                                # The manager chose a path, but it's not enabled.
                                required_inputs = self.pn.inputs.get(t_obj.name, [])
                                input_status = {p: m.get(p, 0) for p in required_inputs}
                                print(
                                    f"  -> [DEBUG ENGINE] Manager suggested {t_obj.name} (Label: {self.pn.labels.get(t_obj.name, '')}) but it is NOT in enabled list: {enabled}")
                                print(f"  -> [DEBUG ENGINE] Required inputs for {t_obj.name}: {input_status}")
                                # Fallback to standard behavior.
                                pass

            # If no conflict found or manager failed, pick first enabled (FIFO/Random)
            if not decision_found or tid is None:
                tid = enabled[0]  # 1.4 XOR logic added

            label = self.pn.labels.get(tid, "")

            if label == "":  # Silent Gateway, instant consume and produce
                self._consume(m, tid)
                self._produce(m, tid)

                loop_prevention_counter += 1

                enabled = [t for t in self.pn.trans_ids if all(m.get(p, 0) > 0 for p in self.pn.inputs.get(t, []))]
            else:  # Real Transition

                label = self.pn.labels.get(tid, tid).strip()
                # print(f"[DEBUG RESOURCE] Asking for resource for task: '{label}' (ID: {tid})")
                # label = self.pn.labels.get(tid, tid)
                # print(f"[DEBUG RESOURCE] Asking for resource for task: '{label}' (ID: {tid})")
                # label = self.pn.labels.get(tid, tid).strip()

                if label.startswith(("A_", "O_")):
                    sec = self.total_sampler.sample(label, kind="total", rng=self.rng, use_qr=False)

                    # optional: damit O_ nicht praktisch 0 bleibt
                    sec = max(sec, 1.0)

                    task_duration = timedelta(seconds=float(sec))

                    # A/O ohne Ressourcenlogik starten (sonst baust du künstliche Bottlenecks)
                    # heapq.heappush(self.queue, Event(self.now, "START", case_id, tid, "System_Auto", task_duration))
                else:
                    # Instance-Index (wie oft kam diese Activity schon im Case vor)
                    instance = self.cases_meta[case_id]["history"].count(label)

                    attrs = self.cases_meta[case_id]["attributes"]
                    ctx = {
                        "case:ApplicationType": attrs.get("case:ApplicationType", "UNK"),
                        "case:RequestedAmount": float(attrs.get("case:RequestedAmount", 0.0)),
                    }

                    sec = self.proc_sampler.sample(
                        label,
                        kind="proc",
                        now=self.now,
                        instance=instance,
                        ctx=ctx,
                        rng=self.rng,
                        use_qr=True  # <— QR aktiv
                    )
                    task_duration = timedelta(seconds=float(sec))

                res = self.resource_manager.assign_resource(label, self.now, task_duration)
                if res is None:
                    # Optional: Print warning only once per activity type to avoid spam
                    print(f"  [WARN] No resource mapping for '{label}'. Assigning 'System_Auto'.")
                    res = "System_Auto"

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
        attributes = self.case_generator.generate_new_case_attributes()

        # 1.4 Decision Point Analysis case history metadata
        self.cases_meta[e.case_id] = {
            "history": [],
            "attributes": attributes,  # CaseGenerator()
            "start_time": self.now
        }

        # 1.2 Instance spawn rates
        if self.next_case_id < self.max_cases:
            next_time = self.spawner.calculate_next_spawn(self.now)
            heapq.heappush(self.queue, Event(next_time, "SPAWN", self.next_case_id + 1))

        self._process_flow(e.case_id)

    def _handle_start(self, e):
        self._consume(self.cases[e.case_id], e.transition_id)
        self._record(e, "start")
        duration = e.duration  # TODO 1.3 Processing times
        heapq.heappush(self.queue, Event(self.now + duration, "COMPLETE", e.case_id, e.transition_id, e.resource))

    def _handle_complete(self, e):
        self._produce(self.cases[e.case_id], e.transition_id)
        self._record(e, "complete")

        # 1.4
        # Add to history for DPManager
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

    def export_log(self, path="simulation_log.csv"):
        if not self.log:
            return
        df = pd.DataFrame(self.log)

        # adding case attributes
        case_attrs = {cid: meta['attributes'] for cid, meta in self.cases_meta.items()}
        attr_df = pd.DataFrame.from_dict(case_attrs, orient='index')
        attr_df.index.name = 'case:concept:name'

        df = df.join(attr_df, on='case:concept:name')

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
        print(f"\nCases:")
        print(f"  Total: {df['case:concept:name'].nunique()}")
        print(f"\nActivities:")
        activity_counts = df['concept:name'].value_counts()
        for act, count in activity_counts.head(10).items():
            print(f"  {act}: {count}")
        print(f"\nResources:")
        resource_counts = df['org:resource'].value_counts()
        for res, count in resource_counts.items():
            print(f"  {res}: {count} tasks")
        print('=' * 60)
