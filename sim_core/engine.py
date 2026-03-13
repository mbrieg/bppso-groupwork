import heapq
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass(order=True)
class Event:
    time: datetime
    type: str = field(compare=False)
    case_id: int = field(compare=False)
    transition_id: str = field(compare=False, default=None)
    resource: str = field(compare=False, default=None)
    duration: timedelta = field(compare=False, default=None)


class Engine:
    def __init__(self, pn, spawner, resource_manager, decision_manager, start_time=None, max_cases=31509, pt_sampler=None):
        self.pn = pn
        self.spawner = spawner
        self.resource_manager = resource_manager
        self.pt = pt_sampler
        self.decision_manager = decision_manager

        self.now = start_time or datetime(2016, 1, 1, 9, 15, 0)
        self.queue = []
        self.cases = {}
        self.log = []
        self.next_case_id = 0
        self.max_cases = max_cases

        # Dict to track the current last activity for each case (DP)
        self.case_last_activity = {}
        self.case_second_last_activity = {}
        self.case_start_times = {}     # When the case is started
        self.case_last_duration = {}

        # case attributes
        self.case_attributes = {}

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
            elif e.type == "RETRY":     # Used when no resource was assigned
                self._process_flow(e.case_id)
            elif e.type == "COMPLETE":
                self._handle_complete(e)
            count += 1

        #Debug
        #print(f"[ENGINE-END] now={self.now} flushing remaining batches")
        self.resource_manager.flush_remaining_batches(self.now)

    def _process_flow(self, case_id):
        m = self.cases[case_id]
        enabled = [t for t in self.pn.trans_ids if all(m.get(p, 0) > 0 for p in self.pn.inputs.get(t, []))]

        while enabled:
            last_act = self.case_last_activity.get(case_id, "START")
            last_dur = self.case_last_duration.get(case_id, 0)
            start_time = self.case_start_times.get(case_id, self.now)

            # DP Integration
            case_ctx = dict(self.case_attributes.get(case_id) or {})
            case_ctx["prev_activity_2"] = self.case_second_last_activity.get(case_id, "START")
            tid = self.decision_manager.get_next_transition(case_id=case_id, enabled_transitions=enabled, last_activity=last_act, last_duration_sec=last_dur, case_start_time=start_time, current_now=self.now, case_context=case_ctx)
            label = self.pn.labels.get(tid, "")

            if label == "":     # Silent Gateway, instant consume and produce
                self._consume(m, tid)
                self._produce(m, tid)
                enabled = [t for t in self.pn.trans_ids if all(m.get(p, 0) > 0 for p in self.pn.inputs.get(t, []))]
            else:
                '''Processing Times'''
                if label.startswith(("A_", "O_")):
                    sec = self.pt.sample(label, kind="total", use_qr=False)
                    duration = timedelta(seconds=float(sec))
                else:
                    # Only Basic, because there are no features about loan type, or amount

                    sec = self.pt.sample(
                        label,
                        kind="total",   # use total to simulate complete behavior, change to "proc" when real procesiing times
                        now=self.now,
                        use_qr=False
                    )
                    duration = timedelta(seconds=float(sec))
                ''' End Processing Times '''

                # Resource allocation
                res_id = self.resource_manager.assign_resource(label, self.now, duration, case_id, tid)
                if res_id is not None:
                    resource = self.resource_manager.get_resource(res_id)
                    if not resource.is_occupied():      # Assign resource NOW
                        resource.pop_task()
                        resource.occupy()
                        if label.startswith('W_'):
                            # give a small, neglegtable delay for the starting time of W_Activities, as their processing time is the relevant
                            heapq.heappush(self.queue, Event(self.now + timedelta(seconds=float(np.random.uniform(0, 1))), "START", case_id, tid, res_id, duration))
                        else:
                            # insert delay for O and A activities
                            heapq.heappush(self.queue, Event(self.now+duration, "COMPLETE", case_id, tid, res_id, duration))
                    else:   # Used for SHQ und ADVANCED allocation
                        pass    # Resources currently busy
                else:   # Find next possible starting time
                    next_avail_time = self.resource_manager.get_earliest_availability(label, self.now)
                    if next_avail_time and next_avail_time > self.now:
                        retry_time = next_avail_time
                    else:
                        retry_time = self.now + timedelta(minutes=15)       # Fallback
                    heapq.heappush(self.queue, Event(retry_time, "RETRY", case_id))
                break

    def _handle_spawn(self, e):
        self.next_case_id += 1
        self.cases[e.case_id] = dict(self.pn.im)
        # DP
        self.case_last_activity[e.case_id] = "START"    # a new memory for each case
        self.case_second_last_activity[e.case_id] = "START"
        self.case_start_times[e.case_id] = self.now
        self.case_last_duration[e.case_id] = 0
        # Case attributes
        if hasattr(self.spawner, "get_case_attributes"):
            self.case_attributes[e.case_id] = self.spawner.get_case_attributes(e.case_id)
        # DP
        if self.next_case_id < self.max_cases:
            next_time = self.spawner.calculate_next_spawn(self.now)
            heapq.heappush(self.queue, Event(next_time, "SPAWN", self.next_case_id + 1))

        self._process_flow(e.case_id)

    def _handle_start(self, e):
        self._consume(self.cases[e.case_id], e.transition_id)
        self._record(e, "start")
        duration = e.duration
        heapq.heappush(self.queue, Event(self.now + duration, "COMPLETE", e.case_id, e.transition_id, e.resource, duration))

    def _handle_complete(self, e):
        label = self.pn.labels.get(e.transition_id, "")

        if not label.startswith("W_"):
            self._consume(self.cases[e.case_id], e.transition_id)

        self._produce(self.cases[e.case_id], e.transition_id)
        self._record(e, "complete")

        resource = self.resource_manager.resources[e.resource]
        resource.release()
        if resource.get_queue_length() > 0:
            next_act = resource.pop_task()
            resource.occupy()
            if label.startswith('W_'):
                heapq.heappush(self.queue,
                               Event(self.now + timedelta(seconds=float(np.random.uniform(0, 1))), "START",
                                     next_act['cid'], next_act['tid'], resource.get_id(), next_act['duration']))
            else:
                heapq.heappush(self.queue, Event(self.now + next_act['duration'], "COMPLETE",
                                                 next_act['cid'], next_act['tid'], resource.get_id(), next_act['duration']))

        if label != "":
            self.case_second_last_activity[e.case_id] = self.case_last_activity.get(e.case_id, "START")
            self.case_last_activity[e.case_id] = label  # write to the memory when is completed
            if e.duration:
                self.case_last_duration[e.case_id] = e.duration.total_seconds()
            else:
                self.case_last_duration[e.case_id] = 0

        self._process_flow(e.case_id)

    def _consume(self, m, tid):
        for p in self.pn.inputs.get(tid, []):
            m[p] -= 1

    def _produce(self, m, tid):
        for p in self.pn.outputs.get(tid, []):
            m[p] = m.get(p, 0) + 1

    def _record(self, e, phase):
        label = self.pn.labels.get(e.transition_id, e.transition_id)
        if label.startswith("O_"):
            origin = "Offer"
        elif label.startswith("W_"):
            origin = "Workflow"
        else:
            origin = "Application"

        row = {
            "case:concept:name": e.case_id,
            "concept:name": label,
            "time:timestamp": self.now,
            "lifecycle:transition": phase,
            "org:resource": e.resource,
            "EventOrigin": origin,
        }

        ctx = self.case_attributes.get(e.case_id)
        if ctx:
            row["case:ApplicationType"] = ctx.get("application_type", "")
            row["case:LoanGoal"] = ctx.get("loan_goal", "")
            row["case:RequestedAmount"] = ctx.get("requested_amount", "")

        self.log.append(row)

    def export_log(self, path="simulation_log.csv"):
        pd.DataFrame(self.log).to_csv(path, index=False)
