import heapq
import pandas as pd
import random
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


class EngineOG:
    def __init__(self, pn, spawner, resource_manager, decision_manager, start_time=None, max_cases=100, pt_sampler = None):
        # Managers
        self.pn = pn
        self.spawner = spawner
        self.resource_manager = resource_manager
        self.pt = pt_sampler
        # TODO insert processing times and decision point managers/interfaces HERE
        self.decision_manager = decision_manager
        # Other stuff
        self.now = start_time or datetime(2016, 1, 1, 9, 15, 0)
        self.queue = []
        self.cases = {}
        self.log = []
        self.next_case_id = 0
        self.max_cases = max_cases

        # ### Dict to track the current last activity for each case (DP)
        self.case_last_activity = {}
        self.case_start_times = {}     # When the case is started
        self.case_last_duration = {}


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
            elif e.type == "RETRY":     # Used when no resource was available yet
                self._process_flow(e.case_id)
            elif e.type == "COMPLETE":
                self._handle_complete(e)
            count += 1

    def _process_flow(self, case_id):
        m = self.cases[case_id]
        enabled = [t for t in self.pn.trans_ids if all(m.get(p, 0) > 0 for p in self.pn.inputs.get(t, []))]

        while enabled:
            last_act = self.case_last_activity.get(case_id, "START")
            last_dur = self.case_last_duration.get(case_id, 0)
            start_time = self.case_start_times.get(case_id, self.now)
            # TODO @Zeynep: Insert get_next_transition() from decision manager --> sid...
            # ### DP Integration
            tid = self.decision_manager.get_next_transition(case_id = case_id, enabled_transitions=enabled, last_activity=last_act,last_duration_sec=last_dur,case_start_time=start_time,current_now=self.now)
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
                    # Processing times: Ich brauche den auskommentierten code
                    # instance = self.cases_meta[case_id]["history"].count(label)

                    # attrs = self.cases_meta[case_id]["attributes"]
                    # ctx = {
                    #    "case:ApplicationType": attrs.get("case:ApplicationType", "UNK"),
                    #    "case:RequestedAmount": float(attrs.get("case:RequestedAmount", 0.0)),
                    #}

                    sec = self.pt.sample(
                        label,
                        kind="proc",
                        now=self.now,
                        # instance=instance,
                        # ctx=ctx,
                        # rng=self.rng,
                        use_qr=False
                    )
                    duration = timedelta(seconds=float(sec))
                ''' End Processing Times '''


                res = self.resource_manager.assign_resource(label, self.now, duration)
                if res:     # Resource is assigned NOW,
                    if label.startswith('W_'):
                        # give a small, neglegtable delay for the starting time of W_Activities, as their processing time is the relevant
                        heapq.heappush(self.queue, Event(self.now + timedelta(seconds=float(np.random.uniform(0, 1))), "START", case_id, tid, res, duration))
                    else:
                        # insert delay for O and A activities
                        heapq.heappush(self.queue, Event(self.now+duration, "COMPLETE", case_id, tid, res, duration))
                else:   # Find next possible starting time
                    next_avail_time = self.resource_manager.get_earliest_availability(label, self.now)
                    retry_time = self.now + timedelta(minutes=15)       # Check if resource is free until then
                    if next_avail_time and next_avail_time > self.now:
                        retry_time = max(retry_time, next_avail_time)
                    heapq.heappush(self.queue, Event(retry_time, "RETRY", case_id))
                break

    def _handle_spawn(self, e):
        self.next_case_id += 1
        self.cases[e.case_id] = dict(self.pn.im)
        #DP
        self.case_last_activity[e.case_id] = "START" # a new memory for each case
        self.case_start_times[e.case_id] = self.now 
        self.case_last_duration[e.case_id] = 0
        #DP
        if self.next_case_id < self.max_cases:
            next_time = self.spawner.calculate_next_spawn(self.now)
            heapq.heappush(self.queue, Event(next_time, "SPAWN", self.next_case_id + 1))

        self._process_flow(e.case_id)

    def _handle_start(self, e):
        self._consume(self.cases[e.case_id], e.transition_id)
        self._record(e, "start")
        duration = e.duration# 1.3 Processing times
        heapq.heappush(self.queue, Event(self.now + duration, "COMPLETE", e.case_id, e.transition_id, e.resource, duration))

    def _handle_complete(self, e):

        label = self.pn.labels.get(e.transition_id, "")

        if not label.startswith("W_"):
            self._consume(self.cases[e.case_id], e.transition_id)

        self._produce(self.cases[e.case_id], e.transition_id)
        self._record(e, "complete")

        if label !="":
            self.case_last_activity[e.case_id] = label # write to the memory when is completed
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
        self.log.append({
            "case:concept:name": e.case_id,
            "concept:name": self.pn.labels.get(e.transition_id, e.transition_id),
            "time:timestamp": self.now,
            "lifecycle:transition": phase,
            "org:resource": e.resource
        })

    def export_log(self, path="simulation_log.csv"):
        pd.DataFrame(self.log).to_csv(path, index=False)
