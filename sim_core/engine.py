import heapq
import pandas as pd
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

@dataclass(order=True)
class Event:
    time: datetime
    type: str = field(compare=False)
    case_id: int = field(compare=False)
    transition_id: str = field(compare=False, default=None)
    resource: str = field(compare=False, default=None)

class Engine:
    def __init__(self, pn, start_time=None):
        self.pn = pn
        self.now = start_time or datetime(2016, 1, 1, 9, 15, 0)
        self.queue = []
        self.cases = {}
        self.log = []
        self.next_case_id = 0
        self.available_resources = ["User_1", "User_2", "User_3"]


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
            elif e.type == "COMPLETE":
                self._handle_complete(e)
            count += 1


    def _process_flow(self, case_id):
        m = self.cases[case_id]
        enabled = [t for t in self.pn.trans_ids if all(m.get(p, 0) > 0 for p in self.pn.inputs.get(t, []))]

        while enabled:
            tid = random.choice(enabled) # 1.4 XOR logic random now
            label = self.pn.labels.get(tid, "")

            if label == "": # Silent Gateway, instant consume and produce
                self._consume(m, tid)
                self._produce(m, tid)
                enabled = [t for t in self.pn.trans_ids if all(m.get(p, 0) > 0 for p in self.pn.inputs.get(t, []))]
            else: # Real Transition
                if self.available_resources:
                    res = self.available_resources.pop(0)
                    heapq.heappush(self.queue, Event(self.now, "START", case_id, tid, res))
                break


    def _handle_spawn(self, e):
        self.next_case_id += 1
        self.cases[e.case_id] = dict(self.pn.im)

        # 1.2 Basic: Static parametric distribution (e.g.: Exponential), only 10 for testing
        if self.next_case_id < 10:
            inter_arrival_time = random.expovariate(1/30) # Average every 30 mins
            next_arrival = self.now + timedelta(minutes=inter_arrival_time)
            heapq.heappush(self.queue, Event(next_arrival, "SPAWN", self.next_case_id + 1))

        self._process_flow(e.case_id)


    def _handle_start(self, e):
        self._consume(self.cases[e.case_id], e.transition_id)
        self._record(e, "start")
        duration = timedelta(minutes=random.randint(5, 15))# 1.3 Processing times
        heapq.heappush(self.queue, Event(self.now + duration, "COMPLETE", e.case_id, e.transition_id, e.resource))


    def _handle_complete(self, e):
        self._produce(self.cases[e.case_id], e.transition_id)
        self._record(e, "complete")
        self.available_resources.append(e.resource)
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