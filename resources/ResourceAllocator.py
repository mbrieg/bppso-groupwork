import random
from enum import Enum
from datetime import timedelta


class Methods(Enum):
    RANDOM = 0
    ROUND_ROBIN = 1
    SHORTEST_QUEUE = 2
    BATCHING = 3
    ADVANCED = 4  # Adapted assignment problem


class ResourceAllocator:
    """
    Assigns tasks to permitted and available resources based on the specified method.
    """

    def __init__(self, resources, availabilities, permissions, method=Methods.RANDOM, delta=1, batch_k=5):
        self.resources = resources
        self.availabilities = availabilities
        self.permissions = permissions
        self.method = method
        self.rr_index = 0
        self.batching_ctr = 0
        self.pending_tasks = []
        self.last_batch_assignments = []
        self.batch_k=batch_k
        self.pending_tids = set() # Needed to handle Retry envents that should not be flushed
        self.assigned_batch_tids = {} # Needed to handle Retry envents that should not be flushed

        if method == Methods.ADVANCED:
            self._predictor = ResourceAllocator._Predictor(delta)

    def allocate_resource(self, act_name, start_time, duration, case_id, tid):


        # Check permissions
        permitted = self.permissions.get_permitted_resources(act_name, self.resources)
        permitted.sort()
        if not permitted:
            return None

        # Choose resource
        selected_id = None
        task_start = None
        if self.method == Methods.RANDOM:
            selected_id = self._allocate_random(permitted, start_time)
        elif self.method == Methods.ROUND_ROBIN:
            selected_id = self._allocate_round_robin(permitted, start_time)
        elif self.method == Methods.SHORTEST_QUEUE:
            selected_id, task_start = self._allocate_shortest_queue(permitted, start_time)
        elif self.method == Methods.BATCHING:
            # Debug Print
            print(f"[BATCH-IN] tid={tid} act={act_name} time={start_time} pending_before={len(self.pending_tasks)}")
            selected_id = self._allocate_batch(act_name, start_time, duration, case_id, tid, permitted)
        elif self.method == Methods.ADVANCED:
            selected_id, task_start = self._allocate_advanced(act_name, permitted, start_time)

        if selected_id and task_start:  # Only used for SHQ and ADVANCED allocation
            act_info = {
                "activity": act_name,
                "start": task_start,
                "duration": duration,
                "cid": case_id,
                "tid": tid
            }
            self.resources[selected_id].add_task(act_info)

        return selected_id

    def _get_available_resources(self, resources, start_time):
        on_shift = []
        for res_id in resources:
            if res_id not in self.resources:
                continue
            if self.availabilities.is_resource_available(res_id, start_time):
                on_shift.append(res_id)
        return on_shift

    def _get_idle_resources(self, on_shift, current_time):
        if not on_shift:
            return None
        idle = []
        for res_id in on_shift:
            if (not self.resources[res_id].is_occupied()
                    and not self.availabilities.is_resource_on_break(res_id, current_time)):
                idle.append(res_id)
        return idle

    def _allocate_random(self, resources, start_time):
        on_shift = self._get_available_resources(resources, start_time)
        idle = self._get_idle_resources(on_shift, start_time)
        if not idle:
            return None
        return random.choice(idle)

    def _allocate_round_robin(self, resources, start_time):
        for i in range(len(resources)):
            selected_id = (self.rr_index + i) % len(resources)
            res_id = resources[selected_id]

            if (self.availabilities.is_resource_available(res_id, start_time)
                    and not self.resources[res_id].is_occupied()
                    and not self.availabilities.is_resource_on_break(res_id, start_time)):
                self.rr_index = (selected_id + 1) % len(resources)
                return res_id

        return None

    def _allocate_shortest_queue(self, resources, current_now):
        resources = [self.resources[res_id] for res_id in resources]
        selected_res = min(resources, key=lambda res: res.get_queue_length())

        task_start = 0.0
        if selected_res.is_occupied():
            task_start = selected_res.get_remaining_working_time(current_now)
        next_time = self.availabilities.get_next_available_time(selected_res.get_id(), current_now)

        return selected_res.get_id(), next_time + timedelta(seconds=task_start)

    def _allocate_batch(self, act_name, start_time, duration, case_id, tid, permitted):
        task = {
            "activity": act_name,
            "arrival_time": start_time,
            "duration": duration,
            "cid": case_id,
            "tid": tid,
            "permitted": list(permitted)
        }

        self.pending_tasks.append(task)
        # Debug Print
        print(f"[BATCH-QUEUE] appended tid={tid} pending_now={len(self.pending_tasks)} batch_k={self.batch_k}")
        if len(self.pending_tasks) < self.batch_k:
            # Debug Pruint
            print(f"[BATCH-WAIT] not enough tasks yet: {len(self.pending_tasks)}/{self.batch_k}")
            return None
        # Debug Print
        print(f"[BATCH-TRIGGER] flushing batch of size {self.batch_k} at {start_time}")

        batch = self.pending_tasks[:self.batch_k]
        self.pending_tasks = self.pending_tasks[self.batch_k:]

        assignments = self._flush_k_batch(batch, start_time)
        self.last_batch_assignments = assignments

        for a in assignments:
            if a["tid"] == tid:
                return a["res_id"]

        return None

    def _get_projected_ready_time(self, res_id, current_time):
        next_time = self.availabilities.get_next_available_time(res_id, current_time)
        if next_time is None:
            return None

        backlog_seconds = self.resources[res_id].get_planned_workload(current_time)
        return next_time + timedelta(seconds=backlog_seconds)

    def _flush_k_batch(self, batch, batch_time):
        assignments = []
        unassigned = []

        ready_times = {
            res_id: self._get_projected_ready_time(res_id, batch_time)
            for res_id in self.resources
        }

        batch_sorted = sorted(
            batch,
            key=lambda t: t["duration"].total_seconds(),
            reverse=True
        )

        for task in batch_sorted:
            best_res_id = None
            best_start = None
            best_end = None

            for res_id in task["permitted"]:
                projected_start = ready_times.get(res_id)
                if projected_start is None:
                    continue

                projected_end = projected_start + task["duration"]

                if best_end is None or projected_end < best_end:
                    best_res_id = res_id
                    best_start = projected_start
                    best_end = projected_end

            if best_res_id is None:
                unassigned.append(task)
                continue
            # Debug
            print(f"[BATCH-ASSIGN] tid={task['tid']} -> res={best_res_id} start={best_start} end={best_end}")

            act_info = {
                "activity": task["activity"],
                "start": best_start,
                "duration": task["duration"],
                "cid": task["cid"],
                "tid": task["tid"]
            }

            self.resources[best_res_id].add_task(act_info)
            ready_times[best_res_id] = best_end

            assignments.append({
                "tid": task["tid"],
                "res_id": best_res_id,
                "start": best_start
            })

        if unassigned:
            self.pending_tasks = unassigned + self.pending_tasks

        self.batching_ctr += 1
        #Debug
        print(f"[BATCH-DONE] assigned={len(assignments)} unassigned={len(unassigned)} total_batches={self.batching_ctr}")
        return assignments

    def flush_remaining_batches(self, current_time):
        if self.method != Methods.BATCHING or not self.pending_tasks:
            return []

        batch = self.pending_tasks
        self.pending_tasks = []
        assignments = self._flush_k_batch(batch, current_time)
        self.last_batch_assignments = assignments
        return assignments

    def reset(self):
        self.rr_index = 0
        self.batching_ctr = 0
        self.pending_tasks.clear()
        self.last_batch_assignments = []

    def _allocate_advanced(self, act_name, permitted, start_time):
        best_res_id = None
        task_start = 0.0
        min_cost = float('inf')

        available = self._get_available_resources(permitted, start_time)
        for res_id in available:    # Calculate costs for available resources
            expected_cost = self._predictor.predict_cost(act_name, res_id)  # TODO: Check what other variables are needed for prediction, e.g. LoanGoal, CaseId, etc.
            remaining_cost = self.resources[res_id].get_remaining_working_time(start_time)
            total_cost = expected_cost + remaining_cost

            if total_cost < min_cost:
                min_cost = total_cost
                best_res_id = res_id
                task_start = remaining_cost

        # Check if dummy value more cost-efficient
        if self._predictor.get_dummy_cost(act_name) < min_cost:
            return None, None

        # Update start time
        return best_res_id, (start_time + timedelta(seconds=task_start))

    def get_next_available_time_adv(self, act_name, current_time):
        total_costs = []
        res_costs = self._predictor.get_all_costs(act_name)
        for res_id, cost in res_costs.items():
            remaining = self.resources[res_id].get_remaining_working_time(current_time)
            total_costs.append(cost + remaining)

        min_cost = min(total_costs)
        return current_time + timedelta(seconds=min_cost)

    class _Predictor:
        """
        costs: dict saving activities to dict of resources and their costs for the activity
        delta: factor variable used for cost estimation
        """
        def __init__(self, delta):
            self._costs = {}
            self._delta = delta

        def predict_cost(self, act_name, res_id):
            """
            Retrieves cost if already predicted, otherwise predicts it.
            """
            if act_name not in self._costs:
                self._costs[act_name] = {}

            # TODO Predict using neural net
            if res_id not in self._costs[act_name]:
                predicted_value = random.uniform(300, 1800)
                self._costs[act_name][res_id] = predicted_value

            return self._costs[act_name][res_id]

        def get_dummy_cost(self, act_name):
            """
            Calculates the cost of the 'Dummy Resource' based on the factorised
            average cost of all authorized resources.
            """
            if act_name not in self._costs or not self._costs[act_name]:
                return 0.0

            res_costs = self._costs[act_name].values()
            dummy_sum = sum(res_costs)

            return self._delta * (1 / len(res_costs)) * dummy_sum

        def get_all_costs(self, act_name):
            """
            Returns a dictionary of all predicted resource costs for a specific activity.
            """
            return self._costs.get(act_name, {})
