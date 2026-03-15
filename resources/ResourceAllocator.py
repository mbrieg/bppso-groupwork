import random
from enum import Enum
from datetime import timedelta
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


class Methods(Enum):
    RANDOM = 0
    ROUND_ROBIN = 1
    SHORTEST_QUEUE = 2
    BATCHING = 3
    ADVANCED_LOCAL = 4  # Greedy assignment
    ADVANCED_GLOBAL = 5  # Adapted assignment problem


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
        self.batch_k = batch_k
        self.pending_task_keys = set()  # Needed to handle Retry events that should not be flushed
        self.assigned_batch_task_keys = {}  # Needed to handle Retry events that should not be flushed
        self.dummy_tasks = {}
        self.pre_assigned_dummies = {}

        if method in [Methods.ADVANCED_GLOBAL, Methods.ADVANCED_LOCAL]:
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
            #print(f"[BATCH-IN] tid={tid} act={act_name} time={start_time} pending_before={len(self.pending_tasks)}")
            selected_id = self._allocate_batch(act_name, start_time, duration, case_id, tid, permitted)
        elif self.method == Methods.ADVANCED_LOCAL:
            selected_id, task_start = self._allocate_advanced_local(act_name, permitted, start_time, duration)
        elif self.method == Methods.ADVANCED_GLOBAL:
            selected_id, task_start = self._allocate_advanced_global(act_name, start_time, case_id, tid, permitted, duration)

        if selected_id and task_start:  # Only used for SHQ and ADVANCED allocation for resources' task queues
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
        on_shift = self._get_available_resources(resources, current_now)
        if not on_shift:
            return None, None

        available_resources = [self.resources[res_id] for res_id in on_shift]
        selected_res = min(available_resources, key=lambda res: res.get_queue_length())
        res_id = selected_res.get_id()
        remaining_working_time = selected_res.get_remaining_working_time(current_now)
        work_done = current_now + timedelta(seconds=remaining_working_time)
        break_end = self.availabilities.is_resource_on_break(res_id, work_done)

        if break_end:
            task_start = break_end
        else:
            task_start = work_done
        return res_id, task_start

    def _allocate_batch(self, act_name, start_time, duration, case_id, tid, permitted):

        task_key = (case_id, tid)

        if task_key in self.assigned_batch_task_keys:
            return self.assigned_batch_task_keys[task_key]

        if task_key in self.pending_task_keys:
            return None

        task = {
            "activity": act_name,
            "arrival_time": start_time,
            "duration": duration,
            "cid": case_id,
            "tid": tid,
            "permitted": list(permitted)
        }
        self.pending_tasks.append(task)
        self.pending_task_keys.add(task_key)
        # Debug Print
        #print(f"[BATCH-QUEUE] appended tid={tid} pending_now={len(self.pending_tasks)} batch_k={self.batch_k}")
        if len(self.pending_tasks) < self.batch_k:
            # Debug Pruint
            #print(f"[BATCH-WAIT] not enough tasks yet: {len(self.pending_tasks)}/{self.batch_k}")
            return None
        # Debug Print
        #print(f"[BATCH-TRIGGER] flushing batch of size {self.batch_k} at {start_time}")

        batch = self.pending_tasks[:self.batch_k]
        self.pending_tasks = self.pending_tasks[self.batch_k:]

        assignments = self._flush_k_batch(batch, start_time)
        self.last_batch_assignments = assignments

        for a in assignments:
            if a["cid"] == case_id and a["tid"] == tid:
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
            task_key = (task["cid"], task["tid"])
            self.pending_task_keys.discard(task_key)
            self.assigned_batch_task_keys[task_key] = best_res_id
            # Debug
            #print(f"[BATCH-ASSIGN] tid={task['tid']} -> res={best_res_id} start={best_start} end={best_end}")

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
                "cid": task["cid"],
                "tid": task["tid"],
                "res_id": best_res_id,
                "start": best_start
            })

        if unassigned:
            self.pending_tasks = unassigned + self.pending_tasks

        self.batching_ctr += 1
        #Debug
        #print(f"[BATCH-DONE] assigned={len(assignments)} unassigned={len(unassigned)} total_batches={self.batching_ctr}")
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
        self.pending_task_keys.clear()
        self.assigned_batch_task_keys = {}
        self.last_batch_assignments = []

    def _allocate_advanced_local(self, act_name, permitted, start_time, duration):
        best_res_id = None
        task_start = 0.0
        min_cost = float('inf')

        available = self._get_available_resources(permitted, start_time)
        for res_id in available:  # Calculate costs for available resources
            expected_cost = duration.total_seconds() # TODO Change backself._predictor.predict_cost(act_name, res_id)
            remaining_cost = self.resources[res_id].get_remaining_working_time(start_time)

            # Check for shift and break violations
            projected_start = start_time + timedelta(seconds=remaining_cost)
            break_end = self.availabilities.is_resource_on_break(res_id, projected_start)

            if break_end:  # Add break delay to total cost
                delay = (break_end - projected_start).total_seconds()
                remaining_cost += delay
                projected_start = break_end

            projected_end = projected_start + timedelta(seconds=expected_cost)
            if not self.availabilities.is_resource_available(res_id, projected_end):
                continue

            total_cost = expected_cost + remaining_cost

            if total_cost < min_cost:
                min_cost = total_cost
                best_res_id = res_id
                task_start = remaining_cost

        # Check if dummy value more cost-efficient
        if self._predictor.get_dummy_cost(act_name) < min_cost or not best_res_id:
            return None, None

        # Update start time
        return best_res_id, (start_time + timedelta(seconds=task_start))

    def _allocate_advanced_global(self, act_name, start_time, case_id, tid, permitted, duration):
        task_key = f"{case_id}_{tid}"

        # Check whether task was already assigned in previous run
        if task_key in self.pre_assigned_dummies:
            res_id = self.pre_assigned_dummies.pop(task_key)['res_id']
            remaining_time = self.resources[res_id].get_remaining_working_time(start_time)
            return res_id, start_time + pd.Timedelta(seconds=remaining_time)

        new_task = {'task_key': task_key, 'act_name': act_name, 'permitted': permitted}

        on_shift = self._get_available_resources(self.resources.keys(), start_time)  # Get all resources for cost matrix
        if not on_shift:
            self.dummy_tasks[task_key] = new_task
            return None, None

        # Add up the expected workload of all tasks already pre-assigned
        pre_queue = {res: 0.0 for res in on_shift}
        for task in self.pre_assigned_dummies.values():
            if task['res_id'] in pre_queue:
                pre_queue[task['res_id']] += task['duration']

        tasks_to_eval = [new_task] + list(self.dummy_tasks.values())
        num_tasks = len(tasks_to_eval)
        num_res = len(on_shift)

        # Build cost matrix
        MAX_COST = 1e9  # Simulates not permitted resources
        cost_matrix = np.full((num_tasks, num_res + num_tasks), MAX_COST)

        for i, task in enumerate(tasks_to_eval):
            for j, res_id in enumerate(on_shift):
                if res_id in task['permitted']:
                    actual_queue = self.resources[res_id].get_remaining_working_time(start_time)
                    total_queue = actual_queue + pre_queue[res_id]  # consider pre-assigned tasks duration

                    projected_start = start_time + timedelta(seconds=total_queue)  # Check for shift violations
                    break_end = self.availabilities.is_resource_on_break(res_id, projected_start)
                    if break_end:
                        delay = (break_end - projected_start).total_seconds()
                        total_queue += delay
                        projected_start = break_end

                    cost = duration.total_seconds() # TODO Change back self._predictor.predict_cost(task['act_name'], res_id)
                    projected_end = projected_start + timedelta(seconds=cost)
                    if self.availabilities.is_resource_available(res_id, projected_end):
                        total_cost = total_queue + cost
                        cost_matrix[i, j] = total_cost

            dummy_cost = self._predictor.get_dummy_cost(task['act_name'])
            cost_matrix[i, num_res + i] = dummy_cost

        # Solve assignments
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_current_res, assigned_current_start = None, None
        for row, col in zip(row_ind, col_ind):
            matched_key = tasks_to_eval[row]['task_key']
            matched_act = tasks_to_eval[row]['act_name']

            if col < num_res:  # Real resource assigned
                res_id = on_shift[col]
                if matched_key == task_key:  # Current task
                    assigned_current_res = res_id
                    remaining = self.resources[res_id].get_remaining_working_time(start_time) + pre_queue[res_id]
                    assigned_current_start = start_time + timedelta(seconds=remaining)
                else:  # Old dummy task
                    self.dummy_tasks.pop(matched_key)
                    self.pre_assigned_dummies[matched_key] = {
                        'res_id': res_id,
                        'duration': self._predictor.predict_cost(matched_act, res_id)
                    }
            else:  # Postpone task
                if matched_key == task_key:
                    self.dummy_tasks[task_key] = new_task

        return assigned_current_res, assigned_current_start

    def get_next_available_time_adv(self, act_name, current_time):
        """
        Finds the best worker based on Total Cost, but wakes the engine up
        the moment their current queue is finished.
        """
        best_remaining = None
        min_total_cost = float('inf')

        res_costs = self._predictor.get_all_costs(act_name)
        for res_id, expected_cost in res_costs.items():
            if self.availabilities.is_resource_available(res_id, current_time):
                remaining = self.resources[res_id].get_remaining_working_time(current_time)
                total_cost = expected_cost + remaining
                if total_cost < min_total_cost:
                    min_total_cost = total_cost
                    best_remaining = remaining

        if best_remaining is not None:
            return current_time + timedelta(seconds=best_remaining)

    class _Predictor:
        """
        costs: dict saving activities to dict of resources and their costs for the activity
        delta: factor variable used for cost estimation
        """

        def __init__(self, delta, averages_path='resources/allocation/resource_activity_averages.csv'):
            self._costs = {}
            self._delta = delta
            self.avg_stats = self._load_averages(averages_path)

        def _load_averages(self, path):
            try:
                df = pd.read_csv(path)
                return df.groupby('Activity').apply(
                    lambda x: dict(zip(x['Resource'], x['AverageDuration'])),
                    include_groups=False
                ).to_dict()
            except FileNotFoundError:
                print(f"Warning: {path} not found. Starting with empty historical stats.")
                return {}

        def predict_cost(self, act_name, res_id):
            """
            Retrieves cost if already predicted, otherwise predicts it.
            """
            if act_name not in self._costs:
                self._costs[act_name] = {}

            if res_id not in self._costs[act_name]:
                predicted_value = 0
                if act_name.startswith('W_'):  # Processing times only available for W activities
                    avg = self.avg_stats.get(act_name, {}).get(res_id)
                    if avg is not None:
                        predicted_value = avg
                    else:  # Fallback
                        activity_data = list(self.avg_stats.get(act_name, {}).values())
                        if activity_data:
                            predicted_value = np.mean(activity_data)
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
