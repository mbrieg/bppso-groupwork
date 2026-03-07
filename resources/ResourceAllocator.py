import random
from enum import Enum


class Methods(Enum):
    RANDOM = 0
    ROUND_ROBIN = 1
    SHORTEST_QUEUE = 2
    BATCHING = 3
    ADVANCED = 4    # Adapted assignment problem


class ResourceAllocator:
    """
    Assigns tasks to permitted and available resources based on the specified method.
    """
    def __init__(self, resources, availabilities, permissions, method=Methods.RANDOM):
        self.resources = resources
        self.availabilities = availabilities
        self.permissions = permissions
        self.method = method
        self.rr_index = 0
        self.batching_ctr = 0

    def allocate_resource(self, act_name, start_time, duration, case_id, tid):
        # Check permissions
        permitted = self.permissions.get_permitted_resources(act_name, self.resources)
        permitted.sort()
        if not permitted:
            return None

        # Choose resource
        selected_id = None
        if self.method == Methods.RANDOM:
            selected_id = self._allocate_random(permitted, start_time)
        elif self.method == Methods.ROUND_ROBIN:
            selected_id = self._allocate_round_robin(permitted, start_time)
        elif self.method == Methods.SHORTEST_QUEUE:
            selected_id = self._allocate_shortest_queue(permitted)
        elif self.method == Methods.BATCHING:
            selected_id = self._allocate_batch(permitted)
        elif self.method == Methods.ADVANCED:
            selected_id = self._allocate_advanced(permitted)

        if selected_id:
            act_info = {
                "activity": act_name,
                "start": start_time,
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
        if not on_shift:
            return None
        return on_shift

    def _get_idle_resources(self, on_shift):
        if not on_shift:
            return None
        idle = []
        for res_id in on_shift:
            if not self.resources[res_id].is_occupied():
                idle.append(res_id)
        return idle

    def _allocate_random(self, resources, start_time):
        on_shift = self._get_available_resources(resources, start_time)
        idle = self._get_idle_resources(on_shift)
        if not idle:
            return None
        return random.choice(idle)

    def _allocate_round_robin(self, resources, start_time):
        for i in range(len(resources)):
            selected_id = (self.rr_index + i) % len(resources)
            res_id = resources[selected_id]

            if self.availabilities.is_resource_available(res_id, start_time) and not self.resources[res_id].is_occupied():
                self.rr_index = (selected_id + 1) % len(resources)
                return res_id

        return None

    def _allocate_shortest_queue(self, resources):
        resources = [self.resources[res_id] for res_id in resources]
        selected_res = min(resources, key=lambda res: res.get_queue_length())
        return selected_res.get_id()

    def _allocate_batch(self, resources):
        # TODO
        raise NotImplementedError

    def _allocate_advanced(self, resources):
        # TODO
        raise NotImplementedError
