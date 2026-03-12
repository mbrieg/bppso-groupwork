class Resource:
    def __init__(self, res_id):
        self._id = res_id
        self._role = None
        self._is_occupied = False
        self._task_queue = []

    def set_role(self, role):
        self._role = role

    def get_role(self):
        return self._role

    def get_id(self):
        return self._id

    def add_task(self, task):
        self._task_queue.append(task)

    def pop_task(self):
        if len(self._task_queue) > 0:
            return self._task_queue.pop(0)
        return None

    def get_queue_length(self):
        return len(self._task_queue)

    def is_occupied(self):
        return self._is_occupied

    def occupy(self):
        self._is_occupied = True

    def release(self):
        self._is_occupied = False

    def reset(self):
        self._is_occupied = False
        self._task_queue.clear()

    def get_remaining_working_time(self, current_now):
        if not self._is_occupied or not self._task_queue:
            return 0.0

        remaining_seconds = 0.0
        for i, task in enumerate(self._task_queue):
            if i == 0:  # Current task
                expected_end = task['start'] + task['duration']
                remaining = (expected_end - current_now).total_seconds()
                remaining_seconds += max(0, remaining)
            else:   # Future already assigned tasks
                remaining_seconds += task['duration'].total_seconds()

        return remaining_seconds
