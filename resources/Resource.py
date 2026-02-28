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
        return self._task_queue.pop(0)

    def get_queue_length(self):
        return len(self._task_queue)

    def is_occupied(self):
        return self._is_occupied

    def occupy(self):
        self._is_occupied = True

    def release(self):
        self._is_occupied = False
