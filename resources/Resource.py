class Resource:
    def __init__(self, res_name):
        self.name = res_name
        self.role = None
        self.occupied_until = None

    def set_role(self, role):
        self.role = role

    def get_role(self):
        return self.role

    def get_id(self):
        return self.name

    def is_occupied(self, current_time):
        if self.occupied_until is None:
            return False
        if current_time >= self.occupied_until:
            self.release()
            return False
        return True

    def occupy(self, until_time):
        self.occupied_until = until_time

    def release(self):
        self.occupied_until = None
