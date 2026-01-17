import random


class ResourceAllocator:
    """
    Assigns tasks to permitted and available resources.
    """
    def __init__(self, resources, availabilities, permissions):
        self.resources = resources
        self.availabilities = availabilities
        self.permissions = permissions

    def allocate_resource(self, act_name, start_time, duration):
        # Check permissions
        permitted = self.permissions.get_permitted_resources(act_name)
        if not permitted:
            return None

        # Check availabilities
        available = []
        for res_id in permitted:
            if res_id not in self.resources:
                continue

            resource = self.resources[res_id]
            if self.availabilities.is_resource_available(res_id, start_time):
                # Check if resource is occupied
                if not resource.is_occupied(start_time):
                    available.append(res_id)

        if not available:
            return None

        selected_id = random.choice(available)      # Random Resource allocation
        end_time = start_time + duration
        self.resources[selected_id].occupy(end_time)

        return selected_id
