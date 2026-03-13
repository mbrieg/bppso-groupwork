import os.path
import pandas as pd

from resources.Resource import Resource
from resources.ResourcePermissions import ResourcePermissions
from resources.ResourceAvailabilities import ResourceAvailabilities
from resources.ResourceAllocator import ResourceAllocator, Methods


class ResourceManager:
    def __init__(self, availabilities='availabilities_basic.csv',
                 permissions='permissions_basic.csv',
                 roles='resource_roles.csv', method=Methods.RANDOM, delta=1, batch_k = 5):
        """
        :param availabilities: CSV file with ['Resource', 'DayId', 'StartTime', 'EndTime'] (basic) or
                                ['Resource', 'DayId', 'StartTime', 'EndTime', 'BreakStart', 'DurationMin'] (advanced)
        :param permissions:    CSV file with ['Activity', 'Resource'] (basic) or ['Activity', 'Role'] (advanced)
        :param roles:          CSV file with ['Resource', 'Role'] (Only used in advanced mode)
        """
        avail_adv = availabilities != 'availabilities_basic.csv'
        perm_adv = permissions != 'permissions_basic.csv'
        self._mode_adv = method == Methods.ADVANCED

        self.availabilities = ResourceAvailabilities(availabilities, avail_adv)
        self.permissions = ResourcePermissions(permissions, perm_adv)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'availabilities', availabilities)
        all_res_names = pd.read_csv(file_path, usecols=['Resource'])
        self.resources = {
            str(res_name): Resource(res_name) for res_name in all_res_names['Resource'].unique()
        }

        if perm_adv:
            self._assign_roles_to_resources(roles)

        self.allocator = ResourceAllocator(
            self.resources,
            self.availabilities,
            self.permissions,
            method,
            delta=delta,
            batch_k=batch_k
        )

    def _assign_roles_to_resources(self, file: str):
        """
        Helper used for initialisation.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        roles_path = os.path.join(current_dir, 'permissions', file)

        if not os.path.exists(roles_path):
            print(f"Warning: Roles file missing at {roles_path}. Defaulting to None.")
            return

        df = pd.read_csv(roles_path)
        role_map = pd.Series(df.Role.values, index=df.Resource).to_dict()

        for res_name, res_obj in self.resources.items():
            res_obj.set_role(role_map.get(res_name))

    def get_resources(self):
        """
        Returns: A list of all resource objects.
        """
        return self.resources.values()

    def get_resource(self, res_id):
        """
        Returns: A resource object.
        """
        return self.resources[res_id]

    def assign_resource(self, act_name, current_time, duration, case_id, tid):
        """
        Returns: Resource name or None if no resource is available.
        """
        return self.allocator.allocate_resource(act_name, current_time, duration, case_id, tid)

    def get_earliest_availability(self, act_name, current_time):
        """
        Returns: Datetime object containing the next possible start time of ANY permitted resource or None.
        """
        if self._mode_adv:  # Advanced allocation approach
            return self.allocator.get_next_available_time_adv(act_name, current_time)

        permitted = self.permissions.get_permitted_resources(act_name, self.resources)
        if not permitted:
            return None

        earliest_time = None
        for res_id in permitted:
            next_time = self.availabilities.get_next_available_time(res_id, current_time)
            if next_time:
                if earliest_time is None or next_time < earliest_time:
                    earliest_time = next_time

        return earliest_time

    def reset_simulation(self):
        """
        Clears all resource states for a new simulation run.
        """
        for res in self.resources.values():
            res.reset()
        self.allocator.reset()

    def flush_remaining_batches(self, current_time):
        #Debug
        #print(f"[RM-FLUSH] current_time={current_time}")
        return self.allocator.flush_remaining_batches(current_time)