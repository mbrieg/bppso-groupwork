import os.path
import pandas as pd

from resources.Resource import Resource
from resources.ResourcePermissions import ResourcePermissions
from resources.ResourceAvailabilities import ResourceAvailabilities
from resources.ResourceAllocator import ResourceAllocator


class ResourceManager:
    def __init__(self, availabilities_csv='weekly_schedule_median.csv',
                 permissions_csv='permissions_basic.csv'):
        """
        :param availabilities_csv:   CSV file with ['Resource', 'DayId', 'StartTime', 'EndTime']
        :param permissions_csv:      CSV file with ['Activity', 'Resource']
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        df_availabilities = pd.read_csv(os.path.join(current_dir, 'availabilities/' + availabilities_csv))
        df_permissions = pd.read_csv(os.path.join(current_dir, 'permissions/' + permissions_csv))

        self.availabilities = ResourceAvailabilities(df_availabilities)
        self.permissions = ResourcePermissions(df_permissions)
        self.resources = {
            str(res_name): Resource(res_name) for res_name in df_availabilities['Resource'].unique()
        }
        self.allocator = ResourceAllocator(
            self.resources,
            self.availabilities,
            self.permissions
        )

    def get_resources(self):
        """
        Returns: A list of all resource objects.
        """
        return self.resources.values()

    def assign_resource(self, act_name, current_time, duration):
        """
        Returns: Resource name or None if no resource is available.
        """
        return self.allocator.allocate_resource(act_name, current_time, duration)

    def get_earliest_availability(self, act_name, current_time):
        """
        Returns: Datetime object containing the next possible start time of ANY permitted resource or None.
        """
        permitted = self.permissions.get_permitted_resources(act_name)
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
            res.release()
