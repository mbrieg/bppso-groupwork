import datetime

import holidays.countries
import pandas as pd
import os.path


class ResourceAvailabilities:
    """
        Manages the working schedules for resources in the simulation.
        Determines when a resource is on shift and when they will be available next.
    """
    def __init__(self, availabilities_file, interval=7):
        """
        Args:
            availabilities_file (str): Filename in 'resources/availabilities' directory.
            interval (int, optional): The cycle length in days. Defaults to 7 (Weekly).
        """
        self.schedule = {}      # {(res_name, day_id) : (start, end)}
        self.start_date = datetime.date(2016, 1, 1)  # Day 0 of simulation
        self.interval = interval

        self.holidays = {
            (1, 1): "New Year's Day",
            (27, 4): "King's Day",
            (5, 5): "Liberation Day",
            (25, 12): "Christmas Day",
            (26, 12): "Boxing Day"
        }
        self.nl_holidays = holidays.countries.Netherlands()

        if availabilities_file is not None:
            self._load_schedule(availabilities_file)

    def _load_schedule(self, file: str):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, 'availabilities', file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Permission file not found at: {path}")
        df = pd.read_csv(path)
        df['StartTime'] = pd.to_datetime(df['StartTime'], format='%H:%M:%S').dt.time
        df['EndTime'] = pd.to_datetime(df['EndTime'], format='%H:%M:%S').dt.time
        for _, row in df.iterrows():
            key = (row['Resource'], row['DayId'])
            self.schedule[key] = (row['StartTime'], row['EndTime'])

    def _get_day_id(self, current_time):
        delta = current_time.date() - self.start_date
        return delta.days % self.interval

    def is_resource_available(self, res_name, current_time):
        holiday = self.nl_holidays.get(current_time.date())
        if ((current_time.date().day, current_time.date().month) in self.holidays
                or holiday == "Hemelvaartsdag"):    # Ascension Day
            if res_name != 'User_1':
                return False

        day_id = self._get_day_id(current_time)
        shift = self.schedule.get((res_name, day_id))

        if not shift:
            return False

        start_time, end_time = shift
        return start_time <= current_time.time() <= end_time

    def get_next_available_time(self, res_name, current_time):
        if self.is_resource_available(res_name, current_time):
            return current_time

        current_day_id = self._get_day_id(current_time)
        shift = self.schedule.get((res_name, current_day_id))
        if shift:
            start_time, end_time = shift
            if current_time.time() < start_time:
                return datetime.datetime.combine(current_time.date(), start_time)

        # Check future days
        for next_day in range(1, self.interval):
            day_id = (current_day_id + next_day) % self.interval
            shift = self.schedule.get((res_name, day_id))

            if shift:
                start_time, end_time = shift
                next_date = current_time.date() + datetime.timedelta(days=next_day)
                return datetime.datetime.combine(next_date, start_time)

        return None
