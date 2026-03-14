import datetime
import holidays.countries
import pandas as pd
import os.path


class ResourceAvailabilities:
    """
        Manages the working schedules for resources in the simulation.
        Determines when a resource is on shift and when they will be available next.
    """
    def __init__(self, availabilities_file, mode, interval=7):
        """
        Args:
            availabilities_file (str): Filename in 'resources/availabilities' directory.
            interval (int, optional): The cycle length in days. Defaults to 7 (Weekly).
        """
        self.schedule = {}
        self.start_date = datetime.date(2016, 1, 1)  # Day 0 of simulation
        self.mode = mode
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
        if self.mode:
            df['BreakStart'] = pd.to_datetime(df['BreakStart'], format='%H:%M:%S').dt.time
            df['DurationMin'] = df['DurationMin'].fillna(0).astype(int)
        else:
            df['BreakStart'] = datetime.time(0, 0)
            df['DurationMin'] = 0

        for _, row in df.iterrows():
            key = (row['Resource'], row['DayId'])
            self.schedule[key] = (row['StartTime'], row['EndTime'], row['BreakStart'], row['DurationMin'])

    def _get_day_id(self, current_time):
        delta = current_time.date() - self.start_date
        return delta.days % self.interval

    def _is_holiday(self, current_date):
        """
        Checks if a given date falls on a defined custom holiday or a Dutch national holiday.
        """
        if (current_date.day, current_date.month) in self.holidays:
            return True
        if self.nl_holidays.get(current_date) == "Hemelvaartsdag":  # Ascension Day
            return True
        return False

    def is_resource_available(self, res_name, current_time):
        """
        Checks if a resource is currently on shift.
        :param res_name: The ID of the resource.
        :param current_time: The current simulation time.
        :return: bool: Returns True if resource is on shift.
                    Returns False if resource is not on shift or on holiday.
        """
        if self._is_holiday(current_time.date()):
            if res_name != 'User_1':
                return False

        day_id = self._get_day_id(current_time)
        shift = self.schedule.get((res_name, day_id))
        if not shift:
            return False

        start_time, end_time, _, _ = shift
        if not (start_time <= current_time.time() <= end_time):
            return False

        return True

    def is_resource_on_break(self, res_name, current_time):
        """
        Determines if a resource is currently on their scheduled break.
        :param res_name: The ID of the resource.
        :param current_time: The current simulation time.
        :return: datetime or None: Returns the datetime of when the break ENDS if they are on break.
                              Returns None if they are not on break or off-shift.
        """
        day_id = self._get_day_id(current_time)
        shift = self.schedule.get((res_name, day_id))
        if not shift:
            return None
        start_time, end_time, break_start, break_duration = shift

        if break_duration > 0:
            break_start_dt = datetime.datetime.combine(current_time.date(), break_start)
            break_end_dt = break_start_dt + datetime.timedelta(minutes=break_duration)

            if break_start_dt <= current_time < break_end_dt:
                return break_end_dt

        return None

    def get_next_available_time(self, res_name, current_time):
        """
        Calculates the exact future datetime when a resource will next be available to take a task.
        It evaluates current breaks, upcoming shifts today, and shifts on future days.

        :param res_name: The ID of the resource.
        :param current_time: The time to search forward from.
        :return: datetime or None: The exact time they will clock in / return from break.
                              Returns None if they are never scheduled again.
        """
        # Check if resource is available but currently on break
        if self.is_resource_available(res_name, current_time):
            break_end = self.is_resource_on_break(res_name, current_time)
            if not break_end:
                return current_time
            else:
                return break_end

        # Check today
        current_day_id = self._get_day_id(current_time)
        if not (self._is_holiday(current_time.date()) and res_name != "User_1"):
            shift = self.schedule.get((res_name, current_day_id))
            if shift:
                start_time, end_time, break_start, break_duration = shift
                if current_time.time() < start_time:    # Shift starts later this day
                    return datetime.datetime.combine(current_time.date(), start_time)
                if break_duration > 0:
                    break_start_dt = datetime.datetime.combine(current_time.date(), break_start)
                    break_end_dt = break_start_dt + datetime.timedelta(minutes=break_duration)
                    if break_start_dt <= current_time < break_end_dt:   # user currently on break
                        return break_end_dt

        # Check future days
        for next_day in range(1, self.interval):
            next_date = current_time.date() + datetime.timedelta(days=next_day)
            if self._is_holiday(next_date) and res_name != "User_1":
                continue

            day_id = (current_day_id + next_day) % self.interval
            shift = self.schedule.get((res_name, day_id))

            if shift:
                start_time = shift[0]
                return datetime.datetime.combine(next_date, start_time)

        return None
