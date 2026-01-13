import datetime
import pandas as pd


class ResourceAvailabilities:
    def __init__(self, df, interval=7):
        self.schedule = {}      # {(res_name, day_id) : (start, end)}
        self.start_date = datetime.date(2016, 1, 1)  # Day 0 of simulation
        self.interval = interval

        if df is not None:
            self.load_schedule(df)

    def load_schedule(self, df):
        df['StartTime'] = pd.to_datetime(df['StartTime'], format='%H:%M:%S').dt.time
        df['EndTime'] = pd.to_datetime(df['EndTime'], format='%H:%M:%S').dt.time
        for _, row in df.iterrows():
            key = (row['Resource'], row['DayId'])
            self.schedule[key] = (row['StartTime'], row['EndTime'])

    def get_day_id(self, current_time):
        delta = current_time.date() - self.start_date
        return delta.days % self.interval

    def is_resource_available(self, res_name, current_time):
        day_id = self.get_day_id(current_time)
        shift = self.schedule.get((res_name, day_id))

        if not shift:
            return False

        start_time, end_time = shift
        return start_time <= current_time.time() <= end_time

    def get_next_available_time(self, res_name, current_time):
        if self.is_resource_available(res_name, current_time):
            return current_time

        current_day_id = self.get_day_id(current_time)
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
