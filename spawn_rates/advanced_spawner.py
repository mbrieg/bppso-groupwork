import random
import math
from datetime import timedelta, datetime
from typing import Optional, Tuple, List, Dict
from .base_spawner import BaseSpawner

class AdvancedSpawner(BaseSpawner):
    NORMAL_APPROX_THRESHOLD = 30.0
    NORMAL_GUARD_FACTOR = 5.0

    def __init__(
            self,
            rate_table: Dict[Tuple[int, int, bool], float],
            holidays: Optional[List] = None,
            seed: Optional[int] = None,
            lookahead_days: int = 7,
    ):
        self.rng = random.Random(seed)
        self.rate_table = rate_table
        self.lookahead_days = lookahead_days

        self.holidays_set = {h.date() if hasattr(h, "date") else h for h in (holidays or [])}

        self._spawn_buffer: List[datetime] = []
        self._current_bin: Optional[Tuple[int, int, bool]] = None

        self._global_mean = sum(rate_table.values()) / len(rate_table) if rate_table else 0.5

    def _poisson_sample(self, lmbda: float) -> int:
        if lmbda <= 0:
            return 0

        if lmbda > self.NORMAL_APPROX_THRESHOLD:
            sample = self.rng.normalvariate(lmbda, math.sqrt(lmbda))
            return max(0, min(round(sample), int(lmbda * self.NORMAL_GUARD_FACTOR)))

        L = math.exp(-lmbda)
        k, p = 0, 1.0
        while p > L:
            k += 1
            p *= self.rng.random()
        return k - 1

    def _get_context(self, dt: datetime) -> Tuple[int, int, bool]:
        return (dt.weekday(), dt.hour, dt.date() in self.holidays_set)

    def _get_rate_with_fallback(self, context: Tuple[int, int, bool]) -> float:
        if context in self.rate_table:
            return self.rate_table[context]

        weekday, hour, is_holiday = context
        candidates = [self.rate_table.get((weekday, h, is_holiday)) for h in (hour - 1, hour + 1)]
        candidates = [x for x in candidates if x is not None]
        if candidates:
            return sum(candidates) / len(candidates)

        if 0 <= hour <= 5:
            return min(self._global_mean, 1.2)

        return self.rate_table.get((weekday, hour, not is_holiday), self._global_mean)

    def _refill_buffer(self, current_time: datetime) -> None:
        context = self._get_context(current_time)
        lmbda = self._get_rate_with_fallback(context)

        hour_end = (current_time + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        remaining_seconds = (hour_end - current_time).total_seconds()

        if remaining_seconds <= 0:
            self._spawn_buffer = []
            self._current_bin = context
            return

        n = self._poisson_sample(lmbda * (remaining_seconds / 3600.0))

        self._spawn_buffer = sorted(
            current_time + timedelta(seconds=self.rng.uniform(0, remaining_seconds))
            for _ in range(n)
        )
        self._current_bin = context

    def calculate_next_spawn(self, current_time: datetime) -> datetime:
        limit = current_time + timedelta(days=self.lookahead_days)

        while current_time < limit:
            ctx = self._get_context(current_time)

            if not self._spawn_buffer or ctx != self._current_bin:
                self._refill_buffer(current_time)

            if self._spawn_buffer:
                return self._spawn_buffer.pop(0)

            current_time = (current_time + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

        return current_time + timedelta(minutes=30)
    
    # Add case attributes to the spawner
    
