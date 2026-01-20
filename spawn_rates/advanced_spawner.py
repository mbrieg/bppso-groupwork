import random
import math
from datetime import timedelta, datetime
from typing import Optional, Dict, Tuple, List
from .base_spawner import BaseSpawner

class AdvancedSpawner(BaseSpawner):
    """
    Advanced Spawn Rate Strategy (Plan A + Holidays)

    Idea:
    - Context-dependent hourly arrival volume:
        N ~ Poisson(lambda(context) * remaining_fraction_of_hour)
    - Place N arrivals uniformly within the remaining part of the current hour
    - Keep an internal buffer of planned spawn timestamps so the engine can
      repeatedly call calculate_next_spawn(current_time) and get the next spawn.
    - Context = (weekday, hour, is_holiday) where weekday is 0..6 (Mon..Sun).
    """

    def __init__(
            self,
            rate_table: Dict[Tuple[int, int, bool], float],
            holidays: Optional[List] = None,
            seed: Optional[int] = None,
            tz_name: str = "Europe/Amsterdam",
            lookahead_days: int = 7,
            normal_approx_threshold: float = 30.0,
            normal_guard_factor: float = 5.0,
    ):
        # Own RNG instance for reproducibility without global side-effects
        self.rng = random.Random(seed)

        self.rate_table = rate_table

        # Normalize holidays to a set of `date` objects for fast lookup
        # Accepts date, datetime, pandas Timestamp, etc.
        self.holidays_set = {
            h.date() if hasattr(h, "date") else h for h in (holidays or [])
        }

        self._spawn_buffer: List[datetime] = []
        self._current_bin: Optional[Tuple[int, int, bool]] = None

        # Global mean for last-resort fallback
        self._global_mean = (
            sum(rate_table.values()) / len(rate_table) if rate_table else 0.5
        )

        # Config
        self.tz_name = tz_name
        self.lookahead_days = lookahead_days
        self.normal_approx_threshold = normal_approx_threshold
        self.normal_guard_factor = normal_guard_factor

    def _poisson_sample(self, lmbda: float) -> int:
        """
        Efficient Poisson sampling:
        - Knuth for small lambdas
        - Normal approximation for large lambdas with a guard against extreme outliers
        """
        if lmbda <= 0:
            return 0

        if lmbda > self.normal_approx_threshold:
            # Poisson(λ) ≈ Normal(λ, λ)
            sample = self.rng.normalvariate(lmbda, math.sqrt(lmbda))
            # Guard: avoid extreme outliers
            return max(0, min(round(sample), int(lmbda * self.normal_guard_factor)))

        # Knuth's algorithm
        L = math.exp(-lmbda)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= self.rng.random()
        return k - 1

    def _get_context(self, dt: datetime) -> Tuple[int, int, bool]:
        """
        Context is computed in the datetime's own timezone.
        Make sure you pass local NL time to this spawner (Europe/Amsterdam),
        or keep your engine clock in that timezone.
        """
        return (dt.weekday(), dt.hour, dt.date() in self.holidays_set)

    def _get_rate_with_fallback(self, context: Tuple[int, int, bool]) -> float:
        """
        Symmetric backoff:
        1) exact (weekday, hour, is_holiday)
        2) try the alternate holiday flag for same weekday/hour
        3) global mean
        """
        if context in self.rate_table:
            return self.rate_table[context]

        weekday, hour, is_holiday = context
        alt_context = (weekday, hour, not is_holiday)
        if alt_context in self.rate_table:
            return self.rate_table[alt_context]

        return self._global_mean

    def _refill_buffer(self, current_time: datetime) -> None:
        """

        - Determine remaining interval in the current hour
        - Sample N arrivals for this remaining interval
        - Place them uniformly within [current_time, hour_end)
        - Store in sorted buffer
        """
        context = self._get_context(current_time)
        lmbda = self._get_rate_with_fallback(context)

        # Define end of current hour
        hour_end = (current_time + timedelta(hours=1)).replace(
            minute=0, second=0, microsecond=0
        )
        remaining_seconds = (hour_end - current_time).total_seconds()

        if remaining_seconds <= 0:
            self._spawn_buffer = []
            self._current_bin = context
            return

        remaining_fraction = remaining_seconds / 3600.0

        # Sample N for the remaining part of the hour
        n = self._poisson_sample(lmbda * remaining_fraction)

        # Uniform placement within remaining interval
        new_spawns = [
            current_time + timedelta(seconds=self.rng.uniform(0, remaining_seconds))
            for _ in range(n)
        ]

        self._spawn_buffer = sorted(new_spawns)
        self._current_bin = context

    def calculate_next_spawn(
            self, current_time: datetime, attributes: Optional[Dict] = None
    ) -> datetime:
        """
        Returns the next spawn timestamp >= current_time.
        Uses a bounded lookahead to skip empty hours.
        """
        lookahead_limit = current_time + timedelta(days=self.lookahead_days)

        while current_time < lookahead_limit:
            current_context = self._get_context(current_time)

            # Refill buffer when entering a new hour or buffer is empty
            if not self._spawn_buffer or current_context != self._current_bin:
                self._refill_buffer(current_time)

            if self._spawn_buffer:
                return self._spawn_buffer.pop(0)

            # No spawns in this hour remainder -> jump to next hour start
            current_time = (current_time + timedelta(hours=1)).replace(
                minute=0, second=0, microsecond=0
            )

        # Absolute fallback if nothing found within lookahead window
        return current_time + timedelta(minutes=30)
