import random
import math
from datetime import timedelta, datetime
from typing import Optional, Tuple, List, Dict
from .base_spawner import BaseSpawner

class AdvancedSpawner(BaseSpawner):
    NORMAL_APPROX_THRESHOLD = 30.0
    NORMAL_GUARD_FACTOR = 5.0

    # Real distributions of case attributes from BPI Challenge 2017 event log
    LOAN_GOAL_DIST = {
    "Car": 0.2960,
    "Home improvement": 0.2434,
    "Existing loan takeover": 0.1778,
    "Other, see explanation": 0.0947,
    "Unknown": 0.0751,
    "Not speficied": 0.0338,
    "Remaining debt home": 0.0267,
    "Extra spending limit": 0.0198,
    "Caravan / Camper": 0.0117,
    "Motorcycle": 0.0087,
    "Boat": 0.0064,
    "Tax payments": 0.0048,
    "Business goal": 0.0010,
    "Debt restructuring": 0.0001,}

    APP_TYPE_DIST = {
    "New credit": 0.8924,
    "Limit raise": 0.1076,}

    AMOUNT_LOG_MEAN = 9.5387
    AMOUNT_LOG_STD  = 0.7092

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

    def get_case_attributes(self, case_id=None) -> Dict:
        """
        Generate case-level attributes sampled from real log distributions.
        Called by AdvancedRouter
        """
        # Sample LoanGoal
        loan_goal = self._weighted_choice(self.LOAN_GOAL_DIST)

        # Sample ApplicationType
        app_type = self._weighted_choice(self.APP_TYPE_DIST)

        # Sample RequestedAmount from lognormal
        raw_amount = self.rng.lognormvariate(self.AMOUNT_LOG_MEAN, self.AMOUNT_LOG_STD)
        # Round to nearest 500 and clamp
        requested_amount = max(500, min(round(raw_amount / 500) * 500, 350000))

        # Amount category (same bins as training)
        if requested_amount <= 5000:    amount_cat = "very_low"
        elif requested_amount <= 10000: amount_cat = "low"
        elif requested_amount <= 20000: amount_cat = "medium"
        elif requested_amount <= 50000: amount_cat = "high"
        else:                           amount_cat = "very_high"

        # Credit score: sampled per case (some cases have no score)
        if self.rng.random() < 0.7:  # ~70% of cases have a credit score
            cs = self.rng.gauss(700, 200)
            cs = max(0, min(1200, cs))
            if cs > 800:   cs_bin = "excellent"
            elif cs > 600: cs_bin = "good"
            elif cs > 400: cs_bin = "fair"
            else:          cs_bin = "poor"
        else:
            cs_bin = "unknown"

        return {
            "loan_goal": loan_goal,
            "application_type": app_type,
            "requested_amount": requested_amount,
            "amount_category": amount_cat,
            "credit_score_bin": cs_bin,
        }

    def _weighted_choice(self, dist: Dict[str, float]) -> str:
        keys = list(dist.keys())
        weights = list(dist.values())
        return self.rng.choices(keys, weights=weights, k=1)[0]

