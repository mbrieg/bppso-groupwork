import random
from datetime import timedelta, datetime
from typing import Optional, Dict
from .base_spawner import BaseSpawner

class StaticSpawner(BaseSpawner):
    def __init__(self, mean_minutes=30, seed: Optional[int] = None):
        self.mean_minutes = mean_minutes
        if seed is not None:
            random.seed(seed)

    def calculate_next_spawn(self, current_time: datetime, attributes: Optional[Dict] = None) -> datetime:
        # Exponentialverteilung für zufällige, aber stetige Ankünfte
        wait_time = random.expovariate(1.0 / self.mean_minutes)
        return current_time + timedelta(minutes=wait_time)