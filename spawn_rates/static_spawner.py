import random
from datetime import timedelta, datetime
from typing import Optional, Dict
from .base_spawner import BaseSpawner

class StaticSpawner(BaseSpawner):

    #Baseline Spawner mit zeitunabhängiger Poisson-Ankunftsrate
    #Interarrival Times ~ Exponential(mean_minutes)

    def __init__(self, mean_minutes:float = 30.0, seed: int | None = None):
        self.rng = random.Random(seed)
        self.mean_minutes = mean_minutes

    def calculate_next_spawn(self, current_time: datetime) -> datetime:
        wait_time = self.rng.expovariate(1.0 / self.mean_minutes)
        return current_time + timedelta(minutes=wait_time)