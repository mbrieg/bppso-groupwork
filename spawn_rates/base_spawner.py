from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict

class BaseSpawner(ABC):
    @abstractmethod
    def calculate_next_spawn(self, current_time: datetime) -> datetime:
        raise NotImplementedError