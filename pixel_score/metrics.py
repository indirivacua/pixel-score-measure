from abc import ABC, abstractmethod
from typing import Optional
import torch

class Metric(ABC):
    def __init__(self):
        self.result: Optional[torch.Tensor] = None
    
    @abstractmethod
    def compute(self) -> torch.Tensor:
        pass
    
    @abstractmethod
    def update(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def reset(self):
        pass
    
    @staticmethod
    @abstractmethod
    def validate_inputs(inputs: torch.Tensor, targets: torch.Tensor):
        pass