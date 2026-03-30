from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class ModelMetrics:
    f1_score: float
    accuracy: float
    precision: float
    recall: float
    train_time: float
    inference_time_ms: float
    tp: Optional[int] = None
    tn: Optional[int] = None
    fp: Optional[int] = None
    fn: Optional[int] = None

    @property
    def has_confusion_matrix(self) -> bool:
        return all(v is not None for v in [self.tp, self.tn, self.fp, self.fn])


class BaseModel(ABC):
    def __init__(self, name: str, seed: int):
        self.name = name
        self.seed = seed
        self._model: Optional[Any] = None

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> ModelMetrics:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass
