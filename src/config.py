from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import random

import numpy as np
import torch


@dataclass
class PhiloxSeedGenerator:
    base_seed: int = 42
    counter: int = 0

    def __post_init__(self):
        self._rng = np.random.Generator(np.random.Philox(self.base_seed))

    def generate_seeds(self, n: int) -> List[int]:
        return [int(self._rng.integers(0, 2**31 - 1)) for _ in range(n)]

    def get_seed(self, index: int) -> int:
        rng = np.random.Generator(np.random.Philox(self.base_seed))
        for _ in range(index + 1):
            seed = int(rng.integers(0, 2**31 - 1))
        return seed

    @staticmethod
    def set_all_seeds(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class PathConfig:
    base_dir: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default=None)
    results_dir: Path = field(default=None)

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = self.base_dir / "data"
        if self.results_dir is None:
            self.results_dir = self.base_dir / "results"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "checkpoints").mkdir(exist_ok=True)
        (self.results_dir / "models").mkdir(exist_ok=True)
        (self.results_dir / "figures").mkdir(exist_ok=True)
        (self.results_dir / "metrics").mkdir(exist_ok=True)

    @property
    def checkpoints_dir(self) -> Path:
        return self.results_dir / "checkpoints"

    @property
    def models_dir(self) -> Path:
        return self.results_dir / "models"

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / "figures"

    @property
    def metrics_dir(self) -> Path:
        return self.results_dir / "metrics"


@dataclass
class TrainingConfig:
    n_seeds: int = 12
    n_folds: int = 5

    tfidf_max_features: int = 10000
    tfidf_ngram_range: tuple = (1, 2)
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.95
    tfidf_sublinear_tf: bool = True

    transformer_max_length: int = 128
    transformer_batch_size: int = 32
    transformer_learning_rate: float = 3e-5
    transformer_epochs: int = 2
    transformer_warmup_ratio: float = 0.06
    transformer_weight_decay: float = 0.01

    @property
    def tfidf_config(self) -> Dict[str, Any]:
        return {
            "max_features": self.tfidf_max_features,
            "ngram_range": self.tfidf_ngram_range,
            "min_df": self.tfidf_min_df,
            "max_df": self.tfidf_max_df,
            "sublinear_tf": self.tfidf_sublinear_tf,
        }

    @property
    def transformer_config(self) -> Dict[str, Any]:
        return {
            "max_length": self.transformer_max_length,
            "batch_size": self.transformer_batch_size,
            "learning_rate": self.transformer_learning_rate,
            "epochs": self.transformer_epochs,
            "warmup_ratio": self.transformer_warmup_ratio,
            "weight_decay": self.transformer_weight_decay,
        }


@dataclass
class DeviceConfig:
    @staticmethod
    def get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        device = DeviceConfig.get_device()
        info = {"device": str(device)}

        if device.type == "cuda":
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = (
                torch.cuda.get_device_properties(0).total_memory / 1e9
            )
        elif device.type == "mps":
            info["gpu_name"] = "Apple Silicon GPU (MPS)"
            info["gpu_memory_gb"] = "Shared with system RAM"

        return info


CLASSICAL_PARAMS = {
    "XGBoost": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "scale_pos_weight": 84218 / 478637,
        "n_jobs": -1,
        "verbosity": 0,
    },
    "LightGBM": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": -1,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "is_unbalance": True,
        "n_jobs": -1,
        "verbose": -1,
    },
    "SVM": {
        "loss": "hinge",
        "penalty": "l2",
        "alpha": 1e-4,
        "max_iter": 1000,
        "class_weight": "balanced",
        "n_jobs": -1,
        "verbose": 0,
    },
    "LogisticRegression": {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
        "class_weight": "balanced",
        "n_jobs": -1,
        "verbose": 0,
    },
    "NaiveBayes": {
        "alpha": 1.0,
    },
}

TRANSFORMER_PATHS = {
    "ELECTRA": "google/electra-small-discriminator",
    "RoBERTa": "roberta-base",
}

CLASSICAL_MODELS = list(CLASSICAL_PARAMS.keys())
TRANSFORMER_MODELS = list(TRANSFORMER_PATHS.keys())
