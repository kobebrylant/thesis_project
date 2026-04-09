import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from .config import PathConfig, CLASSICAL_MODELS, TRANSFORMER_MODELS


@dataclass
class TrainingResult:
    model: str
    model_type: str
    seed: int
    fold: int
    f1_score: float
    accuracy: float
    precision: float
    recall: float
    train_time: float
    inference_time_ms: float
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    tp: Optional[int] = None
    tn: Optional[int] = None
    fp: Optional[int] = None
    fn: Optional[int] = None

    @property
    def run_id(self) -> str:
        return f"{self.model}_seed{self.seed}_fold{self.fold}"

    @property
    def has_confusion_matrix(self) -> bool:
        return all(v is not None for v in [self.tp, self.tn, self.fp, self.fn])

    def to_dict(self) -> Dict:
        result = {
            "model": self.model,
            "type": self.model_type,
            "seed": self.seed,
            "fold": self.fold,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "train_time": self.train_time,
            "inference_time_ms": self.inference_time_ms,
            "timestamp": self.timestamp,
        }
        if self.has_confusion_matrix:
            result["tp"] = self.tp
            result["tn"] = self.tn
            result["fp"] = self.fp
            result["fn"] = self.fn
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> "TrainingResult":
        return cls(
            model=data["model"],
            model_type=data.get("type", "Classical"),
            seed=int(data["seed"]),
            fold=int(data["fold"]),
            f1_score=float(data["f1_score"]),
            accuracy=float(data["accuracy"]),
            precision=float(data["precision"]),
            recall=float(data["recall"]),
            train_time=float(data["train_time"]),
            inference_time_ms=float(data["inference_time_ms"]),
            timestamp=data.get("timestamp", ""),
            tp=int(data["tp"]) if "tp" in data and pd.notna(data.get("tp")) else None,
            tn=int(data["tn"]) if "tn" in data and pd.notna(data.get("tn")) else None,
            fp=int(data["fp"]) if "fp" in data and pd.notna(data.get("fp")) else None,
            fn=int(data["fn"]) if "fn" in data and pd.notna(data.get("fn")) else None,
        )


class CheckpointManager:
    RESULTS_FILE = "training_results.csv"
    PROGRESS_FILE = "training_progress.json"

    def __init__(self, path_config: PathConfig, n_seeds: int = 12, n_folds: int = 5):
        self.path_config = path_config
        self.n_seeds = n_seeds
        self.n_folds = n_folds

        self._results_path = path_config.checkpoints_dir / self.RESULTS_FILE
        self._progress_path = path_config.checkpoints_dir / self.PROGRESS_FILE

        self._results_df: Optional[pd.DataFrame] = None
        self._completed_runs: Set[str] = set()
        self._load()

    def _load(self) -> None:
        if self._results_path.exists():
            self._results_df = pd.read_csv(self._results_path)
            print(f"Loaded {len(self._results_df)} existing results from checkpoint")
        else:
            self._results_df = pd.DataFrame()

        if self._progress_path.exists():
            with open(self._progress_path) as f:
                progress = json.load(f)
                self._completed_runs = set(progress.get("completed_runs", []))
        else:
            if not self._results_df.empty:
                for _, row in self._results_df.iterrows():
                    run_id = (
                        f"{row['model']}_seed{int(row['seed'])}_fold{int(row['fold'])}"
                    )
                    self._completed_runs.add(run_id)

    def import_from_csv(self, csv_path: Path) -> int:
        if not csv_path.exists():
            print(f"File not found: {csv_path}")
            return 0

        imported_df = pd.read_csv(csv_path)
        new_count = 0

        for _, row in imported_df.iterrows():
            result = TrainingResult(
                model=row["model"],
                model_type=row.get("type", "Classical"),
                seed=int(row["seed"]),
                fold=int(row["fold"]),
                f1_score=float(row["f1_score"]),
                accuracy=float(row["accuracy"]),
                precision=float(row["precision"]),
                recall=float(row["recall"]),
                train_time=float(row["train_time"]),
                inference_time_ms=float(row["inference_time_ms"]),
                timestamp=row.get("timestamp", ""),
                tp=int(row["tp"]) if "tp" in row and pd.notna(row.get("tp")) else None,
                tn=int(row["tn"]) if "tn" in row and pd.notna(row.get("tn")) else None,
                fp=int(row["fp"]) if "fp" in row and pd.notna(row.get("fp")) else None,
                fn=int(row["fn"]) if "fn" in row and pd.notna(row.get("fn")) else None,
            )

            if not self.is_completed(result.model, result.seed, result.fold):
                self.save_result(result)
                new_count += 1

        print(f"Imported {new_count} new results")
        return new_count

    def is_completed(self, model: str, seed: int, fold: int) -> bool:
        run_id = f"{model}_seed{seed}_fold{fold}"
        return run_id in self._completed_runs

    def save_result(self, result: TrainingResult) -> None:
        new_row = pd.DataFrame([result.to_dict()])
        if self._results_df.empty:
            self._results_df = new_row
        else:
            self._results_df = pd.concat([self._results_df, new_row], ignore_index=True)

        self._results_df.to_csv(self._results_path, index=False)

        self._completed_runs.add(result.run_id)
        self._save_progress()

    def _save_progress(self) -> None:
        progress = {
            "completed_runs": list(self._completed_runs),
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_seeds": self.n_seeds,
            "n_folds": self.n_folds,
        }
        with open(self._progress_path, "w") as f:
            json.dump(progress, f, indent=2)

    def get_completed_count(self, model: Optional[str] = None) -> int:
        if self._results_df.empty:
            return 0
        if model:
            return len(self._results_df[self._results_df["model"] == model])
        return len(self._results_df)

    def get_results_df(self) -> pd.DataFrame:
        return self._results_df.copy()

    def get_model_summary(self) -> pd.DataFrame:
        if self._results_df.empty:
            return pd.DataFrame()

        summary = (
            self._results_df.groupby(["model", "type"])
            .agg(
                {
                    "f1_score": ["mean", "std", "min", "max"],
                    "accuracy": "mean",
                    "precision": "mean",
                    "recall": "mean",
                    "train_time": "mean",
                }
            )
            .round(4)
        )

        summary.columns = [
            "Avg F1",
            "Std Dev",
            "Min F1",
            "Max F1",
            "Avg Accuracy",
            "Avg Precision",
            "Avg Recall",
            "Avg Time (s)",
        ]
        return summary.sort_values("Avg F1", ascending=False)

    def print_progress(self, seeds: List[int]) -> None:
        print("\n" + "=" * 60)
        print("TRAINING PROGRESS")
        print("=" * 60)

        runs_per_model = len(seeds) * self.n_folds
        all_models = CLASSICAL_MODELS + TRANSFORMER_MODELS

        for model in all_models:
            completed = self.get_completed_count(model)
            pct = (
                min(completed / runs_per_model * 100, 100) if runs_per_model > 0 else 0
            )
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            status = "✓" if completed >= runs_per_model else " "
            print(
                f"  [{status}] {model:20s} [{bar}] {completed:3d}/{runs_per_model} ({pct:.0f}%)"
            )

        total_completed = self.get_completed_count()
        total_needed = len(all_models) * runs_per_model
        print(f"\n  Total: {total_completed}/{total_needed} runs completed")
