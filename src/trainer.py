from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from .config import (
    PhiloxSeedGenerator,
    PathConfig,
    TrainingConfig,
    CLASSICAL_MODELS,
    TRANSFORMER_MODELS,
)
from .checkpoint_manager import CheckpointManager, TrainingResult
from .models.classical import ClassicalModelFactory
from .models.transformers import TransformerModelFactory


@dataclass
class TrainingPlan:
    model_name: str
    model_type: str
    seed: int
    fold: int
    completed: bool = False

    @property
    def run_id(self) -> str:
        return f"{self.model_name}_seed{self.seed}_fold{self.fold}"


class Trainer:
    def __init__(
        self,
        path_config: PathConfig,
        training_config: TrainingConfig,
        checkpoint_manager: CheckpointManager,
    ):
        self.path_config = path_config
        self.training_config = training_config
        self.checkpoint = checkpoint_manager
        self.seed_generator = PhiloxSeedGenerator()

    def generate_seeds(self, n_seeds: int) -> List[int]:
        return self.seed_generator.generate_seeds(n_seeds)

    def get_training_plan(
        self,
        seeds: List[int],
        models: Optional[List[str]] = None,
        include_classical: bool = True,
        include_transformer: bool = True,
    ) -> List[TrainingPlan]:
        if models is None:
            models = []
            if include_classical:
                models.extend(CLASSICAL_MODELS)
            if include_transformer:
                models.extend(TRANSFORMER_MODELS)

        plan = []
        for model in models:
            model_type = "Classical" if model in CLASSICAL_MODELS else "Transformer"
            for seed in seeds:
                for fold in range(self.training_config.n_folds):
                    completed = self.checkpoint.is_completed(model, seed, fold)
                    plan.append(
                        TrainingPlan(
                            model_name=model,
                            model_type=model_type,
                            seed=seed,
                            fold=fold,
                            completed=completed,
                        )
                    )
        return plan

    def train_classical_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seeds: List[int],
        models: Optional[List[str]] = None,
        save_models: bool = True,
    ) -> None:
        if models is None:
            models = CLASSICAL_MODELS

        plan = self.get_training_plan(seeds, models, include_transformer=False)
        remaining = [p for p in plan if not p.completed]

        print("\n" + "=" * 70)
        print("CLASSICAL ML TRAINING")
        print("=" * 70)
        print(
            f"\nTotal runs: {len(plan)}, Completed: {
                len(plan) - len(remaining)
            }, Remaining: {len(remaining)}"
        )

        if not remaining:
            print("All classical models already trained!")
            return

        pbar = tqdm(total=len(remaining), desc="Classical ML Training")

        for model_name in models:
            model_plan = [p for p in remaining if p.model_name == model_name]
            if not model_plan:
                continue

            print(f"\n{'─' * 60}")
            print(f"Training {model_name}")
            print(f"{'─' * 60}")

            for run in model_plan:
                PhiloxSeedGenerator.set_all_seeds(run.seed)

                skf = StratifiedKFold(
                    n_splits=self.training_config.n_folds,
                    shuffle=True,
                    random_state=run.seed,
                )

                splits = list(skf.split(X, y))
                train_idx, val_idx = splits[run.fold]

                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = ClassicalModelFactory.create(
                    model_name,
                    run.seed,
                    self.training_config.tfidf_config,
                )

                print(
                    f"  Seed {run.seed}, Fold {run.fold + 1}/{
                        self.training_config.n_folds
                    } - ",
                    end="",
                )

                metrics = model.train(X_train, y_train, X_val, y_val)

                print(
                    f"F1={metrics.f1_score:.4f}, Acc={metrics.accuracy:.4f}, Time={
                        metrics.train_time:.2f}s"
                )

                if save_models:
                    model_path = (
                        self.path_config.models_dir
                        / f"{model_name}_seed{run.seed}_fold{run.fold}.joblib"
                    )
                    model.save(str(model_path))

                result = TrainingResult(
                    model=model_name,
                    model_type="Classical",
                    seed=run.seed,
                    fold=run.fold,
                    f1_score=metrics.f1_score,
                    accuracy=metrics.accuracy,
                    precision=metrics.precision,
                    recall=metrics.recall,
                    train_time=metrics.train_time,
                    inference_time_ms=metrics.inference_time_ms,
                    tp=metrics.tp,
                    tn=metrics.tn,
                    fp=metrics.fp,
                    fn=metrics.fn,
                )
                self.checkpoint.save_result(result)

                pbar.update(1)

        pbar.close()
        print("\nClassical ML training complete!")

    def train_transformer_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seeds: List[int],
        models: Optional[List[str]] = None,
        save_models: bool = True,
    ) -> None:
        if models is None:
            models = TRANSFORMER_MODELS

        plan = self.get_training_plan(seeds, models, include_classical=False)
        remaining = [p for p in plan if not p.completed]

        print("\n" + "=" * 70)
        print("TRANSFORMER TRAINING")
        print("=" * 70)
        print(
            f"\nTotal runs: {len(plan)}, Completed: {
                len(plan) - len(remaining)
            }, Remaining: {len(remaining)}"
        )

        if not remaining:
            print("All transformer models already trained!")
            return

        for model_name in models:
            model_plan = [p for p in remaining if p.model_name == model_name]
            if not model_plan:
                continue

            print(f"\n{'─' * 60}")
            print(f"Training {model_name}")
            print(f"{'─' * 60}")

            pbar = tqdm(total=len(model_plan), desc=f"Training {model_name}")

            for run in model_plan:
                PhiloxSeedGenerator.set_all_seeds(run.seed)

                skf = StratifiedKFold(
                    n_splits=self.training_config.n_folds,
                    shuffle=True,
                    random_state=run.seed,
                )

                splits = list(skf.split(X, y))
                train_idx, val_idx = splits[run.fold]

                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                print(
                    f"\n  Seed {run.seed}, Fold {run.fold + 1}/{
                        self.training_config.n_folds
                    }"
                )

                model = TransformerModelFactory.create(
                    model_name,
                    run.seed,
                    self.training_config.transformer_config,
                )

                try:
                    metrics = model.train(X_train, y_train, X_val, y_val)

                    print(
                        f"  F1={metrics.f1_score:.4f}, Acc={
                            metrics.accuracy:.4f}, Time={metrics.train_time:.1f}s"
                    )

                    if save_models:
                        model_path = (
                            self.path_config.models_dir
                            / f"{model_name}_seed{run.seed}_fold{run.fold}"
                        )
                        model.save(str(model_path))

                    result = TrainingResult(
                        model=model_name,
                        model_type="Transformer",
                        seed=run.seed,
                        fold=run.fold,
                        f1_score=metrics.f1_score,
                        accuracy=metrics.accuracy,
                        precision=metrics.precision,
                        recall=metrics.recall,
                        train_time=metrics.train_time,
                        inference_time_ms=metrics.inference_time_ms,
                        tp=metrics.tp,
                        tn=metrics.tn,
                        fp=metrics.fp,
                        fn=metrics.fn,
                    )
                    self.checkpoint.save_result(result)

                except Exception as e:
                    print(f"  Error: {e}")
                    continue

                pbar.update(1)

            pbar.close()

        print("\nTransformer training complete!")

    def train_all(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seeds: List[int],
        save_models: bool = True,
        train_classical: bool = True,
        train_transformers: bool = True,
    ) -> None:
        if train_classical:
            self.train_classical_models(X, y, seeds, save_models=save_models)

        if train_transformers:
            self.train_transformer_models(X, y, seeds, save_models=save_models)

        self.checkpoint.print_progress(seeds)
