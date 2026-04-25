import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .base import BaseModel, ModelMetrics
from ..config import CLASSICAL_PARAMS, TrainingConfig


class ClassicalModel(BaseModel):

    def __init__(
        self,
        name: str,
        seed: int,
        tfidf_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, seed)
        self.tfidf_config = tfidf_config or TrainingConfig().tfidf_config
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._model: Optional[Any] = None

    def _create_model(self) -> Any:
        params = CLASSICAL_PARAMS[self.name].copy()

        if self.name in ["XGBoost", "LightGBM", "SVM", "LogisticRegression"]:
            params["random_state"] = self.seed

        model_classes = {
            "XGBoost": XGBClassifier,
            "LightGBM": LGBMClassifier,
            "SVM": SGDClassifier,
            "LogisticRegression": LogisticRegression,
            "NaiveBayes": MultinomialNB,
        }

        return model_classes[self.name](**params)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> ModelMetrics:
        self._vectorizer = TfidfVectorizer(**self.tfidf_config)

        X_train_tfidf = self._vectorizer.fit_transform(X_train)
        X_val_tfidf = self._vectorizer.transform(X_val)

        self._model = self._create_model()

        if self.name == "XGBoost":
            n_pos = int((y_train == 1).sum())
            n_neg = int((y_train == 0).sum())
            if n_pos > 0:
                self._model.set_params(scale_pos_weight=n_neg / n_pos)

        train_start = time.time()
        self._model.fit(X_train_tfidf, y_train)
        train_time = time.time() - train_start

        pred_start = time.time()
        y_pred = self._model.predict(X_val_tfidf)
        inference_time = (time.time() - pred_start) / len(y_val) * 1000

        cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        return ModelMetrics(
            f1_score=f1_score(y_val, y_pred, average="macro"),
            accuracy=accuracy_score(y_val, y_pred),
            precision=precision_score(y_val, y_pred, average="macro"),
            recall=recall_score(y_val, y_pred, average="macro"),
            train_time=train_time,
            inference_time_ms=inference_time,
            tp=int(tp),
            tn=int(tn),
            fp=int(fp),
            fn=int(fn),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or self._vectorizer is None:
            raise RuntimeError("Model not trained or loaded")

        X_tfidf = self._vectorizer.transform(X)
        return self._model.predict(X_tfidf)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or self._vectorizer is None:
            raise RuntimeError("Model not trained or loaded")

        X_tfidf = self._vectorizer.transform(X)
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X_tfidf)
        else:
            preds = self._model.predict(X_tfidf)
            proba = np.zeros((len(preds), 2))
            proba[np.arange(len(preds)), preds] = 1.0
            return proba

    def save(self, path: str) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "model": self._model,
                "tfidf": self._vectorizer,
                "name": self.name,
                "seed": self.seed,
            },
            save_path,
        )

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self._model = data["model"]
        self._vectorizer = data["tfidf"]
        self.name = data.get("name", self.name)
        self.seed = data.get("seed", self.seed)


class ClassicalModelFactory:

    @staticmethod
    def create(
        name: str,
        seed: int,
        tfidf_config: Optional[Dict[str, Any]] = None,
    ) -> ClassicalModel:
        if name not in CLASSICAL_PARAMS:
            raise ValueError(f"Unknown model: {name}. Available: {list(CLASSICAL_PARAMS.keys())}")
        return ClassicalModel(name, seed, tfidf_config)

    @staticmethod
    def available_models() -> list:
        return list(CLASSICAL_PARAMS.keys())
