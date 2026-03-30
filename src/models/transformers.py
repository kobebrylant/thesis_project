import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import EarlyStoppingCallback

from .base import BaseModel, ModelMetrics
from ..config import TRANSFORMER_PATHS, TrainingConfig, DeviceConfig


class ReviewDataset(Dataset):

    def __init__(
        self,
        texts: np.ndarray,
        labels: np.ndarray,
        tokenizer: Any,
        max_length: int = 128,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class TransformerModel(BaseModel):

    def __init__(
        self,
        name: str,
        seed: int,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, seed)
        self.config = config or TrainingConfig().transformer_config
        self.device = DeviceConfig.get_device()
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None
        self._model_path = TRANSFORMER_PATHS.get(name, name)

    def _load_model_and_tokenizer(self) -> None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_path, num_labels=2
        ).to(self.device)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> ModelMetrics:
        from transformers import TrainingArguments, Trainer
        from datasets import Dataset as HFDataset

        self._load_model_and_tokenizer()

        train_enc = self._tokenizer(
            list(X_train),
            truncation=True,
            padding=True,
            max_length=self.config["max_length"],
        )
        val_enc = self._tokenizer(
            list(X_val),
            truncation=True,
            padding=True,
            max_length=self.config["max_length"],
        )

        train_dataset = HFDataset.from_dict({
            "input_ids": train_enc["input_ids"],
            "attention_mask": train_enc["attention_mask"],
            "labels": list(y_train),
        })
        val_dataset = HFDataset.from_dict({
            "input_ids": val_enc["input_ids"],
            "attention_mask": val_enc["attention_mask"],
            "labels": list(y_val),
        })

        training_args = TrainingArguments(
            output_dir=f"./tmp_trainer_{self.name}_{self.seed}",
            num_train_epochs=self.config["epochs"],
            per_device_train_batch_size=self.config["batch_size"],
            per_device_eval_batch_size=self.config["batch_size"],
            learning_rate=self.config["learning_rate"],
            warmup_ratio=self.config["warmup_ratio"],
            weight_decay=self.config["weight_decay"],
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            seed=self.seed,
            report_to="none",
            disable_tqdm=True,
            use_mps_device=(self.device.type == "mps"),
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        train_start = time.time()
        trainer.train()
        train_time = time.time() - train_start

        pred_start = time.time()
        predictions = trainer.predict(val_dataset)
        inference_time = (time.time() - pred_start) / len(y_val) * 1000

        y_pred = np.argmax(predictions.predictions, axis=1)

        cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

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
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not trained or loaded")

        self._model.eval()
        encodings = self._tokenizer(
            list(X),
            truncation=True,
            padding=True,
            max_length=self.config["max_length"],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**encodings)
            predictions = torch.argmax(outputs.logits, dim=-1)

        return predictions.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not trained or loaded")

        self._model.eval()
        encodings = self._tokenizer(
            list(X),
            truncation=True,
            padding=True,
            max_length=self.config["max_length"],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**encodings)
            proba = torch.softmax(outputs.logits, dim=-1)

        return proba.cpu().numpy()

    def save(self, path: str) -> None:
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        if self._model is not None:
            self._model.save_pretrained(save_path)
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(save_path)

    def load(self, path: str) -> None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)


class TransformerModelFactory:

    @staticmethod
    def create(
        name: str,
        seed: int,
        config: Optional[Dict[str, Any]] = None,
    ) -> TransformerModel:
        if name not in TRANSFORMER_PATHS:
            raise ValueError(f"Unknown model: {name}. Available: {list(TRANSFORMER_PATHS.keys())}")
        return TransformerModel(name, seed, config)

    @staticmethod
    def available_models() -> list:
        return list(TRANSFORMER_PATHS.keys())
