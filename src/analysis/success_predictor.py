"""Predict commercial success from pre-release sentiment features.

Provides both:
- Classification: predict success tier (Flop/Moderate/Hit/Blockbuster)
- Regression: predict estimated revenue

Uses leave-one-out cross-validation on training games due to small dataset size.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder, StandardScaler


SUCCESS_TIERS = ["Flop", "Moderate", "Hit", "Blockbuster"]
TIER_ORDER = {tier: i for i, tier in enumerate(SUCCESS_TIERS)}


@dataclass
class PredictionResult:
    """Results from success prediction evaluation."""
    task: str  # "classification" or "regression"
    model_name: str
    n_samples: int
    # Classification metrics
    accuracy: Optional[float] = None
    per_class_report: Optional[str] = None
    confusion_mat: Optional[np.ndarray] = None
    tier_labels: Optional[List[str]] = None
    # Regression metrics
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    # Per-game predictions
    predictions_df: Optional[pd.DataFrame] = None


def build_sentiment_features(
    game_sentiment: pd.DataFrame,
    success_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Join sentiment scores with success metrics to build feature matrix.

    Features extracted per game:
        - ea_predicted_positive_ratio: model's predicted positive %
        - ea_avg_positive_probability: mean predict_proba score
        - ea_actual_positive_ratio: ground-truth EA positive %
        - ea_review_count: number of EA reviews
        - sentiment_confidence: |predicted - 0.5| (how decisive the model is)

    Target:
        - estimated_revenue_usd (regression, Boxleiter: total_reviews x 30 x price)
        - success_tier (classification)

    NOTE: current_price_usd values in game_metadata.csv and game_success_metrics.csv
    were audited and corrected (2026-03-29) to ensure USD pricing. The Steam Store API
    must be called with cc=us to get correct USD prices. See data_gatherer.py and
    success_data_gatherer.py for the fix.
    """
    # Normalize names for join
    sentiment = game_sentiment.copy()
    metrics = success_metrics.copy()

    if "app_id" in sentiment.columns:
        merged = sentiment.merge(metrics, on="app_id", how="inner", suffixes=("", "_m"))
    else:
        def _norm(name):
            name = str(name).lower().strip()
            name = re.sub(r"[^a-z0-9\s]", "", name)
            return re.sub(r"\s+", " ", name).strip()
        sentiment["name_key"] = sentiment["game_name"].apply(_norm)
        metrics["name_key"] = metrics["app_name"].apply(_norm)
        merged = sentiment.merge(metrics, on="name_key", how="inner", suffixes=("", "_m"))

    # Build features
    feature_cols = []

    if "ea_predicted_positive_ratio" in merged.columns:
        feature_cols.append("ea_predicted_positive_ratio")

    if "ea_avg_positive_probability" in merged.columns:
        feature_cols.append("ea_avg_positive_probability")

    if "ea_actual_positive_ratio" in merged.columns:
        feature_cols.append("ea_actual_positive_ratio")

    if "ea_review_count" in merged.columns:
        merged["log_ea_review_count"] = np.log1p(merged["ea_review_count"])
        feature_cols.append("log_ea_review_count")

    if "ea_predicted_positive_ratio" in merged.columns:
        merged["sentiment_confidence"] = (
            merged["ea_predicted_positive_ratio"] - 0.5
        ).abs()
        feature_cols.append("sentiment_confidence")

    merged["feature_cols"] = [feature_cols] * len(merged)

    return merged


class SuccessPredictor:
    """Predict game commercial success from sentiment features."""

    def __init__(self, path_config=None):
        self.path_config = path_config

    def evaluate_classification(
        self,
        game_sentiment: pd.DataFrame,
        success_metrics: pd.DataFrame,
    ) -> PredictionResult:
        """Evaluate success tier prediction using leave-one-out CV."""
        merged = build_sentiment_features(game_sentiment, success_metrics)

        # Filter to games with success tier
        merged = merged.dropna(subset=["success_tier"])
        if len(merged) < 5:
            print(f"Not enough games with success tier data ({len(merged)})")
            return PredictionResult(
                task="classification", model_name="N/A", n_samples=len(merged)
            )

        feature_cols = merged["feature_cols"].iloc[0]
        X = merged[feature_cols].values
        y_labels = merged["success_tier"].values

        le = LabelEncoder()
        le.classes_ = np.array(SUCCESS_TIERS)
        y = le.transform(y_labels)

        # LOO-CV
        loo = LeaveOneOut()
        y_pred = np.zeros_like(y)
        game_names = merged["game_name"].values if "game_name" in merged.columns else merged.index.values

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            clf = LogisticRegression(
                max_iter=1000, multi_class="multinomial", C=1.0
            )
            clf.fit(X_train, y_train)
            y_pred[test_idx] = clf.predict(X_test)

        y_pred_labels = le.inverse_transform(y_pred)

        acc = accuracy_score(y, y_pred)
        report = classification_report(
            y_labels, y_pred_labels, labels=SUCCESS_TIERS, zero_division=0
        )
        cm = confusion_matrix(y_labels, y_pred_labels, labels=SUCCESS_TIERS)

        # Build per-game predictions table
        pred_df = pd.DataFrame({
            "game_name": game_names,
            "actual_tier": y_labels,
            "predicted_tier": y_pred_labels,
            "correct": y_labels == y_pred_labels,
        })

        print(f"\nSuccess Tier Classification (LOO-CV, n={len(merged)})")
        print(f"Accuracy: {acc:.2%}")
        print(f"\n{report}")

        return PredictionResult(
            task="classification",
            model_name="LogisticRegression (LOO-CV)",
            n_samples=len(merged),
            accuracy=acc,
            per_class_report=report,
            confusion_mat=cm,
            tier_labels=SUCCESS_TIERS,
            predictions_df=pred_df,
        )

    def evaluate_regression(
        self,
        game_sentiment: pd.DataFrame,
        success_metrics: pd.DataFrame,
    ) -> PredictionResult:
        """Evaluate revenue prediction using leave-one-out CV."""
        merged = build_sentiment_features(game_sentiment, success_metrics)

        merged = merged.dropna(subset=["estimated_revenue_usd"])
        merged = merged[merged["estimated_revenue_usd"] > 0]

        if len(merged) < 5:
            print(f"Not enough games with revenue data ({len(merged)})")
            return PredictionResult(
                task="regression", model_name="N/A", n_samples=len(merged)
            )

        feature_cols = merged["feature_cols"].iloc[0]
        X = merged[feature_cols].values
        # Use log revenue for better regression behavior
        y = np.log1p(merged["estimated_revenue_usd"].values)

        loo = LeaveOneOut()
        y_pred = np.zeros_like(y)
        game_names = merged["game_name"].values if "game_name" in merged.columns else merged.index.values

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            reg = Ridge(alpha=1.0)
            reg.fit(X_train, y_train)
            y_pred[test_idx] = reg.predict(X_test)

        # Convert back from log space for interpretable metrics
        y_actual = np.expm1(y)
        y_predicted = np.expm1(y_pred)
        y_predicted = np.maximum(y_predicted, 0)  # Clamp negatives

        mae = mean_absolute_error(y_actual, y_predicted)
        rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
        r2 = r2_score(y, y_pred)  # R2 in log space is more meaningful

        pred_df = pd.DataFrame({
            "game_name": game_names,
            "actual_revenue": y_actual,
            "predicted_revenue": y_predicted,
            "absolute_error": np.abs(y_actual - y_predicted),
        })

        print(f"\nRevenue Prediction (LOO-CV, n={len(merged)})")
        print(f"MAE: ${mae:,.0f}")
        print(f"RMSE: ${rmse:,.0f}")
        print(f"R² (log space): {r2:.3f}")

        return PredictionResult(
            task="regression",
            model_name="Ridge (LOO-CV)",
            n_samples=len(merged),
            mae=mae,
            rmse=rmse,
            r2=r2,
            predictions_df=pred_df,
        )

    def predict_for_new_games(
        self,
        training_sentiment: pd.DataFrame,
        training_metrics: pd.DataFrame,
        validation_sentiment: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train on all training games, predict for validation games.

        Returns (tier_predictions_df, revenue_predictions_df).
        """
        # Build training features
        train_merged = build_sentiment_features(training_sentiment, training_metrics)
        train_merged = train_merged.dropna(subset=["success_tier", "estimated_revenue_usd"])

        if len(train_merged) < 5:
            print("Not enough training data for prediction")
            return pd.DataFrame(), pd.DataFrame()

        feature_cols = train_merged["feature_cols"].iloc[0]
        X_train = train_merged[feature_cols].values

        # Build validation features (no targets needed)
        val_sentiment = validation_sentiment.copy()
        if "ea_predicted_positive_ratio" in val_sentiment.columns:
            val_sentiment["sentiment_confidence"] = (
                val_sentiment["ea_predicted_positive_ratio"] - 0.5
            ).abs()
        if "ea_review_count" in val_sentiment.columns:
            val_sentiment["log_ea_review_count"] = np.log1p(val_sentiment["ea_review_count"])

        available_features = [c for c in feature_cols if c in val_sentiment.columns]
        if not available_features:
            print("No matching features between training and validation")
            return pd.DataFrame(), pd.DataFrame()

        X_val = val_sentiment[available_features].values
        X_train_subset = train_merged[available_features].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_subset)
        X_val_scaled = scaler.transform(X_val)

        game_names = val_sentiment["game_name"].values if "game_name" in val_sentiment.columns else val_sentiment.index.values

        # --- Classification ---
        le = LabelEncoder()
        le.classes_ = np.array(SUCCESS_TIERS)
        y_tier = le.transform(train_merged["success_tier"].values)

        clf = LogisticRegression(max_iter=1000, multi_class="multinomial", C=1.0)
        clf.fit(X_train_scaled, y_tier)

        tier_pred = le.inverse_transform(clf.predict(X_val_scaled))
        tier_proba = clf.predict_proba(X_val_scaled)

        tier_df = pd.DataFrame({
            "game_name": game_names,
            "predicted_tier": tier_pred,
        })
        for i, tier in enumerate(SUCCESS_TIERS):
            if i < tier_proba.shape[1]:
                tier_df[f"prob_{tier.lower()}"] = tier_proba[:, i]

        # --- Regression ---
        y_rev = np.log1p(train_merged["estimated_revenue_usd"].values)

        reg = Ridge(alpha=1.0)
        reg.fit(X_train_scaled, y_rev)

        rev_pred = np.expm1(reg.predict(X_val_scaled))
        rev_pred = np.maximum(rev_pred, 0)

        rev_df = pd.DataFrame({
            "game_name": game_names,
            "predicted_revenue_usd": rev_pred,
        })

        print(f"\nPredictions for {len(game_names)} validation games:")
        for i, name in enumerate(game_names):
            print(f"  {name}: {tier_pred[i]} (${rev_pred[i]:,.0f})")

        return tier_df, rev_df

    def save_results(
        self,
        classification_result: Optional[PredictionResult],
        regression_result: Optional[PredictionResult],
        output_dir: Path,
    ):
        """Save prediction results to CSV files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        if classification_result and classification_result.predictions_df is not None:
            classification_result.predictions_df.to_csv(
                output_dir / "success_tier_predictions.csv", index=False
            )
            if classification_result.confusion_mat is not None:
                cm_df = pd.DataFrame(
                    classification_result.confusion_mat,
                    index=SUCCESS_TIERS,
                    columns=SUCCESS_TIERS,
                )
                cm_df.to_csv(output_dir / "success_tier_confusion_matrix.csv")

        if regression_result and regression_result.predictions_df is not None:
            regression_result.predictions_df.to_csv(
                output_dir / "revenue_predictions.csv", index=False
            )

        # Summary
        summary = {}
        if classification_result:
            summary["classification_accuracy"] = classification_result.accuracy
            summary["classification_n_samples"] = classification_result.n_samples
        if regression_result:
            summary["regression_mae"] = regression_result.mae
            summary["regression_rmse"] = regression_result.rmse
            summary["regression_r2"] = regression_result.r2
            summary["regression_n_samples"] = regression_result.n_samples

        if summary:
            pd.DataFrame([summary]).to_csv(
                output_dir / "success_prediction_summary.csv", index=False
            )
            print(f"Results saved to {output_dir}")
