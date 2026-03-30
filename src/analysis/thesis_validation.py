import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

from ..config import PathConfig
from .statistical_tests import CorrelationResult


def _normalize_game_name(name: str) -> str:
    """Normalize game name for matching: lowercase, strip punctuation, collapse whitespace."""
    name = str(name).lower().strip()
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


SUCCESS_METRICS_FILE = Path("data/game_success_metrics.csv")
VALIDATION_DATA_DIR = Path("data/validation")
VALIDATION_SUCCESS_METRICS_FILE = Path("data/validation/game_success_metrics.csv")


@dataclass
class GameSentimentResult:
    game_name: str
    ea_review_count: int
    ea_predicted_positive_ratio: float
    ea_avg_positive_probability: float
    ea_actual_positive_ratio: float
    pr_review_count: int
    pr_actual_positive_ratio: Optional[float]


class ThesisValidator:
    def __init__(self, path_config: PathConfig):
        self.path_config = path_config

    def load_best_model(
        self,
        model_name: Optional[str] = None,
        seed: Optional[int] = None,
        fold: int = 0,
    ) -> Tuple[Any, Any, str]:
        models_dir = self.path_config.models_dir

        candidates = (
            [model_name]
            if model_name
            else ["LogisticRegression", "LightGBM", "SVM", "XGBoost", "NaiveBayes"]
        )

        for name in candidates:
            if name is None:
                continue

            if seed is not None:
                model_path = models_dir / f"{name}_seed{seed}_fold{fold}.joblib"
                if model_path.exists():
                    data = joblib.load(model_path)
                    print(f"Loaded model: {name} (seed={seed}, fold={fold})")
                    return data["model"], data["tfidf"], name
            else:
                import glob

                pattern = str(models_dir / f"{name}_seed*_fold{fold}.joblib")
                matching_files = glob.glob(pattern)
                if matching_files:
                    model_path = Path(matching_files[0])
                    data = joblib.load(model_path)
                    filename = model_path.stem
                    detected_seed = filename.split("_seed")[1].split("_fold")[0]
                    print(f"Loaded model: {name} (seed={detected_seed}, fold={fold})")
                    return data["model"], data["tfidf"], name

        raise FileNotFoundError(
            f"No trained model found in {models_dir}. "
            "Please train classical models first."
        )

    def load_validation_data(self, min_text_length: int = 10) -> pd.DataFrame:
        if not VALIDATION_DATA_DIR.exists():
            raise FileNotFoundError(
                f"Validation data directory not found: {VALIDATION_DATA_DIR}\n"
                "Run 'python data_gatherer.py --validation' first to collect data."
            )

        csv_files = sorted(VALIDATION_DATA_DIR.glob("*_reviews.csv"))

        if not csv_files:
            raise FileNotFoundError(
                f"No review CSV files found in {VALIDATION_DATA_DIR}\n"
                "Run 'python data_gatherer.py --validation' first."
            )

        print(
            f"\nLoading {len(csv_files)} validation CSV files from {
                VALIDATION_DATA_DIR
            }"
        )
        print("(These games were NEVER seen during model training)\n")

        dfs = []
        for file in tqdm(csv_files, desc="Loading validation files"):
            try:
                df = pd.read_csv(file)
                filename = file.name

                if "_early_access_reviews.csv" in filename:
                    game_name = filename.replace("_early_access_reviews.csv", "")
                    review_type = "early_access"
                elif "_post_release_reviews.csv" in filename:
                    game_name = filename.replace("_post_release_reviews.csv", "")
                    review_type = "post_release"
                else:
                    continue

                df["game_name"] = game_name
                df["review_type"] = review_type
                df["source_file"] = filename
                dfs.append(df)

            except Exception as e:
                print(f"Error loading {file}: {e}")

        if not dfs:
            raise FileNotFoundError(
                "No valid review files found in validation directory"
            )

        combined_df = pd.concat(dfs, ignore_index=True)

        combined_df = combined_df.dropna(subset=["review_text"])
        combined_df["review_text"] = combined_df["review_text"].astype(str)
        combined_df["positive"] = combined_df["positive"].astype(int)

        print("Cleaning text...")

        def clean_text(text: str) -> str:
            if pd.isna(text):
                return ""
            text = str(text).lower()
            text = re.sub(r"http\S+|www\S+|https\S+", "", text)
            text = re.sub(r"<.*?>", "", text)
            text = re.sub(r"[^a-zA-Z0-9\s.,!?\'\-]", "", text)
            text = " ".join(text.split())
            return text.strip()

        combined_df["cleaned_text"] = combined_df["review_text"].apply(clean_text)

        original_len = len(combined_df)
        combined_df = combined_df[
            combined_df["cleaned_text"].str.len() > min_text_length
        ]
        removed = original_len - len(combined_df)
        if removed > 0:
            print(f"Removed {removed:,} short reviews")

        print(f"Loaded {len(combined_df):,} validation reviews")
        print(f"Games: {combined_df['game_name'].nunique()}")
        print(
            f"EA reviews: {
                len(combined_df[combined_df['review_type'] == 'early_access']):,}"
        )
        print(
            f"Post-release reviews: {
                len(combined_df[combined_df['review_type'] == 'post_release']):,}"
        )

        return combined_df

    def calculate_game_sentiment(
        self,
        df: pd.DataFrame,
        model: Any,
        tfidf: Any,
    ) -> pd.DataFrame:
        print("\nCalculating game-level sentiment scores...")

        game_col = "game_name" if "game_name" in df.columns else "app_name"

        ea_reviews = df[df["review_type"] == "early_access"].copy()

        if len(ea_reviews) == 0:
            raise ValueError("No early access reviews found in data!")

        game_scores = []

        for game in tqdm(ea_reviews[game_col].unique(), desc="Processing games"):
            game_reviews = ea_reviews[ea_reviews[game_col] == game]

            text_col = (
                "cleaned_text"
                if "cleaned_text" in game_reviews.columns
                else "review_text"
            )
            texts = game_reviews[text_col].values

            X_tfidf = tfidf.transform(texts)
            predictions = model.predict(X_tfidf)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_tfidf)
                avg_positive_prob = proba[:, 1].mean()
            else:
                avg_positive_prob = predictions.mean()

            positive_ratio = predictions.mean()
            total_reviews = len(game_reviews)

            pr_reviews = df[
                (df[game_col] == game) & (df["review_type"] == "post_release")
            ]
            pr_positive_ratio = (
                pr_reviews["positive"].mean() if len(pr_reviews) > 0 else None
            )
            pr_review_count = len(pr_reviews)

            game_scores.append(
                {
                    "game_name": game,
                    "ea_review_count": total_reviews,
                    "ea_predicted_positive_ratio": positive_ratio,
                    "ea_avg_positive_probability": avg_positive_prob,
                    "ea_actual_positive_ratio": game_reviews["positive"].mean(),
                    "pr_review_count": pr_review_count,
                    "pr_actual_positive_ratio": pr_positive_ratio,
                }
            )

        result_df = pd.DataFrame(game_scores)
        result_df = result_df.sort_values(
            "ea_predicted_positive_ratio", ascending=False
        )

        print(f"Calculated sentiment for {len(result_df)} games")
        return result_df

    def validate_thesis(
        self,
        game_sentiment: pd.DataFrame,
        save_results: bool = True,
    ) -> Tuple[CorrelationResult, pd.DataFrame]:
        print("\n" + "=" * 60)
        print("THESIS VALIDATION")
        print("=" * 60)

        valid_games = game_sentiment.dropna(subset=["pr_actual_positive_ratio"])

        print(f"\nGames with both EA and post-release data: {len(valid_games)}")

        if len(valid_games) < 5:
            print("WARNING: Not enough games for reliable correlation analysis!")
            print("Need at least 5 games with both EA and post-release reviews.")
            return None, valid_games

        ea_pred = valid_games["ea_predicted_positive_ratio"].values
        pr_actual = valid_games["pr_actual_positive_ratio"].values

        spearman_r, spearman_p = spearmanr(ea_pred, pr_actual)
        pearson_r, pearson_p = pearsonr(ea_pred, pr_actual)

        result = CorrelationResult(
            spearman_r=spearman_r,
            spearman_p=spearman_p,
            pearson_r=pearson_r,
            pearson_p=pearson_p,
            n_samples=len(valid_games),
        )

        print("\nCan Early Access sentiment predict Post-Release success?")
        print("-" * 40)
        print(result.summary())

        print("\nTHESIS VERDICT:")
        if spearman_p < 0.05 and abs(spearman_r) > 0.3:
            print("  SUPPORTED: Significant positive correlation found!")
            print("  Early access sentiment CAN predict post-release success.")
        elif spearman_p < 0.05:
            print("  PARTIALLY SUPPORTED: Significant but weak correlation.")
        else:
            print("  NOT SUPPORTED: No significant correlation found (p > 0.05).")

        if save_results:
            metrics_dir = self.path_config.metrics_dir

            game_sentiment.to_csv(
                metrics_dir / "game_sentiment_predictions.csv", index=False
            )

            corr_dict = {
                "spearman_r": result.spearman_r,
                "spearman_p": result.spearman_p,
                "pearson_r": result.pearson_r,
                "pearson_p": result.pearson_p,
                "n_games": result.n_samples,
                "effect_size": result.effect_size,
                "significant": result.is_significant,
            }
            pd.DataFrame([corr_dict]).to_csv(
                metrics_dir / "thesis_correlation.csv", index=False
            )

            print(f"\nResults saved to {metrics_dir}")

        return result, valid_games

    def load_success_metrics(self) -> Optional[pd.DataFrame]:
        if not SUCCESS_METRICS_FILE.exists():
            print(f"Success metrics file not found: {SUCCESS_METRICS_FILE}")
            print("Run 'python main.py --gather-success-metrics' first.")
            return None

        df = pd.read_csv(SUCCESS_METRICS_FILE)
        print(f"Loaded success metrics for {len(df)} games")

        if "steamspy_owners_min" in df.columns and "steamspy_owners_max" in df.columns:
            df["owners_midpoint"] = df.apply(
                lambda r: (r["steamspy_owners_min"] + r["steamspy_owners_max"]) / 2
                if pd.notna(r["steamspy_owners_min"])
                and pd.notna(r["steamspy_owners_max"])
                else None,
                axis=1,
            )

        if "steamspy_positive" in df.columns and "steamspy_negative" in df.columns:
            df["review_score"] = df.apply(
                lambda r: r["steamspy_positive"]
                / (r["steamspy_positive"] + r["steamspy_negative"])
                if pd.notna(r["steamspy_positive"])
                and pd.notna(r["steamspy_negative"])
                and (r["steamspy_positive"] + r["steamspy_negative"]) > 0
                else None,
                axis=1,
            )

        return df

    def correlate_with_success_metrics(
        self,
        game_sentiment: pd.DataFrame,
        success_metrics: pd.DataFrame,
    ) -> Dict[str, CorrelationResult]:
        if "app_id" in game_sentiment.columns:
            merged = game_sentiment.merge(
                success_metrics, on="app_id", how="inner", suffixes=("", "_metrics")
            )
        else:
            game_sentiment = game_sentiment.copy()
            success_metrics = success_metrics.copy()

            game_sentiment["name_normalized"] = (
                game_sentiment["game_name"].apply(_normalize_game_name)
            )
            success_metrics["name_normalized"] = (
                success_metrics["app_name"].apply(_normalize_game_name)
            )
            print("  (Using normalized name-based merge — verify matches manually)")

            merged = game_sentiment.merge(
                success_metrics,
                on="name_normalized",
                how="inner",
                suffixes=("", "_metrics"),
            )

        print(f"\nMatched {len(merged)} games between sentiment and success metrics")

        if len(merged) < 5:
            print("WARNING: Not enough matched games for reliable analysis!")
            return {}

        results = {}

        metrics_config = [
            ("owners_midpoint", "Estimated Sales"),
            ("steam_metacritic_score", "Metacritic Score"),
            ("steamspy_avg_playtime", "Average Playtime"),
            ("review_score", "Review Score (Positive Ratio)"),
        ]

        ea_sentiment = merged["ea_predicted_positive_ratio"]

        for metric_col, metric_name in metrics_config:
            if metric_col not in merged.columns:
                print(f"  Skipping {metric_name}: column not found")
                continue

            valid_mask = pd.notna(merged[metric_col]) & pd.notna(ea_sentiment)
            n_valid = valid_mask.sum()

            if n_valid < 5:
                print(f"  Skipping {metric_name}: only {n_valid} valid samples")
                continue

            x = ea_sentiment[valid_mask].values
            y = merged[metric_col][valid_mask].values

            spearman_r, spearman_p = spearmanr(x, y)
            pearson_r, pearson_p = pearsonr(x, y)

            result = CorrelationResult(
                spearman_r=spearman_r,
                spearman_p=spearman_p,
                pearson_r=pearson_r,
                pearson_p=pearson_p,
                n_samples=n_valid,
            )
            results[metric_name] = result

            print(
                f"  {metric_name}: r={spearman_r:.3f}, p={spearman_p:.4f}, n={n_valid}"
            )

        return results

    def run_validation(
        self,
        df: pd.DataFrame,
        model_name: Optional[str] = None,
    ) -> Dict:
        """Full validation pipeline with market success metrics.

        1. Load trained model
        2. Calculate game-level EA sentiment
        3. Correlate with post-release sentiment (existing)
        4. Correlate with market success metrics (new)
        5. Save all results

        Args:
            df: DataFrame with all review data
            model_name: Optional specific model to use

        Returns:
            Dictionary with all validation results
        """
        print("\n" + "=" * 60)
        print("EXTENDED THESIS VALIDATION")
        print("=" * 60)

        results = {}

        try:
            model, tfidf, loaded_model_name = self.load_best_model(model_name)
            results["model_name"] = loaded_model_name
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return results

        game_sentiment = self.calculate_game_sentiment(df, model, tfidf)
        results["game_sentiment"] = game_sentiment

        print("\n" + "-" * 40)
        print("1. CORRELATION WITH POST-RELEASE SENTIMENT")
        print("-" * 40)
        correlation_result, valid_games = self.validate_thesis(
            game_sentiment, save_results=False
        )
        results["post_release_correlation"] = correlation_result
        results["valid_games"] = valid_games

        print("\n" + "-" * 40)
        print("2. CORRELATION WITH MARKET SUCCESS METRICS")
        print("-" * 40)
        success_metrics = self.load_success_metrics()

        if success_metrics is not None:
            success_correlations = self.correlate_with_success_metrics(
                game_sentiment, success_metrics
            )
            results["success_correlations"] = success_correlations
            results["success_metrics"] = success_metrics

        self._save_extended_results(results)

        self._print_extended_summary(results)

        return results

    def _save_extended_results(self, results: Dict):
        metrics_dir = self.path_config.metrics_dir

        if "game_sentiment" in results:
            results["game_sentiment"].to_csv(
                metrics_dir / "game_sentiment_predictions.csv", index=False
            )

        if results.get("post_release_correlation"):
            corr = results["post_release_correlation"]
            corr_dict = {
                "metric": "Post-Release Sentiment",
                "spearman_r": corr.spearman_r,
                "spearman_p": corr.spearman_p,
                "pearson_r": corr.pearson_r,
                "pearson_p": corr.pearson_p,
                "n_samples": corr.n_samples,
                "effect_size": corr.effect_size,
                "significant": corr.is_significant,
            }
            pr_df = pd.DataFrame([corr_dict])
        else:
            pr_df = pd.DataFrame()

        if "success_correlations" in results and results["success_correlations"]:
            rows = []
            for metric_name, corr in results["success_correlations"].items():
                rows.append(
                    {
                        "metric": metric_name,
                        "spearman_r": corr.spearman_r,
                        "spearman_p": corr.spearman_p,
                        "pearson_r": corr.pearson_r,
                        "pearson_p": corr.pearson_p,
                        "n_samples": corr.n_samples,
                        "effect_size": corr.effect_size,
                        "significant": corr.is_significant,
                    }
                )
            success_df = pd.DataFrame(rows)

            if not pr_df.empty:
                all_correlations = pd.concat([pr_df, success_df], ignore_index=True)
            else:
                all_correlations = success_df

            all_correlations.to_csv(
                metrics_dir / "success_correlations.csv", index=False
            )

        summary = {
            "model_used": results.get("model_name", "Unknown"),
            "n_games_sentiment": len(results.get("game_sentiment", [])),
            "n_games_post_release": len(results.get("valid_games", [])),
        }

        if results.get("post_release_correlation"):
            summary["post_release_spearman_r"] = results[
                "post_release_correlation"
            ].spearman_r
            summary["post_release_significant"] = results[
                "post_release_correlation"
            ].is_significant

        if "success_correlations" in results:
            for metric_name, corr in results["success_correlations"].items():
                key = (
                    metric_name.lower()
                    .replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                )
                summary[f"{key}_spearman_r"] = corr.spearman_r
                summary[f"{key}_significant"] = corr.is_significant

        pd.DataFrame([summary]).to_csv(
            metrics_dir / "extended_thesis_summary.csv", index=False
        )

        print(f"\nResults saved to {metrics_dir}")

    def _print_extended_summary(self, results: Dict):
        print("\n" + "=" * 60)
        print("EXTENDED VALIDATION SUMMARY")
        print("=" * 60)

        print(f"\nModel used: {results.get('model_name', 'Unknown')}")
        print(f"Games analyzed: {len(results.get('game_sentiment', []))}")

        if results.get("post_release_correlation"):
            corr = results["post_release_correlation"]
            sig = "***" if corr.is_significant else ""
            print(
                f"\nPost-Release Sentiment: r={corr.spearman_r:.3f} (p={
                    corr.spearman_p:.4f}) {sig}"
            )

        if "success_correlations" in results and results["success_correlations"]:
            print("\nMarket Success Metrics:")
            for metric_name, corr in results["success_correlations"].items():
                sig = "***" if corr.is_significant else ""
                print(
                    f"  {metric_name}: r={corr.spearman_r:.3f} (p={
                        corr.spearman_p:.4f}) {sig}"
                )

        print("\n" + "-" * 40)
        print("THESIS VERDICT")
        print("-" * 40)

        significant_correlations = []
        if (
            results.get("post_release_correlation")
            and results["post_release_correlation"].is_significant
        ):
            significant_correlations.append(
                ("Post-Release Sentiment", results["post_release_correlation"])
            )

        if "success_correlations" in results:
            for name, corr in results["success_correlations"].items():
                if corr.is_significant:
                    significant_correlations.append((name, corr))

        if significant_correlations:
            print(
                f"\nFound {len(significant_correlations)} significant correlation(s):"
            )
            for name, corr in significant_correlations:
                effect = corr.effect_size
                print(f"  - {name}: r={corr.spearman_r:.3f} ({effect} effect)")

            print("\nCONCLUSION: Thesis SUPPORTED")
            print(
                "Early Access sentiment shows significant correlation with market success."
            )
        else:
            print("\nNo significant correlations found (p > 0.05).")
            print("\nCONCLUSION: Thesis NOT SUPPORTED")
            print(
                "Early Access sentiment does not significantly predict market success."
            )

    def evaluate_prediction_accuracy(
        self,
        df: pd.DataFrame,
        game_sentiment: pd.DataFrame,
        model: Any,
        tfidf: Any,
    ) -> Dict:
        """Evaluate how accurately the model predicts sentiment on validation games.

        Returns dict with:
            - per_game_df: per-game accuracy metrics
            - overall_accuracy: per-review accuracy across all validation reviews
            - overall_f1: per-review F1 across all validation reviews
            - mae: mean absolute error of game-level sentiment ratios
            - game_count: number of games evaluated
        """
        from sklearn.metrics import accuracy_score, f1_score

        print("\nEvaluating prediction accuracy on validation games...")

        game_col = "game_name" if "game_name" in df.columns else "app_name"
        ea_reviews = df[df["review_type"] == "early_access"].copy()

        if len(ea_reviews) == 0:
            return {"error": "No early access reviews found"}

        text_col = "cleaned_text" if "cleaned_text" in ea_reviews.columns else "review_text"

        # Per-review predictions across all validation games
        X_all = tfidf.transform(ea_reviews[text_col].values)
        y_true_all = ea_reviews["positive"].values
        y_pred_all = model.predict(X_all)

        overall_accuracy = accuracy_score(y_true_all, y_pred_all)
        overall_f1 = f1_score(y_true_all, y_pred_all, average="weighted")

        # Per-game metrics
        game_rows = []
        for _, row in game_sentiment.iterrows():
            game = row["game_name"]
            game_reviews = ea_reviews[ea_reviews[game_col] == game]
            if len(game_reviews) == 0:
                continue

            X_game = tfidf.transform(game_reviews[text_col].values)
            y_true = game_reviews["positive"].values
            y_pred = model.predict(X_game)

            game_accuracy = accuracy_score(y_true, y_pred)
            game_f1 = f1_score(y_true, y_pred, average="weighted")
            abs_error = abs(row["ea_predicted_positive_ratio"] - row["ea_actual_positive_ratio"])

            game_rows.append({
                "game_name": game,
                "ea_review_count": len(game_reviews),
                "predicted_positive_ratio": row["ea_predicted_positive_ratio"],
                "actual_positive_ratio": row["ea_actual_positive_ratio"],
                "absolute_error": abs_error,
                "per_review_accuracy": game_accuracy,
                "per_review_f1": game_f1,
            })

        per_game_df = pd.DataFrame(game_rows)
        mae = per_game_df["absolute_error"].mean() if len(per_game_df) > 0 else None

        print(f"\n  Per-review accuracy (all games pooled): {overall_accuracy:.2%}")
        print(f"  Per-review F1 (weighted): {overall_f1:.2%}")
        print(f"  Game-level sentiment MAE: {mae:.4f}" if mae else "")
        print(f"  Games evaluated: {len(per_game_df)}")

        if len(per_game_df) > 0:
            print("\n  Per-game breakdown:")
            for _, row in per_game_df.iterrows():
                print(
                    f"    {row['game_name']}: "
                    f"pred={row['predicted_positive_ratio']:.2%}, "
                    f"actual={row['actual_positive_ratio']:.2%}, "
                    f"err={row['absolute_error']:.4f}, "
                    f"acc={row['per_review_accuracy']:.2%}"
                )

        return {
            "per_game_df": per_game_df,
            "overall_accuracy": overall_accuracy,
            "overall_f1": overall_f1,
            "mae": mae,
            "game_count": len(per_game_df),
        }

    def run_validation_on_new_games(
        self,
        model_name: Optional[str] = None,
    ) -> Dict:
        """Run thesis validation on NEW games (never seen during training).

        This is the CORRECT way to validate the thesis:
        1. Load model trained on original games
        2. Load validation data (completely new games)
        3. Predict EA sentiment on new games
        3.5. Evaluate prediction accuracy
        4. Correlate with success metrics

        Args:
            model_name: Optional specific model to use

        Returns:
            Dictionary with all validation results
        """
        print("\n" + "=" * 70)
        print("THESIS VALIDATION ON UNSEEN GAMES")
        print("=" * 70)
        print("\nThis validation uses games the model has NEVER seen during training.")
        print("This is the scientifically valid way to test the thesis hypothesis.\n")

        results = {}
        results["validation_type"] = "unseen_games"

        try:
            model, tfidf, loaded_model_name = self.load_best_model(model_name)
            results["model_name"] = loaded_model_name
            print(f"Loaded model: {loaded_model_name}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return results

        try:
            validation_df = self.load_validation_data()
            results["n_validation_reviews"] = len(validation_df)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return results

        print("\n" + "-" * 40)
        print("1. PREDICTING EA SENTIMENT ON NEW GAMES")
        print("-" * 40)
        game_sentiment = self.calculate_game_sentiment(validation_df, model, tfidf)
        results["game_sentiment"] = game_sentiment

        print("\n" + "-" * 40)
        print("1.5. PREDICTION ACCURACY EVALUATION")
        print("-" * 40)
        accuracy_results = self.evaluate_prediction_accuracy(
            validation_df, game_sentiment, model, tfidf
        )
        results["prediction_accuracy"] = accuracy_results

        # Save accuracy results
        if accuracy_results.get("per_game_df") is not None:
            metrics_dir = self.path_config.metrics_dir
            accuracy_results["per_game_df"].to_csv(
                metrics_dir / "validation_prediction_accuracy.csv", index=False
            )

        print("\n" + "-" * 40)
        print("2. CORRELATION WITH POST-RELEASE SENTIMENT")
        print("-" * 40)
        correlation_result, valid_games = self.validate_thesis(
            game_sentiment, save_results=False
        )
        results["post_release_correlation"] = correlation_result
        results["valid_games"] = valid_games

        print("\n" + "-" * 40)
        print("3. CORRELATION WITH MARKET SUCCESS METRICS")
        print("-" * 40)

        success_metrics = None
        if VALIDATION_SUCCESS_METRICS_FILE.exists():
            success_metrics = pd.read_csv(VALIDATION_SUCCESS_METRICS_FILE)
            print(f"Loaded validation success metrics: {len(success_metrics)} games")
        else:
            print(
                f"Validation success metrics not found at {
                    VALIDATION_SUCCESS_METRICS_FILE
                }"
            )
            print("Trying main success metrics file...")
            success_metrics = self.load_success_metrics()

        if success_metrics is not None:
            if "steamspy_owners_min" in success_metrics.columns:
                success_metrics["owners_midpoint"] = success_metrics.apply(
                    lambda r: (r["steamspy_owners_min"] + r["steamspy_owners_max"]) / 2
                    if pd.notna(r.get("steamspy_owners_min"))
                    and pd.notna(r.get("steamspy_owners_max"))
                    else None,
                    axis=1,
                )
            if "steamspy_positive" in success_metrics.columns:
                success_metrics["review_score"] = success_metrics.apply(
                    lambda r: r["steamspy_positive"]
                    / (r["steamspy_positive"] + r["steamspy_negative"])
                    if pd.notna(r.get("steamspy_positive"))
                    and pd.notna(r.get("steamspy_negative"))
                    and (r.get("steamspy_positive", 0) + r.get("steamspy_negative", 0))
                    > 0
                    else None,
                    axis=1,
                )

            success_correlations = self.correlate_with_success_metrics(
                game_sentiment, success_metrics
            )
            results["success_correlations"] = success_correlations
            results["success_metrics"] = success_metrics
        else:
            print("\nNo success metrics available.")
            print("Run 'python main.py --gather-success-metrics' for validation games.")

        self._save_validation_results(results)
        self._print_validation_summary(results)

        return results

    def _save_validation_results(self, results: Dict):
        metrics_dir = self.path_config.metrics_dir

        if "game_sentiment" in results:
            results["game_sentiment"].to_csv(
                metrics_dir / "validation_game_sentiment.csv", index=False
            )

        rows = []
        if results.get("post_release_correlation"):
            corr = results["post_release_correlation"]
            rows.append(
                {
                    "metric": "Post-Release Sentiment",
                    "spearman_r": corr.spearman_r,
                    "spearman_p": corr.spearman_p,
                    "pearson_r": corr.pearson_r,
                    "pearson_p": corr.pearson_p,
                    "n_samples": corr.n_samples,
                    "effect_size": corr.effect_size,
                    "significant": corr.is_significant,
                }
            )

        if "success_correlations" in results:
            for metric_name, corr in results["success_correlations"].items():
                rows.append(
                    {
                        "metric": metric_name,
                        "spearman_r": corr.spearman_r,
                        "spearman_p": corr.spearman_p,
                        "pearson_r": corr.pearson_r,
                        "pearson_p": corr.pearson_p,
                        "n_samples": corr.n_samples,
                        "effect_size": corr.effect_size,
                        "significant": corr.is_significant,
                    }
                )

        if rows:
            pd.DataFrame(rows).to_csv(
                metrics_dir / "validation_correlations.csv", index=False
            )

        summary = {
            "validation_type": "unseen_games",
            "model_used": results.get("model_name", "Unknown"),
            "n_validation_games": len(results.get("game_sentiment", [])),
            "n_validation_reviews": results.get("n_validation_reviews", 0),
        }

        if results.get("post_release_correlation"):
            summary["post_release_r"] = results["post_release_correlation"].spearman_r
            summary["post_release_p"] = results["post_release_correlation"].spearman_p
            summary["post_release_significant"] = results[
                "post_release_correlation"
            ].is_significant

        if "success_correlations" in results:
            for metric_name, corr in results["success_correlations"].items():
                key = (
                    metric_name.lower()
                    .replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                )
                summary[f"{key}_r"] = corr.spearman_r
                summary[f"{key}_p"] = corr.spearman_p
                summary[f"{key}_significant"] = corr.is_significant

        pd.DataFrame([summary]).to_csv(
            metrics_dir / "validation_summary.csv", index=False
        )

        print(f"\nResults saved to {metrics_dir}")

    def _print_validation_summary(self, results: Dict):
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)

        print(f"\nModel used: {results.get('model_name', 'Unknown')}")
        print(f"Validation games: {len(results.get('game_sentiment', []))}")
        print(f"Total reviews analyzed: {results.get('n_validation_reviews', 0):,}")

        if (
            results.get("game_sentiment") is not None
            and len(results["game_sentiment"]) > 0
        ):
            gs = results["game_sentiment"]
            print(f"\nGames analyzed:")
            for _, row in gs.iterrows():
                print(
                    f"  - {row['game_name']}: {row['ea_review_count']} EA reviews, "
                    f"predicted {row['ea_predicted_positive_ratio']:.1%} positive"
                )

        if results.get("post_release_correlation"):
            corr = results["post_release_correlation"]
            sig = "***" if corr.is_significant else ""
            print(
                f"\nPost-Release Sentiment: r={corr.spearman_r:.3f} (p={
                    corr.spearman_p:.4f}) {sig}"
            )

        if "success_correlations" in results and results["success_correlations"]:
            print("\nMarket Success Metrics:")
            for metric_name, corr in results["success_correlations"].items():
                sig = "***" if corr.is_significant else ""
                print(
                    f"  {metric_name}: r={corr.spearman_r:.3f} (p={
                        corr.spearman_p:.4f}) {sig}"
                )

        print("\n" + "-" * 40)
        print("THESIS VERDICT ")
        print("-" * 40)

        significant_correlations = []
        if (
            results.get("post_release_correlation")
            and results["post_release_correlation"].is_significant
        ):
            significant_correlations.append(
                ("Post-Release Sentiment", results["post_release_correlation"])
            )

        if "success_correlations" in results:
            for name, corr in results["success_correlations"].items():
                if corr.is_significant:
                    significant_correlations.append((name, corr))

        if significant_correlations:
            print(
                f"\nFound {len(significant_correlations)} significant correlation(s):"
            )
            for name, corr in significant_correlations:
                effect = corr.effect_size
                print(f"  - {name}: r={corr.spearman_r:.3f} ({effect} effect)")

            print("correlation with market success. The model generalizes!")
        else:
            print("\nNo significant correlations found (p > 0.05).")
