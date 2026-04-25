import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

from ..config import PathConfig
from .statistical_tests import CorrelationResult


CLASSICAL_MODEL_NAMES = [
    "LogisticRegression",
    "LightGBM",
    "SVM",
    "XGBoost",
    "NaiveBayes",
]
TRANSFORMER_MODEL_NAMES = ["ELECTRA", "RoBERTa"]
TRANSFORMER_INFERENCE_BATCH_SIZE = 64


class SentimentPredictor:
    """Unified prediction adapter for classical (sklearn + TF-IDF) and transformer models.

    Exposes `predict(texts)` and `predict_proba(texts)` so validation code can treat
    both families identically. Transformer inference is batched to avoid OOM on large
    validation sets.
    """

    def __init__(
        self,
        name: str,
        kind: str,
        model: Any,
        tfidf: Any = None,
        tokenizer: Any = None,
        device: Any = None,
        max_length: int = 128,
    ):
        self.name = name
        self.kind = kind
        self.model = model
        self.tfidf = tfidf
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def predict(self, texts: np.ndarray) -> np.ndarray:
        if self.kind == "classical":
            return self.model.predict(self.tfidf.transform(texts))
        return np.argmax(self._transformer_logits(texts), axis=-1)

    def predict_proba(self, texts: np.ndarray) -> np.ndarray:
        if self.kind == "classical":
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(self.tfidf.transform(texts))
            preds = self.model.predict(self.tfidf.transform(texts)).astype(float)
            proba = np.zeros((len(preds), 2), dtype=float)
            proba[:, 1] = preds
            proba[:, 0] = 1.0 - preds
            return proba
        import torch

        logits = self._transformer_logits(texts)
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)

    def _transformer_logits(self, texts: np.ndarray) -> np.ndarray:
        import torch

        self.model.eval()
        all_logits: List[np.ndarray] = []
        batch_size = TRANSFORMER_INFERENCE_BATCH_SIZE
        text_list = [str(t) for t in texts]
        with torch.no_grad():
            for i in range(0, len(text_list), batch_size):
                batch = text_list[i : i + batch_size]
                encodings = self.tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**encodings)
                all_logits.append(outputs.logits.cpu().numpy())
        return np.concatenate(all_logits, axis=0)


def _normalize_game_name(name: str) -> str:
    """Normalize game name for matching: lowercase, strip punctuation, collapse whitespace."""
    name = str(name).lower().strip()
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _format_feature_name(col: str) -> str:
    """Human-readable label for a metadata feature column."""
    if col.startswith("genre_"):
        return "Genre: " + col[len("genre_"):].replace("_", " ").title()
    if col.startswith("tag_"):
        return "Tag: " + col[len("tag_"):].replace("_", " ").title()
    return col.replace("_", " ").title()


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float, int]:
    """Spearman ρ(x, y | z) via rank residualization.

    Addresses the studio-capability / engagement-volume confound: if a feature's
    raw correlation with revenue is driven by review volume, partialling out
    log_ea_review_count will attenuate it toward zero.
    """
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    n = int(mask.sum())
    if n < 4:
        return float("nan"), float("nan"), n
    rx = pd.Series(x[mask]).rank().to_numpy()
    ry = pd.Series(y[mask]).rank().to_numpy()
    rz = pd.Series(z[mask]).rank().to_numpy()
    if np.std(rz) == 0:
        r, p = pearsonr(rx, ry)
        return float(r), float(p), n
    Z = np.column_stack([np.ones_like(rz), rz])
    beta_x, *_ = np.linalg.lstsq(Z, rx, rcond=None)
    beta_y, *_ = np.linalg.lstsq(Z, ry, rcond=None)
    resid_x = rx - Z @ beta_x
    resid_y = ry - Z @ beta_y
    if np.std(resid_x) == 0 or np.std(resid_y) == 0:
        return float("nan"), float("nan"), n
    r, p = pearsonr(resid_x, resid_y)
    return float(r), float(p), n


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

    def load_predictor(
        self,
        model_name: Optional[str] = None,
        seed: Optional[int] = None,
        fold: int = 0,
    ) -> SentimentPredictor:
        models_dir = self.path_config.models_dir

        if model_name:
            candidates = [model_name]
        else:
            candidates = TRANSFORMER_MODEL_NAMES + CLASSICAL_MODEL_NAMES

        for name in candidates:
            if name in TRANSFORMER_MODEL_NAMES:
                predictor = self._load_transformer(models_dir, name, seed, fold)
                if predictor is not None:
                    return predictor
            elif name in CLASSICAL_MODEL_NAMES:
                predictor = self._load_classical(models_dir, name, seed, fold)
                if predictor is not None:
                    return predictor
            else:
                raise ValueError(
                    f"Unknown model name '{name}'. "
                    f"Available: {TRANSFORMER_MODEL_NAMES + CLASSICAL_MODEL_NAMES}"
                )

        raise FileNotFoundError(
            f"No trained model found in {models_dir} matching {candidates}. "
            "Train models first (python main.py)."
        )

    def _load_classical(
        self,
        models_dir: Path,
        name: str,
        seed: Optional[int],
        fold: int,
    ) -> Optional[SentimentPredictor]:
        if seed is not None:
            path = models_dir / f"{name}_seed{seed}_fold{fold}.joblib"
            if not path.exists():
                return None
            matched_seed = seed
        else:
            matches = sorted(models_dir.glob(f"{name}_seed*_fold{fold}.joblib"))
            if not matches:
                return None
            path = matches[0]
            matched_seed = path.stem.split("_seed")[1].split("_fold")[0]

        data = joblib.load(path)
        print(f"Loaded model: {name} (seed={matched_seed}, fold={fold})")
        return SentimentPredictor(
            name=name,
            kind="classical",
            model=data["model"],
            tfidf=data["tfidf"],
        )

    def _load_transformer(
        self,
        models_dir: Path,
        name: str,
        seed: Optional[int],
        fold: int,
    ) -> Optional[SentimentPredictor]:
        if seed is not None:
            path = models_dir / f"{name}_seed{seed}_fold{fold}"
            if not (path / "config.json").exists():
                return None
            matched_seed = seed
        else:
            matches = sorted(
                p for p in models_dir.glob(f"{name}_seed*_fold{fold}")
                if (p / "config.json").exists()
            )
            if not matches:
                return None
            path = matches[0]
            matched_seed = path.name.split("_seed")[1].split("_fold")[0]

        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        from ..config import DeviceConfig, TrainingConfig

        device = DeviceConfig.get_device()
        tokenizer = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForSequenceClassification.from_pretrained(
            str(path), use_safetensors=True
        ).to(device)

        print(f"Loaded model: {name} (seed={matched_seed}, fold={fold}) on {device}")
        return SentimentPredictor(
            name=name,
            kind="transformer",
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=TrainingConfig().transformer_max_length,
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
            f"\nLoading {len(csv_files)} validation CSV files from {VALIDATION_DATA_DIR}"
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
            f"EA reviews: {len(combined_df[combined_df['review_type'] == 'early_access']):,}"
        )
        print(
            f"Post-release reviews: {len(combined_df[combined_df['review_type'] == 'post_release']):,}"
        )

        return combined_df

    def calculate_game_sentiment(
        self,
        df: pd.DataFrame,
        predictor: SentimentPredictor,
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

            proba = predictor.predict_proba(texts)
            positive_probs = proba[:, 1]
            predictions = np.argmax(proba, axis=-1)
            avg_positive_prob = positive_probs.mean()

            positive_ratio = predictions.mean()
            total_reviews = len(game_reviews)

            # --- Text-based features ---
            raw_texts = game_reviews["review_text"].values
            word_counts = np.array([len(str(t).split()) for t in raw_texts])
            mean_review_length = word_counts.mean()
            std_review_length = word_counts.std()
            long_review_ratio = (word_counts > 50).mean()

            # Sentiment variance (controversy signal)
            sentiment_std = positive_probs.std()

            # Negative vs positive review length
            pos_mask = predictions == 1
            neg_mask = predictions == 0
            mean_pos_length = word_counts[pos_mask].mean() if pos_mask.any() else 0.0
            mean_neg_length = word_counts[neg_mask].mean() if neg_mask.any() else 0.0
            neg_pos_length_ratio = (
                mean_neg_length / mean_pos_length
                if mean_pos_length > 0
                else 0.0
            )

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
                    "app_id": game_reviews["app_id"].iloc[0],
                    "ea_review_count": total_reviews,
                    "ea_predicted_positive_ratio": positive_ratio,
                    "ea_avg_positive_probability": avg_positive_prob,
                    "ea_actual_positive_ratio": game_reviews["positive"].mean(),
                    "ea_sentiment_std": sentiment_std,
                    "ea_mean_review_length": mean_review_length,
                    "ea_std_review_length": std_review_length,
                    "ea_long_review_ratio": long_review_ratio,
                    "ea_mean_pos_review_length": mean_pos_length,
                    "ea_mean_neg_review_length": mean_neg_length,
                    "ea_neg_pos_length_ratio": neg_pos_length_ratio,
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

        if "steamspy_owners_min" in merged.columns and "steamspy_owners_max" in merged.columns:
            merged["owners_midpoint"] = (
                merged["steamspy_owners_min"] + merged["steamspy_owners_max"]
            ) / 2

        if len(merged) < 5:
            print("WARNING: Not enough matched games for reliable analysis!")
            return {}

        results = {}

        success_metrics_config = [
            ("owners_midpoint", "Estimated Sales (Owners)"),
            ("steamspy_owners_min", "Owner Estimate (Min)"),
            ("steam_metacritic_score", "Metacritic Score"),
            ("steamspy_avg_playtime", "Average Playtime"),
            ("review_score", "Review Score (Positive Ratio)"),
            ("estimated_revenue_usd", "Estimated Revenue"),
        ]

        sentiment_features = [
            ("ea_predicted_positive_ratio", "Predicted Sentiment"),
            ("ea_sentiment_std", "Sentiment Variance"),
            ("ea_mean_review_length", "Mean Review Length"),
            ("ea_long_review_ratio", "Long Review Ratio"),
            ("ea_neg_pos_length_ratio", "Neg/Pos Length Ratio"),
            ("log_ea_review_count", "Log Review Count"),
        ]

        # Compute derived columns for correlation
        if "ea_review_count" in merged.columns:
            merged["log_ea_review_count"] = np.log1p(merged["ea_review_count"])

        for metric_col, metric_name in success_metrics_config:
            if metric_col not in merged.columns:
                print(f"  Skipping {metric_name}: column not found")
                continue

            for feat_col, feat_name in sentiment_features:
                if feat_col not in merged.columns:
                    continue

                valid_mask = pd.notna(merged[metric_col]) & pd.notna(merged[feat_col])
                n_valid = valid_mask.sum()

                if n_valid < 5:
                    continue

                x = merged[feat_col][valid_mask].values
                y = merged[metric_col][valid_mask].values

                spearman_r, spearman_p = spearmanr(x, y)
                pearson_r, pearson_p = pearsonr(x, y)

                partial_r = partial_p = None
                control_var = None
                if metric_col in ("estimated_revenue_usd", "owners_midpoint",
                                  "steamspy_owners_min") \
                        and feat_col != "log_ea_review_count" \
                        and "log_ea_review_count" in merged.columns:
                    z = merged["log_ea_review_count"][valid_mask].values
                    p_r, p_p, _ = _partial_spearman(x, y, z)
                    if not np.isnan(p_r):
                        partial_r, partial_p = p_r, p_p
                        control_var = "log_ea_review_count"

                result = CorrelationResult(
                    spearman_r=spearman_r,
                    spearman_p=spearman_p,
                    pearson_r=pearson_r,
                    pearson_p=pearson_p,
                    n_samples=n_valid,
                    partial_r=partial_r,
                    partial_p=partial_p,
                    control_var=control_var,
                )
                key = f"{feat_name} → {metric_name}"
                results[key] = result

                sig = "***" if spearman_p < 0.05 else ""
                print(
                    f"  {key}: r={spearman_r:.3f}, p={spearman_p:.4f}, n={n_valid} {sig}"
                )

        return results

    def correlate_metadata_with_success(
        self,
        game_sentiment: pd.DataFrame,
        success_metrics: pd.DataFrame,
    ) -> Dict[str, CorrelationResult]:
        """Spearman correlations between game metadata features and market outcomes.

        Parses genre flags, tag flags, price tier, release year, platform support, etc.
        from `game_metadata.csv` and `game_success_metrics.csv`, then correlates each
        metadata feature against each market-success target. Only features with at least
        MIN_VARIANCE non-constant values are included (binary flags with ≤1 positive
        case are skipped).
        """
        from .success_predictor import (
            GENRE_FLAGS,
            TAG_FLAGS,
            METADATA_FEATURES,
            _enrich_metadata_features,
            _load_game_metadata_for,
        )
        from scipy.stats import spearmanr, pearsonr
        import numpy as np
        import pandas as pd
        from .statistical_tests import CorrelationResult
        from ..config import PathConfig

        if "app_id" in game_sentiment.columns and "app_id" in success_metrics.columns:
            merged = game_sentiment.merge(
                success_metrics, on="app_id", how="inner", suffixes=("", "_metrics")
            )
        else:
            gs = game_sentiment.copy()
            sm = success_metrics.copy()
            gs["name_normalized"] = gs["game_name"].apply(_normalize_game_name)
            sm["name_normalized"] = sm["app_name"].apply(_normalize_game_name)
            merged = gs.merge(sm, on="name_normalized", how="inner", suffixes=("", "_metrics"))

        meta_df = _load_game_metadata_for(merged)
        if meta_df is not None:
            if "app_id" in merged.columns and "app_id" in meta_df.columns:
                merged = merged.merge(meta_df, on="app_id", how="left", suffixes=("", "_meta"))
            else:
                merged = merged.copy()
                if "name_normalized" not in merged.columns:
                    merged["name_normalized"] = merged.get(
                        "game_name", merged.get("app_name")
                    ).apply(_normalize_game_name)
                meta_df = meta_df.copy()
                meta_df["name_normalized"] = meta_df["app_name"].apply(_normalize_game_name)
                merged = merged.merge(
                    meta_df.drop(columns=["app_id"], errors="ignore"),
                    on="name_normalized",
                    how="left",
                    suffixes=("", "_meta"),
                )

        merged = _enrich_metadata_features(merged)

        if "steamspy_owners_min" in merged.columns and "steamspy_owners_max" in merged.columns:
            merged["owners_midpoint"] = (
                merged["steamspy_owners_min"] + merged["steamspy_owners_max"]
            ) / 2

        if "ea_review_count" in merged.columns and "log_ea_review_count" not in merged.columns:
            merged["log_ea_review_count"] = np.log1p(
                pd.to_numeric(merged["ea_review_count"], errors="coerce")
            )

        if len(merged) < 5:
            print("WARNING: Not enough matched games for metadata correlation analysis!")
            return {}

        print(f"\nRunning metadata correlations on {len(merged)} games")

        success_metrics_config = [
            ("owners_midpoint", "Estimated Sales (Owners)"),
            ("steamspy_owners_min", "Owner Estimate (Min)"),
            ("steam_metacritic_score", "Metacritic Score"),
            ("steamspy_avg_playtime", "Average Playtime"),
            ("review_score", "Review Score (Positive Ratio)"),
            ("estimated_revenue_usd", "Estimated Revenue"),
        ]

        # Only keep metadata features actually present in the merge AND with enough
        # variation to produce a meaningful correlation.
        metadata_feats = []
        for col in METADATA_FEATURES:
            if col not in merged.columns:
                continue
            series = pd.to_numeric(merged[col], errors="coerce")
            nonnull = series.dropna()
            if len(nonnull) < 5:
                continue
            # Binary flags: require ≥2 positives AND ≥2 negatives
            unique_vals = nonnull.unique()
            if set(unique_vals).issubset({0, 1}):
                if nonnull.sum() < 2 or (len(nonnull) - nonnull.sum()) < 2:
                    continue
            else:
                if nonnull.nunique() < 2:
                    continue
            metadata_feats.append((col, _format_feature_name(col)))

        results: Dict[str, CorrelationResult] = {}

        for metric_col, metric_name in success_metrics_config:
            if metric_col not in merged.columns:
                continue

            for feat_col, feat_name in metadata_feats:
                x_raw = pd.to_numeric(merged[feat_col], errors="coerce")
                y_raw = pd.to_numeric(merged[metric_col], errors="coerce")
                valid_mask = x_raw.notna() & y_raw.notna()
                n_valid = int(valid_mask.sum())
                if n_valid < 5:
                    continue

                x = x_raw[valid_mask].values
                y = y_raw[valid_mask].values
                if np.std(x) == 0 or np.std(y) == 0:
                    continue

                spearman_r, spearman_p = spearmanr(x, y)
                pearson_r, pearson_p = pearsonr(x, y)
                if not np.isfinite(spearman_r) or not np.isfinite(pearson_r):
                    continue

                partial_r = partial_p = None
                control_var = None
                if metric_col in ("estimated_revenue_usd", "owners_midpoint",
                                  "steamspy_owners_min") \
                        and "log_ea_review_count" in merged.columns:
                    z_raw = pd.to_numeric(merged["log_ea_review_count"], errors="coerce")
                    z = z_raw[valid_mask].values
                    p_r, p_p, _ = _partial_spearman(x, y, z)
                    if not np.isnan(p_r):
                        partial_r, partial_p = p_r, p_p
                        control_var = "log_ea_review_count"

                key = f"{feat_name} → {metric_name}"
                results[key] = CorrelationResult(
                    spearman_r=float(spearman_r),
                    spearman_p=float(spearman_p),
                    pearson_r=float(pearson_r),
                    pearson_p=float(pearson_p),
                    n_samples=n_valid,
                    partial_r=partial_r,
                    partial_p=partial_p,
                    control_var=control_var,
                )

                if spearman_p < 0.05:
                    print(
                        f"  {key}: ρ={spearman_r:+.3f}, p={spearman_p:.4f}, n={n_valid}  ***"
                    )

        print(f"\nMetadata correlations: {len(results)} feature×target pairs evaluated")
        sig_count = sum(1 for r in results.values() if r.is_significant)
        print(f"  Significant (p<0.05): {sig_count}")

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
            predictor = self.load_predictor(model_name)
            results["model_name"] = predictor.name
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return results

        game_sentiment = self.calculate_game_sentiment(df, predictor)
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

            print("\n" + "-" * 40)
            print("3. CORRELATION WITH GAME METADATA FEATURES")
            print("-" * 40)
            metadata_correlations = self.correlate_metadata_with_success(
                game_sentiment, success_metrics
            )
            results["metadata_correlations"] = metadata_correlations

        self._save_extended_results(results)

        self._print_extended_summary(results)

        return results

    def _save_metadata_correlations(
        self,
        metadata_correlations: Dict[str, "CorrelationResult"],
        output_path: Path,
    ):
        """Write metadata correlation results to CSV, sorted by |spearman_r|."""
        rows = []
        for name, corr in metadata_correlations.items():
            feat, target = name.split(" → ", 1) if " → " in name else (name, "")
            rows.append(
                {
                    "feature": feat,
                    "target": target,
                    "spearman_r": corr.spearman_r,
                    "spearman_p": corr.spearman_p,
                    "pearson_r": corr.pearson_r,
                    "pearson_p": corr.pearson_p,
                    "partial_r": corr.partial_r,
                    "partial_p": corr.partial_p,
                    "control_var": corr.control_var,
                    "n_samples": corr.n_samples,
                    "effect_size": corr.effect_size,
                    "significant": corr.is_significant,
                    "abs_spearman_r": abs(corr.spearman_r),
                }
            )
        if not rows:
            return
        df = pd.DataFrame(rows).sort_values(
            ["significant", "abs_spearman_r"], ascending=[False, False]
        )
        df = df.drop(columns=["abs_spearman_r"])
        df.to_csv(output_path, index=False)
        print(f"\nMetadata correlations saved to {output_path}")
        sig = df[df["significant"]]
        if len(sig) > 0:
            print(f"\nTop significant metadata correlations ({len(sig)} total):")
            for _, r in sig.head(15).iterrows():
                print(
                    f"  {r['feature']} → {r['target']}: "
                    f"ρ={r['spearman_r']:+.3f}, p={r['spearman_p']:.4f}, n={r['n_samples']}"
                )

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

        if results.get("metadata_correlations"):
            self._save_metadata_correlations(
                results["metadata_correlations"],
                metrics_dir / "metadata_correlations.csv",
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
                f"\nPost-Release Sentiment: r={corr.spearman_r:.3f} (p={corr.spearman_p:.4f}) {sig}"
            )

        if "success_correlations" in results and results["success_correlations"]:
            print("\nMarket Success Metrics:")
            for metric_name, corr in results["success_correlations"].items():
                sig = "***" if corr.is_significant else ""
                print(
                    f"  {metric_name}: r={corr.spearman_r:.3f} (p={corr.spearman_p:.4f}) {sig}"
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
        predictor: SentimentPredictor,
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
        y_true_all = ea_reviews["positive"].values
        y_pred_all = predictor.predict(ea_reviews[text_col].values)

        overall_accuracy = accuracy_score(y_true_all, y_pred_all)
        overall_f1 = f1_score(y_true_all, y_pred_all, average="weighted")

        # Per-game metrics
        game_rows = []
        for _, row in game_sentiment.iterrows():
            game = row["game_name"]
            game_reviews = ea_reviews[ea_reviews[game_col] == game]
            if len(game_reviews) == 0:
                continue

            y_true = game_reviews["positive"].values
            y_pred = predictor.predict(game_reviews[text_col].values)

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
            predictor = self.load_predictor(model_name)
            results["model_name"] = predictor.name
            print(f"Loaded model: {predictor.name}")
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
        game_sentiment = self.calculate_game_sentiment(validation_df, predictor)
        results["game_sentiment"] = game_sentiment

        print("\n" + "-" * 40)
        print("1.5. PREDICTION ACCURACY EVALUATION")
        print("-" * 40)
        accuracy_results = self.evaluate_prediction_accuracy(
            validation_df, game_sentiment, predictor
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
                f"Validation success metrics not found at {VALIDATION_SUCCESS_METRICS_FILE}"
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

            print("\n" + "-" * 40)
            print("4. CORRELATION WITH GAME METADATA FEATURES")
            print("-" * 40)
            metadata_correlations = self.correlate_metadata_with_success(
                game_sentiment, success_metrics
            )
            results["metadata_correlations"] = metadata_correlations
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
                        "partial_r": corr.partial_r,
                        "partial_p": corr.partial_p,
                        "control_var": corr.control_var,
                        "n_samples": corr.n_samples,
                        "effect_size": corr.effect_size,
                        "significant": corr.is_significant,
                    }
                )

        if rows:
            pd.DataFrame(rows).to_csv(
                metrics_dir / "validation_correlations.csv", index=False
            )

        if results.get("metadata_correlations"):
            self._save_metadata_correlations(
                results["metadata_correlations"],
                metrics_dir / "validation_metadata_correlations.csv",
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
                f"\nPost-Release Sentiment: r={corr.spearman_r:.3f} (p={corr.spearman_p:.4f}) {sig}"
            )

        if "success_correlations" in results and results["success_correlations"]:
            print("\nMarket Success Metrics:")
            for metric_name, corr in results["success_correlations"].items():
                sig = "***" if corr.is_significant else ""
                print(
                    f"  {metric_name}: r={corr.spearman_r:.3f} (p={corr.spearman_p:.4f}) {sig}"
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
