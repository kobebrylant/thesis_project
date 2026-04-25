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


STEAM_FEATURES = [
    "ea_predicted_positive_ratio",
    "ea_avg_positive_probability",
    "ea_actual_positive_ratio",
    "log_ea_review_count",
    "sentiment_confidence",
    "ea_sentiment_std",
    "ea_mean_review_length",
    "ea_long_review_ratio",
    "ea_neg_pos_length_ratio",
]
YOUTUBE_FEATURES = [
    "yt_predicted_positive_ratio",
    "yt_avg_positive_probability",
    "yt_sentiment_std",
    "yt_mean_comment_length",
    "yt_long_comment_ratio",
    "log_yt_comment_count",
    "yt_sentiment_confidence",
]

# Binary genre flags derived from `steamspy_genre` (comma-separated SteamSpy field).
GENRE_FLAGS = [
    "Action",
    "Adventure",
    "Indie",
    "RPG",
    "Simulation",
    "Strategy",
    "Casual",
    "Early Access",
    "Massively Multiplayer",
    "Sports",
    "Racing",
]

# Binary flags derived from top-10 SteamSpy tags (`steamspy_tags`, pipe-separated).
# Chosen to cover recurring, high-signal engagement patterns in our sample.
TAG_FLAGS = [
    "Multiplayer",
    "Singleplayer",
    "Co-op",
    "Online Co-Op",
    "Open World",
    "Survival",
    "Crafting",
    "Building",
    "Sandbox",
    "Roguelike",
    "Roguelite",
    "Pixel Graphics",
    "2D",
    "3D",
    "Atmospheric",
    "Story Rich",
    "Difficult",
    "Procedural Generation",
    "Horror",
    "Action RPG",
]

METADATA_FEATURES = [
    "release_year",
    "years_since_release",
    "release_month",
    "release_is_q4",
    "ea_cohort_post_2020",
    "price_usd",
    "price_tier",
    "num_genres",
    "num_categories",
    "num_platforms",
    "num_languages",
    "num_top_tags",
    "num_publishers",
    "num_developers",
    "is_self_published",
    "ea_to_pr_review_ratio",
    "required_age",
    "supports_windows",
    "supports_mac",
    "supports_linux",
    "has_multiplayer_category",
    "has_controller_support",
    "has_cloud_saves",
] + [f"genre_{g.lower().replace(' ', '_')}" for g in GENRE_FLAGS] \
  + [f"tag_{t.lower().replace(' ', '_').replace('-', '_')}" for t in TAG_FLAGS]


FEATURE_SETS = ("steam", "youtube", "metadata", "steam+metadata", "combined")


def _parse_release_year(val) -> Optional[int]:
    """Parse Steam release date strings like 'Sep 10, 2024' or '2024-09-10' into a year."""
    if pd.isna(val) or not str(val).strip():
        return None
    try:
        return pd.to_datetime(val, errors="coerce").year or None
    except Exception:
        return None


def _price_tier(price: Optional[float]) -> Optional[int]:
    """Bucket price into ordinal tiers: 0=free, 1=<$10, 2=<$20, 3=<$30, 4=$30+."""
    if price is None or pd.isna(price):
        return None
    if price <= 0:
        return 0
    if price < 10:
        return 1
    if price < 20:
        return 2
    if price < 30:
        return 3
    return 4


def _split_field(val, sep: str) -> List[str]:
    """Split a pipe/comma-separated field into a clean list of tokens."""
    if pd.isna(val) or not str(val).strip():
        return []
    return [t.strip() for t in str(val).split(sep) if t.strip()]


def _enrich_metadata_features(merged: pd.DataFrame) -> pd.DataFrame:
    """Derive numeric/binary feature columns from the merged metadata fields.

    Mutates and returns `merged`. Safe to call when columns are missing — the
    corresponding derived features will simply not be created.
    """
    release_col = next(
        (c for c in ("steam_release_date", "release_date") if c in merged.columns),
        None,
    )
    if release_col:
        release_parsed = pd.to_datetime(merged[release_col], errors="coerce")
        merged["release_year"] = release_parsed.dt.year
        merged["release_month"] = release_parsed.dt.month
        merged["release_is_q4"] = release_parsed.dt.month.apply(
            lambda m: int(m in (10, 11, 12)) if pd.notna(m) else None
        )
        merged["ea_cohort_post_2020"] = merged["release_year"].apply(
            lambda y: int(y >= 2020) if pd.notna(y) else None
        )
        current_year = pd.Timestamp.now().year
        merged["years_since_release"] = merged["release_year"].apply(
            lambda y: current_year - y if pd.notna(y) else None
        )

    price_col = next(
        (c for c in ("steam_current_price_usd", "current_price_usd") if c in merged.columns),
        None,
    )
    if price_col:
        merged["price_usd"] = pd.to_numeric(merged[price_col], errors="coerce")
        merged["price_tier"] = merged["price_usd"].apply(_price_tier)

    if "required_age" in merged.columns:
        merged["required_age"] = pd.to_numeric(
            merged["required_age"], errors="coerce"
        ).fillna(0)

    if "genres" in merged.columns:
        genre_lists = merged["genres"].apply(lambda v: _split_field(v, "|"))
        merged["num_genres"] = genre_lists.apply(len)
        for g in GENRE_FLAGS:
            col = f"genre_{g.lower().replace(' ', '_')}"
            merged[col] = genre_lists.apply(lambda lst: int(g in lst)).astype(int)

    if "categories" in merged.columns:
        cat_lists = merged["categories"].apply(lambda v: _split_field(v, "|"))
        merged["num_categories"] = cat_lists.apply(len)
        merged["has_multiplayer_category"] = cat_lists.apply(
            lambda lst: int(any("Multi-player" in c or "Co-op" in c for c in lst))
        )
        merged["has_controller_support"] = cat_lists.apply(
            lambda lst: int(any("Controller" in c for c in lst))
        )
        merged["has_cloud_saves"] = cat_lists.apply(
            lambda lst: int(any("Cloud" in c for c in lst))
        )

    if "platforms" in merged.columns:
        plat_lists = merged["platforms"].apply(lambda v: _split_field(v, "|"))
        merged["num_platforms"] = plat_lists.apply(len)
        merged["supports_windows"] = plat_lists.apply(lambda lst: int("windows" in lst))
        merged["supports_mac"] = plat_lists.apply(lambda lst: int("mac" in lst))
        merged["supports_linux"] = plat_lists.apply(lambda lst: int("linux" in lst))

    if "steamspy_languages" in merged.columns:
        lang_lists = merged["steamspy_languages"].apply(lambda v: _split_field(v, ","))
        merged["num_languages"] = lang_lists.apply(len)

    if "steamspy_tags" in merged.columns:
        tag_lists = merged["steamspy_tags"].apply(lambda v: _split_field(v, "|"))
        merged["num_top_tags"] = tag_lists.apply(len)
        for t in TAG_FLAGS:
            col = f"tag_{t.lower().replace(' ', '_').replace('-', '_')}"
            merged[col] = tag_lists.apply(lambda lst: int(t in lst)).astype(int)

    if "developer" in merged.columns and "publisher" in merged.columns:
        dev_lists = merged["developer"].apply(lambda v: _split_field(v, ","))
        pub_lists = merged["publisher"].apply(lambda v: _split_field(v, ","))
        merged["num_developers"] = dev_lists.apply(len)
        merged["num_publishers"] = pub_lists.apply(len)
        merged["is_self_published"] = [
            int(bool(ds) and bool(ps) and set(ds) == set(ps))
            for ds, ps in zip(dev_lists, pub_lists)
        ]

    if "pr_review_count" in merged.columns and "ea_review_count" in merged.columns:
        ea = pd.to_numeric(merged["ea_review_count"], errors="coerce")
        pr = pd.to_numeric(merged["pr_review_count"], errors="coerce")
        merged["ea_to_pr_review_ratio"] = (pr / ea).where(ea > 0)

    return merged


def _normalize_name(name: str) -> str:
    name = str(name).lower().strip()
    name = re.sub(r"[^a-z0-9\s]", "", name)
    return re.sub(r"\s+", " ", name).strip()


def _load_game_metadata_for(merged: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Load and concatenate both training + validation game_metadata.csv files.

    Returns None when no metadata file is available. Deduplicates on app_id,
    keeping the first occurrence (training takes precedence over validation
    if a game appears in both).
    """
    paths = [
        Path("data/game_metadata.csv"),
        Path("data/validation/game_metadata.csv"),
    ]
    frames = []
    for p in paths:
        if p.exists():
            try:
                frames.append(pd.read_csv(p))
            except Exception as e:
                print(f"  Warning: failed to read {p}: {e}")
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    if "app_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["app_id"], keep="first")
    return combined


def build_sentiment_features(
    game_sentiment: pd.DataFrame,
    success_metrics: pd.DataFrame,
    youtube_sentiment: Optional[pd.DataFrame] = None,
    feature_set: str = "steam",
) -> pd.DataFrame:
    """Join sentiment with success metrics (and optionally YouTube features) for the feature matrix.

    Args:
        game_sentiment: per-game Steam EA sentiment (from validation/training).
        success_metrics: SteamSpy/Steam metrics with success_tier & estimated_revenue_usd.
        youtube_sentiment: optional per-game YouTube comment sentiment.
        feature_set: one of "steam", "youtube", "combined". Selects which columns end
            up in `feature_cols` — the target columns must still be present in the merge.

    Target:
        - estimated_revenue_usd (regression, Boxleiter: total_reviews x 30 x price)
        - success_tier (classification)
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"feature_set must be one of {FEATURE_SETS}, got {feature_set}"
        )

    sentiment = game_sentiment.copy()
    metrics = success_metrics.copy()

    if "app_id" in sentiment.columns and "app_id" in metrics.columns:
        merged = sentiment.merge(metrics, on="app_id", how="inner", suffixes=("", "_m"))
    else:
        sentiment["name_key"] = sentiment["game_name"].apply(_normalize_name)
        metrics["name_key"] = metrics["app_name"].apply(_normalize_name)
        merged = sentiment.merge(metrics, on="name_key", how="inner", suffixes=("", "_m"))

    if youtube_sentiment is not None and len(youtube_sentiment) > 0:
        yt = youtube_sentiment.copy()
        if "app_id" in merged.columns and "app_id" in yt.columns:
            merged = merged.merge(
                yt, on="app_id", how="left", suffixes=("", "_yt")
            )
        else:
            if "name_key" not in merged.columns:
                merged["name_key"] = merged.get(
                    "game_name", merged.get("app_name")
                ).apply(_normalize_name)
            yt["name_key"] = yt.get("app_name", yt.get("game_name")).apply(_normalize_name)
            merged = merged.merge(yt, on="name_key", how="left", suffixes=("", "_yt"))

    metadata_df = _load_game_metadata_for(merged)
    if metadata_df is not None:
        merge_on = "app_id" if "app_id" in merged.columns and "app_id" in metadata_df.columns else None
        if merge_on:
            merged = merged.merge(
                metadata_df, on="app_id", how="left", suffixes=("", "_meta")
            )
        else:
            if "name_key" not in merged.columns:
                merged["name_key"] = merged.get(
                    "game_name", merged.get("app_name")
                ).apply(_normalize_name)
            metadata_df = metadata_df.copy()
            metadata_df["name_key"] = metadata_df["app_name"].apply(_normalize_name)
            merged = merged.merge(
                metadata_df.drop(columns=["app_id"], errors="ignore"),
                on="name_key",
                how="left",
                suffixes=("", "_meta"),
            )

    merged = _enrich_metadata_features(merged)

    if "ea_predicted_positive_ratio" in merged.columns:
        merged["sentiment_confidence"] = (
            merged["ea_predicted_positive_ratio"] - 0.5
        ).abs()
    if "ea_review_count" in merged.columns:
        merged["log_ea_review_count"] = np.log1p(merged["ea_review_count"])
    if "yt_predicted_positive_ratio" in merged.columns:
        merged["yt_sentiment_confidence"] = (
            merged["yt_predicted_positive_ratio"] - 0.5
        ).abs()
    if "yt_comment_count" in merged.columns:
        merged["log_yt_comment_count"] = np.log1p(merged["yt_comment_count"])

    if feature_set == "steam":
        candidates = STEAM_FEATURES
    elif feature_set == "youtube":
        candidates = YOUTUBE_FEATURES
    elif feature_set == "metadata":
        candidates = METADATA_FEATURES
    elif feature_set == "steam+metadata":
        candidates = STEAM_FEATURES + METADATA_FEATURES
    else:  # combined
        candidates = STEAM_FEATURES + YOUTUBE_FEATURES + METADATA_FEATURES

    feature_cols = [c for c in candidates if c in merged.columns]

    merged["feature_cols"] = [feature_cols] * len(merged)
    merged.attrs["feature_set"] = feature_set

    return merged


class SuccessPredictor:
    """Predict game commercial success from sentiment features."""

    def __init__(self, path_config=None):
        self.path_config = path_config

    def evaluate_classification(
        self,
        game_sentiment: pd.DataFrame,
        success_metrics: pd.DataFrame,
        youtube_sentiment: Optional[pd.DataFrame] = None,
        feature_set: str = "steam",
    ) -> PredictionResult:
        """Evaluate success tier prediction using leave-one-out CV."""
        merged = build_sentiment_features(
            game_sentiment, success_metrics, youtube_sentiment, feature_set
        )

        # Filter to games with success tier
        merged = merged.dropna(subset=["success_tier"])
        feature_cols_preview = merged["feature_cols"].iloc[0] if len(merged) > 0 else []
        if feature_cols_preview:
            merged = merged.dropna(subset=feature_cols_preview)
        if len(merged) < 5:
            print(
                f"Not enough games with success tier + {feature_set} features "
                f"(n={len(merged)})"
            )
            return PredictionResult(
                task="classification",
                model_name=f"LogisticRegression [{feature_set}]",
                n_samples=len(merged),
            )

        feature_cols = merged["feature_cols"].iloc[0]
        X = merged[feature_cols].values
        y_labels = merged["success_tier"].values

        le = LabelEncoder()
        le.classes_ = np.array(SUCCESS_TIERS)
        y = le.transform(y_labels)

        unique_tiers, tier_counts = np.unique(y_labels, return_counts=True)
        if len(unique_tiers) < 2:
            print(
                f"  WARNING: only one tier present ({unique_tiers[0]}); "
                "classification is degenerate."
            )
        elif tier_counts.max() / len(y_labels) > 0.8:
            majority = unique_tiers[tier_counts.argmax()]
            print(
                f"  WARNING: class imbalance — {tier_counts.max()}/{len(y_labels)} "
                f"are '{majority}'. LOO accuracy will be dominated by the majority class; "
                "treat this metric as a sanity check, not a real result."
            )

        # LOO-CV (falls back to majority class when a fold has a single training class)
        loo = LeaveOneOut()
        y_pred = np.zeros_like(y)
        game_names = merged["game_name"].values if "game_name" in merged.columns else merged.index.values

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]

            if len(np.unique(y_train)) < 2:
                y_pred[test_idx] = y_train[0]
                continue

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            clf = LogisticRegression(max_iter=1000, C=1.0)
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

        print(
            f"\nSuccess Tier Classification [{feature_set}] "
            f"(LOO-CV, n={len(merged)}, features={len(feature_cols)})"
        )
        print(f"Accuracy: {acc:.2%}")
        print(f"\n{report}")

        return PredictionResult(
            task="classification",
            model_name=f"LogisticRegression [{feature_set}]",
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
        youtube_sentiment: Optional[pd.DataFrame] = None,
        feature_set: str = "steam",
    ) -> PredictionResult:
        """Evaluate revenue prediction using leave-one-out CV."""
        merged = build_sentiment_features(
            game_sentiment, success_metrics, youtube_sentiment, feature_set
        )

        merged = merged.dropna(subset=["estimated_revenue_usd"])
        merged = merged[merged["estimated_revenue_usd"] > 0]

        feature_cols_preview = merged["feature_cols"].iloc[0] if len(merged) > 0 else []
        if feature_cols_preview:
            merged = merged.dropna(subset=feature_cols_preview)

        if len(merged) < 5:
            print(
                f"Not enough games with revenue + {feature_set} features "
                f"(n={len(merged)})"
            )
            return PredictionResult(
                task="regression",
                model_name=f"Ridge [{feature_set}]",
                n_samples=len(merged),
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

        print(
            f"\nRevenue Prediction [{feature_set}] "
            f"(LOO-CV, n={len(merged)}, features={len(feature_cols)})"
        )
        print(f"MAE: ${mae:,.0f}")
        print(f"RMSE: ${rmse:,.0f}")
        print(f"R² (log space): {r2:.3f}")

        return PredictionResult(
            task="regression",
            model_name=f"Ridge [{feature_set}]",
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

        clf = LogisticRegression(max_iter=1000, C=1.0)
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

    def run_ablation(
        self,
        game_sentiment: pd.DataFrame,
        success_metrics: pd.DataFrame,
        youtube_sentiment: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compare Steam-only, YouTube-only, and combined feature sets on both tasks.

        Runs evaluate_classification and evaluate_regression three times each and
        returns a summary DataFrame. Prints a concise side-by-side table.
        """
        rows = []
        for feature_set in FEATURE_SETS:
            print("\n" + "=" * 60)
            print(f"ABLATION — feature_set = {feature_set}")
            print("=" * 60)

            cls = self.evaluate_classification(
                game_sentiment, success_metrics, youtube_sentiment, feature_set
            )
            reg = self.evaluate_regression(
                game_sentiment, success_metrics, youtube_sentiment, feature_set
            )
            rows.append({
                "feature_set": feature_set,
                "cls_n": cls.n_samples,
                "cls_accuracy": cls.accuracy,
                "reg_n": reg.n_samples,
                "reg_mae_usd": reg.mae,
                "reg_rmse_usd": reg.rmse,
                "reg_r2_log": reg.r2,
            })

        ablation_df = pd.DataFrame(rows)

        print("\n" + "=" * 60)
        print("ABLATION SUMMARY")
        print("=" * 60)
        print(ablation_df.to_string(index=False))

        baseline_mask = ablation_df["feature_set"] == "steam"
        if baseline_mask.any():
            base = ablation_df[baseline_mask].iloc[0]
            print("\nΔ vs baseline (steam):")
            for _, row in ablation_df.iterrows():
                if row["feature_set"] == "steam":
                    continue
                parts = [f"{row['feature_set']}:"]
                if pd.notna(base["cls_accuracy"]) and pd.notna(row["cls_accuracy"]):
                    parts.append(
                        f"Δacc={row['cls_accuracy'] - base['cls_accuracy']:+.2%}"
                    )
                if pd.notna(base["reg_r2_log"]) and pd.notna(row["reg_r2_log"]):
                    parts.append(
                        f"ΔR²={row['reg_r2_log'] - base['reg_r2_log']:+.3f}"
                    )
                if pd.notna(base["reg_mae_usd"]) and pd.notna(row["reg_mae_usd"]):
                    parts.append(
                        f"ΔMAE=${row['reg_mae_usd'] - base['reg_mae_usd']:+,.0f}"
                    )
                print("  " + " ".join(parts))

        return ablation_df

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
