"""Run trained sentiment classifier on collected YouTube comments.

Loads the best trained classical model, applies it to data/youtube_data.csv,
and computes per-game YouTube sentiment features. Output mirrors the
EA-review feature set so it can be merged with success metrics.

Usage:
    python youtube_sentiment.py
    python youtube_sentiment.py --model LogisticRegression
    python youtube_sentiment.py --output results/metrics/youtube_sentiment.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PathConfig
from src.data_loader import TextPreprocessor
from src.analysis.thesis_validation import ThesisValidator


YOUTUBE_DATA_FILE = Path("data/youtube_data.csv")
DEFAULT_OUTPUT = Path("results/metrics/youtube_sentiment.csv")


def compute_game_features(
    game_df: pd.DataFrame,
    model,
    tfidf,
) -> dict:
    """Compute sentiment features for a single game's YouTube comments."""
    texts = game_df["cleaned_text"].values
    raw_texts = game_df["text"].values

    X = tfidf.transform(texts)
    predictions = model.predict(X)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        positive_probs = proba[:, 1]
        avg_positive_prob = positive_probs.mean()
    else:
        positive_probs = predictions.astype(float)
        avg_positive_prob = predictions.mean()

    word_counts = np.array([len(str(t).split()) for t in raw_texts])
    pos_mask = predictions == 1
    neg_mask = predictions == 0

    return {
        "yt_comment_count": len(game_df),
        "yt_predicted_positive_ratio": predictions.mean(),
        "yt_avg_positive_probability": avg_positive_prob,
        "yt_sentiment_std": positive_probs.std(),
        "yt_mean_comment_length": word_counts.mean(),
        "yt_std_comment_length": word_counts.std(),
        "yt_long_comment_ratio": (word_counts > 50).mean(),
        "yt_mean_pos_comment_length": word_counts[pos_mask].mean() if pos_mask.any() else 0.0,
        "yt_mean_neg_comment_length": word_counts[neg_mask].mean() if neg_mask.any() else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Model name (e.g., LogisticRegression)")
    parser.add_argument("--seed", type=int, default=None, help="Specific seed to load")
    parser.add_argument("--fold", type=int, default=0, help="Fold to load (default: 0)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--min-comments", type=int, default=10,
                        help="Skip games with fewer than N comments")
    parser.add_argument("--min-text-length", type=int, default=10,
                        help="Skip comments shorter than N characters")
    args = parser.parse_args()

    if not YOUTUBE_DATA_FILE.exists():
        raise FileNotFoundError(f"YouTube data not found at {YOUTUBE_DATA_FILE}")

    # Load model
    path_config = PathConfig()
    validator = ThesisValidator(path_config)
    model, tfidf, model_name = validator.load_best_model(
        model_name=args.model, seed=args.seed, fold=args.fold,
    )
    print(f"Using model: {model_name}\n")

    # Load and clean YouTube comments
    print(f"Loading {YOUTUBE_DATA_FILE}...")
    df = pd.read_csv(YOUTUBE_DATA_FILE)
    print(f"Loaded {len(df):,} comments across {df['app_name'].nunique()} games")

    preprocessor = TextPreprocessor()
    df["cleaned_text"] = df["text"].apply(preprocessor.clean_text)

    before = len(df)
    df = df[df["cleaned_text"].str.len() > args.min_text_length]
    print(f"Removed {before - len(df):,} short comments (<= {args.min_text_length} chars)")
    print(f"Remaining: {len(df):,} comments\n")

    # Per-game feature computation
    print("Computing per-game sentiment features...")
    rows = []
    for (app_id, app_name), game_df in df.groupby(["app_id", "app_name"]):
        if len(game_df) < args.min_comments:
            print(f"  Skipping {app_name}: only {len(game_df)} comments")
            continue
        features = compute_game_features(game_df, model, tfidf)
        rows.append({"app_id": app_id, "app_name": app_name, **features})

    result_df = pd.DataFrame(rows).sort_values(
        "yt_predicted_positive_ratio", ascending=False,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)
    print(f"\nSaved features for {len(result_df)} games to {args.output}")

    # Summary
    print("\nPer-game summary (sorted by predicted positive ratio):")
    print(result_df[
        ["app_name", "yt_comment_count", "yt_predicted_positive_ratio",
         "yt_sentiment_std", "yt_mean_comment_length"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
