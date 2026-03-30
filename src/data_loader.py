import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import PathConfig


TRAINING_PARQUET = Path("data/processed/training_dataset.parquet")
VALIDATION_PARQUET = Path("data/processed/validation_dataset.parquet")


@dataclass
class ReviewData:
    df: pd.DataFrame
    X: np.ndarray  # cleaned text
    y: np.ndarray  # labels

    @property
    def n_samples(self) -> int:
        return len(self.df)

    @property
    def n_positive(self) -> int:
        return int(self.y.sum())

    @property
    def n_negative(self) -> int:
        return self.n_samples - self.n_positive

    @property
    def positive_ratio(self) -> float:
        return self.n_positive / self.n_samples if self.n_samples > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"Total reviews: {self.n_samples:,}",
            f"Positive: {self.n_positive:,} ({self.positive_ratio * 100:.1f}%)",
            f"Negative: {self.n_negative:,} ({(1 - self.positive_ratio) * 100:.1f}%)",
            f"Unique games: {self.df['game_name'].nunique()}",
        ]
        return "\n".join(lines)


class TextPreprocessor:
    @staticmethod
    def clean_text(text: str) -> str:
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s.,!?\'\-]", "", text)
        text = " ".join(text.split())
        return text.strip()


class DataLoader:
    def __init__(self, path_config: PathConfig):
        self.path_config = path_config
        self.preprocessor = TextPreprocessor()

    def _load_from_parquet(self, parquet_path: Path) -> Optional[pd.DataFrame]:
        """Try loading from pre-built parquet file."""
        if not parquet_path.exists():
            return None
        print(f"Loading pre-processed dataset from {parquet_path}")
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df):,} reviews ({df['game_name'].nunique()} games)")
        return df

    def _load_from_raw_csvs(self, data_dir: Path, min_text_length: int) -> pd.DataFrame:
        """Fallback: load from raw CSVs with on-the-fly cleaning."""
        csv_files = sorted(data_dir.glob("*_reviews.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No review CSV files found in {data_dir}")

        print(f"[Fallback] Loading {len(csv_files)} raw CSVs from {data_dir}")
        print("  Hint: run 'python -m src.data_pipeline' to build parquet for faster loading.")

        dfs = []
        for file in tqdm(csv_files, desc="Loading files"):
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

        combined_df = pd.concat(dfs, ignore_index=True)

        combined_df = combined_df.dropna(subset=["review_text"])
        combined_df["review_text"] = combined_df["review_text"].astype(str)
        combined_df["positive"] = combined_df["positive"].astype(int)

        if "written_during_early_access" in combined_df.columns:
            combined_df = combined_df.drop(columns=["written_during_early_access"])

        print("Cleaning text...")
        combined_df["cleaned_text"] = combined_df["review_text"].apply(
            self.preprocessor.clean_text
        )

        original_len = len(combined_df)
        combined_df = combined_df[
            combined_df["cleaned_text"].str.len() > min_text_length
        ]
        removed = original_len - len(combined_df)
        if removed > 0:
            print(f"Removed {removed:,} short reviews")

        return combined_df

    def load_all_reviews(self, min_text_length: int = 10) -> ReviewData:
        """Load all review data. Uses parquet if available, falls back to raw CSVs."""
        # Try parquet first
        df = self._load_from_parquet(TRAINING_PARQUET)

        if df is None:
            df = self._load_from_raw_csvs(self.path_config.data_dir, min_text_length)

        X = df["cleaned_text"].values
        y = df["positive"].values

        print(f"Loaded {len(df):,} reviews")

        return ReviewData(df=df, X=X, y=y)

    def get_game_data(self, data: ReviewData, game_name: str) -> Optional[ReviewData]:
        mask = data.df["game_name"] == game_name
        if not mask.any():
            return None

        game_df = data.df[mask].copy()
        return ReviewData(
            df=game_df,
            X=game_df["cleaned_text"].values,
            y=game_df["positive"].values,
        )

    def get_early_access_data(self, data: ReviewData) -> ReviewData:
        mask = data.df["review_type"] == "early_access"
        ea_df = data.df[mask].copy()
        return ReviewData(
            df=ea_df,
            X=ea_df["cleaned_text"].values,
            y=ea_df["positive"].values,
        )

    def get_post_release_data(self, data: ReviewData) -> ReviewData:
        mask = data.df["review_type"] == "post_release"
        pr_df = data.df[mask].copy()
        return ReviewData(
            df=pr_df,
            X=pr_df["cleaned_text"].values,
            y=pr_df["positive"].values,
        )
