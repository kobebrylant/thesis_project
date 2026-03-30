"""ETL Transform + Load pipeline.

Reads raw per-game CSVs and metadata, cleans text, deduplicates,
and outputs Gold Standard parquet files ready for model consumption.

Usage:
    python -m src.data_pipeline              # Build if stale
    python -m src.data_pipeline --force      # Force rebuild
    python -m src.data_pipeline --status     # Show manifest info
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from .data_loader import TextPreprocessor


PROCESSED_DIR = Path("data/processed")
TRAINING_PARQUET = PROCESSED_DIR / "training_dataset.parquet"
VALIDATION_PARQUET = PROCESSED_DIR / "validation_dataset.parquet"
MANIFEST_FILE = PROCESSED_DIR / "manifest.json"


def _hash_file(path: Path) -> str:
    """Fast hash of file contents for change detection."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_source_hashes(raw_dir: Path) -> Dict[str, str]:
    """Hash all CSVs in a directory."""
    hashes = {}
    for csv_file in sorted(raw_dir.glob("*.csv")):
        # Skip metadata and success metrics files
        if csv_file.name in ("game_metadata.csv", "game_success_metrics.csv"):
            continue
        hashes[str(csv_file)] = _hash_file(csv_file)
    return hashes


def _load_manifest() -> Optional[Dict]:
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, "r") as f:
            return json.load(f)
    return None


def _save_manifest(manifest: Dict):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def is_stale(raw_dirs: List[Path]) -> bool:
    """Check if processed parquet needs rebuilding."""
    manifest = _load_manifest()
    if manifest is None:
        return True

    if not TRAINING_PARQUET.exists():
        return True

    current_hashes = {}
    for raw_dir in raw_dirs:
        if raw_dir.exists():
            current_hashes.update(_collect_source_hashes(raw_dir))

    return current_hashes != manifest.get("source_hashes", {})


def _load_raw_csvs(raw_dir: Path) -> pd.DataFrame:
    """Load all review CSVs from a directory into a single DataFrame."""
    csv_files = sorted(raw_dir.glob("*_reviews.csv"))

    if not csv_files:
        print(f"  No review CSVs found in {raw_dir}")
        return pd.DataFrame()

    dfs = []
    for file in tqdm(csv_files, desc=f"Loading {raw_dir.name}"):
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
            print(f"  Error loading {file}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def _load_metadata(metadata_file: Path) -> Optional[pd.DataFrame]:
    """Load game metadata CSV for join."""
    if not metadata_file.exists():
        return None
    df = pd.read_csv(metadata_file)
    # Ensure app_id is int for join
    df["app_id"] = df["app_id"].astype(int)
    return df


def _clean_and_transform(
    df: pd.DataFrame,
    metadata_df: Optional[pd.DataFrame] = None,
    min_text_length: int = 10,
) -> pd.DataFrame:
    """Apply text cleaning, deduplication, and metadata enrichment."""
    if df.empty:
        return df

    # Drop rows without text
    df = df.dropna(subset=["review_text"])
    df["review_text"] = df["review_text"].astype(str)
    df["positive"] = df["positive"].astype(int)

    # Drop the written_during_early_access column if it survived from old CSVs
    if "written_during_early_access" in df.columns:
        df = df.drop(columns=["written_during_early_access"])

    # Clean text
    preprocessor = TextPreprocessor()
    print("  Cleaning text...")
    df["cleaned_text"] = df["review_text"].apply(preprocessor.clean_text)

    # Remove short reviews
    original_len = len(df)
    df = df[df["cleaned_text"].str.len() > min_text_length]
    removed = original_len - len(df)
    if removed > 0:
        print(f"  Removed {removed:,} short reviews (< {min_text_length} chars)")

    # Deduplicate on (app_id, review_text)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["app_id", "review_text"])
    deduped = before_dedup - len(df)
    if deduped > 0:
        print(f"  Removed {deduped:,} duplicate reviews")

    # Join metadata if available
    if metadata_df is not None and "app_id" in df.columns:
        df["app_id"] = df["app_id"].astype(int)
        meta_cols = ["app_id", "developer", "publisher", "release_date", "current_price_usd"]
        available_cols = [c for c in meta_cols if c in metadata_df.columns]
        df = df.merge(metadata_df[available_cols], on="app_id", how="left")

    return df


def build_dataset(force: bool = False) -> Dict[str, int]:
    """Main ETL entry point. Build processed parquet files from raw CSVs.

    Returns dict with row counts per dataset.
    """
    raw_training_dir = Path("data")
    raw_validation_dir = Path("data/validation")

    if not force and not is_stale([raw_training_dir, raw_validation_dir]):
        manifest = _load_manifest()
        print("Datasets are up to date (no source changes detected).")
        print(f"  Training: {manifest.get('training_rows', '?')} rows")
        print(f"  Validation: {manifest.get('validation_rows', '?')} rows")
        print(f"  Built: {manifest.get('build_time', '?')}")
        return {
            "training_rows": manifest.get("training_rows", 0),
            "validation_rows": manifest.get("validation_rows", 0),
        }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}

    # --- Training dataset ---
    print("\n=== Building Training Dataset ===")
    training_df = _load_raw_csvs(raw_training_dir)
    if not training_df.empty:
        metadata_df = _load_metadata(Path("data/game_metadata.csv"))
        training_df = _clean_and_transform(training_df, metadata_df)
        training_df.to_parquet(TRAINING_PARQUET, index=False)
        stats["training_rows"] = len(training_df)
        print(f"  Saved {len(training_df):,} rows to {TRAINING_PARQUET}")
        print(f"  Games: {training_df['game_name'].nunique()}")
        print(f"  EA reviews: {len(training_df[training_df['review_type'] == 'early_access']):,}")
        print(f"  Post-release: {len(training_df[training_df['review_type'] == 'post_release']):,}")
    else:
        stats["training_rows"] = 0
        print("  No training data found.")

    # --- Validation dataset ---
    print("\n=== Building Validation Dataset ===")
    validation_df = _load_raw_csvs(raw_validation_dir)
    if not validation_df.empty:
        metadata_df = _load_metadata(Path("data/validation/game_metadata.csv"))
        validation_df = _clean_and_transform(validation_df, metadata_df)
        validation_df.to_parquet(VALIDATION_PARQUET, index=False)
        stats["validation_rows"] = len(validation_df)
        print(f"  Saved {len(validation_df):,} rows to {VALIDATION_PARQUET}")
        print(f"  Games: {validation_df['game_name'].nunique()}")
    else:
        stats["validation_rows"] = 0
        print("  No validation data found.")

    # --- Save manifest ---
    source_hashes = {}
    for d in [raw_training_dir, raw_validation_dir]:
        if d.exists():
            source_hashes.update(_collect_source_hashes(d))

    manifest = {
        "build_time": datetime.now().isoformat(),
        "training_rows": stats["training_rows"],
        "validation_rows": stats["validation_rows"],
        "source_hashes": source_hashes,
    }
    _save_manifest(manifest)

    print(f"\nManifest saved to {MANIFEST_FILE}")
    return stats


def show_status():
    """Print current dataset status."""
    manifest = _load_manifest()
    if manifest is None:
        print("No processed dataset found. Run: python -m src.data_pipeline")
        return

    print("=== Dataset Status ===")
    print(f"Build time: {manifest.get('build_time', '?')}")
    print(f"Training rows: {manifest.get('training_rows', '?')}")
    print(f"Validation rows: {manifest.get('validation_rows', '?')}")
    print(f"Source files tracked: {len(manifest.get('source_hashes', {}))}")

    raw_dirs = [Path("data"), Path("data/validation")]
    if is_stale(raw_dirs):
        print("\nStatus: STALE — source files have changed. Run rebuild.")
    else:
        print("\nStatus: UP TO DATE")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build processed datasets from raw CSVs")
    parser.add_argument("--force", action="store_true", help="Force rebuild")
    parser.add_argument("--status", action="store_true", help="Show dataset status")
    args = parser.parse_args()

    if args.status:
        show_status()
    else:
        build_dataset(force=args.force)
