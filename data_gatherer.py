"""Collect Steam review data and game metadata for sentiment analysis.

ETL Extract phase: fetches raw reviews and metadata from Steam APIs,
saves per-game CSVs and a shared metadata cache.
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from src.api_client import ResilientAPIClient


DATA_DIR = "data"
VALIDATION_DATA_DIR = "data/validation"
GAME_REGISTRY_PATH = Path("data/game_registry.json")
METADATA_FILE = Path("data/game_metadata.csv")
VALIDATION_METADATA_FILE = Path("data/validation/game_metadata.csv")

METADATA_FIELDNAMES = [
    "app_id",
    "app_name",
    "developer",
    "publisher",
    "release_date",
    "current_price_usd",
    "genres",
    "categories",
    "platforms",
    "required_age",
    "short_description",
]


def load_game_registry() -> Dict[str, List[Dict]]:
    """Load game definitions from JSON registry."""
    if not GAME_REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Game registry not found at {GAME_REGISTRY_PATH}. "
            "Create it or run with --init-registry."
        )
    with open(GAME_REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_data_dir(data_dir: str = DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)


def load_existing_metadata(metadata_file: Path) -> Dict[int, Dict]:
    """Load already-collected metadata keyed by app_id."""
    existing = {}
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[int(row["app_id"])] = row
    return existing


def save_metadata(metadata_records: Dict[int, Dict], metadata_file: Path):
    """Save metadata dict to CSV."""
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=METADATA_FIELDNAMES, extrasaction="ignore"
        )
        writer.writeheader()
        for record in metadata_records.values():
            for col in METADATA_FIELDNAMES:
                record.setdefault(col, "")
            writer.writerow(record)


def fetch_game_metadata(
    app_id: int,
    app_name: str,
    client: ResilientAPIClient,
) -> Dict:
    """Fetch developer, publisher, release date, and price from Steam Store API.

    Single HTTP request per game — results are cached to game_metadata.csv.
    """
    url = f"https://store.steampowered.com/api/appdetails"
    params = {"appids": str(app_id), "cc": "us", "l": "en"}

    record = {
        "app_id": app_id,
        "app_name": app_name,
        "developer": "",
        "publisher": "",
        "release_date": "",
        "current_price_usd": "",
        "genres": "",
        "categories": "",
        "platforms": "",
        "required_age": "",
        "short_description": "",
    }

    try:
        response = client.get(url, params=params)
        if response.status_code != 200:
            print(f"  Metadata fetch failed for {app_name}: HTTP {response.status_code}")
            return record

        data = response.json()
        app_data = data.get(str(app_id), {})
        if not app_data.get("success"):
            print(f"  Steam Store returned success=false for {app_name}")
            return record

        details = app_data.get("data", {})

        developers = details.get("developers", [])
        record["developer"] = ", ".join(developers) if developers else ""

        publishers = details.get("publishers", [])
        record["publisher"] = ", ".join(publishers) if publishers else ""

        release_info = details.get("release_date", {})
        record["release_date"] = release_info.get("date", "")

        price_info = details.get("price_overview", {})
        if price_info:
            # Price is in cents
            record["current_price_usd"] = price_info.get("initial", 0) / 100
        elif details.get("is_free"):
            record["current_price_usd"] = 0.0

        genres = details.get("genres", [])
        record["genres"] = "|".join(g.get("description", "") for g in genres if g.get("description"))

        categories = details.get("categories", [])
        record["categories"] = "|".join(c.get("description", "") for c in categories if c.get("description"))

        platforms = details.get("platforms", {})
        plat_list = [p for p, enabled in platforms.items() if enabled]
        record["platforms"] = "|".join(plat_list)

        record["required_age"] = details.get("required_age", 0)
        record["short_description"] = details.get("short_description", "")

    except Exception as e:
        print(f"  Error fetching metadata for {app_name}: {e}")

    return record


def collect_metadata_for_games(
    games: List[Dict],
    metadata_file: Path,
    client: ResilientAPIClient,
):
    """Fetch and cache metadata for all games, skipping already-collected ones."""
    existing = load_existing_metadata(metadata_file)
    updated = False

    for game in games:
        app_id = game["app_id"]
        if app_id in existing:
            continue

        print(f"  Fetching metadata for {game['name']}...")
        record = fetch_game_metadata(app_id, game["name"], client)
        existing[app_id] = record
        updated = True

    if updated:
        save_metadata(existing, metadata_file)
        print(f"  Metadata saved to {metadata_file}")


def get_reviews_batch(
    client: ResilientAPIClient,
    app_id: str,
    cursor: str = "*",
    language: str = "english",
    start_date: Optional[int] = None,
    end_date: Optional[int] = None,
    date_range_type: bool = False,
) -> Optional[Dict]:
    url = f"https://store.steampowered.com/appreviews/{app_id}"
    params = {
        "cursor": cursor,
        "filter": "recent",
        "language": language,
        "purchase_type": "all",
        "num_per_page": 100,
        "json": 1,
    }

    if date_range_type and start_date and end_date:
        params["start_date"] = start_date
        params["end_date"] = end_date
        params["date_range_type"] = "include"

    try:
        response = client.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(
                f"Error fetching reviews for app {app_id}: HTTP {response.status_code}"
            )
            return None
    except Exception as e:
        print(f"Request error for app {app_id}: {e}")
        return None


def collect_early_access_reviews(
    client: ResilientAPIClient,
    app_id: int,
    app_name: str,
    start_date: int,
    end_date: int,
    max_reviews: int = 500,
) -> List[Dict]:
    """Collect EA-period reviews using Steam's ``written_during_early_access`` flag.

    The flag is Steam's authoritative marker for whether a review was
    submitted while the game was in Early Access. We also pass the
    registry's EA window via ``date_range_type=include`` so Steam
    prefilters server-side to that range. Timestamp membership in
    [start_date, end_date] is accepted as a fallback for the small
    number of games where the flag is missing.
    """
    reviews = []
    cursor = "*"
    empty_batches = 0
    max_empty_batches = 5
    effective_end = end_date if end_date != 9999999999 else 2_000_000_000

    print(f"Collecting early access reviews for {app_name} (App ID: {app_id})...")

    while len(reviews) < max_reviews:
        response = get_reviews_batch(
            client=client,
            app_id=str(app_id),
            cursor=cursor,
            start_date=start_date,
            end_date=end_date,
            date_range_type=True,
        )

        if not response or not response.get("success"):
            success_val = response.get('success') if response else 'No data'
            print(f"Failed to fetch reviews. Success: {success_val}")
            break

        batch_reviews = response.get("reviews", [])
        if not batch_reviews:
            print(f"No more reviews available. Total collected: {len(reviews)}")
            break

        batch_count = 0
        for review in batch_reviews:
            ts = review.get("timestamp_created", 0)
            flagged_ea = review.get("written_during_early_access", False)
            in_window = start_date <= ts <= effective_end

            # Accept if Steam tagged it as EA, or (fallback) if timestamp
            # is inside the registry-declared EA window. The flag is the
            # authoritative signal; the timestamp check is a safety net.
            if not (flagged_ea or in_window):
                continue

            reviews.append(
                {
                    "app_id": app_id,
                    "app_name": app_name,
                    "review_text": review.get("review", ""),
                    "positive": 1 if review.get("voted_up") else 0,
                }
            )
            batch_count += 1

            if len(reviews) >= max_reviews:
                break

        if batch_count == 0:
            empty_batches += 1
            print(
                f"Warning: No EA reviews in this batch "
                f"(empty batch {empty_batches}/{max_empty_batches})"
            )
            if empty_batches >= max_empty_batches:
                print(
                    "Breaking: no more EA reviews available for this game."
                )
                break
        else:
            empty_batches = 0

        print(f"Collected {len(reviews)} early access reviews so far...")

        new_cursor = response.get("cursor")
        if not new_cursor or new_cursor == cursor:
            print("No more pages available.")
            break
        cursor = new_cursor

    print(f"Total early access reviews collected: {len(reviews)}\n")
    return reviews


def collect_post_release_reviews(
    client: ResilientAPIClient,
    app_id: int,
    app_name: str,
    max_reviews: int = 500,
    end_date: Optional[int] = None,
) -> List[Dict]:
    """Collect post-release reviews using timestamp-based filtering.

    A review is post-release if its ``timestamp_created`` is strictly
    greater than the EA ``end_date`` (i.e. after full launch). This
    avoids relying on the unreliable ``written_during_early_access``
    flag.
    """
    reviews = []
    cursor = "*"
    empty_batches = 0
    max_empty_batches = 3

    print(f"Collecting post-release reviews for {app_name} (App ID: {app_id})...")

    while len(reviews) < max_reviews:
        response = get_reviews_batch(
            client=client,
            app_id=str(app_id),
            cursor=cursor,
            start_date=None,
            end_date=None,
            date_range_type=False,
        )

        if not response or not response.get("success"):
            success_val = response.get('success') if response else 'No response'
            print(f"Failed to fetch reviews. Success: {success_val}")
            break

        batch_reviews = response.get("reviews", [])
        if not batch_reviews:
            print(f"No more reviews available. Total collected: {len(reviews)}")
            break

        batch_count = 0
        for review in batch_reviews:
            ts = review.get("timestamp_created", 0)
            flagged_ea = review.get("written_during_early_access", False)

            # Timestamp-based: post-release if after end_date (when known).
            # Fallback to the unreliable flag when no end_date provided.
            if end_date is not None:
                if ts <= end_date:
                    continue
            else:
                if flagged_ea:
                    continue

            reviews.append(
                {
                    "app_id": app_id,
                    "app_name": app_name,
                    "review_text": review.get("review", ""),
                    "positive": 1 if review.get("voted_up") else 0,
                }
            )
            batch_count += 1

            if len(reviews) >= max_reviews:
                break

        if batch_count == 0:
            empty_batches += 1
            print(
                f"Warning: No post-release reviews in this batch "
                f"(empty batch {empty_batches}/{max_empty_batches})"
            )
            if empty_batches >= max_empty_batches:
                print(
                    "Breaking: Game likely still in Early Access "
                    "or no post-release reviews found."
                )
                break
        else:
            empty_batches = 0

        print(f"Collected {len(reviews)} post-release reviews so far...")

        new_cursor = response.get("cursor")
        if not new_cursor or new_cursor == cursor:
            print("No more pages available.")
            break
        cursor = new_cursor

    print(f"Total post-release reviews collected: {len(reviews)}\n")
    return reviews


def save_to_csv(reviews: List[Dict], filename: str, data_dir: str = DATA_DIR):
    if not reviews:
        print(f"No reviews to save to {filename}")
        return

    filepath = os.path.join(data_dir, filename)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=reviews[0].keys())
        writer.writeheader()
        writer.writerows(reviews)

    print(f"Saved {len(reviews)} reviews to {filepath}\n")


def file_exists(filename: str, data_dir: str = DATA_DIR) -> bool:
    filepath = os.path.join(data_dir, filename)
    return os.path.exists(filepath)


def _safe_filename(name: str) -> str:
    """Replace filesystem-unsafe characters in game names."""
    return name.replace(":", "_")


def collect_data_for_games(
    games: List[Dict],
    data_dir: str,
    metadata_file: Path,
    client: ResilientAPIClient,
    max_reviews: int = 10000,
):
    """Collect reviews + metadata for a list of games."""
    ensure_data_dir(data_dir)

    print("\n--- Collecting game metadata ---")
    collect_metadata_for_games(games, metadata_file, client)

    print("\n--- Collecting reviews ---")
    for game in games:
        app_id = game["app_id"]
        app_name = game["name"]
        start_date = game.get("start_date")
        end_date = game.get("end_date")

        safe_name = _safe_filename(app_name)
        ea_filename = f"{safe_name}_early_access_reviews.csv"
        post_filename = f"{safe_name}_post_release_reviews.csv"

        if file_exists(ea_filename, data_dir):
            print(f"Skipping {app_name} - early access reviews already exist\n")
        else:
            ea_reviews = collect_early_access_reviews(
                client=client,
                app_id=app_id,
                app_name=app_name,
                start_date=start_date,
                end_date=end_date,
                max_reviews=max_reviews,
            )
            if ea_reviews:
                save_to_csv(ea_reviews, ea_filename, data_dir)

        # Skip post-release reviews for games still in EA
        if end_date == 9999999999:
            print(f"Skipping post-release reviews for {app_name} - still in Early Access\n")
            continue

        if file_exists(post_filename, data_dir):
            print(f"Skipping {app_name} - post-release reviews already exist\n")
        else:
            post_reviews = collect_post_release_reviews(
                client=client,
                app_id=app_id,
                app_name=app_name,
                max_reviews=max_reviews,
                end_date=end_date,
            )
            if post_reviews:
                save_to_csv(post_reviews, post_filename, data_dir)


def collect_training_data(client: ResilientAPIClient):
    """Collect data for training (original games)."""
    print("=" * 70)
    print("COLLECTING TRAINING DATA")
    print("=" * 70)

    registry = load_game_registry()
    games = registry["training_games"]
    collect_data_for_games(games, DATA_DIR, METADATA_FILE, client)


def collect_validation_data(client: ResilientAPIClient):
    """Collect data for thesis validation (NEW games, never seen during training)."""
    print("\n" + "=" * 70)
    print("COLLECTING VALIDATION DATA (NEW GAMES FOR THESIS)")
    print("=" * 70)
    print("These games will be used for thesis validation ONLY.")
    print("Models have NEVER seen this data during training.\n")

    registry = load_game_registry()
    games = registry["validation_games"]
    collect_data_for_games(games, VALIDATION_DATA_DIR, VALIDATION_METADATA_FILE, client)


def collect_youtube_data(
    games: List[Dict],
    data_dir: str,
    metadata_file: Path,
    max_videos: int = 3,
    max_comments_per_video: int = 100,
):
    """Collect YouTube trailer comment data for all games."""
    from src.sources.youtube_source import YouTubeSource

    ensure_data_dir(data_dir)

    youtube = YouTubeSource()
    output_file = os.path.join(data_dir, "youtube_data.csv")

    existing_app_ids = set()
    if os.path.exists(output_file):
        import pandas as pd
        existing_df = pd.read_csv(output_file)
        existing_app_ids = set(existing_df["app_id"].unique())
        print(f"Found existing YouTube data for {len(existing_app_ids)} games")

    all_records = []
    for game in games:
        app_id = game["app_id"]
        if app_id in existing_app_ids:
            print(f"Skipping {game['name']} - YouTube data already exists")
            continue

        print(f"\n[YouTube] Collecting for {game['name']}...")

        try:
            records = youtube.fetch(
                app_id=app_id,
                app_name=game["name"],
                max_videos=max_videos,
                max_comments_per_video=max_comments_per_video,
            )
            all_records.extend(records)
            print(f"  Collected {len(records)} records")
        except Exception as e:
            print(f"  Error: {e}")

    if all_records:
        _save_source_records(all_records, output_file, append=os.path.exists(output_file))
        print(f"\nYouTube data saved to {output_file}")
    else:
        print("\nNo new YouTube data collected")


def _save_source_records(records: list, filepath: str, append: bool = False):
    """Save SourceRecord instances to CSV."""
    if not records:
        return

    fieldnames = [
        "app_id", "app_name", "text", "sentiment_source",
        "timestamp", "is_pre_release", "author_type", "source_url",
    ]

    mode = "a" if append else "w"
    write_header = not append

    with open(filepath, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect data from Steam and YouTube for sentiment analysis"
    )

    parser.add_argument(
        "--training",
        action="store_true",
        help="Collect data for training games",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Collect data for validation games (unseen)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Collect for both training and validation games",
    )
    parser.add_argument(
        "--steam",
        action="store_true",
        help="Collect Steam reviews (default if no source specified)",
    )
    parser.add_argument(
        "--youtube",
        action="store_true",
        help="Collect YouTube trailer comments (requires YOUTUBE_API_KEY)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Force re-collection (ignore existing files)",
    )

    args = parser.parse_args()

    if not args.training and not args.validation and not args.all:
        print("No dataset specified. Use --help for options.")
        print("Defaulting to --all (both training and validation)\n")
        args.all = True

    if not args.steam and not args.youtube:
        args.steam = True

    registry = load_game_registry()

    datasets = []
    if args.all or args.training:
        datasets.append(("training", registry["training_games"], DATA_DIR, METADATA_FILE))
    if args.all or args.validation:
        datasets.append(("validation", registry["validation_games"], VALIDATION_DATA_DIR, VALIDATION_METADATA_FILE))

    with ResilientAPIClient() as client:
        for dataset_name, games, data_dir, metadata_file in datasets:
            print("\n" + "=" * 70)
            print(f"COLLECTING {dataset_name.upper()} DATA ({len(games)} games)")
            print("=" * 70)

            if args.steam:
                print("\n--- Steam Reviews ---")
                collect_data_for_games(games, data_dir, metadata_file, client)

            if args.youtube:
                print("\n--- YouTube Comments ---")
                collect_youtube_data(games, data_dir, metadata_file)

    print("\n" + "=" * 70)
    print("DATA COLLECTION COMPLETE")
    print("=" * 70)

    for dataset_name, _, data_dir, _ in datasets:
        print(f"  {dataset_name}: {data_dir}/")

    sources = []
    if args.steam:
        sources.append("Steam")
    if args.youtube:
        sources.append("YouTube")
    print(f"  Sources: {', '.join(sources)}")


if __name__ == "__main__":
    main()
