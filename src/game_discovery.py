"""Programmatic game discovery for expanding the training/validation dataset.

Uses SteamSpy API to find Early Access games that have left EA,
filters by review count, and balances Indie/AAA mix.

Usage:
    python -m src.game_discovery --discover --min-reviews 500
    python -m src.game_discovery --show-candidates
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set

from .api_client import ResilientAPIClient


GAME_REGISTRY_PATH = Path("data/game_registry.json")

# Owner count thresholds for Indie vs AAA classification
INDIE_MAX_OWNERS = 500_000
AAA_MIN_OWNERS = 500_000


def load_existing_app_ids() -> Set[int]:
    """Get all app_ids already in the registry."""
    if not GAME_REGISTRY_PATH.exists():
        return set()
    with open(GAME_REGISTRY_PATH, "r") as f:
        registry = json.load(f)
    ids = set()
    for key in ("training_games", "validation_games"):
        for game in registry.get(key, []):
            ids.add(game["app_id"])
    return ids


def discover_ea_games(
    client: ResilientAPIClient,
    min_reviews: int = 500,
    max_candidates: int = 100,
    target_indie_ratio: float = 0.6,
) -> List[Dict]:
    """Discover Early Access games from SteamSpy.

    Strategy:
        1. Query SteamSpy genre=Early+Access for top games by owners
        2. Filter by minimum review count
        3. Exclude games already in registry
        4. Balance Indie/AAA by owner count

    Returns list of candidate game dicts.
    """
    existing_ids = load_existing_app_ids()
    candidates = []

    print("Querying SteamSpy for Early Access games...")
    url = "https://steamspy.com/api.php"

    # SteamSpy doesn't have a perfect EA-specific endpoint,
    # but we can query by tag
    for page in range(0, 5):
        params = {
            "request": "tag",
            "tag": "Early Access",
        }

        try:
            response = client.get(url, params=params)
            if response.status_code != 200:
                print(f"  SteamSpy returned HTTP {response.status_code}")
                break

            data = response.json()
            if not data:
                break

            for app_id_str, game_data in data.items():
                try:
                    app_id = int(app_id_str)
                except ValueError:
                    continue

                if app_id in existing_ids:
                    continue

                name = game_data.get("name", "Unknown")
                positive = game_data.get("positive", 0) or 0
                negative = game_data.get("negative", 0) or 0
                total_reviews = positive + negative

                if total_reviews < min_reviews:
                    continue

                # Parse owner range
                owners_str = game_data.get("owners", "0 .. 0")
                owners_str_clean = owners_str.replace(",", "")
                parts = owners_str_clean.split(" .. ")
                try:
                    owners_min = int(parts[0])
                    owners_max = int(parts[1]) if len(parts) > 1 else owners_min
                    owners_mid = (owners_min + owners_max) // 2
                except (ValueError, IndexError):
                    owners_mid = 0

                category = "indie" if owners_mid < AAA_MIN_OWNERS else "aaa"

                candidates.append({
                    "app_id": app_id,
                    "name": name,
                    "total_reviews": total_reviews,
                    "positive_reviews": positive,
                    "owners_midpoint": owners_mid,
                    "category": category,
                })

        except Exception as e:
            print(f"  Error querying SteamSpy: {e}")
            break

    # Sort by total reviews (most data first)
    candidates.sort(key=lambda x: x["total_reviews"], reverse=True)

    # Balance Indie/AAA
    indie_target = int(max_candidates * target_indie_ratio)
    aaa_target = max_candidates - indie_target

    indie_games = [c for c in candidates if c["category"] == "indie"][:indie_target]
    aaa_games = [c for c in candidates if c["category"] == "aaa"][:aaa_target]

    balanced = indie_games + aaa_games
    balanced.sort(key=lambda x: x["total_reviews"], reverse=True)

    print(f"\nFound {len(candidates)} candidates total")
    print(f"After balancing: {len(indie_games)} indie + {len(aaa_games)} AAA = {len(balanced)}")

    return balanced


def add_to_registry(
    candidates: List[Dict],
    target: str = "training_games",
):
    """Add discovered games to the registry JSON.

    Note: start_date and end_date must be manually verified
    before the games can be used for data collection.
    """
    if not GAME_REGISTRY_PATH.exists():
        registry = {"training_games": [], "validation_games": []}
    else:
        with open(GAME_REGISTRY_PATH, "r") as f:
            registry = json.load(f)

    existing_ids = {g["app_id"] for g in registry.get(target, [])}

    added = 0
    for candidate in candidates:
        if candidate["app_id"] in existing_ids:
            continue

        registry[target].append({
            "app_id": candidate["app_id"],
            "name": candidate["name"],
            "start_date": None,  # Must be filled manually
            "end_date": None,    # Must be filled manually
            "_discovered": True,
            "_owners_midpoint": candidate.get("owners_midpoint"),
            "_total_reviews": candidate.get("total_reviews"),
            "_category": candidate.get("category"),
        })
        added += 1

    with open(GAME_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"Added {added} new games to '{target}' in {GAME_REGISTRY_PATH}")
    print("NOTE: start_date and end_date must be manually set before data collection.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Discover EA games for dataset expansion")
    parser.add_argument("--discover", action="store_true", help="Run discovery")
    parser.add_argument("--min-reviews", type=int, default=500, help="Min review count filter")
    parser.add_argument("--max-candidates", type=int, default=50, help="Max games to discover")
    parser.add_argument("--add", action="store_true", help="Add candidates to registry")
    parser.add_argument("--show-candidates", action="store_true", help="Show current candidates")
    args = parser.parse_args()

    if args.discover or args.show_candidates:
        with ResilientAPIClient() as client:
            candidates = discover_ea_games(
                client,
                min_reviews=args.min_reviews,
                max_candidates=args.max_candidates,
            )

        if candidates:
            print(f"\nTop candidates:")
            for c in candidates[:20]:
                print(
                    f"  {c['name']} (ID: {c['app_id']}): "
                    f"{c['total_reviews']} reviews, "
                    f"~{c['owners_midpoint']:,} owners, "
                    f"{c['category']}"
                )

        if args.add and candidates:
            add_to_registry(candidates)
    else:
        parser.print_help()
