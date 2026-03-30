"""Collect market success metrics from SteamSpy, Steam Store, and SteamCharts."""
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import csv
import re
import time

import requests

# Import game lists from data_gatherer
from data_gatherer import load_game_registry

_registry = load_game_registry()
EARLY_ACCESS_GAMES = _registry["training_games"]
VALIDATION_GAMES = _registry["validation_games"]


DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / "game_success_metrics.csv"
VALIDATION_DATA_DIR = Path("data/validation")
VALIDATION_OUTPUT_FILE = VALIDATION_DATA_DIR / "game_success_metrics.csv"


@dataclass
class GameSuccessMetrics:
    """Container for all success metrics per game."""
    app_id: int
    app_name: str
    collection_date: str
    # SteamSpy data
    steamspy_owners_min: Optional[int] = None
    steamspy_owners_max: Optional[int] = None
    steamspy_players_forever: Optional[int] = None
    steamspy_avg_playtime: Optional[int] = None  # in minutes
    steamspy_positive: Optional[int] = None
    steamspy_negative: Optional[int] = None
    # Steam Store data
    steam_metacritic_score: Optional[int] = None
    steam_total_reviews: Optional[int] = None
    steam_release_date: Optional[str] = None
    steam_current_price_usd: Optional[float] = None
    # SteamCharts data
    steamcharts_peak_all_time: Optional[int] = None
    # Derived metrics
    estimated_revenue_usd: Optional[float] = None
    success_tier: Optional[str] = None  # Flop, Moderate, Hit, Blockbuster
    # Status tracking
    data_complete: bool = False
    last_error: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def owners_midpoint(self) -> Optional[int]:
        """Calculate midpoint of estimated owners range."""
        if self.steamspy_owners_min is not None and self.steamspy_owners_max is not None:
            return (self.steamspy_owners_min + self.steamspy_owners_max) // 2
        return None

    @property
    def review_score(self) -> Optional[float]:
        """Calculate positive review ratio."""
        if self.steamspy_positive is not None and self.steamspy_negative is not None:
            total = self.steamspy_positive + self.steamspy_negative
            if total > 0:
                return self.steamspy_positive / total
        return None

    def compute_estimated_revenue(self, review_multiplier: float = 30.0):
        """Compute estimated revenue using Boxleiter method.

        Formula: total_reviews × review_multiplier × price
        The review_multiplier (default 30) represents the ratio of buyers
        to reviewers, per GameDiscoverCo / Simon Carless research.

        NOTE: current_price_usd values were audited and corrected (2026-03-29).
        All Steam Store API calls now use cc=us&l=en to ensure USD prices.
        Any future metadata update should validate prices via the cc=us parameter.
        """
        if self.steam_total_reviews and self.steam_current_price_usd is not None:
            estimated_sales = self.steam_total_reviews * review_multiplier
            self.estimated_revenue_usd = estimated_sales * self.steam_current_price_usd
        elif self.steam_total_reviews and self.steam_current_price_usd == 0:
            # Free-to-play: revenue = 0 from base sales
            self.estimated_revenue_usd = 0.0

    def compute_success_tier(self):
        """Classify game into success tier based on estimated revenue.

        Tiers (USD):
            Flop:        < $100,000
            Moderate:    $100,000 - $1,000,000
            Hit:         $1,000,000 - $10,000,000
            Blockbuster: > $10,000,000
        """
        if self.estimated_revenue_usd is None:
            return
        rev = self.estimated_revenue_usd
        if rev < 100_000:
            self.success_tier = "Flop"
        elif rev < 1_000_000:
            self.success_tier = "Moderate"
        elif rev < 10_000_000:
            self.success_tier = "Hit"
        else:
            self.success_tier = "Blockbuster"


class SteamSpyCollector:
    """Collect data from SteamSpy API."""

    BASE_URL = "https://steamspy.com/api.php"
    RATE_LIMIT_SECONDS = 1.0

    def __init__(self):
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.RATE_LIMIT_SECONDS:
            time.sleep(self.RATE_LIMIT_SECONDS - elapsed)
        self.last_request_time = time.time()

    def get_app_details(self, app_id: int) -> Optional[Dict]:
        """Get app details from SteamSpy.

        Returns dict with: owners, players_forever, average_forever, positive, negative
        """
        self._rate_limit()

        url = f"{self.BASE_URL}?request=appdetails&appid={app_id}"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data or 'appid' not in data:
                return None

            return data
        except requests.RequestException as e:
            print(f"  SteamSpy error for {app_id}: {e}")
            return None

    def parse_owners(self, owners_str: str) -> tuple:
        """Parse owners string like '1,000,000 .. 2,000,000' into (min, max)."""
        if not owners_str or owners_str == "0":
            return (None, None)

        # Remove commas and parse
        owners_str = owners_str.replace(",", "")
        match = re.search(r'(\d+)\s*\.\.\s*(\d+)', owners_str)
        if match:
            return (int(match.group(1)), int(match.group(2)))

        # Single value
        try:
            val = int(owners_str)
            return (val, val)
        except ValueError:
            return (None, None)


class SteamStoreCollector:
    """Collect data from Steam Store API."""

    BASE_URL = "https://store.steampowered.com/api/appdetails"
    RATE_LIMIT_SECONDS = 0.5

    def __init__(self):
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.RATE_LIMIT_SECONDS:
            time.sleep(self.RATE_LIMIT_SECONDS - elapsed)
        self.last_request_time = time.time()

    def get_app_details(self, app_id: int) -> Optional[Dict]:
        """Get app details from Steam Store.

        Returns dict with: metacritic score, total reviews, release date
        """
        self._rate_limit()

        url = f"{self.BASE_URL}?appids={app_id}&cc=us&l=en"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            app_data = data.get(str(app_id), {})
            if not app_data.get('success'):
                return None

            return app_data.get('data', {})
        except requests.RequestException as e:
            print(f"  Steam Store error for {app_id}: {e}")
            return None


class SteamChartsCollector:
    """Collect data from SteamCharts (web scraping)."""

    BASE_URL = "https://steamcharts.com/app"
    RATE_LIMIT_SECONDS = 2.0

    def __init__(self):
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.RATE_LIMIT_SECONDS:
            time.sleep(self.RATE_LIMIT_SECONDS - elapsed)
        self.last_request_time = time.time()

    def get_peak_players(self, app_id: int) -> Optional[int]:
        """Get all-time peak concurrent players from SteamCharts.

        Returns peak player count or None if unavailable.
        """
        self._rate_limit()

        url = f"{self.BASE_URL}/{app_id}"
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 404:
                return None

            response.raise_for_status()

            # Look for "All-Time Peak" pattern in the HTML
            # Pattern: <span class="num">123,456</span>
            content = response.text

            # Look for the all-time peak section
            peak_match = re.search(
                r'All-Time Peak.*?<span[^>]*class="num"[^>]*>([\d,]+)</span>',
                content,
                re.IGNORECASE | re.DOTALL
            )

            if peak_match:
                peak_str = peak_match.group(1).replace(",", "")
                return int(peak_str)

            return None

        except requests.RequestException as e:
            print(f"  SteamCharts error for {app_id}: {e}")
            return None
        except ValueError as e:
            print(f"  SteamCharts parse error for {app_id}: {e}")
            return None


class SuccessMetricsCollector:
    """Coordinate collection from all sources."""

    def __init__(self, skip_steamcharts: bool = False):
        self.steamspy = SteamSpyCollector()
        self.steam_store = SteamStoreCollector()
        self.steamcharts = SteamChartsCollector() if not skip_steamcharts else None
        self.skip_steamcharts = skip_steamcharts

    def collect_for_game(self, app_id: int, app_name: str) -> GameSuccessMetrics:
        """Collect all available metrics for a single game."""
        metrics = GameSuccessMetrics(
            app_id=app_id,
            app_name=app_name,
            collection_date=datetime.now().strftime("%Y-%m-%d"),
        )

        errors = []

        # Collect from SteamSpy
        print(f"  Fetching SteamSpy data...")
        steamspy_data = self.steamspy.get_app_details(app_id)
        if steamspy_data:
            # Parse owners
            owners_str = steamspy_data.get('owners', '')
            owners_min, owners_max = self.steamspy.parse_owners(owners_str)
            metrics.steamspy_owners_min = owners_min
            metrics.steamspy_owners_max = owners_max

            # Other metrics
            metrics.steamspy_players_forever = steamspy_data.get('players_forever')
            metrics.steamspy_avg_playtime = steamspy_data.get('average_forever')
            metrics.steamspy_positive = steamspy_data.get('positive')
            metrics.steamspy_negative = steamspy_data.get('negative')
        else:
            errors.append("SteamSpy failed")

        # Collect from Steam Store
        print(f"  Fetching Steam Store data...")
        store_data = self.steam_store.get_app_details(app_id)
        if store_data:
            # Metacritic
            metacritic = store_data.get('metacritic', {})
            metrics.steam_metacritic_score = metacritic.get('score')

            # Total reviews (recommendations)
            recs = store_data.get('recommendations', {})
            metrics.steam_total_reviews = recs.get('total')

            # Release date
            release_info = store_data.get('release_date', {})
            metrics.steam_release_date = release_info.get('date')

            # Price (in cents from API, convert to USD)
            price_info = store_data.get('price_overview', {})
            if price_info:
                metrics.steam_current_price_usd = price_info.get('initial', 0) / 100
            elif store_data.get('is_free'):
                metrics.steam_current_price_usd = 0.0
        else:
            errors.append("Steam Store failed")

        # Collect from SteamCharts (optional)
        if not self.skip_steamcharts:
            print(f"  Fetching SteamCharts data...")
            peak = self.steamcharts.get_peak_players(app_id)
            if peak is not None:
                metrics.steamcharts_peak_all_time = peak
            else:
                errors.append("SteamCharts failed")

        # Compute derived metrics
        metrics.compute_estimated_revenue()
        metrics.compute_success_tier()

        # Determine completeness
        has_steamspy = metrics.steamspy_owners_min is not None
        has_store = metrics.steam_total_reviews is not None or metrics.steam_metacritic_score is not None
        metrics.data_complete = has_steamspy and has_store

        if errors:
            metrics.last_error = "; ".join(errors)

        return metrics

    def collect_all(
        self,
        games: List[Dict],
        resume: bool = True,
        output_file: Path = OUTPUT_FILE,
    ) -> List[GameSuccessMetrics]:
        """Collect metrics for all games.

        Args:
            games: List of game dicts with 'app_id' and 'name'
            resume: Skip games already in output file
            output_file: Where to save results

        Returns:
            List of GameSuccessMetrics
        """
        # Load existing data if resuming
        existing = {}
        if resume and output_file.exists():
            existing = self._load_existing(output_file)
            print(f"Found existing data for {len(existing)} games")

        results = []
        total = len(games)

        for i, game in enumerate(games, 1):
            app_id = game['app_id']
            app_name = game['name']

            # Skip if already collected
            if app_id in existing:
                print(f"[{i}/{total}] Skipping {app_name} (already collected)")
                results.append(existing[app_id])
                continue

            print(f"\n[{i}/{total}] Collecting metrics for {app_name} (ID: {app_id})")

            try:
                metrics = self.collect_for_game(app_id, app_name)
                results.append(metrics)

                # Save incrementally
                self._save_results(results, output_file)
                print(f"  Complete: {metrics.data_complete}, Errors: {metrics.last_error or 'None'}")

            except Exception as e:
                print(f"  ERROR: {e}")
                # Create error entry
                metrics = GameSuccessMetrics(
                    app_id=app_id,
                    app_name=app_name,
                    collection_date=datetime.now().strftime("%Y-%m-%d"),
                    data_complete=False,
                    last_error=str(e),
                )
                results.append(metrics)
                self._save_results(results, output_file)

        return results

    def _load_existing(self, output_file: Path = OUTPUT_FILE) -> Dict[int, GameSuccessMetrics]:
        """Load existing metrics from CSV."""
        existing = {}
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    app_id = int(row['app_id'])

                    # Convert types
                    metrics = GameSuccessMetrics(
                        app_id=app_id,
                        app_name=row['app_name'],
                        collection_date=row['collection_date'],
                        steamspy_owners_min=int(row['steamspy_owners_min']) if row.get('steamspy_owners_min') else None,
                        steamspy_owners_max=int(row['steamspy_owners_max']) if row.get('steamspy_owners_max') else None,
                        steamspy_players_forever=int(row['steamspy_players_forever']) if row.get('steamspy_players_forever') else None,
                        steamspy_avg_playtime=int(row['steamspy_avg_playtime']) if row.get('steamspy_avg_playtime') else None,
                        steamspy_positive=int(row['steamspy_positive']) if row.get('steamspy_positive') else None,
                        steamspy_negative=int(row['steamspy_negative']) if row.get('steamspy_negative') else None,
                        steam_metacritic_score=int(row['steam_metacritic_score']) if row.get('steam_metacritic_score') else None,
                        steam_total_reviews=int(row['steam_total_reviews']) if row.get('steam_total_reviews') else None,
                        steam_release_date=row.get('steam_release_date') or None,
                        steam_current_price_usd=float(row['steam_current_price_usd']) if row.get('steam_current_price_usd') else None,
                        steamcharts_peak_all_time=int(row['steamcharts_peak_all_time']) if row.get('steamcharts_peak_all_time') else None,
                        estimated_revenue_usd=float(row['estimated_revenue_usd']) if row.get('estimated_revenue_usd') else None,
                        success_tier=row.get('success_tier') or None,
                        data_complete=row.get('data_complete', '').lower() == 'true',
                        last_error=row.get('last_error') or None,
                    )
                    existing[app_id] = metrics
        except Exception as e:
            print(f"Warning: Could not load existing data: {e}")

        return existing

    def _save_results(self, results: List[GameSuccessMetrics], output_file: Path = OUTPUT_FILE):
        """Save results to CSV."""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            'app_id', 'app_name', 'collection_date',
            'steamspy_owners_min', 'steamspy_owners_max',
            'steamspy_players_forever', 'steamspy_avg_playtime',
            'steamspy_positive', 'steamspy_negative',
            'steam_metacritic_score', 'steam_total_reviews', 'steam_release_date',
            'steam_current_price_usd',
            'steamcharts_peak_all_time',
            'estimated_revenue_usd', 'success_tier',
            'data_complete', 'last_error',
        ]

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for metrics in results:
                row = metrics.to_dict()
                # Convert None to empty string for CSV
                row = {k: ('' if v is None else v) for k, v in row.items()}
                writer.writerow(row)


def collect_training_metrics():
    """Collect metrics for training games."""
    print("=" * 60)
    print("TRAINING GAMES - SUCCESS METRICS")
    print("=" * 60)
    print(f"\nCollecting metrics for {len(EARLY_ACCESS_GAMES)} training games...")
    print(f"Output file: {OUTPUT_FILE}")

    collector = SuccessMetricsCollector(skip_steamcharts=False)
    results = collector.collect_all(EARLY_ACCESS_GAMES, resume=True, output_file=OUTPUT_FILE)

    _print_summary(results, OUTPUT_FILE)
    return results


def collect_validation_metrics():
    """Collect metrics for validation games (NEW, unseen during training)."""
    print("\n" + "=" * 60)
    print("VALIDATION GAMES - SUCCESS METRICS")
    print("=" * 60)
    print("\nThese are NEW games never seen during model training.")
    print(f"Collecting metrics for {len(VALIDATION_GAMES)} validation games...")
    print(f"Output file: {VALIDATION_OUTPUT_FILE}")

    collector = SuccessMetricsCollector(skip_steamcharts=False)
    results = collector.collect_all(VALIDATION_GAMES, resume=True, output_file=VALIDATION_OUTPUT_FILE)

    _print_summary(results, VALIDATION_OUTPUT_FILE)
    return results


def _print_summary(results: List[GameSuccessMetrics], output_file: Path):
    """Print collection summary."""
    complete = sum(1 for r in results if r.data_complete)
    has_metacritic = sum(1 for r in results if r.steam_metacritic_score is not None)
    has_owners = sum(1 for r in results if r.steamspy_owners_min is not None)
    has_peak = sum(1 for r in results if r.steamcharts_peak_all_time is not None)
    has_revenue = sum(1 for r in results if r.estimated_revenue_usd is not None)

    # Count success tiers
    tier_counts = {}
    for r in results:
        if r.success_tier:
            tier_counts[r.success_tier] = tier_counts.get(r.success_tier, 0) + 1

    print("\n" + "-" * 40)
    print("COLLECTION SUMMARY")
    print("-" * 40)
    print(f"Total games: {len(results)}")
    print(f"Complete data: {complete}")
    print(f"Has Metacritic: {has_metacritic}")
    print(f"Has Owner estimates: {has_owners}")
    print(f"Has Peak Players: {has_peak}")
    print(f"Has Estimated Revenue: {has_revenue}")
    if tier_counts:
        print(f"Success Tiers: {tier_counts}")
    print(f"Results saved to: {output_file}")


def main():
    """Main entry point for success metrics collection."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect Steam success metrics for games"
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="Collect metrics for training games",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Collect metrics for validation games (NEW games for thesis)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Collect metrics for both training and validation games",
    )

    args = parser.parse_args()

    # Default: collect validation metrics only
    if not args.training and not args.validation and not args.all:
        print("No arguments specified. Use --help for options.")
        print("Defaulting to --validation (new games for thesis)\n")
        args.validation = True

    if args.all or args.training:
        collect_training_metrics()

    if args.all or args.validation:
        collect_validation_metrics()

    print("\n" + "=" * 60)
    print("ALL COLLECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
