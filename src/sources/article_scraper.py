"""Automatic preview/review article scraper from trusted gaming outlets.

Searches established gaming publications for pre-release previews and reviews
of games in the registry. Results are saved to data/external/preview_articles.csv
for manual review before use in training.

Supported outlets:
    - IGN (ign.com)
    - PC Gamer (pcgamer.com)
    - Eurogamer (eurogamer.net)
    - Rock Paper Shotgun (rockpapershotgun.com)
    - Destructoid (destructoid.com)

Usage:
    python -m src.sources.article_scraper --training
    python -m src.sources.article_scraper --all
    python -m src.sources.article_scraper --game "Hades"
"""

import csv
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote_plus, urljoin, urlparse

import requests

OUTPUT_FILE = Path("data/external/preview_articles.csv")
OUTPUT_FIELDNAMES = ["app_id", "app_name", "source_url", "publication", "date", "text", "scraped_at"]

# Conservative rate limit for scraping (requests per second per domain)
SCRAPE_RATE = 0.4  # ~1 request every 2.5 seconds per domain

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class ArticleResult:
    app_id: int
    app_name: str
    source_url: str
    publication: str
    date: str        # ISO format or empty string
    text: str
    scraped_at: str  # ISO format


# ---------------------------------------------------------------------------
# BeautifulSoup helper (optional dep — graceful fallback)
# ---------------------------------------------------------------------------

def _get_soup(html: str):
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup(html, "html.parser")
    except ImportError:
        raise ImportError(
            "beautifulsoup4 is required for article scraping.\n"
            "Install it with: pip install beautifulsoup4 lxml"
        )


def _extract_text_from_soup(soup, content_selectors: List[str]) -> str:
    """Try each CSS selector in order, return first non-empty result."""
    for selector in content_selectors:
        container = soup.select_one(selector)
        if container:
            # Remove script, style, nav, aside noise
            for tag in container.find_all(["script", "style", "aside", "nav", "figure"]):
                tag.decompose()
            text = container.get_text(separator=" ", strip=True)
            if len(text) > 200:
                return _clean_text(text)
    return ""


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def _parse_date_str(raw: str) -> str:
    """Try to parse a date string to ISO format YYYY-MM-DD. Returns raw if fails."""
    raw = raw.strip()
    formats = [
        "%B %d, %Y",   # January 15, 2023
        "%b %d, %Y",   # Jan 15, 2023
        "%d %B %Y",    # 15 January 2023
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Try dateutil as last resort
    try:
        from dateutil import parser as dateutil_parser
        return dateutil_parser.parse(raw).strftime("%Y-%m-%d")
    except Exception:
        return raw


# ---------------------------------------------------------------------------
# Per-outlet scrapers
# ---------------------------------------------------------------------------

class BaseOutletScraper:
    name: str
    domain: str
    _last_request: float = 0.0

    def _get(self, url: str, session: requests.Session, timeout: int = 20) -> Optional[requests.Response]:
        """Rate-limited GET with polite delay."""
        elapsed = time.time() - self._last_request
        min_interval = 1.0 / SCRAPE_RATE
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request = time.time()

        try:
            resp = session.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
            if resp.status_code == 200:
                return resp
            print(f"    [{self.name}] HTTP {resp.status_code} for {url}")
            return None
        except requests.RequestException as e:
            print(f"    [{self.name}] Request error: {e}")
            return None

    def search(self, game_name: str, app_id: int, session: requests.Session) -> List[ArticleResult]:
        raise NotImplementedError


class IGNScraper(BaseOutletScraper):
    name = "IGN"
    domain = "ign.com"

    def search(self, game_name: str, app_id: int, session: requests.Session) -> List[ArticleResult]:
        results = []
        query = quote_plus(f"{game_name} preview OR review OR early access")
        search_url = f"https://www.ign.com/search?q={query}&type=article"

        resp = self._get(search_url, session)
        if not resp:
            return results

        soup = _get_soup(resp.text)
        links = []
        # IGN search results: article links in search-results section
        for a in soup.select("a[href]"):
            href = a["href"]
            if not href.startswith("http"):
                href = urljoin("https://www.ign.com", href)
            # Filter to article-like URLs for this game
            if "/articles/" in href or "/reviews/" in href or "/previews/" in href:
                if game_name.split()[0].lower() in href.lower() or game_name.replace(" ", "-").lower() in href.lower():
                    if href not in links:
                        links.append(href)

        for url in links[:5]:
            article = self._scrape_article(url, app_id, game_name, session)
            if article:
                results.append(article)

        return results

    def _scrape_article(self, url: str, app_id: int, game_name: str, session: requests.Session) -> Optional[ArticleResult]:
        resp = self._get(url, session)
        if not resp:
            return None

        soup = _get_soup(resp.text)
        text = _extract_text_from_soup(soup, [
            "article .article-content",
            "div[class*='article-body']",
            "div[class*='content-body']",
            "main article",
            "article",
        ])
        if not text:
            return None

        # Date
        date = ""
        time_tag = soup.find("time")
        if time_tag:
            date = _parse_date_str(time_tag.get("datetime") or time_tag.get_text())

        return ArticleResult(
            app_id=app_id,
            app_name=game_name,
            source_url=url,
            publication="IGN",
            date=date,
            text=text[:8000],
            scraped_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )


class PCGamerScraper(BaseOutletScraper):
    name = "PC Gamer"
    domain = "pcgamer.com"

    def search(self, game_name: str, app_id: int, session: requests.Session) -> List[ArticleResult]:
        results = []
        query = quote_plus(game_name)
        search_url = f"https://www.pcgamer.com/search/?searchTerm={query}"

        resp = self._get(search_url, session)
        if not resp:
            return results

        soup = _get_soup(resp.text)
        links = []
        for a in soup.select("a[href]"):
            href = a["href"]
            if not href.startswith("http"):
                href = urljoin("https://www.pcgamer.com", href)
            # PC Gamer review/preview slugs
            if "pcgamer.com/" in href and ("/review" in href or "/preview" in href or "-review" in href or "-preview" in href):
                if href not in links:
                    links.append(href)

        for url in links[:5]:
            article = self._scrape_article(url, app_id, game_name, session)
            if article:
                results.append(article)

        return results

    def _scrape_article(self, url: str, app_id: int, game_name: str, session: requests.Session) -> Optional[ArticleResult]:
        resp = self._get(url, session)
        if not resp:
            return None

        soup = _get_soup(resp.text)
        text = _extract_text_from_soup(soup, [
            "div.article-body",
            "div[class*='article-body']",
            "div.content",
            "article",
        ])
        if not text:
            return None

        date = ""
        time_tag = soup.find("time")
        if time_tag:
            date = _parse_date_str(time_tag.get("datetime") or time_tag.get_text())

        return ArticleResult(
            app_id=app_id,
            app_name=game_name,
            source_url=url,
            publication="PC Gamer",
            date=date,
            text=text[:8000],
            scraped_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )


class EurogamerScraper(BaseOutletScraper):
    name = "Eurogamer"
    domain = "eurogamer.net"

    def search(self, game_name: str, app_id: int, session: requests.Session) -> List[ArticleResult]:
        results = []
        query = quote_plus(game_name)
        search_url = f"https://www.eurogamer.net/?q={query}"

        resp = self._get(search_url, session)
        if not resp:
            return results

        soup = _get_soup(resp.text)
        links = []
        for a in soup.select("a[href]"):
            href = a["href"]
            if not href.startswith("http"):
                href = urljoin("https://www.eurogamer.net", href)
            if "eurogamer.net" in href and ("/review" in href or "/preview" in href or "/feature" in href):
                if href not in links:
                    links.append(href)

        for url in links[:5]:
            article = self._scrape_article(url, app_id, game_name, session)
            if article:
                results.append(article)

        return results

    def _scrape_article(self, url: str, app_id: int, game_name: str, session: requests.Session) -> Optional[ArticleResult]:
        resp = self._get(url, session)
        if not resp:
            return None

        soup = _get_soup(resp.text)
        text = _extract_text_from_soup(soup, [
            "div.article_body_content",
            "div[class*='article_body']",
            "div[class*='article-body']",
            "article",
        ])
        if not text:
            return None

        date = ""
        time_tag = soup.find("time")
        if time_tag:
            date = _parse_date_str(time_tag.get("datetime") or time_tag.get_text())

        return ArticleResult(
            app_id=app_id,
            app_name=game_name,
            source_url=url,
            publication="Eurogamer",
            date=date,
            text=text[:8000],
            scraped_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )


class RPSScraper(BaseOutletScraper):
    name = "Rock Paper Shotgun"
    domain = "rockpapershotgun.com"

    def search(self, game_name: str, app_id: int, session: requests.Session) -> List[ArticleResult]:
        results = []
        query = quote_plus(game_name)
        search_url = f"https://www.rockpapershotgun.com/?s={query}"

        resp = self._get(search_url, session)
        if not resp:
            return results

        soup = _get_soup(resp.text)
        links = []
        for a in soup.select("a[href]"):
            href = a["href"]
            if not href.startswith("http"):
                href = urljoin("https://www.rockpapershotgun.com", href)
            if "rockpapershotgun.com" in href and any(
                kw in href for kw in ["/review", "/preview", "/impressions", "/hands-on"]
            ):
                if href not in links:
                    links.append(href)

        for url in links[:5]:
            article = self._scrape_article(url, app_id, game_name, session)
            if article:
                results.append(article)

        return results

    def _scrape_article(self, url: str, app_id: int, game_name: str, session: requests.Session) -> Optional[ArticleResult]:
        resp = self._get(url, session)
        if not resp:
            return None

        soup = _get_soup(resp.text)
        text = _extract_text_from_soup(soup, [
            "div.article-content",
            "div[class*='article_body']",
            "div[class*='entry-content']",
            "article",
        ])
        if not text:
            return None

        date = ""
        time_tag = soup.find("time")
        if time_tag:
            date = _parse_date_str(time_tag.get("datetime") or time_tag.get_text())

        return ArticleResult(
            app_id=app_id,
            app_name=game_name,
            source_url=url,
            publication="Rock Paper Shotgun",
            date=date,
            text=text[:8000],
            scraped_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )


class DestructoidScraper(BaseOutletScraper):
    name = "Destructoid"
    domain = "destructoid.com"

    def search(self, game_name: str, app_id: int, session: requests.Session) -> List[ArticleResult]:
        results = []
        query = quote_plus(game_name)
        search_url = f"https://www.destructoid.com/?s={query}"

        resp = self._get(search_url, session)
        if not resp:
            return results

        soup = _get_soup(resp.text)
        links = []
        for a in soup.select("a[href]"):
            href = a["href"]
            if not href.startswith("http"):
                href = urljoin("https://www.destructoid.com", href)
            if "destructoid.com" in href and ("/review" in href or "/preview" in href):
                if href not in links:
                    links.append(href)

        for url in links[:5]:
            article = self._scrape_article(url, app_id, game_name, session)
            if article:
                results.append(article)

        return results

    def _scrape_article(self, url: str, app_id: int, game_name: str, session: requests.Session) -> Optional[ArticleResult]:
        resp = self._get(url, session)
        if not resp:
            return None

        soup = _get_soup(resp.text)
        text = _extract_text_from_soup(soup, [
            "div.article-content",
            "div[class*='c-entry-content']",
            "div[class*='article_body']",
            "article",
        ])
        if not text:
            return None

        date = ""
        time_tag = soup.find("time")
        if time_tag:
            date = _parse_date_str(time_tag.get("datetime") or time_tag.get_text())

        return ArticleResult(
            app_id=app_id,
            app_name=game_name,
            source_url=url,
            publication="Destructoid",
            date=date,
            text=text[:8000],
            scraped_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

ALL_SCRAPERS = [
    IGNScraper(),
    PCGamerScraper(),
    EurogamerScraper(),
    RPSScraper(),
    DestructoidScraper(),
]


def scrape_articles_for_games(
    games: List[dict],
    release_dates: Optional[dict] = None,
    pre_release_only: bool = False,
    output_file: Path = OUTPUT_FILE,
) -> List[ArticleResult]:
    """Scrape preview/review articles for a list of games.

    Args:
        games: List of dicts with 'app_id' and 'name' keys.
        release_dates: Optional dict mapping app_id -> ISO date string.
        pre_release_only: If True, skip articles published after release date.
        output_file: Path to CSV output file.

    Returns:
        List of ArticleResult instances collected.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing URLs to avoid duplicates
    existing_urls: set = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_urls.add(row.get("source_url", ""))
        print(f"Found {len(existing_urls)} existing articles in {output_file}")

    session = requests.Session()
    session.headers.update(HEADERS)

    all_new: List[ArticleResult] = []

    for game in games:
        app_id = game["app_id"]
        game_name = game["name"]
        release_date = (release_dates or {}).get(app_id)

        print(f"\n[Articles] {game_name} (app_id={app_id})")

        for scraper in ALL_SCRAPERS:
            print(f"  Searching {scraper.name}...")
            try:
                results = scraper.search(game_name, app_id, session)
            except Exception as e:
                print(f"    {scraper.name} error: {e}")
                continue

            new_for_outlet = 0
            for article in results:
                if article.source_url in existing_urls:
                    continue
                # Pre-release filter
                if pre_release_only and release_date and article.date:
                    try:
                        art_date = datetime.strptime(article.date[:10], "%Y-%m-%d")
                        rel_date = datetime.strptime(release_date[:10], "%Y-%m-%d")
                        if art_date >= rel_date:
                            continue
                    except ValueError:
                        pass

                existing_urls.add(article.source_url)
                all_new.append(article)
                new_for_outlet += 1

            print(f"    {new_for_outlet} new articles")

    if all_new:
        _save_articles(all_new, output_file, append=output_file.exists())
        print(f"\nSaved {len(all_new)} new articles to {output_file}")
    else:
        print("\nNo new articles collected.")

    return all_new


def _save_articles(articles: List[ArticleResult], path: Path, append: bool = False):
    mode = "a" if append else "w"
    write_header = not (append and path.exists() and path.stat().st_size > 0)

    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for a in articles:
            writer.writerow({
                "app_id": a.app_id,
                "app_name": a.app_name,
                "source_url": a.source_url,
                "publication": a.publication,
                "date": a.date,
                "text": a.text,
                "scraped_at": a.scraped_at,
            })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Scrape preview articles from trusted gaming outlets")
    parser.add_argument("--training", action="store_true", help="Scrape for training games")
    parser.add_argument("--validation", action="store_true", help="Scrape for validation games")
    parser.add_argument("--all", action="store_true", help="Scrape for all games")
    parser.add_argument("--game", type=str, help="Scrape for a single game by name")
    parser.add_argument("--pre-release-only", action="store_true", help="Only keep pre-release articles")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE), help="Output CSV path")
    args = parser.parse_args()

    registry_path = Path("data/game_registry.json")
    with open(registry_path) as f:
        registry = json.load(f)

    # Load release dates from metadata
    release_dates = {}
    metadata_path = Path("data/game_metadata.csv")
    if metadata_path.exists():
        with open(metadata_path) as f:
            for row in csv.DictReader(f):
                try:
                    release_dates[int(row["app_id"])] = row.get("release_date", "")
                except (ValueError, KeyError):
                    pass

    games = []
    if args.game:
        name_lower = args.game.lower()
        for g in registry["training_games"] + registry["validation_games"]:
            if name_lower in g["name"].lower():
                games.append(g)
        if not games:
            print(f"No game found matching '{args.game}'")
            return
    elif args.all or (not args.training and not args.validation):
        games = registry["training_games"] + registry["validation_games"]
    else:
        if args.training:
            games += registry["training_games"]
        if args.validation:
            games += registry["validation_games"]

    print(f"Scraping articles for {len(games)} games from {len(ALL_SCRAPERS)} outlets...")
    scrape_articles_for_games(
        games,
        release_dates=release_dates,
        pre_release_only=args.pre_release_only,
        output_file=Path(args.output),
    )


if __name__ == "__main__":
    main()
