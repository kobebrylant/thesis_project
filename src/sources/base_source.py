"""Abstract base class for all data sources.

All sources produce SourceRecord instances with a unified schema,
regardless of whether they come from Steam, Reddit, YouTube, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class SourceRecord:
    """Unified schema for text content from any source.

    This is the common output format that all sources produce.
    The data pipeline merges these into a single dataset.
    """
    app_id: int
    app_name: str
    text: str
    sentiment_source: str  # steam_review, reddit_post, reddit_comment, youtube_comment, preview_article
    timestamp: Optional[str] = None  # ISO format datetime
    is_pre_release: Optional[bool] = None
    author_type: str = "player"  # player, critic, community
    source_url: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class BaseSource(ABC):
    """Abstract base for data sources.

    Subclasses implement fetch() to collect raw data from their API,
    and transform() to convert it to SourceRecord instances.
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Identifier for this source (e.g., 'steam', 'reddit', 'youtube')."""
        ...

    @abstractmethod
    def fetch(self, app_id: int, app_name: str, **kwargs) -> List[SourceRecord]:
        """Fetch and transform data for a single game.

        Args:
            app_id: Steam App ID (canonical game identifier)
            app_name: Game name for search queries

        Returns:
            List of SourceRecord instances
        """
        ...

    def fetch_batch(self, games: List[Dict], **kwargs) -> List[SourceRecord]:
        """Fetch data for multiple games.

        Default implementation iterates; subclasses may optimize.
        """
        all_records = []
        for game in games:
            try:
                records = self.fetch(
                    app_id=game["app_id"],
                    app_name=game["name"],
                    **kwargs,
                )
                all_records.extend(records)
                print(f"  [{self.source_name}] {game['name']}: {len(records)} records")
            except Exception as e:
                print(f"  [{self.source_name}] {game['name']}: ERROR - {e}")
        return all_records

    @staticmethod
    def is_before_release(timestamp_str: Optional[str], release_date_str: Optional[str]) -> Optional[bool]:
        """Determine if content was created before game release."""
        if not timestamp_str or not release_date_str:
            return None
        try:
            content_date = datetime.fromisoformat(timestamp_str)
            release_date = datetime.fromisoformat(release_date_str)
            return content_date < release_date
        except (ValueError, TypeError):
            return None
