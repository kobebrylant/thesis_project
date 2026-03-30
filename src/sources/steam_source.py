"""Steam review data source.

Wraps the existing data_gatherer functionality into the BaseSource interface.
This is the primary data source for the project.
"""

from typing import Dict, List

from ..api_client import ResilientAPIClient
from .base_source import BaseSource, SourceRecord


class SteamSource(BaseSource):
    """Collect Steam user reviews via the Steam Store Reviews API."""

    def __init__(self, client: ResilientAPIClient = None):
        self.client = client or ResilientAPIClient()

    @property
    def source_name(self) -> str:
        return "steam"

    def fetch(
        self,
        app_id: int,
        app_name: str,
        start_date: int = None,
        end_date: int = None,
        max_reviews: int = 10000,
        review_type: str = "early_access",
    ) -> List[SourceRecord]:
        """Fetch reviews from Steam.

        Args:
            app_id: Steam App ID
            app_name: Game name
            start_date: Unix timestamp for EA start
            end_date: Unix timestamp for EA end
            max_reviews: Maximum reviews to collect
            review_type: "early_access" or "post_release"
        """
        from data_gatherer import (
            collect_early_access_reviews,
            collect_post_release_reviews,
        )

        if review_type == "early_access" and start_date and end_date:
            raw_reviews = collect_early_access_reviews(
                client=self.client,
                app_id=app_id,
                app_name=app_name,
                start_date=start_date,
                end_date=end_date,
                max_reviews=max_reviews,
            )
        else:
            raw_reviews = collect_post_release_reviews(
                client=self.client,
                app_id=app_id,
                app_name=app_name,
                max_reviews=max_reviews,
            )

        records = []
        for review in raw_reviews:
            is_pre = review_type == "early_access"
            records.append(
                SourceRecord(
                    app_id=app_id,
                    app_name=app_name,
                    text=review.get("review_text", ""),
                    sentiment_source=f"steam_{review_type}",
                    is_pre_release=is_pre,
                    author_type="player",
                )
            )

        return records
