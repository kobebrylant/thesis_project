"""Preview article / critic content data source.

Reads from a manually curated CSV file since professional preview articles
are behind varied HTML structures and paywalls, making automated scraping fragile.

Expected CSV format (data/external/preview_articles.csv):
    app_id,app_name,source_url,publication,date,text

Example row:
    1086940,Baldur's Gate 3,https://example.com/bg3-preview,IGN,2023-06-15,"The early access..."
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

from .base_source import BaseSource, SourceRecord


DEFAULT_PREVIEW_FILE = Path("data/external/preview_articles.csv")


class PreviewSource(BaseSource):
    """Load pre-release critic previews from a curated CSV."""

    def __init__(self, csv_path: Optional[Path] = None):
        self.csv_path = csv_path or DEFAULT_PREVIEW_FILE

    @property
    def source_name(self) -> str:
        return "preview"

    def fetch(
        self,
        app_id: int,
        app_name: str,
        release_date: Optional[str] = None,
    ) -> List[SourceRecord]:
        """Load preview articles for a specific game from CSV."""
        if not self.csv_path.exists():
            return []

        records = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row.get("app_id", 0)) != app_id:
                    continue

                timestamp = row.get("date", "")
                is_pre = self.is_before_release(timestamp, release_date) if timestamp else True

                records.append(
                    SourceRecord(
                        app_id=app_id,
                        app_name=app_name,
                        text=row.get("text", ""),
                        sentiment_source="preview_article",
                        timestamp=timestamp,
                        is_pre_release=is_pre,
                        author_type="critic",
                        source_url=row.get("source_url", ""),
                    )
                )

        return records

    def fetch_all(self) -> List[SourceRecord]:
        """Load all preview articles from CSV (all games)."""
        if not self.csv_path.exists():
            print(
                f"Preview articles CSV not found at {self.csv_path}\n"
                f"Create it with columns: app_id,app_name,source_url,publication,date,text"
            )
            return []

        records = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(
                    SourceRecord(
                        app_id=int(row.get("app_id", 0)),
                        app_name=row.get("app_name", ""),
                        text=row.get("text", ""),
                        sentiment_source="preview_article",
                        timestamp=row.get("date", ""),
                        is_pre_release=True,
                        author_type="critic",
                        source_url=row.get("source_url", ""),
                    )
                )

        print(f"Loaded {len(records)} preview articles from {self.csv_path}")
        return records
