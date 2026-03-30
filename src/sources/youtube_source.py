"""YouTube comment data source via YouTube Data API v3.

Collects comments from game reveal/gameplay trailers.

Setup:
    pip install google-api-python-client
    Set environment variable: YOUTUBE_API_KEY
    Get a key at: https://console.cloud.google.com/apis/credentials

API Quota: 10,000 units/day
    - Search: 100 units per request
    - Comment threads: 1 unit per request
    - Budget: ~50 games per day (2 searches + 50 comment pages each)
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

from .base_source import BaseSource, SourceRecord


class YouTubeSource(BaseSource):
    """Collect comments from game trailers on YouTube."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("YOUTUBE_API_KEY")
        self._service = None

    @property
    def source_name(self) -> str:
        return "youtube"

    def _get_service(self):
        """Lazy-initialize YouTube API client."""
        if self._service is not None:
            return self._service

        if not self.api_key:
            raise RuntimeError(
                "YouTube API key not configured.\n"
                "Set YOUTUBE_API_KEY environment variable.\n"
                "Get a key at: https://console.cloud.google.com/apis/credentials"
            )

        try:
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError(
                "google-api-python-client is required for YouTube data collection.\n"
                "Install it with: pip install google-api-python-client"
            )

        self._service = build("youtube", "v3", developerKey=self.api_key)
        return self._service

    def _search_trailer(self, game_name: str, max_results: int = 3) -> List[str]:
        """Search for official game trailers, return video IDs."""
        service = self._get_service()

        query = f"{game_name} official trailer"
        request = service.search().list(
            q=query,
            part="id",
            type="video",
            maxResults=max_results,
            order="relevance",
        )
        response = request.execute()

        video_ids = []
        for item in response.get("items", []):
            video_ids.append(item["id"]["videoId"])

        return video_ids

    def _get_comments(
        self, video_id: str, max_comments: int = 100
    ) -> List[Dict]:
        """Fetch top-level comment threads for a video."""
        service = self._get_service()
        comments = []
        page_token = None

        while len(comments) < max_comments:
            request = service.commentThreads().list(
                videoId=video_id,
                part="snippet",
                maxResults=min(100, max_comments - len(comments)),
                order="relevance",
                pageToken=page_token,
                textFormat="plainText",
            )

            try:
                response = request.execute()
            except Exception as e:
                print(f"    Comments disabled or error for video {video_id}: {e}")
                break

            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "text": snippet.get("textDisplay", ""),
                    "published_at": snippet.get("publishedAt", ""),
                    "like_count": snippet.get("likeCount", 0),
                })

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        return comments

    def fetch(
        self,
        app_id: int,
        app_name: str,
        max_videos: int = 3,
        max_comments_per_video: int = 100,
        release_date: Optional[str] = None,
    ) -> List[SourceRecord]:
        """Search for game trailers and collect comments.

        Args:
            app_id: Steam App ID
            app_name: Game name (used as search query)
            max_videos: Number of trailer videos to search
            max_comments_per_video: Max comments per video
            release_date: ISO date for pre-release filtering
        """
        video_ids = self._search_trailer(app_name, max_results=max_videos)
        records = []

        for video_id in video_ids:
            comments = self._get_comments(video_id, max_comments=max_comments_per_video)

            for comment in comments:
                timestamp = comment.get("published_at", "")
                is_pre = self.is_before_release(timestamp, release_date) if timestamp else None

                records.append(
                    SourceRecord(
                        app_id=app_id,
                        app_name=app_name,
                        text=comment["text"],
                        sentiment_source="youtube_comment",
                        timestamp=timestamp,
                        is_pre_release=is_pre,
                        author_type="community",
                        source_url=f"https://youtube.com/watch?v={video_id}",
                    )
                )

        return records
