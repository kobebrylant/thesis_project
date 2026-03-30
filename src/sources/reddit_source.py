"""Reddit data source via PRAW (Python Reddit API Wrapper).

Collects pre-release discussion from gaming subreddits.
Requires Reddit API credentials (client_id, client_secret, user_agent).

Setup:
    pip install praw
    Set environment variables: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
"""

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .base_source import BaseSource, SourceRecord


# Default subreddits for gaming discussion
GAMING_SUBREDDITS = [
    "Games",
    "pcgaming",
    "Steam",
    "indiegaming",
    "truegaming",
]


class RedditSource(BaseSource):
    """Collect pre-release discussion from Reddit gaming subreddits."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
        subreddits: Optional[List[str]] = None,
    ):
        self.client_id = client_id or os.environ.get("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("REDDIT_CLIENT_SECRET")
        self.user_agent = user_agent or os.environ.get(
            "REDDIT_USER_AGENT", "thesis-sentiment-analysis/1.0"
        )
        self.subreddits = subreddits or GAMING_SUBREDDITS
        self._reddit = None

    @property
    def source_name(self) -> str:
        return "reddit"

    def _get_client(self):
        """Lazy-initialize PRAW client."""
        if self._reddit is not None:
            return self._reddit

        if not self.client_id or not self.client_secret:
            raise RuntimeError(
                "Reddit API credentials not configured.\n"
                "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables,\n"
                "or pass them to RedditSource(client_id=..., client_secret=...).\n"
                "Get credentials at: https://www.reddit.com/prefs/apps"
            )

        try:
            import praw
        except ImportError:
            raise ImportError(
                "PRAW is required for Reddit data collection.\n"
                "Install it with: pip install praw"
            )

        self._reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent,
        )
        return self._reddit

    def fetch(
        self,
        app_id: int,
        app_name: str,
        max_posts: int = 100,
        max_comments_per_post: int = 20,
        release_date: Optional[str] = None,
    ) -> List[SourceRecord]:
        """Search Reddit for pre-release discussion about a game.

        Args:
            app_id: Steam App ID
            app_name: Game name (used as search query)
            max_posts: Maximum posts to retrieve per subreddit
            max_comments_per_post: Max top-level comments per post
            release_date: ISO date string for pre-release filtering
        """
        reddit = self._get_client()
        records = []

        for subreddit_name in self.subreddits:
            try:
                subreddit = reddit.subreddit(subreddit_name)
                search_results = subreddit.search(
                    app_name,
                    sort="relevance",
                    time_filter="all",
                    limit=max_posts,
                )

                for post in search_results:
                    post_time = datetime.fromtimestamp(
                        post.created_utc, tz=timezone.utc
                    ).isoformat()

                    is_pre = self.is_before_release(post_time, release_date)

                    # Post title + body
                    post_text = f"{post.title}\n{post.selftext}".strip()
                    if post_text:
                        records.append(
                            SourceRecord(
                                app_id=app_id,
                                app_name=app_name,
                                text=post_text,
                                sentiment_source="reddit_post",
                                timestamp=post_time,
                                is_pre_release=is_pre,
                                author_type="community",
                                source_url=f"https://reddit.com{post.permalink}",
                            )
                        )

                    # Top-level comments
                    post.comments.replace_more(limit=0)
                    for comment in post.comments[:max_comments_per_post]:
                        comment_time = datetime.fromtimestamp(
                            comment.created_utc, tz=timezone.utc
                        ).isoformat()
                        is_pre_comment = self.is_before_release(comment_time, release_date)

                        if comment.body and comment.body != "[deleted]":
                            records.append(
                                SourceRecord(
                                    app_id=app_id,
                                    app_name=app_name,
                                    text=comment.body,
                                    sentiment_source="reddit_comment",
                                    timestamp=comment_time,
                                    is_pre_release=is_pre_comment,
                                    author_type="community",
                                    source_url=f"https://reddit.com{comment.permalink}",
                                )
                            )

            except Exception as e:
                print(f"    r/{subreddit_name}: {e}")

        return records
