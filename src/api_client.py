"""Shared resilient HTTP client with exponential backoff and per-domain rate limiting."""

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests


@dataclass
class RetryConfig:
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter_max: float = 0.5
    retryable_status_codes: tuple = (429, 500, 502, 503, 504)


@dataclass
class RateLimitConfig:
    """Per-domain rate limit configuration."""
    requests_per_second: float = 1.0

    @property
    def min_interval(self) -> float:
        return 1.0 / self.requests_per_second


# Sensible defaults for known APIs
DEFAULT_RATE_LIMITS: Dict[str, RateLimitConfig] = {
    "store.steampowered.com": RateLimitConfig(requests_per_second=1.0),
    "steamspy.com": RateLimitConfig(requests_per_second=1.0),
    "steamcharts.com": RateLimitConfig(requests_per_second=0.5),
    "oauth.reddit.com": RateLimitConfig(requests_per_second=1.0),
    "www.googleapis.com": RateLimitConfig(requests_per_second=5.0),
}


class ResilientAPIClient:
    """HTTP client with exponential backoff and per-domain rate limiting.

    Usage:
        client = ResilientAPIClient()
        response = client.get("https://store.steampowered.com/api/appdetails", params={...})
        data = response.json()
    """

    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        rate_limits: Optional[Dict[str, RateLimitConfig]] = None,
        default_timeout: int = 30,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        self.retry_config = retry_config or RetryConfig()
        self.rate_limits = rate_limits or DEFAULT_RATE_LIMITS.copy()
        self.default_timeout = default_timeout
        self.default_headers = default_headers or {}
        self._last_request_time: Dict[str, float] = {}
        self._session = requests.Session()
        if self.default_headers:
            self._session.headers.update(self.default_headers)

    def _get_domain(self, url: str) -> str:
        return urlparse(url).netloc

    def _enforce_rate_limit(self, domain: str) -> None:
        if domain not in self.rate_limits:
            return

        min_interval = self.rate_limits[domain].min_interval
        last_time = self._last_request_time.get(domain, 0)
        elapsed = time.time() - last_time

        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            time.sleep(sleep_time)

        self._last_request_time[domain] = time.time()

    def _calculate_backoff(self, attempt: int) -> float:
        cfg = self.retry_config
        delay = cfg.base_delay * (cfg.backoff_factor ** attempt)
        delay = min(delay, cfg.max_delay)
        jitter = random.uniform(0, cfg.jitter_max)
        return delay + jitter

    def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> requests.Response:
        return self._request("GET", url, params=params, headers=headers, timeout=timeout)

    def _request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> requests.Response:
        domain = self._get_domain(url)
        timeout = timeout or self.default_timeout
        cfg = self.retry_config
        last_exception = None

        for attempt in range(cfg.max_retries + 1):
            self._enforce_rate_limit(domain)

            try:
                response = self._session.request(
                    method,
                    url,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                )

                if response.status_code not in cfg.retryable_status_codes:
                    return response

                # Retryable status code
                if attempt < cfg.max_retries:
                    # Honor Retry-After header if present (common with 429)
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = self._calculate_backoff(attempt)
                    else:
                        delay = self._calculate_backoff(attempt)

                    print(
                        f"  HTTP {response.status_code} from {domain}, "
                        f"retrying in {delay:.1f}s (attempt {attempt + 1}/{cfg.max_retries})"
                    )
                    time.sleep(delay)
                else:
                    return response

            except requests.exceptions.Timeout as e:
                last_exception = e
                if attempt < cfg.max_retries:
                    delay = self._calculate_backoff(attempt)
                    print(
                        f"  Timeout from {domain}, "
                        f"retrying in {delay:.1f}s (attempt {attempt + 1}/{cfg.max_retries})"
                    )
                    time.sleep(delay)

            except requests.exceptions.ConnectionError as e:
                last_exception = e
                if attempt < cfg.max_retries:
                    delay = self._calculate_backoff(attempt)
                    print(
                        f"  Connection error from {domain}, "
                        f"retrying in {delay:.1f}s (attempt {attempt + 1}/{cfg.max_retries})"
                    )
                    time.sleep(delay)

            except requests.exceptions.RequestException as e:
                raise

        raise requests.exceptions.RetryError(
            f"Max retries ({cfg.max_retries}) exceeded for {url}"
        ) from last_exception

    def set_rate_limit(self, domain: str, requests_per_second: float) -> None:
        self.rate_limits[domain] = RateLimitConfig(requests_per_second=requests_per_second)

    def close(self) -> None:
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
