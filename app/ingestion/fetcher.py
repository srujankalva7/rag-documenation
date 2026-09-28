"""Polite HTTP client for approved FastAPI documentation pages."""

from __future__ import annotations

import time
from collections.abc import Callable

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.ingestion.models import Source


class RetryableHTTPStatusError(RuntimeError):
    """A temporary server response that may succeed on a later attempt."""


class UnexpectedContentTypeError(RuntimeError):
    """Raised when a documentation URL does not return HTML."""


class FastAPIDocumentFetcher:
    """Fetch approved sources with retries, timeouts, and request spacing."""

    def __init__(
        self,
        *,
        delay_seconds: float = 0.5,
        timeout_seconds: float = 20.0,
        user_agent: str = (
            "rag-documentation/0.1 (educational FastAPI documentation indexing project)"
        ),
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        client: httpx.Client | None = None,
    ) -> None:
        if delay_seconds < 0:
            raise ValueError("delay_seconds must be non-negative")
        self.delay_seconds = delay_seconds
        self._sleep = sleep
        self._monotonic = monotonic
        self._next_request_at = 0.0
        self._owns_client = client is None
        self._client = client or httpx.Client(
            headers={"User-Agent": user_agent, "Accept": "text/html"},
            timeout=timeout_seconds,
            follow_redirects=True,
        )

    def __enter__(self) -> FastAPIDocumentFetcher:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _wait_for_request_slot(self) -> None:
        now = self._monotonic()
        remaining = self._next_request_at - now
        if remaining > 0:
            self._sleep(remaining)
        self._next_request_at = self._monotonic() + self.delay_seconds

    @retry(
        retry=retry_if_exception_type(
            (httpx.TimeoutException, httpx.NetworkError, RetryableHTTPStatusError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def fetch(self, source: Source) -> str:
        """Download one source, retrying only temporary failures."""

        source.validate()
        self._wait_for_request_slot()
        response = self._client.get(source.url)

        if response.status_code == 429 or response.status_code >= 500:
            raise RetryableHTTPStatusError(
                f"temporary HTTP {response.status_code} for {source.url}"
            )

        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "text/html" not in content_type:
            raise UnexpectedContentTypeError(
                f"expected HTML from {source.url}, received {content_type or 'unknown'}"
            )
        return response.text
