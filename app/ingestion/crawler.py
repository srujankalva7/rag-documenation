"""Bounded, same-site discovery for public HTML websites."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

from app.ingestion.site import SiteConfig, SiteSource, normalize_page_url, resolve_link

SKIPPED_EXTENSIONS = {
    ".7z",
    ".avi",
    ".css",
    ".csv",
    ".doc",
    ".docx",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".mov",
    ".mp3",
    ".mp4",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".svg",
    ".tar",
    ".webp",
    ".xls",
    ".xlsx",
    ".xml",
    ".zip",
}


@dataclass(frozen=True, slots=True)
class CrawledPage:
    source: SiteSource
    html: str
    depth: int


@dataclass(frozen=True, slots=True)
class CrawlFailure:
    url: str
    error: str


@dataclass(slots=True)
class CrawlResult:
    pages: list[CrawledPage] = field(default_factory=list)
    failures: list[CrawlFailure] = field(default_factory=list)
    skipped: int = 0


class SiteCrawler:
    """Discover and download a bounded set of same-site HTML pages."""

    def __init__(
        self,
        *,
        delay_seconds: float = 0.5,
        timeout_seconds: float = 20.0,
        max_response_bytes: int = 2_000_000,
        user_agent: str = "rag-documentation/0.1 generic-site-ingestion",
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if delay_seconds < 0:
            raise ValueError("delay_seconds must be non-negative")
        if max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        self.delay_seconds = delay_seconds
        self.max_response_bytes = max_response_bytes
        self.user_agent = user_agent
        self._sleep = sleep
        self._monotonic = monotonic
        self._next_request_at = 0.0
        self._owns_client = client is None
        self._client = client or httpx.Client(
            headers={"User-Agent": user_agent, "Accept": "text/html,application/xml"},
            timeout=timeout_seconds,
            follow_redirects=False,
        )

    def __enter__(self) -> SiteCrawler:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _wait(self) -> None:
        now = self._monotonic()
        remaining = self._next_request_at - now
        if remaining > 0:
            self._sleep(remaining)
        self._next_request_at = self._monotonic() + self.delay_seconds

    def _get(self, config: SiteConfig, url: str) -> httpx.Response:
        current_url = url
        for _ in range(6):
            config.validate_url(current_url)
            self._wait()
            response = self._client.get(current_url)
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    response.raise_for_status()
                current_url = resolve_link(str(response.url), location)
                continue
            response.raise_for_status()
            config.validate_url(str(response.url))
            if len(response.content) > self.max_response_bytes:
                raise ValueError(f"response exceeds size limit: {url}")
            return response
        raise ValueError(f"too many redirects: {url}")

    def _robots(self, config: SiteConfig) -> tuple[RobotFileParser, list[str]]:
        robots_url = urljoin(config.homepage_url, "/robots.txt")
        parser = RobotFileParser()
        parser.set_url(robots_url)
        sitemap_urls = []
        try:
            response = self._get(config, robots_url)
            lines = response.text.splitlines()
            parser.parse(lines)
            for line in lines:
                name, separator, value = line.partition(":")
                if separator and name.strip().lower() == "sitemap":
                    sitemap = normalize_page_url(value.strip())
                    try:
                        config.validate_url(sitemap)
                    except ValueError:
                        continue
                    else:
                        sitemap_urls.append(sitemap)
        except (httpx.HTTPError, ValueError):
            parser.parse([])
        return parser, sitemap_urls

    def _sitemap_pages(
        self,
        config: SiteConfig,
        initial_urls: list[str],
    ) -> list[str]:
        queue = deque(initial_urls)
        visited = set()
        pages = []
        while queue and len(visited) < 20 and len(pages) < config.max_pages:
            sitemap_url = queue.popleft()
            if sitemap_url in visited:
                continue
            try:
                config.validate_url(sitemap_url)
            except ValueError:
                continue
            visited.add(sitemap_url)
            try:
                response = self._get(config, sitemap_url)
                root = ET.fromstring(response.text)
            except (httpx.HTTPError, ET.ParseError, ValueError):
                continue
            locations = [
                normalize_page_url(element.text)
                for element in root.iter()
                if element.tag.rsplit("}", 1)[-1] == "loc" and element.text
            ]
            if root.tag.rsplit("}", 1)[-1] == "sitemapindex":
                queue.extend(locations)
            else:
                pages.extend(url for url in locations if config.allows_url(url))
        return pages[: config.max_pages]

    @staticmethod
    def _is_page_url(url: str) -> bool:
        path = urlparse(url).path.lower()
        return not any(path.endswith(extension) for extension in SKIPPED_EXTENSIONS)

    @staticmethod
    def _page_title(soup: BeautifulSoup, fallback: str) -> str:
        heading = soup.find("h1")
        if heading and heading.get_text(" ", strip=True):
            return heading.get_text(" ", strip=True)
        if soup.title and soup.title.get_text(" ", strip=True):
            return soup.title.get_text(" ", strip=True)
        return fallback

    def crawl(self, config: SiteConfig) -> CrawlResult:
        config.validate()
        robots, advertised_sitemaps = self._robots(config)
        sitemap_candidates = advertised_sitemaps
        if config.sitemap_url:
            sitemap_candidates.insert(0, config.sitemap_url)
        if not sitemap_candidates:
            sitemap_candidates.append(urljoin(config.homepage_url, "/sitemap.xml"))
        sitemap_pages = self._sitemap_pages(config, sitemap_candidates)

        queue = deque([(config.homepage_url, 0)])
        queue.extend((url, 0) for url in sitemap_pages)
        queued = {url for url, _ in queue}
        max_queued_urls = max(config.max_pages * 10, 100)
        visited = set()
        result = CrawlResult()

        while queue and len(result.pages) < config.max_pages:
            url, depth = queue.popleft()
            if url in visited:
                continue
            visited.add(url)
            is_homepage = url == config.homepage_url
            if (
                depth > config.max_depth
                or (not is_homepage and not config.allows_url(url))
                or not self._is_page_url(url)
                or not robots.can_fetch(self.user_agent, url)
            ):
                result.skipped += 1
                continue
            try:
                response = self._get(config, url)
                content_type = response.headers.get("content-type", "").lower()
                if "text/html" not in content_type:
                    result.skipped += 1
                    continue
                soup = BeautifulSoup(response.text, "html.parser")
                title = self._page_title(soup, urlparse(url).path or config.site_id)
                if config.allows_url(str(response.url)):
                    source = SiteSource.from_url(config, str(response.url), title)
                    result.pages.append(CrawledPage(source, response.text, depth))
            except (httpx.HTTPError, ValueError) as error:
                result.failures.append(
                    CrawlFailure(url, f"{type(error).__name__}: {error}")
                )
                continue

            if depth >= config.max_depth:
                continue
            for link in soup.select("a[href]"):
                rel = {str(value).lower() for value in link.get("rel", [])}
                if "nofollow" in rel:
                    continue
                href = str(link.get("href", "")).strip()
                if not href or href.startswith(("mailto:", "tel:", "javascript:")):
                    continue
                candidate = resolve_link(str(response.url), href)
                if (
                    len(queued) < max_queued_urls
                    and candidate not in visited
                    and candidate not in queued
                    and config.allows_url(candidate)
                    and self._is_page_url(candidate)
                ):
                    queue.append((candidate, depth + 1))
                    queued.add(candidate)
        return result
