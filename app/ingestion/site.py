"""Configuration and URL rules for generic public website ingestion."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse

from app.ingestion.models import ID_PATTERN, SourceValidationError

HOST_PATTERN = re.compile(
    r"(?=.{1,253}\Z)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z]{2,63}\Z"
)


def _public_hostname(hostname: str) -> bool:
    value = hostname.rstrip(".").lower()
    if value == "localhost" or value.endswith(".localhost"):
        return False
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return bool(HOST_PATTERN.fullmatch(value))
    return address.is_global


def normalize_page_url(url: str) -> str:
    """Return a stable crawl URL without fragments or tracking differences."""
    value = urldefrag(url.strip()).url
    parsed = urlparse(value)
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    last_segment = path.rsplit("/", 1)[-1]
    if path != "/" and "." not in last_segment:
        path = path.rstrip("/") + "/"
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            "",
            "",
            "",
        )
    )


@dataclass(frozen=True, slots=True)
class SiteConfig:
    site_id: str
    homepage_url: str
    allowed_domains: tuple[str, ...]
    max_pages: int = 100
    max_depth: int = 3
    include_paths: tuple[str, ...] = ()
    exclude_paths: tuple[str, ...] = (
        "/login",
        "/logout",
        "/account",
        "/admin",
        "/search",
    )
    sitemap_url: str | None = None

    @classmethod
    def create(
        cls,
        homepage_url: str,
        *,
        site_id: str | None = None,
        allowed_domains: tuple[str, ...] | None = None,
        max_pages: int = 100,
        max_depth: int = 3,
        include_paths: tuple[str, ...] = (),
        exclude_paths: tuple[str, ...] | None = None,
        sitemap_url: str | None = None,
    ) -> SiteConfig:
        normalized_homepage = normalize_page_url(homepage_url)
        hostname = (urlparse(normalized_homepage).hostname or "").lower()
        resolved_id = site_id or re.sub(r"[^a-z0-9]+", "-", hostname).strip("-")
        resolved_excludes = (
            ("/login", "/logout", "/account", "/admin", "/search")
            if exclude_paths is None
            else exclude_paths
        )
        resolved_domains = tuple(
            domain.rstrip(".").lower() for domain in (allowed_domains or (hostname,))
        )
        config = cls(
            site_id=resolved_id,
            homepage_url=normalized_homepage,
            allowed_domains=resolved_domains,
            max_pages=max_pages,
            max_depth=max_depth,
            include_paths=include_paths,
            exclude_paths=resolved_excludes,
            sitemap_url=(normalize_page_url(sitemap_url) if sitemap_url else None),
        )
        config.validate()
        return config

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> SiteConfig:
        allowed = value.get("allowed_domains")
        return cls.create(
            str(value.get("homepage_url", "")),
            site_id=str(value.get("site_id", "")) or None,
            allowed_domains=(
                tuple(str(item) for item in allowed)
                if isinstance(allowed, list)
                else None
            ),
            max_pages=int(value.get("max_pages", 100)),
            max_depth=int(value.get("max_depth", 3)),
            include_paths=tuple(str(item) for item in value.get("include_paths", [])),
            exclude_paths=tuple(str(item) for item in value["exclude_paths"])
            if "exclude_paths" in value
            else None,
            sitemap_url=(
                str(value["sitemap_url"]) if value.get("sitemap_url") else None
            ),
        )

    def validate(self) -> None:
        if not ID_PATTERN.fullmatch(self.site_id):
            raise SourceValidationError(f"invalid site id: {self.site_id!r}")
        if not 1 <= self.max_pages <= 10_000:
            raise SourceValidationError("max_pages must be between 1 and 10000")
        if not 0 <= self.max_depth <= 20:
            raise SourceValidationError("max_depth must be between 0 and 20")
        if not self.allowed_domains:
            raise SourceValidationError("at least one allowed domain is required")
        for domain in self.allowed_domains:
            if not _public_hostname(domain):
                raise SourceValidationError(f"domain is not public: {domain!r}")
        self.validate_url(self.homepage_url)
        if self.sitemap_url:
            self.validate_url(self.sitemap_url)

    def validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        if parsed.scheme not in {"http", "https"}:
            raise SourceValidationError("site URLs must use HTTP or HTTPS")
        if parsed.username or parsed.password:
            raise SourceValidationError("site URLs cannot contain credentials")
        if parsed.port not in {None, 80, 443}:
            raise SourceValidationError("site URLs cannot use a non-standard port")
        if not _public_hostname(hostname):
            raise SourceValidationError("site URL must use a public hostname")
        if not any(
            hostname == domain or hostname.endswith(f".{domain}")
            for domain in self.allowed_domains
        ):
            raise SourceValidationError(f"URL is outside allowed domains: {url}")

    def allows_url(self, url: str) -> bool:
        try:
            self.validate_url(url)
        except (SourceValidationError, ValueError):
            return False
        path = urlparse(url).path or "/"
        if self.include_paths and not any(
            path.startswith(item) for item in self.include_paths
        ):
            return False
        return not any(path.startswith(item) for item in self.exclude_paths)

    def to_dict(self) -> dict[str, object]:
        return {
            "site_id": self.site_id,
            "homepage_url": self.homepage_url,
            "allowed_domains": list(self.allowed_domains),
            "max_pages": self.max_pages,
            "max_depth": self.max_depth,
            "include_paths": list(self.include_paths),
            "exclude_paths": list(self.exclude_paths),
            "sitemap_url": self.sitemap_url,
        }


@dataclass(frozen=True, slots=True)
class SiteSource:
    id: str
    title: str
    url: str
    category: str
    site_id: str
    config: SiteConfig

    @classmethod
    def from_url(
        cls, config: SiteConfig, url: str, title: str = "Untitled"
    ) -> SiteSource:
        normalized = normalize_page_url(url)
        path = urlparse(normalized).path.strip("/") or "home"
        slug = re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-")[-48:] or "home"
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]
        source = cls(
            id=f"{config.site_id}-{slug}-{digest}",
            title=title.strip() or "Untitled",
            url=normalized,
            category="website",
            site_id=config.site_id,
            config=config,
        )
        source.validate()
        return source

    def validate(self) -> None:
        if not ID_PATTERN.fullmatch(self.id):
            raise SourceValidationError(f"invalid source id: {self.id!r}")
        self.config.validate_url(self.url)


def load_site_config(path: Path) -> SiteConfig:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("site configuration must be a JSON object")
    return SiteConfig.from_dict(value)


def resolve_link(base_url: str, href: str) -> str:
    return normalize_page_url(urljoin(base_url, href))
