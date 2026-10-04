"""Typed models shared by the documentation ingestion pipeline."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

ALLOWED_CATEGORIES = {"tutorial", "advanced", "deployment", "how-to", "reference"}
SOURCE_KEYS = {"id", "title", "url", "category"}
ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


class SourceValidationError(ValueError):
    """Raised when a source manifest entry is unsafe or malformed."""


@dataclass(frozen=True, slots=True)
class Source:
    """An approved documentation source from the source manifest."""

    id: str
    title: str
    url: str
    category: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Source:
        extra_or_missing = set(value) ^ SOURCE_KEYS
        if extra_or_missing:
            raise SourceValidationError(
                "source must contain exactly id, title, url, and category; "
                f"difference: {sorted(extra_or_missing)}"
            )

        source = cls(
            id=str(value["id"]),
            title=str(value["title"]),
            url=str(value["url"]),
            category=str(value["category"]),
        )
        source.validate()
        return source

    def validate(self) -> None:
        if not ID_PATTERN.fullmatch(self.id):
            raise SourceValidationError(f"invalid source id: {self.id!r}")
        if not self.title.strip():
            raise SourceValidationError("source title cannot be empty")
        if self.category not in ALLOWED_CATEGORIES:
            raise SourceValidationError(f"unsupported category: {self.category!r}")

        parsed = urlparse(self.url)
        if parsed.scheme != "https" or parsed.hostname != "fastapi.tiangolo.com":
            raise SourceValidationError(
                "only https://fastapi.tiangolo.com documentation is allowed"
            )
        if parsed.username or parsed.password or parsed.port:
            raise SourceValidationError(
                "source URL cannot contain credentials or a port"
            )
        if parsed.query or parsed.fragment:
            raise SourceValidationError(
                "source URL must not contain a query or fragment"
            )


@dataclass(frozen=True, slots=True)
class CodeBlock:
    language: str | None
    code: str


@dataclass(frozen=True, slots=True)
class Section:
    heading: str
    heading_level: int
    anchor: str | None
    source_url: str
    content: str
    code_blocks: list[CodeBlock] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    title: str
    url: str
    category: str
    content_hash: str
    scraped_at: str
    sections: list[Section]

    @classmethod
    def create(
        cls,
        *,
        source: Source,
        content_hash: str,
        sections: list[Section],
        scraped_at: datetime | None = None,
    ) -> Document:
        timestamp = scraped_at or datetime.now(UTC)
        return cls(
            id=source.id,
            title=source.title,
            url=source.url,
            category=source.category,
            content_hash=content_hash,
            scraped_at=timestamp.isoformat().replace("+00:00", "Z"),
            sections=sections,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
