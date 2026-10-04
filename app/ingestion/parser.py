"""Extract structured, citation-ready content from FastAPI documentation HTML."""

from __future__ import annotations

import hashlib
import json
import re
from urllib.parse import urldefrag

from bs4 import BeautifulSoup, Tag

from app.ingestion.models import CodeBlock, Document, Section, Source

ARTICLE_SELECTORS = ("article", "main article", "main")
REMOVE_SELECTORS = (
    "nav",
    "footer",
    "script",
    "style",
    "button",
    "form",
    ".md-header",
    ".md-footer",
    ".md-sidebar",
    ".toc",
    ".headerlink",
    "[data-clipboard-target]",
)
CONTENT_TAGS = {"h1", "h2", "h3", "h4", "p", "ul", "ol", "table", "pre"}
CONTAINER_TAGS = {"ul", "ol", "table", "pre"}


class DocumentParseError(ValueError):
    """Raised when a page does not contain usable documentation content."""


def _normalize_text(value: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in value.splitlines()]
    return "\n".join(line for line in lines if line)


def _element_text(element: Tag) -> str:
    if element.name in {"ul", "ol"}:
        items = [
            _normalize_text(item.get_text(" ", strip=True))
            for item in element.find_all("li", recursive=False)
        ]
        prefix = "- " if element.name == "ul" else ""
        return "\n".join(
            f"{prefix}{item}" if prefix else f"{index}. {item}"
            for index, item in enumerate(items, start=1)
            if item
        )

    if element.name == "table":
        rows = []
        for row in element.find_all("tr"):
            cells = [
                _normalize_text(cell.get_text(" ", strip=True))
                for cell in row.find_all(["th", "td"], recursive=False)
            ]
            if cells:
                rows.append(" | ".join(cells))
        return "\n".join(rows)

    return _normalize_text(element.get_text(" ", strip=True))


def _is_nested_content(element: Tag) -> bool:
    return any(
        isinstance(parent, Tag) and parent.name in CONTAINER_TAGS
        for parent in element.parents
    )


def _language_for(pre: Tag) -> str | None:
    code = pre.find("code")
    candidates = [pre, code] if isinstance(code, Tag) else [pre]
    for candidate in candidates:
        for class_name in candidate.get("class", []):
            if class_name.startswith("language-"):
                return class_name.removeprefix("language-")
    return None


def calculate_content_hash(sections: list[Section]) -> str:
    """Return a stable hash that excludes scrape time."""

    serialized = json.dumps(
        [
            {
                "heading": section.heading,
                "heading_level": section.heading_level,
                "anchor": section.anchor,
                "source_url": section.source_url,
                "content": section.content,
                "code_blocks": [
                    {"language": block.language, "code": block.code}
                    for block in section.code_blocks
                ],
            }
            for section in sections
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class FastAPIDocumentParser:
    """Parse one FastAPI documentation page into ordered sections."""

    def parse(self, source: Source, html: str) -> Document:
        source.validate()
        if not html.strip():
            raise DocumentParseError(f"empty HTML for {source.id}")

        soup = BeautifulSoup(html, "html.parser")
        article = next(
            (
                match
                for selector in ARTICLE_SELECTORS
                if (match := soup.select_one(selector))
            ),
            None,
        )
        if article is None:
            raise DocumentParseError(f"no documentation article found for {source.id}")

        for selector in REMOVE_SELECTORS:
            for element in article.select(selector):
                element.decompose()

        sections: list[Section] = []
        heading = source.title
        heading_level = 1
        anchor: str | None = None
        text_blocks: list[str] = []
        code_blocks: list[CodeBlock] = []

        def flush() -> None:
            nonlocal text_blocks, code_blocks
            content = "\n\n".join(block for block in text_blocks if block).strip()
            if not content and not code_blocks:
                return
            base_url = urldefrag(source.url).url
            section_url = f"{base_url}#{anchor}" if anchor else base_url
            sections.append(
                Section(
                    heading=heading,
                    heading_level=heading_level,
                    anchor=anchor,
                    source_url=section_url,
                    content=content,
                    code_blocks=list(code_blocks),
                )
            )
            text_blocks = []
            code_blocks = []

        for element in article.find_all(CONTENT_TAGS):
            if _is_nested_content(element):
                continue

            if element.name in {"h1", "h2", "h3", "h4"}:
                flush()
                heading = _normalize_text(element.get_text(" ", strip=True))
                heading_level = int(element.name[1])
                anchor = element.get("id") or None
                continue

            if element.name == "pre":
                code_element = element.find("code")
                code = (code_element or element).get_text("", strip=False).strip("\n")
                if code.strip():
                    code_blocks.append(
                        CodeBlock(language=_language_for(element), code=code)
                    )
                continue

            text = _element_text(element)
            if text:
                text_blocks.append(text)

        flush()
        if not sections:
            raise DocumentParseError(
                f"no usable documentation sections for {source.id}"
            )

        content_hash = calculate_content_hash(sections)
        return Document.create(
            source=source,
            content_hash=content_hash,
            sections=sections,
        )
