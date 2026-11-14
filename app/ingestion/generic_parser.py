"""Extract citation-ready sections from ordinary public HTML pages."""

from __future__ import annotations

from urllib.parse import urldefrag

from bs4 import BeautifulSoup, Tag

from app.ingestion.models import CodeBlock, Document, Section
from app.ingestion.parser import (
    CONTAINER_TAGS,
    DocumentParseError,
    _element_text,
    _language_for,
    _normalize_text,
    calculate_content_hash,
)
from app.ingestion.site import SiteSource

MAIN_SELECTORS = (
    "main article",
    "article",
    "main",
    "[role='main']",
    ".documentation",
    ".docs-content",
    ".content",
    "body",
)
REMOVE_SELECTORS = (
    "nav",
    "footer",
    "aside",
    "script",
    "style",
    "noscript",
    "template",
    "button",
    "form",
    "iframe",
    "svg",
    "[hidden]",
    "[aria-hidden='true']",
    ".navigation",
    ".navbar",
    ".sidebar",
    ".footer",
    ".cookie-banner",
    ".breadcrumbs",
    ".table-of-contents",
    ".toc",
)
CONTENT_TAGS = frozenset(
    {"h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "table", "pre"}
)


def _is_nested_content(element: Tag) -> bool:
    return any(
        isinstance(parent, Tag) and parent.name in CONTAINER_TAGS
        for parent in element.parents
    )


class GenericDocumentParser:
    """Parse a generic page while removing common website chrome."""

    def parse(self, source: SiteSource, html: str) -> Document:
        source.validate()
        if not html.strip():
            raise DocumentParseError(f"empty HTML for {source.id}")

        soup = BeautifulSoup(html, "html.parser")
        content_root = next(
            (
                match
                for selector in MAIN_SELECTORS
                if (match := soup.select_one(selector))
            ),
            None,
        )
        if content_root is None:
            raise DocumentParseError(f"no page content found for {source.id}")
        for selector in REMOVE_SELECTORS:
            for element in content_root.select(selector):
                element.decompose()

        sections: list[Section] = []
        heading = source.title
        heading_level = 1
        anchor: str | None = None
        text_blocks: list[str] = []
        code_blocks: list[CodeBlock] = []

        def flush() -> None:
            nonlocal text_blocks, code_blocks
            content = "\n\n".join(text_blocks).strip()
            if not content and not code_blocks:
                return
            base_url = urldefrag(source.url).url
            source_url = f"{base_url}#{anchor}" if anchor else base_url
            sections.append(
                Section(
                    heading=heading,
                    heading_level=heading_level,
                    anchor=anchor,
                    source_url=source_url,
                    content=content,
                    code_blocks=list(code_blocks),
                )
            )
            text_blocks = []
            code_blocks = []

        for element in content_root.descendants:
            if not isinstance(element, Tag) or element.name not in CONTENT_TAGS:
                continue
            if _is_nested_content(element):
                continue
            if element.name.startswith("h"):
                flush()
                heading = _normalize_text(element.get_text(" ", strip=True))
                heading_level = int(element.name[1])
                anchor = element.get("id") or None
                continue
            if element.name == "pre":
                code_element = element.find("code")
                code = (code_element or element).get_text("", strip=False).strip("\n")
                if code.strip():
                    code_blocks.append(CodeBlock(_language_for(element), code))
                continue
            text = _element_text(element)
            if text:
                text_blocks.append(text)

        flush()
        if not sections:
            raise DocumentParseError(f"no usable content for {source.id}")
        return Document.create(
            source=source,
            content_hash=calculate_content_hash(sections),
            sections=sections,
        )
