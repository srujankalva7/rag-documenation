from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.models import Source
from app.ingestion.parser import DocumentParseError, FastAPIDocumentParser

FIXTURE = Path("tests/fixtures/fastapi_page.html")
SOURCE = Source(
    id="request-body",
    title="Request Body",
    url="https://fastapi.tiangolo.com/tutorial/body/",
    category="tutorial",
)


def test_parser_extracts_sections_code_and_citation_urls() -> None:
    document = FastAPIDocumentParser().parse(
        SOURCE,
        FIXTURE.read_text(encoding="utf-8"),
    )

    assert [section.heading for section in document.sections] == [
        "Request Body",
        "Create your data model",
        "Benefits",
    ]
    model_section = document.sections[1]
    assert model_section.source_url.endswith("#create-your-data-model")
    assert "Import BaseModel" in model_section.content
    assert "FastAPI reads the model" in model_section.content
    assert model_section.code_blocks[0].language == "python"
    assert "    name: str" in model_section.code_blocks[0].code
    assert "FastAPI Tutorial Advanced Search" not in str(document.to_dict())


def test_parser_hash_is_stable_when_scrape_time_changes() -> None:
    parser = FastAPIDocumentParser()
    html = FIXTURE.read_text(encoding="utf-8")

    first = parser.parse(SOURCE, html)
    second = parser.parse(SOURCE, html)

    assert first.content_hash == second.content_hash


def test_parser_rejects_page_without_article_or_main() -> None:
    with pytest.raises(DocumentParseError):
        FastAPIDocumentParser().parse(SOURCE, "<html><body>missing</body></html>")
