from __future__ import annotations

from pathlib import Path

from app.ingestion.models import Source
from app.ingestion.parser import FastAPIDocumentParser
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.storage import DocumentStorage


class FakeFetcher:
    def __init__(self, html: str) -> None:
        self.html = html

    def fetch(self, source: Source) -> str:
        if source.id == "broken-page":
            raise RuntimeError("simulated failure")
        return self.html


def test_pipeline_continues_after_one_page_fails(tmp_path: Path) -> None:
    html = Path("tests/fixtures/fastapi_page.html").read_text(encoding="utf-8")
    sources = [
        Source(
            id="request-body",
            title="Request Body",
            url="https://fastapi.tiangolo.com/tutorial/body/",
            category="tutorial",
        ),
        Source(
            id="broken-page",
            title="Broken Page",
            url="https://fastapi.tiangolo.com/tutorial/broken/",
            category="tutorial",
        ),
    ]
    storage = DocumentStorage(
        raw_dir=tmp_path / "raw",
        documents_dir=tmp_path / "documents",
        versions_dir=tmp_path / "versions",
    )
    pipeline = IngestionPipeline(
        FakeFetcher(html),
        FastAPIDocumentParser(),
        storage,
    )

    summary = pipeline.run(sources)

    assert [result.status for result in summary.results] == ["created", "failed"]
    assert summary.failed == 1
    assert (tmp_path / "documents/request-body.json").exists()
