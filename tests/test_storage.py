from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from app.ingestion.models import Source
from app.ingestion.parser import FastAPIDocumentParser
from app.ingestion.storage import DocumentStorage

SOURCE = Source(
    id="request-body",
    title="Request Body",
    url="https://fastapi.tiangolo.com/tutorial/body/",
    category="tutorial",
)


def test_storage_skips_unchanged_and_archives_changed_document(tmp_path: Path) -> None:
    storage = DocumentStorage(
        raw_dir=tmp_path / "raw",
        documents_dir=tmp_path / "documents",
        versions_dir=tmp_path / "versions",
    )
    html = Path("tests/fixtures/fastapi_page.html").read_text(encoding="utf-8")
    original = FastAPIDocumentParser().parse(SOURCE, html)

    assert storage.save_document(SOURCE, original) == "created"
    assert storage.save_document(SOURCE, original) == "unchanged"

    changed = replace(original, content_hash="f" * 64)
    assert storage.save_document(SOURCE, changed) == "updated"
    archived = tmp_path / "versions" / SOURCE.id / f"{original.content_hash}.json"
    assert archived.exists()
