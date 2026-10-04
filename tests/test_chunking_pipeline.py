from __future__ import annotations

import shutil
from pathlib import Path

from app.chunking.chunker import DocumentChunker
from app.chunking.models import ChunkingConfig
from app.chunking.pipeline import ChunkingPipeline


def test_pipeline_creates_then_skips_unchanged_output(tmp_path: Path) -> None:
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    shutil.copy(
        "tests/fixtures/normalized_document.json",
        documents_dir / "request-body.json",
    )
    pipeline = ChunkingPipeline(
        chunker=DocumentChunker(ChunkingConfig(max_tokens=100, overlap_tokens=10)),
        documents_dir=documents_dir,
        output_dir=tmp_path / "chunks",
    )

    first = pipeline.run()
    second = pipeline.run()

    assert first.results[0].status == "created"
    assert second.results[0].status == "unchanged"
    assert first.chunk_count > 0
    assert (tmp_path / "chunks/request-body.json").exists()


def test_pipeline_reports_bad_document_and_continues(tmp_path: Path) -> None:
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    (documents_dir / "bad.json").write_text("{}", encoding="utf-8")
    shutil.copy(
        "tests/fixtures/normalized_document.json",
        documents_dir / "request-body.json",
    )
    pipeline = ChunkingPipeline(
        chunker=DocumentChunker(),
        documents_dir=documents_dir,
        output_dir=tmp_path / "chunks",
    )

    summary = pipeline.run()

    assert [result.status for result in summary.results] == ["failed", "created"]
    assert summary.failed == 1
