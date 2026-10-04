from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from app.indexing.pipeline import IndexingPipeline
from app.indexing.store import SQLiteVectorIndex


@dataclass
class FakeEmbeddingProvider:
    model_name: str = "fake-local-model"
    calls: list[list[str]] = field(default_factory=list)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text)), float(sum(map(ord, text)) % 997)] for text in texts]


def make_chunk(
    chunk_id: str,
    content_hash: str,
    text: str,
    *,
    document_id: str = "request-body",
    chunk_index: int = 0,
    code_blocks: list[dict[str, str | None]] | None = None,
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "content_hash": content_hash,
        "document_id": document_id,
        "document_hash": f"{document_id}-hash",
        "title": document_id.replace("-", " ").title(),
        "category": "tutorial",
        "section": "Example",
        "section_index": 0,
        "chunk_index": chunk_index,
        "section_chunk_index": chunk_index,
        "source_url": f"https://fastapi.tiangolo.com/tutorial/{document_id}/#example",
        "text": text,
        "code_blocks": code_blocks or [],
        "token_count": max(len(text.split()), 1),
    }


def write_document(
    path: Path, document_id: str, chunks: list[dict[str, object]]
) -> None:
    path.write_text(
        json.dumps(
            {
                "document_id": document_id,
                "document_hash": f"{document_id}-hash",
                "chunking": {
                    "max_tokens": 600,
                    "overlap_tokens": 75,
                    "encoding_name": "cl100k_base",
                },
                "chunks": chunks,
            }
        ),
        encoding="utf-8",
    )


def table_count(path: Path, table: str) -> int:
    with sqlite3.connect(path) as connection:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def test_pipeline_indexes_incrementally_and_removes_stale_data(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    index_path = tmp_path / "index" / "vectors.sqlite3"
    document_path = chunks_dir / "request-body.json"
    first_chunk = make_chunk("request-body:0", "hash-a", "First chunk")
    second_chunk = make_chunk("request-body:1", "hash-b", "Second chunk", chunk_index=1)
    write_document(document_path, "request-body", [first_chunk, second_chunk])
    provider = FakeEmbeddingProvider()
    pipeline = IndexingPipeline(
        provider=provider,
        store=SQLiteVectorIndex(index_path),
        chunks_dir=chunks_dir,
        batch_size=1,
    )

    first = pipeline.run()
    second = pipeline.run()

    assert first.embedded == 2
    assert second.embedded == 0
    assert second.unchanged == 2
    assert len(provider.calls) == 2
    assert table_count(index_path, "chunks") == 2
    assert table_count(index_path, "embeddings") == 2

    changed_chunk = make_chunk("request-body:0", "hash-c", "Changed chunk")
    write_document(document_path, "request-body", [changed_chunk])
    changed = pipeline.run()

    assert changed.embedded == 1
    assert changed.removed == 1
    assert changed.orphaned_embeddings_removed == 2
    assert table_count(index_path, "chunks") == 1
    assert table_count(index_path, "embeddings") == 1


def test_pipeline_deduplicates_embeddings_by_content_hash(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    index_path = tmp_path / "vectors.sqlite3"
    chunks = [
        make_chunk("request-body:0", "shared-hash", "Same content"),
        make_chunk("request-body:1", "shared-hash", "Same content", chunk_index=1),
    ]
    write_document(chunks_dir / "request-body.json", "request-body", chunks)
    provider = FakeEmbeddingProvider()

    summary = IndexingPipeline(
        provider=provider,
        store=SQLiteVectorIndex(index_path),
        chunks_dir=chunks_dir,
    ).run()

    assert summary.chunks == 2
    assert summary.embedded == 1
    assert provider.calls == [["Same content"]]
    assert table_count(index_path, "chunks") == 2
    assert table_count(index_path, "embeddings") == 1


def test_dry_run_does_not_create_index_or_call_provider(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    index_path = tmp_path / "vectors.sqlite3"
    write_document(
        chunks_dir / "request-body.json",
        "request-body",
        [make_chunk("request-body:0", "hash-a", "Chunk text")],
    )
    provider = FakeEmbeddingProvider()

    summary = IndexingPipeline(
        provider=provider,
        store=SQLiteVectorIndex(index_path),
        chunks_dir=chunks_dir,
    ).run(dry_run=True)

    assert summary.dry_run is True
    assert summary.embedded == 1
    assert provider.calls == []
    assert not index_path.exists()


def test_pipeline_embeds_and_stores_code_only_chunks(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    index_path = tmp_path / "vectors.sqlite3"
    code_blocks = [{"language": "python", "code": "app = FastAPI()"}]
    write_document(
        chunks_dir / "request-body.json",
        "request-body",
        [
            make_chunk(
                "request-body:0",
                "code-hash",
                "",
                code_blocks=code_blocks,
            )
        ],
    )
    provider = FakeEmbeddingProvider()

    summary = IndexingPipeline(
        provider=provider,
        store=SQLiteVectorIndex(index_path),
        chunks_dir=chunks_dir,
    ).run()

    assert summary.embedded == 1
    assert provider.calls == [["```python\napp = FastAPI()\n```"]]
    with sqlite3.connect(index_path) as connection:
        stored = connection.execute(
            "SELECT text, code_blocks_json FROM chunks"
        ).fetchone()
    assert stored == ("", '[{"language":"python","code":"app = FastAPI()"}]')
