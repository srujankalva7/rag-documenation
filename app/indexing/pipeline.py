"""Incrementally embed chunk files and synchronize the local vector index."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.indexing.models import IndexRecord, records_from_chunked_document
from app.indexing.providers import EmbeddingProvider
from app.indexing.store import SQLiteVectorIndex


@dataclass(frozen=True, slots=True)
class IndexingSummary:
    documents: int
    chunks: int
    embedded: int
    unchanged: int
    removed: int
    orphaned_embeddings_removed: int
    dry_run: bool = False


@dataclass(slots=True)
class IndexingPipeline:
    provider: EmbeddingProvider
    store: SQLiteVectorIndex
    chunks_dir: Path
    batch_size: int = 32

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

    def _files(self, document_id: str | None) -> list[Path]:
        if document_id:
            path = self.chunks_dir / f"{document_id}.json"
            if not path.exists():
                raise ValueError(f"unknown document id: {document_id}")
            return [path]
        files = sorted(self.chunks_dir.glob("*.json"))
        if not files:
            raise ValueError(f"no chunk files found in {self.chunks_dir}")
        return files

    def _load(self, document_id: str | None) -> tuple[list[IndexRecord], set[str]]:
        records: list[IndexRecord] = []
        document_ids: set[str] = set()
        seen_ids: set[str] = set()
        for path in self._files(document_id):
            value = json.loads(path.read_text(encoding="utf-8"))
            file_records = records_from_chunked_document(value)
            duplicate_ids = seen_ids.intersection(
                record.chunk_id for record in file_records
            )
            if duplicate_ids:
                duplicate = sorted(duplicate_ids)[0]
                raise ValueError(f"duplicate chunk ID across files: {duplicate}")
            seen_ids.update(record.chunk_id for record in file_records)
            records.extend(file_records)
            document_ids.add(file_records[0].document_id)
        return records, document_ids

    def run(
        self,
        *,
        document_id: str | None = None,
        dry_run: bool = False,
        rebuild: bool = False,
    ) -> IndexingSummary:
        records, document_ids = self._load(document_id)
        desired_by_hash: dict[str, IndexRecord] = {}
        for record in records:
            desired_by_hash.setdefault(record.content_hash, record)

        transient = dry_run and not self.store.path.exists()
        with self.store.connect(transient=transient) as connection:
            if rebuild and not dry_run:
                self.store.clear(connection)
            current_chunks = {} if rebuild else self.store.chunk_hashes(connection)
            indexed_hashes = (
                set()
                if rebuild
                else self.store.indexed_hashes(connection, self.provider.model_name)
            )
            hashes_to_embed = sorted(set(desired_by_hash).difference(indexed_hashes))
            unchanged = sum(
                current_chunks.get(record.chunk_id) == record.content_hash
                for record in records
            )

            if dry_run:
                existing_ids = set(current_chunks)
                if document_id:
                    rows = connection.execute(
                        "SELECT chunk_id FROM chunks WHERE document_id IN ({})".format(
                            ",".join("?" for _ in document_ids)
                        ),
                        tuple(sorted(document_ids)),
                    )
                    existing_ids = {str(row["chunk_id"]) for row in rows}
                removed = len(
                    existing_ids.difference(record.chunk_id for record in records)
                )
                return IndexingSummary(
                    documents=len(document_ids),
                    chunks=len(records),
                    embedded=len(hashes_to_embed),
                    unchanged=unchanged,
                    removed=removed,
                    orphaned_embeddings_removed=0,
                    dry_run=True,
                )

            for start in range(0, len(hashes_to_embed), self.batch_size):
                batch_hashes = hashes_to_embed[start : start + self.batch_size]
                vectors = self.provider.embed(
                    [
                        desired_by_hash[content_hash].text
                        if not desired_by_hash[content_hash].code_blocks
                        else desired_by_hash[content_hash].embedding_text
                        for content_hash in batch_hashes
                    ]
                )
                if len(vectors) != len(batch_hashes):
                    raise ValueError(
                        "embedding provider returned an unexpected vector count"
                    )
                self.store.insert_embeddings(
                    connection,
                    self.provider.model_name,
                    zip(batch_hashes, vectors, strict=True),
                )

            self.store.upsert_chunks(connection, records, self.provider.model_name)
            removed = self.store.delete_stale_chunks(
                connection,
                {record.chunk_id for record in records},
                document_ids if document_id else None,
            )
            orphaned = self.store.delete_orphaned_embeddings(connection)
            return IndexingSummary(
                documents=len(document_ids),
                chunks=len(records),
                embedded=len(hashes_to_embed),
                unchanged=unchanged,
                removed=removed,
                orphaned_embeddings_removed=orphaned,
            )
