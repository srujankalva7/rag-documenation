"""Persistent, local vector storage backed by SQLite."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from app.indexing.models import IndexRecord

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS embeddings (
    content_hash TEXT NOT NULL,
    model_name TEXT NOT NULL,
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    vector_json TEXT NOT NULL,
    PRIMARY KEY (content_hash, model_name)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    model_name TEXT NOT NULL,
    document_id TEXT NOT NULL,
    document_hash TEXT NOT NULL,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    section TEXT NOT NULL,
    section_index INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    section_chunk_index INTEGER NOT NULL,
    source_url TEXT NOT NULL,
    text TEXT NOT NULL,
    code_blocks_json TEXT NOT NULL DEFAULT '[]',
    token_count INTEGER NOT NULL,
    FOREIGN KEY (content_hash, model_name)
        REFERENCES embeddings(content_hash, model_name)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash);
"""


class SQLiteVectorIndex:
    """Store normalized vectors and their source chunk metadata."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def connect(self, *, transient: bool = False) -> sqlite3.Connection:
        if transient:
            connection = sqlite3.connect(":memory:")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.executescript(SCHEMA)
        columns = {
            str(row["name"]) for row in connection.execute("PRAGMA table_info(chunks)")
        }
        if "code_blocks_json" not in columns:
            connection.execute(
                "ALTER TABLE chunks ADD COLUMN "
                "code_blocks_json TEXT NOT NULL DEFAULT '[]'"
            )
        return connection

    def indexed_hashes(
        self, connection: sqlite3.Connection, model_name: str
    ) -> set[str]:
        rows = connection.execute(
            "SELECT content_hash FROM embeddings WHERE model_name = ?",
            (model_name,),
        )
        return {str(row["content_hash"]) for row in rows}

    def chunk_hashes(self, connection: sqlite3.Connection) -> dict[str, str]:
        rows = connection.execute("SELECT chunk_id, content_hash FROM chunks")
        return {str(row["chunk_id"]): str(row["content_hash"]) for row in rows}

    def insert_embeddings(
        self,
        connection: sqlite3.Connection,
        model_name: str,
        values: Iterable[tuple[str, list[float]]],
    ) -> None:
        rows = []
        for content_hash, vector in values:
            if not vector:
                raise ValueError("embedding vectors cannot be empty")
            rows.append(
                (
                    content_hash,
                    model_name,
                    len(vector),
                    json.dumps(vector, separators=(",", ":")),
                )
            )
        connection.executemany(
            """
            INSERT OR IGNORE INTO embeddings
                (content_hash, model_name, dimension, vector_json)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )

    def upsert_chunks(
        self,
        connection: sqlite3.Connection,
        records: Iterable[IndexRecord],
        model_name: str,
    ) -> None:
        connection.executemany(
            """
            INSERT INTO chunks (
                chunk_id, content_hash, model_name, document_id, document_hash,
                title, category, section, section_index, chunk_index,
                section_chunk_index, source_url, text, code_blocks_json, token_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                content_hash = excluded.content_hash,
                model_name = excluded.model_name,
                document_id = excluded.document_id,
                document_hash = excluded.document_hash,
                title = excluded.title,
                category = excluded.category,
                section = excluded.section,
                section_index = excluded.section_index,
                chunk_index = excluded.chunk_index,
                section_chunk_index = excluded.section_chunk_index,
                source_url = excluded.source_url,
                text = excluded.text,
                code_blocks_json = excluded.code_blocks_json,
                token_count = excluded.token_count
            """,
            [
                (
                    record.chunk_id,
                    record.content_hash,
                    model_name,
                    record.document_id,
                    record.document_hash,
                    record.title,
                    record.category,
                    record.section,
                    record.section_index,
                    record.chunk_index,
                    record.section_chunk_index,
                    record.source_url,
                    record.text,
                    json.dumps(
                        [
                            {"language": block.language, "code": block.code}
                            for block in record.code_blocks
                        ],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    record.token_count,
                )
                for record in records
            ],
        )

    def delete_stale_chunks(
        self,
        connection: sqlite3.Connection,
        desired_ids: set[str],
        document_ids: set[str] | None = None,
    ) -> int:
        if document_ids is None:
            rows = connection.execute("SELECT chunk_id FROM chunks").fetchall()
        else:
            if not document_ids:
                return 0
            placeholders = ",".join("?" for _ in document_ids)
            rows = connection.execute(
                f"SELECT chunk_id FROM chunks WHERE document_id IN ({placeholders})",
                tuple(sorted(document_ids)),
            ).fetchall()
        stale_ids = sorted(
            str(row["chunk_id"]) for row in rows if row["chunk_id"] not in desired_ids
        )
        connection.executemany(
            "DELETE FROM chunks WHERE chunk_id = ?",
            [(chunk_id,) for chunk_id in stale_ids],
        )
        return len(stale_ids)

    def delete_orphaned_embeddings(self, connection: sqlite3.Connection) -> int:
        cursor = connection.execute(
            """
            DELETE FROM embeddings
            WHERE NOT EXISTS (
                SELECT 1 FROM chunks
                WHERE chunks.content_hash = embeddings.content_hash
                  AND chunks.model_name = embeddings.model_name
            )
            """
        )
        return max(cursor.rowcount, 0)

    def clear(self, connection: sqlite3.Connection) -> None:
        connection.execute("DELETE FROM chunks")
        connection.execute("DELETE FROM embeddings")
