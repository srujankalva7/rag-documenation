"""SQLite persistence for exact and semantic answer-cache entries."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from app.cache.models import CacheEntry

SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    normalized_question TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    question TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    answer_json TEXT NOT NULL,
    source_versions_json TEXT NOT NULL,
    generation_cost_usd REAL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    invalidated_at REAL,
    invalidation_reason TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_cache_exact
ON cache_entries(normalized_question, scope_key, provider, model)
WHERE invalidated_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_cache_semantic
ON cache_entries(scope_key, provider, model, embedding_model, expires_at);

CREATE TABLE IF NOT EXISTS cache_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    reason TEXT,
    entry_id INTEGER,
    similarity REAL,
    created_at REAL NOT NULL
);
"""


class SQLiteSemanticCache:
    def __init__(self, path: Path) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.executescript(SCHEMA)
        return connection

    @staticmethod
    def _entry(row: sqlite3.Row) -> CacheEntry:
        return CacheEntry(
            entry_id=int(row["id"]),
            normalized_question=str(row["normalized_question"]),
            scope_key=str(row["scope_key"]),
            question=str(row["question"]),
            embedding_model=str(row["embedding_model"]),
            embedding=[float(value) for value in json.loads(row["embedding_json"])],
            provider=str(row["provider"]),
            model=str(row["model"]),
            answer_payload=json.loads(row["answer_json"]),
            source_versions=json.loads(row["source_versions_json"]),
            generation_cost_usd=(
                float(row["generation_cost_usd"])
                if row["generation_cost_usd"] is not None
                else None
            ),
            created_at=float(row["created_at"]),
            expires_at=float(row["expires_at"]),
        )

    def find_exact(
        self,
        normalized_question: str,
        scope_key: str,
        provider: str,
        model: str,
        *,
        now: float | None = None,
    ) -> CacheEntry | None:
        timestamp = time.time() if now is None else now
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM cache_entries
                WHERE normalized_question = ? AND scope_key = ?
                  AND provider = ? AND model = ?
                  AND invalidated_at IS NULL AND expires_at > ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (normalized_question, scope_key, provider, model, timestamp),
            ).fetchone()
        return self._entry(row) if row is not None else None

    def semantic_candidates(
        self,
        scope_key: str,
        provider: str,
        model: str,
        embedding_model: str,
        *,
        now: float | None = None,
    ) -> list[CacheEntry]:
        timestamp = time.time() if now is None else now
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM cache_entries
                WHERE scope_key = ? AND provider = ? AND model = ?
                  AND embedding_model = ? AND invalidated_at IS NULL
                  AND expires_at > ?
                ORDER BY created_at DESC
                """,
                (scope_key, provider, model, embedding_model, timestamp),
            ).fetchall()
        return [self._entry(row) for row in rows]

    def put(
        self,
        *,
        normalized_question: str,
        scope_key: str,
        question: str,
        embedding_model: str,
        embedding: list[float],
        provider: str,
        model: str,
        answer_payload: dict[str, Any],
        source_versions: dict[str, str],
        generation_cost_usd: float | None,
        ttl_seconds: int,
        now: float | None = None,
    ) -> int:
        if not embedding:
            raise ValueError("cache embedding cannot be empty")
        timestamp = time.time() if now is None else now
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE cache_entries
                SET invalidated_at = ?, invalidation_reason = 'replaced'
                WHERE normalized_question = ? AND scope_key = ?
                  AND provider = ? AND model = ? AND invalidated_at IS NULL
                """,
                (timestamp, normalized_question, scope_key, provider, model),
            )
            cursor = connection.execute(
                """
                INSERT INTO cache_entries (
                    normalized_question, scope_key, question, embedding_model,
                    embedding_json, provider, model, answer_json,
                    source_versions_json, generation_cost_usd, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_question,
                    scope_key,
                    question,
                    embedding_model,
                    json.dumps(embedding, separators=(",", ":")),
                    provider,
                    model,
                    json.dumps(answer_payload, separators=(",", ":")),
                    json.dumps(source_versions, sort_keys=True, separators=(",", ":")),
                    generation_cost_usd,
                    timestamp,
                    timestamp + ttl_seconds,
                ),
            )
            return int(cursor.lastrowid)

    def invalidate(self, entry_id: int, reason: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE cache_entries
                SET invalidated_at = ?, invalidation_reason = ?
                WHERE id = ? AND invalidated_at IS NULL
                """,
                (time.time(), reason, entry_id),
            )

    def matching_entry_ids(
        self,
        *,
        document_ids: set[str] | None = None,
        provider: str | None = None,
        model: str | None = None,
        created_before: float | None = None,
        expired_before: float | None = None,
    ) -> list[int]:
        """Return active entries matching every supplied lifecycle filter."""
        clauses = ["invalidated_at IS NULL"]
        parameters: list[object] = []
        if provider is not None:
            clauses.append("provider = ?")
            parameters.append(provider)
        if model is not None:
            clauses.append("model = ?")
            parameters.append(model)
        if created_before is not None:
            clauses.append("created_at < ?")
            parameters.append(created_before)
        if expired_before is not None:
            clauses.append("expires_at <= ?")
            parameters.append(expired_before)

        query = (
            "SELECT id, source_versions_json FROM cache_entries WHERE "
            + " AND ".join(clauses)
            + " ORDER BY id"
        )
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()

        matches = []
        for row in rows:
            if document_ids:
                source_versions = json.loads(row["source_versions_json"])
                if document_ids.isdisjoint(source_versions):
                    continue
            matches.append(int(row["id"]))
        return matches

    def invalidate_entries(
        self,
        entry_ids: list[int],
        reason: str,
        *,
        now: float | None = None,
    ) -> int:
        """Invalidate selected entries while retaining their audit history."""
        if not reason.strip():
            raise ValueError("invalidation reason cannot be empty")
        if not entry_ids:
            return 0
        timestamp = time.time() if now is None else now
        placeholders = ",".join("?" for _ in entry_ids)
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT id FROM cache_entries WHERE id IN ({placeholders}) "
                "AND invalidated_at IS NULL",
                entry_ids,
            ).fetchall()
            active_ids = [int(row["id"]) for row in rows]
            if not active_ids:
                return 0
            active_placeholders = ",".join("?" for _ in active_ids)
            connection.execute(
                f"UPDATE cache_entries SET invalidated_at = ?, "
                f"invalidation_reason = ? WHERE id IN ({active_placeholders})",
                (timestamp, reason, *active_ids),
            )
            connection.executemany(
                """
                INSERT INTO cache_events
                    (event_type, reason, entry_id, similarity, created_at)
                VALUES ('invalidated', ?, ?, NULL, ?)
                """,
                [(reason, entry_id, timestamp) for entry_id in active_ids],
            )
        return len(active_ids)

    def record_event(
        self,
        event_type: str,
        *,
        reason: str | None = None,
        entry_id: int | None = None,
        similarity: float | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO cache_events
                    (event_type, reason, entry_id, similarity, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (event_type, reason, entry_id, similarity, time.time()),
            )
