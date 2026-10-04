"""Rank indexed documentation chunks by cosine similarity."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from app.indexing.providers import EmbeddingProvider
from app.indexing.store import SQLiteVectorIndex
from app.retrieval.models import RetrievalError, SearchResult


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise RetrievalError(
            f"embedding dimension mismatch: query has {len(left)}, "
            f"index has {len(right)}"
        )
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )


@dataclass(slots=True)
class VectorRetriever:
    provider: EmbeddingProvider
    store: SQLiteVectorIndex

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        category: str | None = None,
    ) -> list[SearchResult]:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if category is not None:
            category = category.strip()
            if not category:
                raise ValueError("category cannot be empty")
        if not self.store.path.exists():
            raise RetrievalError(
                f"vector index not found at {self.store.path}; "
                "run python -m scripts.index_documents"
            )

        with self.store.connect() as connection:
            chunk_count = int(
                connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            )
            if chunk_count == 0:
                raise RetrievalError(
                    "vector index is empty; run python -m scripts.index_documents"
                )
            model_names = {
                str(row["model_name"])
                for row in connection.execute(
                    "SELECT DISTINCT model_name FROM embeddings"
                )
            }
            if self.provider.model_name not in model_names:
                available = ", ".join(sorted(model_names)) or "none"
                raise RetrievalError(
                    f"index does not contain model {self.provider.model_name!r}; "
                    f"available models: {available}. Rebuild the index with the "
                    "same --model used for search"
                )

            statement = """
                SELECT
                    c.chunk_id, c.document_id, c.title, c.category, c.section,
                    c.source_url, c.text, c.code_blocks_json, c.token_count,
                    e.dimension, e.vector_json
                FROM chunks AS c
                JOIN embeddings AS e
                  ON e.content_hash = c.content_hash
                 AND e.model_name = c.model_name
                WHERE c.model_name = ?
            """
            parameters: list[str] = [self.provider.model_name]
            if category is not None:
                statement += " AND c.category = ?"
                parameters.append(category)
            rows = connection.execute(statement, parameters).fetchall()

        query_vectors = self.provider.embed([query])
        if len(query_vectors) != 1 or not query_vectors[0]:
            raise RetrievalError("embedding provider did not return one query vector")
        query_vector = query_vectors[0]
        results = []
        for row in rows:
            stored_vector = json.loads(str(row["vector_json"]))
            if not isinstance(stored_vector, list):
                raise RetrievalError(f"invalid vector for chunk {row['chunk_id']}")
            dimension = int(row["dimension"])
            if len(stored_vector) != dimension:
                raise RetrievalError(
                    f"invalid vector dimension for chunk {row['chunk_id']}"
                )
            raw_code_blocks = json.loads(str(row["code_blocks_json"]))
            results.append(
                SearchResult(
                    chunk_id=str(row["chunk_id"]),
                    document_id=str(row["document_id"]),
                    title=str(row["title"]),
                    category=str(row["category"]),
                    section=str(row["section"]),
                    source_url=str(row["source_url"]),
                    text=str(row["text"]),
                    code_blocks=raw_code_blocks,
                    token_count=int(row["token_count"]),
                    score=cosine_similarity(query_vector, stored_vector),
                )
            )
        results.sort(key=lambda result: (-result.score, result.chunk_id))
        return results[:top_k]
