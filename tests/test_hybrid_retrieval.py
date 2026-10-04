from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from app.indexing.models import IndexRecord
from app.indexing.store import SQLiteVectorIndex
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.keyword import KeywordRetriever
from app.retrieval.service import VectorRetriever


@dataclass
class FakeEmbeddingProvider:
    query_vector: list[float]
    model_name: str = "fake-model"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.query_vector for _ in texts]


def make_record(chunk_id: str, title: str, section: str, text: str) -> IndexRecord:
    return IndexRecord(
        chunk_id=chunk_id,
        content_hash=f"{chunk_id}-hash",
        document_id=chunk_id.split(":")[0],
        document_hash="document-hash",
        title=title,
        category="tutorial",
        section=section,
        section_index=0,
        chunk_index=0,
        section_chunk_index=0,
        source_url=f"https://fastapi.tiangolo.com/{chunk_id}/",
        text=text,
        code_blocks=(),
        token_count=10,
    )


def seed_index(
    path: Path, records_and_vectors: list[tuple[IndexRecord, list[float]]]
) -> SQLiteVectorIndex:
    store = SQLiteVectorIndex(path)
    with store.connect() as connection:
        store.insert_embeddings(
            connection,
            "fake-model",
            [(record.content_hash, vector) for record, vector in records_and_vectors],
        )
        store.upsert_chunks(
            connection,
            [record for record, _ in records_and_vectors],
            "fake-model",
        )
    return store


def test_hybrid_search_fixes_request_body_vector_ranking(tmp_path: Path) -> None:
    records_and_vectors = [
        (
            make_record(
                "body-multiple:0",
                "Body - Multiple Parameters",
                "Mix Path, Query and body parameters",
                "Mix request body and query parameters.",
            ),
            [1.0, 0.0],
        ),
        (
            make_record(
                "stream-json:0",
                "Stream JSON Lines",
                "Stream JSON Lines",
                "Send a stream of JSON lines.",
            ),
            [0.99, 0.01],
        ),
        (
            make_record(
                "first-steps:0",
                "First Steps",
                "Create a FastAPI instance",
                "Create the application.",
            ),
            [0.98, 0.02],
        ),
        (
            make_record(
                "direct-request:0",
                "Using the Request Directly",
                "Request object",
                "Read the request body directly.",
            ),
            [0.97, 0.03],
        ),
        (
            make_record(
                "request-body:0",
                "Request Body",
                "Create your data model",
                "Declare a request body with a Pydantic model.",
            ),
            [0.96, 0.04],
        ),
    ]
    store = seed_index(tmp_path / "vectors.sqlite3", records_and_vectors)
    provider = FakeEmbeddingProvider([1.0, 0.0])
    vector = VectorRetriever(provider, store)
    query = "How do I create a request body in FastAPI?"

    vector_results = vector.search(query, top_k=5)
    hybrid_results = HybridRetriever(
        vector_retriever=vector,
        keyword_retriever=KeywordRetriever(store),
    ).search(query, top_k=5)

    assert vector_results[-1].chunk_id == "request-body:0"
    assert hybrid_results[0].result.chunk_id == "request-body:0"
    assert hybrid_results[0].title_phrase_matches == 1
    assert hybrid_results[0].vector_rank == 5
    assert hybrid_results[0].keyword_rank == 1


def test_hybrid_result_serializes_component_scores(tmp_path: Path) -> None:
    record = make_record(
        "request-body:0",
        "Request Body",
        "Request Body",
        "Declare a request body.",
    )
    store = seed_index(tmp_path / "vectors.sqlite3", [(record, [1.0, 0.0])])
    result = HybridRetriever(
        VectorRetriever(FakeEmbeddingProvider([1.0, 0.0]), store),
        KeywordRetriever(store),
    ).search("request body", top_k=1)[0]

    value = result.to_dict()

    assert value["chunk_id"] == "request-body:0"
    assert value["vector_rank"] == 1
    assert value["keyword_rank"] == 1
    assert value["vector_score"] == pytest.approx(1.0)


def test_hybrid_rejects_invalid_configuration(tmp_path: Path) -> None:
    store = SQLiteVectorIndex(tmp_path / "vectors.sqlite3")
    vector = VectorRetriever(FakeEmbeddingProvider([1.0]), store)

    with pytest.raises(ValueError, match="at least one"):
        HybridRetriever(
            vector,
            KeywordRetriever(store),
            vector_weight=0,
            keyword_weight=0,
        )
