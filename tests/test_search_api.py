from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.generation.models import GenerationOutput, GenerationUsage
from app.indexing.models import IndexCodeBlock, IndexRecord
from app.indexing.store import SQLiteVectorIndex
from app.retrieval.models import HybridSearchResult


@dataclass
class FakeEmbeddingProvider:
    query_vector: list[float]
    model_name: str = "fake-model"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.query_vector for _ in texts]


@dataclass
class FakeGenerationProvider:
    provider_name: str = "fake"
    model_name: str = "fake-model"
    calls: int = 0

    def generate(
        self,
        question: str,
        contexts: list[HybridSearchResult],
    ) -> GenerationOutput:
        del question, contexts
        self.calls += 1
        return GenerationOutput(
            answer="Declare a Pydantic model and use it as a parameter. [1]",
            usage=GenerationUsage(input_tokens=80, output_tokens=12),
            estimated_cost_usd=0.00002,
        )


def make_record(
    chunk_id: str,
    title: str,
    section: str,
    text: str,
    vector: list[float],
    *,
    category: str = "tutorial",
    code_blocks: tuple[IndexCodeBlock, ...] = (),
) -> tuple[IndexRecord, list[float]]:
    return (
        IndexRecord(
            chunk_id=chunk_id,
            content_hash=f"{chunk_id}-hash",
            document_id=chunk_id.split(":")[0],
            document_hash="document-hash",
            title=title,
            category=category,
            section=section,
            section_index=0,
            chunk_index=0,
            section_chunk_index=0,
            source_url=f"https://fastapi.tiangolo.com/{chunk_id}/",
            text=text,
            code_blocks=code_blocks,
            token_count=10,
        ),
        vector,
    )


def seed_index(
    path: Path,
    records_and_vectors: list[tuple[IndexRecord, list[float]]],
) -> None:
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


def test_health_reports_index_availability(tmp_path: Path) -> None:
    index_path = tmp_path / "vectors.sqlite3"
    client = TestClient(
        create_app(
            index_path=index_path,
            provider=FakeEmbeddingProvider([1.0, 0.0]),
        )
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "index_available": False,
        "index_path": str(index_path),
    }


def test_search_returns_hybrid_results_with_citations(tmp_path: Path) -> None:
    index_path = tmp_path / "vectors.sqlite3"
    code = IndexCodeBlock(language="python", code="item: Item")
    seed_index(
        index_path,
        [
            make_record(
                "body-multiple:0",
                "Body - Multiple Parameters",
                "Mix parameters",
                "Mix request body and query parameters.",
                [1.0, 0.0],
            ),
            make_record(
                "request-body:0",
                "Request Body",
                "Declare it as a parameter",
                "Declare a request body with a Pydantic model.",
                [0.9, 0.1],
                code_blocks=(code,),
            ),
        ],
    )
    client = TestClient(
        create_app(
            index_path=index_path,
            provider=FakeEmbeddingProvider([1.0, 0.0]),
        )
    )

    response = client.post(
        "/search",
        json={
            "query": "How do I create a request body?",
            "top_k": 2,
            "category": "tutorial",
        },
    )

    assert response.status_code == 200
    value = response.json()
    assert value["mode"] == "hybrid"
    assert value["count"] == 2
    assert value["results"][0]["title"] == "Request Body"
    assert value["results"][0]["vector_rank"] is not None
    assert value["results"][0]["keyword_rank"] is not None
    assert value["results"][0]["source_url"].startswith("https://fastapi.tiangolo.com/")
    assert "item: Item" in value["results"][0]["content"]


def test_search_supports_vector_and_keyword_modes(tmp_path: Path) -> None:
    index_path = tmp_path / "vectors.sqlite3"
    seed_index(
        index_path,
        [
            make_record(
                "request-body:0",
                "Request Body",
                "Request Body",
                "Declare a request body.",
                [1.0, 0.0],
            )
        ],
    )
    client = TestClient(
        create_app(
            index_path=index_path,
            provider=FakeEmbeddingProvider([1.0, 0.0]),
        )
    )

    vector = client.post("/search", json={"query": "request body", "mode": "vector"})
    keyword = client.post("/search", json={"query": "request body", "mode": "keyword"})

    assert vector.status_code == 200
    assert vector.json()["results"][0]["vector_rank"] is None
    assert keyword.status_code == 200
    assert keyword.json()["mode"] == "keyword"


def test_search_validates_requests(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            index_path=tmp_path / "vectors.sqlite3",
            provider=FakeEmbeddingProvider([1.0]),
        )
    )

    assert client.post("/search", json={"query": "   "}).status_code == 422
    assert (
        client.post("/search", json={"query": "question", "top_k": 0}).status_code
        == 422
    )
    assert (
        client.post(
            "/search",
            json={
                "query": "question",
                "vector_weight": 0,
                "keyword_weight": 0,
            },
        ).status_code
        == 422
    )


def test_search_returns_service_unavailable_when_index_is_missing(
    tmp_path: Path,
) -> None:
    client = TestClient(
        create_app(
            index_path=tmp_path / "missing.sqlite3",
            provider=FakeEmbeddingProvider([1.0]),
        )
    )

    response = client.post("/search", json={"query": "request body"})

    assert response.status_code == 503
    assert "vector index not found" in response.json()["detail"]


def test_ask_returns_grounded_answer_citations_and_usage(tmp_path: Path) -> None:
    index_path = tmp_path / "vectors.sqlite3"
    seed_index(
        index_path,
        [
            make_record(
                "request-body:0",
                "Request Body",
                "Declare it as a parameter",
                "Declare a request body with a Pydantic model.",
                [1.0, 0.0],
            )
        ],
    )
    generation_provider = FakeGenerationProvider()
    client = TestClient(
        create_app(
            index_path=index_path,
            provider=FakeEmbeddingProvider([1.0, 0.0]),
            generation_provider=generation_provider,
        )
    )

    response = client.post(
        "/ask",
        json={
            "query": "How do I create a request body?",
            "top_k": 1,
            "category": "tutorial",
        },
    )

    assert response.status_code == 200
    value = response.json()
    assert value["supported"] is True
    assert value["answer"].endswith("[1]")
    assert value["citations"][0]["title"] == "Request Body"
    assert value["retrieval_results"][0]["source_url"].startswith(
        "https://fastapi.tiangolo.com/"
    )
    assert value["usage"] == {
        "input_tokens": 80,
        "output_tokens": 12,
        "total_tokens": 92,
    }
    assert value["estimated_cost_usd"] == 0.00002
    assert value["latency_ms"] >= 0
    assert generation_provider.calls == 1


def test_ask_declines_low_scoring_evidence_without_generation(tmp_path: Path) -> None:
    index_path = tmp_path / "vectors.sqlite3"
    seed_index(
        index_path,
        [
            make_record(
                "request-body:0",
                "Request Body",
                "Request Body",
                "Request body documentation.",
                [1.0, 0.0],
            )
        ],
    )
    generation_provider = FakeGenerationProvider()
    client = TestClient(
        create_app(
            index_path=index_path,
            provider=FakeEmbeddingProvider([1.0, 0.0]),
            generation_provider=generation_provider,
        )
    )

    response = client.post(
        "/ask",
        json={"query": "What is the weather?", "minimum_score": 1.0},
    )

    assert response.status_code == 200
    value = response.json()
    assert value["supported"] is False
    assert value["citations"] == []
    assert value["usage"]["total_tokens"] == 0
    assert value["estimated_cost_usd"] == 0.0
    assert generation_provider.calls == 0
