from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from app.indexing.models import IndexCodeBlock, IndexRecord
from app.indexing.store import SQLiteVectorIndex
from app.retrieval.models import RetrievalError
from app.retrieval.service import VectorRetriever, cosine_similarity


@dataclass
class FakeEmbeddingProvider:
    query_vector: list[float]
    model_name: str = "fake-model"
    calls: list[list[str]] = field(default_factory=list)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self.query_vector for _ in texts]


def make_record(
    chunk_id: str,
    *,
    category: str = "tutorial",
    text: str = "Documentation text",
    code_blocks: tuple[IndexCodeBlock, ...] = (),
) -> IndexRecord:
    return IndexRecord(
        chunk_id=chunk_id,
        content_hash=f"{chunk_id}-hash",
        document_id=chunk_id.split(":")[0],
        document_hash="document-hash",
        title=chunk_id.split(":")[0].replace("-", " ").title(),
        category=category,
        section="Example",
        section_index=0,
        chunk_index=0,
        section_chunk_index=0,
        source_url=f"https://fastapi.tiangolo.com/{chunk_id}/",
        text=text,
        code_blocks=code_blocks,
        token_count=10,
    )


def seed_index(
    path: Path,
    records_and_vectors: list[tuple[IndexRecord, list[float]]],
    *,
    model_name: str = "fake-model",
) -> SQLiteVectorIndex:
    store = SQLiteVectorIndex(path)
    with store.connect() as connection:
        store.insert_embeddings(
            connection,
            model_name,
            [(record.content_hash, vector) for record, vector in records_and_vectors],
        )
        store.upsert_chunks(
            connection,
            [record for record, _ in records_and_vectors],
            model_name,
        )
    return store


def test_cosine_similarity() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_search_ranks_results_and_respects_top_k(tmp_path: Path) -> None:
    request_body = make_record("request-body:0")
    dependencies = make_record("dependencies:0")
    testing = make_record("testing:0")
    store = seed_index(
        tmp_path / "vectors.sqlite3",
        [
            (request_body, [1.0, 0.0]),
            (dependencies, [0.8, 0.2]),
            (testing, [0.0, 1.0]),
        ],
    )
    provider = FakeEmbeddingProvider([1.0, 0.0])

    results = VectorRetriever(provider, store).search("request body", top_k=2)

    assert [result.chunk_id for result in results] == [
        "request-body:0",
        "dependencies:0",
    ]
    assert results[0].score == pytest.approx(1.0)
    assert provider.calls == [["request body"]]


def test_search_filters_by_category(tmp_path: Path) -> None:
    tutorial = make_record("request-body:0", category="tutorial")
    advanced = make_record("deployment:0", category="deployment")
    store = seed_index(
        tmp_path / "vectors.sqlite3",
        [(tutorial, [1.0, 0.0]), (advanced, [1.0, 0.0])],
    )

    results = VectorRetriever(FakeEmbeddingProvider([1.0, 0.0]), store).search(
        "deploy", category="deployment"
    )

    assert [result.chunk_id for result in results] == ["deployment:0"]


def test_search_returns_code_only_content(tmp_path: Path) -> None:
    code = IndexCodeBlock(language="python", code="app = FastAPI()")
    record = make_record("first-steps:0", text="", code_blocks=(code,))
    store = seed_index(tmp_path / "vectors.sqlite3", [(record, [1.0, 0.0])])

    result = VectorRetriever(FakeEmbeddingProvider([1.0, 0.0]), store).search(
        "create app"
    )[0]

    assert result.text == ""
    assert result.content == "```python\napp = FastAPI()\n```"


@pytest.mark.parametrize("query,top_k", [("", 5), ("question", 0), ("question", -1)])
def test_search_rejects_invalid_input(tmp_path: Path, query: str, top_k: int) -> None:
    retriever = VectorRetriever(
        FakeEmbeddingProvider([1.0]),
        SQLiteVectorIndex(tmp_path / "missing.sqlite3"),
    )

    with pytest.raises(ValueError):
        retriever.search(query, top_k=top_k)


def test_search_reports_missing_index(tmp_path: Path) -> None:
    retriever = VectorRetriever(
        FakeEmbeddingProvider([1.0]),
        SQLiteVectorIndex(tmp_path / "missing.sqlite3"),
    )

    with pytest.raises(RetrievalError, match="vector index not found"):
        retriever.search("request body")


def test_search_reports_embedding_model_mismatch(tmp_path: Path) -> None:
    record = make_record("request-body:0")
    store = seed_index(
        tmp_path / "vectors.sqlite3",
        [(record, [1.0, 0.0])],
        model_name="indexed-model",
    )

    with pytest.raises(RetrievalError, match="does not contain model"):
        VectorRetriever(FakeEmbeddingProvider([1.0, 0.0]), store).search("request body")


def test_search_reports_dimension_mismatch(tmp_path: Path) -> None:
    record = make_record("request-body:0")
    store = seed_index(tmp_path / "vectors.sqlite3", [(record, [1.0, 0.0])])

    with pytest.raises(RetrievalError, match="dimension mismatch"):
        VectorRetriever(FakeEmbeddingProvider([1.0]), store).search("request body")
