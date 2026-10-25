from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.cache.normalize import normalize_question
from app.cache.service import CachedAnswerService
from app.cache.store import SQLiteSemanticCache
from app.generation.models import GenerationOutput, GenerationUsage
from app.generation.service import AnswerService
from app.retrieval.models import HybridSearchResult, SearchResult


def make_result(document_hash: str = "doc-v1") -> HybridSearchResult:
    del document_hash
    result = SearchResult(
        chunk_id="request-body:0",
        document_id="request-body",
        title="Request Body",
        category="tutorial",
        section="Declare it as a parameter",
        source_url="https://fastapi.tiangolo.com/tutorial/body/",
        text="Use a Pydantic model for the request body.",
        code_blocks=[],
        token_count=10,
        score=0.8,
    )
    return HybridSearchResult(
        result=result,
        score=0.04,
        vector_rank=1,
        keyword_rank=1,
        vector_score=0.8,
        keyword_score=10.0,
    )


@dataclass
class CountingRetriever:
    calls: int = 0

    def search(
        self, query: str, *, top_k: int, category: str | None
    ) -> list[HybridSearchResult]:
        del query, top_k, category
        self.calls += 1
        return [make_result()]


@dataclass
class CountingGenerator:
    provider_name: str = "fake"
    model_name: str = "fake-model"
    calls: int = 0

    def generate(
        self, question: str, contexts: list[HybridSearchResult]
    ) -> GenerationOutput:
        del question, contexts
        self.calls += 1
        return GenerationOutput(
            answer="Use a Pydantic model. [1]",
            usage=GenerationUsage(input_tokens=100, output_tokens=20),
            estimated_cost_usd=0.001,
        )


@dataclass
class MappingEmbeddingProvider:
    values: dict[str, list[float]]
    model_name: str = "fake-embedding"
    calls: list[str] = field(default_factory=list)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.extend(texts)
        return [self.values.get(text, [1.0, 0.0]) for text in texts]


def build_service(
    cache_path: Path,
    embeddings: MappingEmbeddingProvider,
    versions: dict[str, str],
) -> tuple[CachedAnswerService, CountingRetriever, CountingGenerator]:
    retriever = CountingRetriever()
    generator = CountingGenerator()
    answer_service = AnswerService(retriever, generator)
    service = CachedAnswerService(
        answer_service,
        SQLiteSemanticCache(cache_path),
        embeddings,
        lambda: dict(versions),
    )
    return service, retriever, generator


def test_normalization_preserves_technical_identifiers() -> None:
    assert normalize_question("  How does GET   /items/{item_id}?  ") == (
        "how does get /items/{item_id}"
    )


def test_exact_hit_skips_embedding_retrieval_and_generation(tmp_path: Path) -> None:
    embeddings = MappingEmbeddingProvider({})
    service, retriever, generator = build_service(
        tmp_path / "cache.sqlite3", embeddings, {"request-body": "doc-v1"}
    )

    first = service.ask("How do I create a request body?")
    second = service.ask("  HOW do I create a request body?  ")

    assert first.cache_status == "miss"
    assert second.cache_status == "exact_hit"
    assert second.usage.total_tokens == 0
    assert second.estimated_cost_usd == 0.0
    assert second.avoided_generation_cost_usd == 0.001
    assert len(embeddings.calls) == 1
    assert retriever.calls == 1
    assert generator.calls == 1


def test_semantic_hit_reuses_paraphrased_question(tmp_path: Path) -> None:
    first_question = "How do I create a request body?"
    paraphrase = "How can a FastAPI endpoint accept JSON input?"
    embeddings = MappingEmbeddingProvider(
        {first_question: [1.0, 0.0], paraphrase: [0.999, 0.01]}
    )
    service, retriever, generator = build_service(
        tmp_path / "cache.sqlite3", embeddings, {"request-body": "doc-v1"}
    )

    service.ask(first_question)
    cached = service.ask(paraphrase)

    assert cached.cache_status == "semantic_hit"
    assert cached.cache_similarity is not None
    assert cached.cache_similarity >= 0.95
    assert retriever.calls == 1
    assert generator.calls == 1


def test_meaningfully_different_question_is_not_reused(tmp_path: Path) -> None:
    first_question = "How do I create a request body?"
    different = "How do I configure CORS?"
    embeddings = MappingEmbeddingProvider(
        {first_question: [1.0, 0.0], different: [0.0, 1.0]}
    )
    service, retriever, generator = build_service(
        tmp_path / "cache.sqlite3", embeddings, {"request-body": "doc-v1"}
    )

    service.ask(first_question)
    result = service.ask(different)

    assert result.cache_status == "miss"
    assert retriever.calls == 2
    assert generator.calls == 2


def test_document_version_change_invalidates_exact_match(tmp_path: Path) -> None:
    versions = {"request-body": "doc-v1"}
    embeddings = MappingEmbeddingProvider({})
    service, retriever, generator = build_service(
        tmp_path / "cache.sqlite3", embeddings, versions
    )
    service.ask("How do I create a request body?")

    versions["request-body"] = "doc-v2"
    result = service.ask("How do I create a request body?")

    assert result.cache_status == "miss"
    assert retriever.calls == 2
    assert generator.calls == 2
