"""Exact-first, version-aware semantic caching around answer generation."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from time import perf_counter
from typing import Protocol

from app.cache.models import CacheEntry, CacheStatus
from app.cache.normalize import cache_scope, normalize_question
from app.cache.store import SQLiteSemanticCache
from app.generation.models import AnswerResult, Citation, GenerationUsage
from app.retrieval.models import HybridSearchResult, SearchResult


class AnswerProvider(Protocol):
    @property
    def provider(self): ...

    def ask(
        self,
        question: str,
        *,
        top_k: int,
        category: str | None,
        minimum_score: float,
    ) -> AnswerResult: ...


class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return -1.0
    return dot / (left_norm * right_norm)


def _serialize_answer(answer: AnswerResult) -> dict[str, object]:
    results = []
    for item in answer.retrieval_results:
        result = item.result
        results.append(
            {
                "result": {
                    "chunk_id": result.chunk_id,
                    "document_id": result.document_id,
                    "title": result.title,
                    "category": result.category,
                    "section": result.section,
                    "source_url": result.source_url,
                    "text": result.text,
                    "code_blocks": result.code_blocks,
                    "token_count": result.token_count,
                    "score": result.score,
                },
                "score": item.score,
                "vector_rank": item.vector_rank,
                "keyword_rank": item.keyword_rank,
                "vector_score": item.vector_score,
                "keyword_score": item.keyword_score,
                "title_phrase_matches": item.title_phrase_matches,
            }
        )
    return {
        "question": answer.question,
        "answer": answer.answer,
        "supported": answer.supported,
        "citations": [
            {
                "number": value.number,
                "chunk_id": value.chunk_id,
                "title": value.title,
                "section": value.section,
                "source_url": value.source_url,
            }
            for value in answer.citations
        ],
        "retrieval_results": results,
        "provider": answer.provider,
        "model": answer.model,
    }


def _deserialize_answer(entry: CacheEntry) -> AnswerResult:
    value = entry.answer_payload
    results = []
    for item in value["retrieval_results"]:
        result = SearchResult(**item["result"])
        results.append(
            HybridSearchResult(
                result=result,
                score=item["score"],
                vector_rank=item["vector_rank"],
                keyword_rank=item["keyword_rank"],
                vector_score=item["vector_score"],
                keyword_score=item["keyword_score"],
                title_phrase_matches=item["title_phrase_matches"],
            )
        )
    return AnswerResult(
        question=value["question"],
        answer=value["answer"],
        supported=value["supported"],
        citations=[Citation(**citation) for citation in value["citations"]],
        retrieval_results=results,
        provider=value["provider"],
        model=value["model"],
        usage=GenerationUsage(),
        estimated_cost_usd=0.0,
        latency_ms=0.0,
    )


class CachedAnswerService:
    def __init__(
        self,
        answer_service: AnswerProvider,
        cache: SQLiteSemanticCache,
        embedding_provider: EmbeddingProvider,
        current_document_hashes: Callable[[], dict[str, str]],
        *,
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 86_400,
    ) -> None:
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be positive")
        self.answer_service = answer_service
        self.cache = cache
        self.embedding_provider = embedding_provider
        self.current_document_hashes = current_document_hashes
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds

    @property
    def _provider_name(self) -> str:
        return self.answer_service.provider.provider_name

    @property
    def _model_name(self) -> str:
        return self.answer_service.provider.model_name

    @staticmethod
    def _is_current(entry: CacheEntry, current: dict[str, str]) -> bool:
        return bool(entry.source_versions) and all(
            current.get(document_id) == document_hash
            for document_id, document_hash in entry.source_versions.items()
        )

    def _hit(
        self,
        entry: CacheEntry,
        status: CacheStatus,
        started: float,
        similarity: float,
        question: str,
    ) -> AnswerResult:
        cached = _deserialize_answer(entry)
        return replace(
            cached,
            question=question,
            latency_ms=(perf_counter() - started) * 1000,
            cache_status=status.value,
            cache_similarity=similarity,
            cached_at=datetime.fromtimestamp(entry.created_at, UTC).isoformat(),
            avoided_generation_cost_usd=entry.generation_cost_usd,
        )

    def ask(
        self,
        question: str,
        *,
        top_k: int = 5,
        category: str | None = None,
        minimum_score: float = 0.02,
        use_cache: bool = True,
    ) -> AnswerResult:
        if not use_cache:
            return self.answer_service.ask(
                question,
                top_k=top_k,
                category=category,
                minimum_score=minimum_score,
            )

        started = perf_counter()
        normalized = normalize_question(question)
        scope = cache_scope(
            category=category,
            top_k=top_k,
            minimum_score=minimum_score,
        )
        current_versions = self.current_document_hashes()
        exact = self.cache.find_exact(
            normalized,
            scope,
            self._provider_name,
            self._model_name,
        )
        if exact is not None:
            if self._is_current(exact, current_versions):
                self.cache.record_event(
                    "exact_hit", entry_id=exact.entry_id, similarity=1.0
                )
                return self._hit(exact, CacheStatus.exact_hit, started, 1.0, question)
            self.cache.invalidate(exact.entry_id, "document_version_changed")
            self.cache.record_event(
                "rejected",
                reason="document_version_changed",
                entry_id=exact.entry_id,
            )

        query_embedding = self.embedding_provider.embed([question])[0]
        candidates = self.cache.semantic_candidates(
            scope,
            self._provider_name,
            self._model_name,
            self.embedding_provider.model_name,
        )
        ranked = sorted(
            (
                (_cosine_similarity(query_embedding, entry.embedding), entry)
                for entry in candidates
            ),
            key=lambda value: value[0],
            reverse=True,
        )
        for similarity, entry in ranked:
            if similarity < self.similarity_threshold:
                break
            if not self._is_current(entry, current_versions):
                self.cache.invalidate(entry.entry_id, "document_version_changed")
                self.cache.record_event(
                    "rejected",
                    reason="document_version_changed",
                    entry_id=entry.entry_id,
                    similarity=similarity,
                )
                continue
            self.cache.record_event(
                "semantic_hit", entry_id=entry.entry_id, similarity=similarity
            )
            return self._hit(
                entry, CacheStatus.semantic_hit, started, similarity, question
            )
        if ranked:
            self.cache.record_event(
                "rejected",
                reason="below_similarity_threshold",
                entry_id=ranked[0][1].entry_id,
                similarity=ranked[0][0],
            )

        generated = self.answer_service.ask(
            question,
            top_k=top_k,
            category=category,
            minimum_score=minimum_score,
        )
        self.cache.record_event("miss", reason="no_reusable_entry")
        if generated.supported:
            source_versions = {
                item.result.document_id: current_versions[item.result.document_id]
                for item in generated.retrieval_results
                if item.result.document_id in current_versions
            }
            if source_versions:
                self.cache.put(
                    normalized_question=normalized,
                    scope_key=scope,
                    question=question,
                    embedding_model=self.embedding_provider.model_name,
                    embedding=query_embedding,
                    provider=generated.provider,
                    model=generated.model,
                    answer_payload=_serialize_answer(generated),
                    source_versions=source_versions,
                    generation_cost_usd=generated.estimated_cost_usd,
                    ttl_seconds=self.ttl_seconds,
                )
        return replace(generated, cache_status=CacheStatus.miss.value)
