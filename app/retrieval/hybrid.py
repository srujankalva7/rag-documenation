"""Fuse semantic and keyword rankings with Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.keyword import KeywordRetriever, title_phrase_match_count
from app.retrieval.models import HybridSearchResult, SearchResult
from app.retrieval.service import VectorRetriever


@dataclass(slots=True)
class HybridRetriever:
    vector_retriever: VectorRetriever
    keyword_retriever: KeywordRetriever
    vector_weight: float = 1.0
    keyword_weight: float = 1.0
    rrf_k: int = 60
    candidate_k: int = 50
    title_match_boost: float = 0.01

    def __post_init__(self) -> None:
        if self.vector_weight < 0 or self.keyword_weight < 0:
            raise ValueError("retrieval weights cannot be negative")
        if self.vector_weight == 0 and self.keyword_weight == 0:
            raise ValueError("at least one retrieval weight must be positive")
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        if self.candidate_k < 1:
            raise ValueError("candidate_k must be positive")
        if self.title_match_boost < 0:
            raise ValueError("title_match_boost cannot be negative")

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        category: str | None = None,
    ) -> list[HybridSearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        candidate_k = max(top_k, self.candidate_k)
        vector_results = self.vector_retriever.search(
            query, top_k=candidate_k, category=category
        )
        keyword_results = self.keyword_retriever.search(
            query, top_k=candidate_k, category=category
        )
        vector_ranks = {
            result.chunk_id: rank for rank, result in enumerate(vector_results, start=1)
        }
        keyword_ranks = {
            result.chunk_id: rank
            for rank, result in enumerate(keyword_results, start=1)
        }
        vector_by_id = {result.chunk_id: result for result in vector_results}
        keyword_by_id = {result.chunk_id: result for result in keyword_results}
        all_ids = set(vector_by_id).union(keyword_by_id)

        fused = []
        for chunk_id in all_ids:
            vector_rank = vector_ranks.get(chunk_id)
            keyword_rank = keyword_ranks.get(chunk_id)
            vector_result = vector_by_id.get(chunk_id)
            keyword_result = keyword_by_id.get(chunk_id)
            result: SearchResult | None = vector_result or keyword_result
            if result is None:  # Defensive: all_ids comes from these two mappings.
                continue
            phrase_matches = title_phrase_match_count(query, result.title)
            score = self.title_match_boost * phrase_matches
            if vector_rank is not None:
                score += self.vector_weight / (self.rrf_k + vector_rank)
            if keyword_rank is not None:
                score += self.keyword_weight / (self.rrf_k + keyword_rank)
            fused.append(
                HybridSearchResult(
                    result=result,
                    score=score,
                    vector_rank=vector_rank,
                    keyword_rank=keyword_rank,
                    vector_score=vector_result.score if vector_result else None,
                    keyword_score=keyword_result.score if keyword_result else None,
                    title_phrase_matches=phrase_matches,
                )
            )
        fused.sort(key=lambda item: (-item.score, item.result.chunk_id))
        return fused[:top_k]
