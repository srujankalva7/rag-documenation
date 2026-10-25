"""Retrieve evidence and generate a citation-grounded answer."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from app.generation.models import (
    AnswerResult,
    Citation,
    GenerationProvider,
    GenerationUsage,
)
from app.retrieval.models import HybridSearchResult

UNSUPPORTED_ANSWER = (
    "I don't have enough evidence in the indexed FastAPI documentation to answer "
    "that question."
)


class RetrievalProvider(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int,
        category: str | None,
    ) -> list[HybridSearchResult]: ...


@dataclass(slots=True)
class AnswerService:
    retriever: RetrievalProvider
    provider: GenerationProvider

    def ask(
        self,
        question: str,
        *,
        top_k: int = 5,
        category: str | None = None,
        minimum_score: float = 0.02,
    ) -> AnswerResult:
        if minimum_score < 0:
            raise ValueError("minimum_score cannot be negative")
        started = perf_counter()
        results = self.retriever.search(
            question,
            top_k=top_k,
            category=category,
        )
        if not results or results[0].score < minimum_score:
            return AnswerResult(
                question=question,
                answer=UNSUPPORTED_ANSWER,
                supported=False,
                citations=[],
                retrieval_results=results,
                provider=self.provider.provider_name,
                model=self.provider.model_name,
                usage=GenerationUsage(),
                estimated_cost_usd=0.0,
                latency_ms=(perf_counter() - started) * 1000,
            )

        generated = self.provider.generate(question, results)
        citations = []
        for number, item in enumerate(results, start=1):
            result = item.result
            citations.append(
                Citation(
                    number=number,
                    chunk_id=result.chunk_id,
                    title=result.title,
                    section=result.section,
                    source_url=result.source_url,
                )
            )
        return AnswerResult(
            question=question,
            answer=generated.answer,
            supported=True,
            citations=citations,
            retrieval_results=results,
            provider=self.provider.provider_name,
            model=self.provider.model_name,
            usage=generated.usage,
            estimated_cost_usd=generated.estimated_cost_usd,
            latency_ms=(perf_counter() - started) * 1000,
        )
