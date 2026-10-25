"""Models used by grounded answer generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.retrieval.models import HybridSearchResult


@dataclass(frozen=True, slots=True)
class GenerationUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class GenerationOutput:
    answer: str
    usage: GenerationUsage = GenerationUsage()
    estimated_cost_usd: float | None = None


class GenerationProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    def generate(
        self,
        question: str,
        contexts: list[HybridSearchResult],
    ) -> GenerationOutput: ...


@dataclass(frozen=True, slots=True)
class Citation:
    number: int
    chunk_id: str
    title: str
    section: str
    source_url: str


@dataclass(frozen=True, slots=True)
class AnswerResult:
    question: str
    answer: str
    supported: bool
    citations: list[Citation]
    retrieval_results: list[HybridSearchResult]
    provider: str
    model: str
    usage: GenerationUsage
    estimated_cost_usd: float | None
    latency_ms: float
