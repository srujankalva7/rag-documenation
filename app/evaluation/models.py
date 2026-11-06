"""Typed evaluation inputs and outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    question: str
    expected_document_ids: tuple[str, ...]
    category: str | None = None
    expect_supported: bool = True

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvaluationCase:
        case_id = str(value.get("id", "")).strip()
        question = str(value.get("question", "")).strip()
        document_ids = tuple(
            str(item).strip() for item in value.get("expected_document_ids", [])
        )
        if not case_id or not question:
            raise ValueError("evaluation cases require non-empty id and question")
        if not document_ids:
            raise ValueError(f"evaluation case {case_id} requires an expected document")
        if any(not item for item in document_ids):
            raise ValueError(f"evaluation case {case_id} has an empty document id")
        return cls(
            case_id=case_id,
            question=question,
            expected_document_ids=document_ids,
            category=value.get("category"),
            expect_supported=bool(value.get("expect_supported", True)),
        )


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    case_id: str
    preset: str
    repetition: int
    expected_document_ids: tuple[str, ...]
    retrieved_document_ids: tuple[str, ...]
    relevant_rank: int | None
    hit_at_3: bool
    hit_at_5: bool
    reciprocal_rank: float
    citation_correct: bool
    supported_correct: bool
    supported: bool
    cache_status: str
    server_latency_ms: float
    wall_latency_ms: float
    estimated_generation_cost_usd: float
    estimated_avoided_generation_cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    preset: str
    questions: int
    hit_rate_at_3: float
    hit_rate_at_5: float
    mean_reciprocal_rank: float
    citation_accuracy: float
    supported_accuracy: float
    average_server_latency_ms: float
    p95_server_latency_ms: float
    exact_cache_hit_rate: float
    semantic_cache_hit_rate: float
    total_estimated_generation_cost_usd: float
    total_estimated_avoided_generation_cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
