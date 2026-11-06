"""Calculate per-question and aggregate RAG metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from app.evaluation.models import (
    EvaluationCase,
    EvaluationResult,
    EvaluationSummary,
)


def _relevant_rank(retrieved: Sequence[str], expected: set[str]) -> int | None:
    return next(
        (
            rank
            for rank, document_id in enumerate(retrieved, start=1)
            if document_id in expected
        ),
        None,
    )


def evaluate_response(
    case: EvaluationCase,
    preset: str,
    repetition: int,
    response: dict[str, Any],
    *,
    wall_latency_ms: float,
) -> EvaluationResult:
    retrieval_results = response.get("retrieval_results", [])
    retrieved = tuple(str(item["document_id"]) for item in retrieval_results)
    expected = set(case.expected_document_ids)
    rank = _relevant_rank(retrieved, expected)
    relevant_chunks = {
        str(item["chunk_id"])
        for item in retrieval_results
        if str(item["document_id"]) in expected
    }
    cited_chunks = {str(item["chunk_id"]) for item in response.get("citations", [])}
    cache = response.get("cache", {})
    cost = response.get("estimated_cost_usd")
    avoided = cache.get("avoided_generation_cost_usd")
    supported = bool(response.get("supported"))
    return EvaluationResult(
        case_id=case.case_id,
        preset=preset,
        repetition=repetition,
        expected_document_ids=case.expected_document_ids,
        retrieved_document_ids=retrieved,
        relevant_rank=rank,
        hit_at_3=rank is not None and rank <= 3,
        hit_at_5=rank is not None and rank <= 5,
        reciprocal_rank=0.0 if rank is None else 1.0 / rank,
        citation_correct=bool(relevant_chunks & cited_chunks),
        supported_correct=supported == case.expect_supported,
        supported=supported,
        cache_status=str(cache.get("status", "unknown")),
        server_latency_ms=float(response.get("latency_ms", 0.0)),
        wall_latency_ms=wall_latency_ms,
        estimated_generation_cost_usd=float(cost or 0.0),
        estimated_avoided_generation_cost_usd=float(avoided or 0.0),
    )


def _rate(results: Sequence[EvaluationResult], attribute: str) -> float:
    return sum(bool(getattr(result, attribute)) for result in results) / len(results)


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def summarize(results: Sequence[EvaluationResult], preset: str) -> EvaluationSummary:
    selected = [result for result in results if result.preset == preset]
    if not selected:
        raise ValueError(f"no evaluation results for preset {preset}")
    count = len(selected)
    return EvaluationSummary(
        preset=preset,
        questions=count,
        hit_rate_at_3=_rate(selected, "hit_at_3"),
        hit_rate_at_5=_rate(selected, "hit_at_5"),
        mean_reciprocal_rank=sum(item.reciprocal_rank for item in selected) / count,
        citation_accuracy=_rate(selected, "citation_correct"),
        supported_accuracy=_rate(selected, "supported_correct"),
        average_server_latency_ms=(
            sum(item.server_latency_ms for item in selected) / count
        ),
        p95_server_latency_ms=_p95([item.server_latency_ms for item in selected]),
        exact_cache_hit_rate=(
            sum(item.cache_status == "exact_hit" for item in selected) / count
        ),
        semantic_cache_hit_rate=(
            sum(item.cache_status == "semantic_hit" for item in selected) / count
        ),
        total_estimated_generation_cost_usd=sum(
            item.estimated_generation_cost_usd for item in selected
        ),
        total_estimated_avoided_generation_cost_usd=sum(
            item.estimated_avoided_generation_cost_usd for item in selected
        ),
    )
