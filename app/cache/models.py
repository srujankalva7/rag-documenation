"""Models stored by the semantic answer cache."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class CacheStatus(StrEnum):
    disabled = "disabled"
    miss = "miss"
    exact_hit = "exact_hit"
    semantic_hit = "semantic_hit"


@dataclass(frozen=True, slots=True)
class CacheEntry:
    entry_id: int
    normalized_question: str
    scope_key: str
    question: str
    embedding_model: str
    embedding: list[float]
    provider: str
    model: str
    answer_payload: dict[str, Any]
    source_versions: dict[str, str]
    generation_cost_usd: float | None
    created_at: float
    expires_at: float
