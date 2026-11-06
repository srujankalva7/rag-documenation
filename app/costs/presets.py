"""Cost-control presets for retrieval, caching, and answer generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CostPreset(StrEnum):
    economy = "economy"
    balanced = "balanced"
    accuracy = "accuracy"


@dataclass(frozen=True, slots=True)
class PresetSettings:
    name: CostPreset
    top_k: int
    minimum_score: float
    candidate_k: int
    cache_similarity_threshold: float
    cache_ttl_seconds: int
    generation_provider: str
    generation_model: str


PRESETS = {
    CostPreset.economy: PresetSettings(
        name=CostPreset.economy,
        top_k=3,
        minimum_score=0.025,
        candidate_k=25,
        cache_similarity_threshold=0.92,
        cache_ttl_seconds=604_800,
        generation_provider="extractive",
        generation_model="retrieved-evidence",
    ),
    CostPreset.balanced: PresetSettings(
        name=CostPreset.balanced,
        top_k=5,
        minimum_score=0.02,
        candidate_k=50,
        cache_similarity_threshold=0.95,
        cache_ttl_seconds=86_400,
        generation_provider="ollama",
        generation_model="qwen2.5:7b",
    ),
    CostPreset.accuracy: PresetSettings(
        name=CostPreset.accuracy,
        top_k=8,
        minimum_score=0.015,
        candidate_k=100,
        cache_similarity_threshold=0.98,
        cache_ttl_seconds=21_600,
        generation_provider="ollama",
        generation_model="qwen2.5:14b",
    ),
}


def get_preset(value: CostPreset | str) -> PresetSettings:
    return PRESETS[CostPreset(value)]
