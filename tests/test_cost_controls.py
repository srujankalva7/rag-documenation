from __future__ import annotations

from app.costs.presets import CostPreset, get_preset


def test_presets_increase_retrieval_effort() -> None:
    economy = get_preset(CostPreset.economy)
    balanced = get_preset(CostPreset.balanced)
    accuracy = get_preset(CostPreset.accuracy)

    assert economy.top_k < balanced.top_k < accuracy.top_k
    assert economy.candidate_k < balanced.candidate_k < accuracy.candidate_k
    assert economy.generation_provider == "extractive"
    assert balanced.generation_model == "qwen2.5:7b"
    assert accuracy.generation_model == "qwen2.5:14b"


def test_economy_uses_longer_cache_lifetime() -> None:
    economy = get_preset("economy")
    accuracy = get_preset("accuracy")

    assert economy.cache_ttl_seconds > accuracy.cache_ttl_seconds
    assert economy.cache_similarity_threshold < accuracy.cache_similarity_threshold
