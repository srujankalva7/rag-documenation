from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from app.cache.store import SQLiteSemanticCache
from scripts.invalidate_cache import run


def add_entry(
    cache: SQLiteSemanticCache,
    question: str,
    document_id: str,
    *,
    provider: str = "ollama",
    model: str = "qwen2.5:7b",
    now: float = 1_000.0,
    ttl_seconds: int = 3_600,
) -> int:
    return cache.put(
        normalized_question=question.lower(),
        scope_key="tutorial",
        question=question,
        embedding_model="fake-embedding",
        embedding=[1.0, 0.0],
        provider=provider,
        model=model,
        answer_payload={},
        source_versions={document_id: f"{document_id}-hash"},
        generation_cost_usd=0.0,
        ttl_seconds=ttl_seconds,
        now=now,
    )


def arguments(cache_path: Path, **overrides: object) -> Namespace:
    values = {
        "cache_path": cache_path,
        "document_id": [],
        "provider": None,
        "model": None,
        "older_than_days": None,
        "expired": False,
        "all": False,
        "apply": False,
        "reason": "manual_invalidation",
    }
    values.update(overrides)
    return Namespace(**values)


def test_document_invalidation_preserves_unrelated_entries(tmp_path: Path) -> None:
    cache = SQLiteSemanticCache(tmp_path / "cache.sqlite3")
    body_id = add_entry(cache, "request body", "request-body")
    cors_id = add_entry(cache, "configure cors", "cors")

    matches = cache.matching_entry_ids(document_ids={"request-body"})
    invalidated = cache.invalidate_entries(matches, "document_changed", now=2_000)

    assert matches == [body_id]
    assert invalidated == 1
    assert cache.matching_entry_ids() == [cors_id]


def test_expired_provider_and_model_filters_are_combined(tmp_path: Path) -> None:
    cache = SQLiteSemanticCache(tmp_path / "cache.sqlite3")
    expired_id = add_entry(
        cache,
        "expired ollama",
        "request-body",
        now=100,
        ttl_seconds=10,
    )
    add_entry(
        cache,
        "active ollama",
        "request-body",
        now=150,
        ttl_seconds=1_000,
    )
    add_entry(
        cache,
        "expired other provider",
        "request-body",
        provider="extractive",
        model="retrieved-evidence",
        now=100,
        ttl_seconds=10,
    )

    matches = cache.matching_entry_ids(
        provider="ollama",
        model="qwen2.5:7b",
        expired_before=200,
    )

    assert matches == [expired_id]


def test_cli_previews_before_applying_invalidation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_path = tmp_path / "cache.sqlite3"
    cache = SQLiteSemanticCache(cache_path)
    entry_id = add_entry(cache, "request body", "request-body")
    preview = arguments(cache_path, document_id=["request-body"])

    assert run(preview, now=2_000) == 0
    assert cache.matching_entry_ids() == [entry_id]
    assert "Preview only" in capsys.readouterr().out

    apply = arguments(
        cache_path,
        document_id=["request-body"],
        apply=True,
        reason="documentation_updated",
    )
    assert run(apply, now=2_000) == 0
    assert cache.matching_entry_ids() == []
    assert "Invalidated entries: 1" in capsys.readouterr().out


def test_cli_requires_filter_or_explicit_all(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.sqlite3"
    SQLiteSemanticCache(cache_path).connect().close()

    with pytest.raises(ValueError, match="select at least one filter"):
        run(arguments(cache_path), now=2_000)
