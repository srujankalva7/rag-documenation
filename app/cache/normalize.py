"""Stable normalization and scoping for cached questions."""

from __future__ import annotations

import json
import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")
_TRAILING_QUESTION_PUNCTUATION = re.compile(r"[?!]+$")


def normalize_question(question: str) -> str:
    """Normalize presentation differences without changing technical identifiers."""
    value = unicodedata.normalize("NFKC", question).casefold().strip()
    value = _WHITESPACE.sub(" ", value)
    return _TRAILING_QUESTION_PUNCTUATION.sub("", value).rstrip()


def cache_scope(
    *,
    category: str | None,
    top_k: int,
    minimum_score: float,
) -> str:
    """Return a deterministic key for settings that can change an answer."""
    return json.dumps(
        {
            "category": category or "*",
            "minimum_score": round(minimum_score, 8),
            "top_k": top_k,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
