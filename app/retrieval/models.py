"""Models returned by documentation retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class RetrievalError(RuntimeError):
    """Raised when the vector index cannot satisfy a search."""


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk_id: str
    document_id: str
    title: str
    category: str
    section: str
    source_url: str
    text: str
    code_blocks: list[dict[str, str | None]]
    token_count: int
    score: float

    @property
    def content(self) -> str:
        rendered_code = "\n\n".join(
            f"```{block.get('language') or ''}\n{block['code']}\n```"
            for block in self.code_blocks
        )
        return f"{self.text.strip()}\n\n{rendered_code}".strip()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["content"] = self.content
        return value


@dataclass(frozen=True, slots=True)
class HybridSearchResult:
    result: SearchResult
    score: float
    vector_rank: int | None
    keyword_rank: int | None
    vector_score: float | None
    keyword_score: float | None
    title_phrase_matches: int = 0

    def to_dict(self) -> dict[str, Any]:
        value = self.result.to_dict()
        value.update(
            {
                "score": self.score,
                "vector_rank": self.vector_rank,
                "keyword_rank": self.keyword_rank,
                "vector_score": self.vector_score,
                "keyword_score": self.keyword_score,
                "title_phrase_matches": self.title_phrase_matches,
            }
        )
        return value
