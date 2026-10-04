"""BM25-style keyword retrieval over the local chunk index."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass

from app.indexing.store import SQLiteVectorIndex
from app.retrieval.models import RetrievalError, SearchResult

TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "do",
    "does",
    "for",
    "how",
    "i",
    "in",
    "is",
    "of",
    "the",
    "to",
    "what",
    "with",
}


def tokenize(value: str, *, remove_stopwords: bool = False) -> list[str]:
    tokens = TOKEN_PATTERN.findall(value.casefold())
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    return tokens


def title_phrase_match_count(query: str, title: str) -> int:
    query_tokens = tokenize(query, remove_stopwords=True)
    title_text = " ".join(tokenize(title))
    phrases = {
        " ".join(query_tokens[index : index + 2])
        for index in range(len(query_tokens) - 1)
    }
    return sum(phrase in title_text for phrase in phrases)


@dataclass(slots=True)
class KeywordRetriever:
    store: SQLiteVectorIndex
    k1: float = 1.5
    b: float = 0.75
    title_weight: int = 4
    section_weight: int = 2
    title_term_boost: float = 1.5
    title_phrase_boost: float = 3.0

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        category: str | None = None,
    ) -> list[SearchResult]:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if category is not None:
            category = category.strip()
            if not category:
                raise ValueError("category cannot be empty")
        if not self.store.path.exists():
            raise RetrievalError(
                f"vector index not found at {self.store.path}; "
                "run python -m scripts.index_documents"
            )

        with self.store.connect() as connection:
            statement = """
                SELECT chunk_id, document_id, title, category, section,
                       source_url, text, code_blocks_json, token_count
                FROM chunks
            """
            parameters: list[str] = []
            if category is not None:
                statement += " WHERE category = ?"
                parameters.append(category)
            rows = connection.execute(statement, parameters).fetchall()
        if not rows:
            return []

        query_tokens = tokenize(query, remove_stopwords=True)
        if not query_tokens:
            raise ValueError("query must contain at least one searchable term")

        document_tokens: list[list[str]] = []
        document_frequencies: Counter[str] = Counter()
        for row in rows:
            code_blocks = json.loads(str(row["code_blocks_json"]))
            code = " ".join(str(block["code"]) for block in code_blocks)
            tokens = (
                tokenize(str(row["title"])) * self.title_weight
                + tokenize(str(row["section"])) * self.section_weight
                + tokenize(f"{row['text']} {code}")
            )
            document_tokens.append(tokens)
            document_frequencies.update(set(tokens))

        average_length = sum(map(len, document_tokens)) / len(document_tokens)
        document_count = len(rows)
        results = []
        for row, tokens in zip(rows, document_tokens, strict=True):
            frequencies = Counter(tokens)
            length_ratio = len(tokens) / average_length if average_length else 1.0
            score = 0.0
            for term in query_tokens:
                frequency = frequencies[term]
                if frequency == 0:
                    continue
                document_frequency = document_frequencies[term]
                inverse_frequency = math.log(
                    1
                    + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                denominator = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                score += inverse_frequency * frequency * (self.k1 + 1) / denominator

            title_tokens = set(tokenize(str(row["title"])))
            score += self.title_term_boost * len(
                title_tokens.intersection(query_tokens)
            )
            score += self.title_phrase_boost * title_phrase_match_count(
                query, str(row["title"])
            )
            if score <= 0:
                continue
            code_blocks = json.loads(str(row["code_blocks_json"]))
            results.append(
                SearchResult(
                    chunk_id=str(row["chunk_id"]),
                    document_id=str(row["document_id"]),
                    title=str(row["title"]),
                    category=str(row["category"]),
                    section=str(row["section"]),
                    source_url=str(row["source_url"]),
                    text=str(row["text"]),
                    code_blocks=code_blocks,
                    token_count=int(row["token_count"]),
                    score=score,
                )
            )
        results.sort(key=lambda result: (-result.score, result.chunk_id))
        return results[:top_k]
