"""HTTP request and response models for documentation search."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator


class SearchMode(StrEnum):
    hybrid = "hybrid"
    vector = "vector"
    keyword = "keyword"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    category: str | None = Field(default=None, max_length=100)
    mode: SearchMode = SearchMode.hybrid
    vector_weight: float = Field(default=1.0, ge=0)
    keyword_weight: float = Field(default=1.0, ge=0)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    candidate_k: int = Field(default=50, ge=1, le=500)
    title_match_boost: float = Field(default=0.01, ge=0)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query cannot be empty")
        return value

    @field_validator("category")
    @classmethod
    def validate_category(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("category cannot be empty")
        return value

    @model_validator(mode="after")
    def validate_hybrid_weights(self) -> SearchRequest:
        if (
            self.mode == SearchMode.hybrid
            and self.vector_weight == 0
            and self.keyword_weight == 0
        ):
            raise ValueError("at least one hybrid retrieval weight must be positive")
        return self


class CodeBlockResponse(BaseModel):
    language: str | None
    code: str


class SearchResultResponse(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    category: str
    section: str
    source_url: str
    text: str
    code_blocks: list[CodeBlockResponse]
    content: str
    token_count: int
    score: float
    vector_rank: int | None = None
    keyword_rank: int | None = None
    vector_score: float | None = None
    keyword_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    count: int
    results: list[SearchResultResponse]


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)
    category: str | None = Field(default=None, max_length=100)
    minimum_score: float = Field(default=0.02, ge=0)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query cannot be empty")
        return value

    @field_validator("category")
    @classmethod
    def validate_category(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("category cannot be empty")
        return value


class CitationResponse(BaseModel):
    number: int
    chunk_id: str
    title: str
    section: str
    source_url: str


class UsageResponse(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class AskResponse(BaseModel):
    query: str
    answer: str
    supported: bool
    citations: list[CitationResponse]
    retrieval_results: list[SearchResultResponse]
    provider: str
    model: str
    usage: UsageResponse
    estimated_cost_usd: float | None
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    index_available: bool
    index_path: str
