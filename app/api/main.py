"""FastAPI application exposing documentation retrieval."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.api.models import (
    HealthResponse,
    SearchMode,
    SearchRequest,
    SearchResponse,
    SearchResultResponse,
)
from app.indexing.providers import EmbeddingProvider, FastEmbedEmbeddingProvider
from app.indexing.store import SQLiteVectorIndex
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.keyword import KeywordRetriever
from app.retrieval.models import HybridSearchResult, RetrievalError, SearchResult
from app.retrieval.service import VectorRetriever

DEFAULT_INDEX_PATH = Path("data/index/vectors.sqlite3")
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def _response_item(
    value: SearchResult | HybridSearchResult,
) -> SearchResultResponse:
    if isinstance(value, HybridSearchResult):
        source = value.result
        source_value = source.to_dict()
        source_value["score"] = value.score
        return SearchResultResponse(
            **source_value,
            vector_rank=value.vector_rank,
            keyword_rank=value.keyword_rank,
            vector_score=value.vector_score,
            keyword_score=value.keyword_score,
        )
    return SearchResultResponse(**value.to_dict())


def create_app(
    *,
    index_path: Path | None = None,
    provider: EmbeddingProvider | None = None,
) -> FastAPI:
    resolved_path = index_path or Path(
        os.environ.get("RAG_INDEX_PATH", str(DEFAULT_INDEX_PATH))
    )
    embedding_provider = provider or FastEmbedEmbeddingProvider(
        os.environ.get("RAG_EMBEDDING_MODEL", DEFAULT_MODEL)
    )
    store = SQLiteVectorIndex(resolved_path)
    vector_retriever = VectorRetriever(embedding_provider, store)
    keyword_retriever = KeywordRetriever(store)

    application = FastAPI(
        title="FastAPI Documentation RAG",
        description="Search indexed FastAPI documentation with hybrid retrieval.",
        version="0.1.0",
    )

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            index_available=resolved_path.is_file(),
            index_path=str(resolved_path),
        )

    @application.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        try:
            if request.mode == SearchMode.vector:
                results: list[SearchResult | HybridSearchResult] = (
                    vector_retriever.search(
                        request.query,
                        top_k=request.top_k,
                        category=request.category,
                    )
                )
            elif request.mode == SearchMode.keyword:
                results = keyword_retriever.search(
                    request.query,
                    top_k=request.top_k,
                    category=request.category,
                )
            else:
                results = HybridRetriever(
                    vector_retriever,
                    keyword_retriever,
                    vector_weight=request.vector_weight,
                    keyword_weight=request.keyword_weight,
                    rrf_k=request.rrf_k,
                    candidate_k=request.candidate_k,
                    title_match_boost=request.title_match_boost,
                ).search(
                    request.query,
                    top_k=request.top_k,
                    category=request.category,
                )
        except RetrievalError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        response_results = [_response_item(result) for result in results]
        return SearchResponse(
            query=request.query,
            mode=request.mode,
            count=len(response_results),
            results=response_results,
        )

    return application


app = create_app()
