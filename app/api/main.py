"""FastAPI application exposing documentation retrieval."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.api.models import (
    AskRequest,
    AskResponse,
    CacheMetadataResponse,
    CitationResponse,
    CostControlsResponse,
    HealthResponse,
    SearchMode,
    SearchRequest,
    SearchResponse,
    SearchResultResponse,
    UsageResponse,
)
from app.cache.service import CachedAnswerService
from app.cache.store import SQLiteSemanticCache
from app.costs.presets import CostPreset, PresetSettings, get_preset
from app.generation.models import GenerationProvider
from app.generation.providers import (
    ExtractiveGenerationProvider,
    OllamaGenerationProvider,
    OpenAIGenerationProvider,
)
from app.generation.service import AnswerService
from app.indexing.providers import EmbeddingProvider, FastEmbedEmbeddingProvider
from app.indexing.store import SQLiteVectorIndex
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.keyword import KeywordRetriever
from app.retrieval.models import HybridSearchResult, RetrievalError, SearchResult
from app.retrieval.service import VectorRetriever

DEFAULT_INDEX_PATH = Path("data/index/vectors.sqlite3")
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_CACHE_PATH = Path("data/cache/semantic_cache.sqlite3")


def _optional_float(name: str) -> float | None:
    value = os.environ.get(name)
    return float(value) if value else None


def _optional_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def _environment_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _default_generation_provider() -> GenerationProvider:
    provider_name = os.environ.get("RAG_GENERATION_PROVIDER", "extractive")
    if provider_name == "extractive":
        return ExtractiveGenerationProvider()
    if provider_name == "ollama":
        return OllamaGenerationProvider(
            model_name=os.environ.get("RAG_GENERATION_MODEL", DEFAULT_OLLAMA_MODEL),
            base_url=os.environ.get("RAG_OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            timeout_seconds=_optional_float("RAG_OLLAMA_TIMEOUT_SECONDS") or 120.0,
        )
    if provider_name == "openai":
        return OpenAIGenerationProvider(
            model_name=os.environ.get("RAG_GENERATION_MODEL", DEFAULT_OPENAI_MODEL),
            input_cost_per_million=_optional_float("RAG_INPUT_COST_PER_MILLION"),
            output_cost_per_million=_optional_float("RAG_OUTPUT_COST_PER_MILLION"),
        )
    raise ValueError(f"unknown RAG_GENERATION_PROVIDER: {provider_name}")


def _preset_generation_provider(settings: PresetSettings) -> GenerationProvider:
    if os.environ.get("RAG_GENERATION_PROVIDER"):
        return _default_generation_provider()
    if settings.generation_provider == "extractive":
        return ExtractiveGenerationProvider()
    return OllamaGenerationProvider(
        model_name=os.environ.get(
            f"RAG_{settings.name.value.upper()}_GENERATION_MODEL",
            settings.generation_model,
        ),
        base_url=os.environ.get("RAG_OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        timeout_seconds=_optional_float("RAG_OLLAMA_TIMEOUT_SECONDS") or 120.0,
    )


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
    generation_provider: GenerationProvider | None = None,
    cache_path: Path | None = None,
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
    configured_cache_path = os.environ.get("RAG_CACHE_PATH")
    resolved_cache_path = cache_path or (
        Path(configured_cache_path)
        if configured_cache_path
        else (
            resolved_path.parent / "semantic_cache.sqlite3"
            if index_path is not None
            else DEFAULT_CACHE_PATH
        )
    )
    cached_answer_services = {}
    for preset in CostPreset:
        settings = get_preset(preset)
        answer_service = AnswerService(
            HybridRetriever(
                vector_retriever,
                keyword_retriever,
                candidate_k=settings.candidate_k,
            ),
            generation_provider or _preset_generation_provider(settings),
        )
        cached_answer_services[preset] = CachedAnswerService(
            answer_service,
            SQLiteSemanticCache(resolved_cache_path),
            embedding_provider,
            store.current_document_hashes,
        )
    cache_enabled = _environment_flag("RAG_CACHE_ENABLED")

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

    @application.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        settings = get_preset(request.preset)
        top_k = request.top_k if request.top_k is not None else settings.top_k
        minimum_score = (
            request.minimum_score
            if request.minimum_score is not None
            else settings.minimum_score
        )
        try:
            result = cached_answer_services[request.preset].ask(
                request.query,
                top_k=top_k,
                category=request.category,
                minimum_score=minimum_score,
                use_cache=request.use_cache and cache_enabled,
                similarity_threshold=(
                    _optional_float("RAG_CACHE_SIMILARITY_THRESHOLD")
                    or settings.cache_similarity_threshold
                ),
                ttl_seconds=_optional_int(
                    "RAG_CACHE_TTL_SECONDS", settings.cache_ttl_seconds
                ),
            )
        except RetrievalError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

        return AskResponse(
            query=result.question,
            answer=result.answer,
            supported=result.supported,
            citations=[
                CitationResponse(
                    number=citation.number,
                    chunk_id=citation.chunk_id,
                    title=citation.title,
                    section=citation.section,
                    source_url=citation.source_url,
                )
                for citation in result.citations
            ],
            retrieval_results=[
                _response_item(item) for item in result.retrieval_results
            ],
            provider=result.provider,
            model=result.model,
            usage=UsageResponse(
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                total_tokens=result.usage.total_tokens,
            ),
            estimated_cost_usd=result.estimated_cost_usd,
            latency_ms=result.latency_ms,
            cache=CacheMetadataResponse(
                status=result.cache_status,
                hit=result.cache_status in {"exact_hit", "semantic_hit"},
                similarity=result.cache_similarity,
                cached_at=result.cached_at,
                avoided_generation_cost_usd=result.avoided_generation_cost_usd,
            ),
            cost_controls=CostControlsResponse(
                preset=settings.name,
                top_k=top_k,
                minimum_score=minimum_score,
                candidate_k=settings.candidate_k,
                cache_similarity_threshold=(
                    _optional_float("RAG_CACHE_SIMILARITY_THRESHOLD")
                    or settings.cache_similarity_threshold
                ),
                cache_ttl_seconds=_optional_int(
                    "RAG_CACHE_TTL_SECONDS", settings.cache_ttl_seconds
                ),
            ),
        )

    return application


app = create_app()
