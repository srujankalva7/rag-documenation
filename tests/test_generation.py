from __future__ import annotations

from dataclasses import dataclass, field

from app.generation.models import GenerationOutput, GenerationUsage
from app.generation.prompts import SYSTEM_PROMPT, build_grounded_prompt
from app.generation.providers import (
    ExtractiveGenerationProvider,
    OpenAIGenerationProvider,
)
from app.generation.service import UNSUPPORTED_ANSWER, AnswerService
from app.retrieval.models import HybridSearchResult, SearchResult


def make_hybrid_result(
    *,
    score: float = 0.04,
    source_url: str = "https://fastapi.tiangolo.com/tutorial/body/#request-body",
) -> HybridSearchResult:
    result = SearchResult(
        chunk_id="request-body:0",
        document_id="request-body",
        title="Request Body",
        category="tutorial",
        section="Request Body",
        source_url=source_url,
        text="Declare a request body with a Pydantic model.",
        code_blocks=[{"language": "python", "code": "item: Item"}],
        token_count=12,
        score=0.8,
    )
    return HybridSearchResult(
        result=result,
        score=score,
        vector_rank=1,
        keyword_rank=1,
        vector_score=0.8,
        keyword_score=8.0,
        title_phrase_matches=1,
    )


@dataclass
class StubRetriever:
    results: list[HybridSearchResult]

    def search(
        self,
        query: str,
        *,
        top_k: int,
        category: str | None,
    ) -> list[HybridSearchResult]:
        del query, category
        return self.results[:top_k]


@dataclass
class FakeGenerationProvider:
    provider_name: str = "fake"
    model_name: str = "fake-model"
    calls: list[tuple[str, list[HybridSearchResult]]] = field(default_factory=list)

    def generate(
        self,
        question: str,
        contexts: list[HybridSearchResult],
    ) -> GenerationOutput:
        self.calls.append((question, contexts))
        return GenerationOutput(
            answer="Use a Pydantic model. [1]",
            usage=GenerationUsage(input_tokens=100, output_tokens=20),
            estimated_cost_usd=0.00004,
        )


def test_grounded_prompt_contains_rules_sources_and_question() -> None:
    prompt = build_grounded_prompt("How do I declare a body?", [make_hybrid_result()])

    assert "only the supplied FastAPI documentation" in SYSTEM_PROMPT
    assert "How do I declare a body?" in prompt
    assert "[Source 1]" in prompt
    assert "https://fastapi.tiangolo.com/tutorial/body/" in prompt
    assert "item: Item" in prompt


def test_answer_service_generates_answer_with_usage_and_citations() -> None:
    result = make_hybrid_result()
    provider = FakeGenerationProvider()
    service = AnswerService(StubRetriever([result]), provider)

    answer = service.ask("How do I declare a request body?")

    assert answer.supported is True
    assert answer.answer == "Use a Pydantic model. [1]"
    assert answer.citations[0].source_url == result.result.source_url
    assert answer.usage.total_tokens == 120
    assert answer.estimated_cost_usd == 0.00004
    assert answer.latency_ms >= 0
    assert len(provider.calls) == 1


def test_answer_service_declines_without_calling_generator() -> None:
    provider = FakeGenerationProvider()
    service = AnswerService(StubRetriever([make_hybrid_result(score=0.01)]), provider)

    answer = service.ask("What is the weather?", minimum_score=0.02)

    assert answer.supported is False
    assert answer.answer == UNSUPPORTED_ANSWER
    assert answer.citations == []
    assert answer.usage.total_tokens == 0
    assert answer.estimated_cost_usd == 0.0
    assert provider.calls == []


def test_citation_numbers_match_context_numbers() -> None:
    first = make_hybrid_result()
    duplicate = make_hybrid_result()
    provider = FakeGenerationProvider()
    service = AnswerService(StubRetriever([first, duplicate]), provider)

    answer = service.ask("How do I declare a request body?")

    assert [citation.number for citation in answer.citations] == [1, 2]


def test_extractive_provider_returns_zero_cost_evidence() -> None:
    output = ExtractiveGenerationProvider().generate(
        "How do I declare a request body?", [make_hybrid_result()]
    )

    assert "Declare a request body" in output.answer
    assert output.answer.endswith("[1]")
    assert output.usage.total_tokens == 0
    assert output.estimated_cost_usd == 0.0


def test_openai_provider_reports_usage_and_configured_cost() -> None:
    class Usage:
        input_tokens = 1000
        output_tokens = 200

    class Response:
        output_text = "Use a Pydantic model. [1]"
        usage = Usage()

    class Responses:
        def create(self, **kwargs: object) -> Response:
            assert kwargs["model"] == "test-model"
            assert "Question:" in str(kwargs["input"])
            return Response()

    class Client:
        responses = Responses()

    provider = OpenAIGenerationProvider(
        model_name="test-model",
        api_key="test-key",
        input_cost_per_million=1.0,
        output_cost_per_million=2.0,
    )
    provider._client = Client()

    output = provider.generate("How do I declare a body?", [make_hybrid_result()])

    assert output.answer.endswith("[1]")
    assert output.usage.total_tokens == 1200
    assert output.estimated_cost_usd == 0.0014
