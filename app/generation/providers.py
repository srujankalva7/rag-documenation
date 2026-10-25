"""Configurable local and hosted answer-generation providers."""

from __future__ import annotations

import os

from app.generation.models import GenerationOutput, GenerationUsage
from app.generation.prompts import SYSTEM_PROMPT, build_grounded_prompt
from app.retrieval.models import HybridSearchResult


class ExtractiveGenerationProvider:
    """Return retrieved evidence directly without an LLM call or API cost."""

    def __init__(self, max_characters: int = 1600) -> None:
        if max_characters < 1:
            raise ValueError("max_characters must be positive")
        self.max_characters = max_characters

    @property
    def provider_name(self) -> str:
        return "extractive"

    @property
    def model_name(self) -> str:
        return "retrieved-evidence"

    def generate(
        self,
        question: str,
        contexts: list[HybridSearchResult],
    ) -> GenerationOutput:
        del question
        if not contexts:
            raise ValueError("at least one context is required")
        evidence = contexts[0].result.content.strip()
        if len(evidence) > self.max_characters:
            evidence = evidence[: self.max_characters].rsplit(" ", 1)[0].rstrip() + "…"
        return GenerationOutput(
            answer=f"{evidence}\n\n[1]",
            estimated_cost_usd=0.0,
        )


class OpenAIGenerationProvider:
    """Generate grounded answers with the OpenAI Responses API."""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str | None = None,
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI generation")
        self._model_name = model_name
        self._api_key = resolved_key
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self._client = None

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model_name

    def _estimated_cost(self, usage: GenerationUsage) -> float | None:
        if self.input_cost_per_million is None or self.output_cost_per_million is None:
            return None
        return (
            usage.input_tokens * self.input_cost_per_million
            + usage.output_tokens * self.output_cost_per_million
        ) / 1_000_000

    def generate(
        self,
        question: str,
        contexts: list[HybridSearchResult],
    ) -> GenerationOutput:
        if self._client is None:
            try:
                from openai import OpenAI
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "hosted generation is not installed; run "
                    "python -m pip install -e '.[hosted]'"
                ) from error
            self._client = OpenAI(api_key=self._api_key)
        response = self._client.responses.create(
            model=self.model_name,
            instructions=SYSTEM_PROMPT,
            input=build_grounded_prompt(question, contexts),
        )
        answer = response.output_text.strip()
        if not answer:
            raise RuntimeError("generation provider returned an empty answer")
        raw_usage = getattr(response, "usage", None)
        usage = GenerationUsage(
            input_tokens=int(getattr(raw_usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(raw_usage, "output_tokens", 0) or 0),
        )
        return GenerationOutput(
            answer=answer,
            usage=usage,
            estimated_cost_usd=self._estimated_cost(usage),
        )
