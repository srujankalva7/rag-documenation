from __future__ import annotations

from app.api.main import _default_generation_provider
from app.generation.providers import OllamaGenerationProvider


def test_default_generation_provider_supports_ollama(monkeypatch) -> None:
    monkeypatch.setenv("RAG_GENERATION_PROVIDER", "ollama")
    monkeypatch.setenv("RAG_GENERATION_MODEL", "llama3.2:3b")
    monkeypatch.setenv("RAG_OLLAMA_BASE_URL", "http://localhost:11434/")
    monkeypatch.setenv("RAG_OLLAMA_TIMEOUT_SECONDS", "30")

    provider = _default_generation_provider()

    assert isinstance(provider, OllamaGenerationProvider)
    assert provider.provider_name == "ollama"
    assert provider.model_name == "llama3.2:3b"
    assert provider.base_url == "http://localhost:11434"
