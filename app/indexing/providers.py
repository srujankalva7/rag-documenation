"""Configurable embedding providers."""

from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    """Minimal interface required by the indexing pipeline."""

    @property
    def model_name(self) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedEmbeddingProvider:
    """Create embeddings locally with FastEmbed's ONNX runtime."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        self._model_name = model_name
        self._model = None

    @property
    def model_name(self) -> str:
        # Version the embedding input format so an existing text-only vector is
        # never reused after code blocks become part of the embedded content.
        return f"{self._model_name}::text-and-code-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "local embeddings are not installed; run "
                    "python -m pip install -e '.[local]'"
                ) from error

            self._model = TextEmbedding(model_name=self._model_name)
        embeddings = self._model.embed(texts)
        return [[float(value) for value in row] for row in embeddings]
