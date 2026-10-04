"""Embedding and vector-indexing support."""

from app.indexing.pipeline import IndexingPipeline
from app.indexing.providers import FastEmbedEmbeddingProvider
from app.indexing.store import SQLiteVectorIndex

__all__ = [
    "IndexingPipeline",
    "SQLiteVectorIndex",
    "FastEmbedEmbeddingProvider",
]
