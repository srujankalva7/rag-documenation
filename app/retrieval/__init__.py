"""Vector retrieval over indexed documentation chunks."""

from app.retrieval.models import RetrievalError, SearchResult
from app.retrieval.service import VectorRetriever

__all__ = ["RetrievalError", "SearchResult", "VectorRetriever"]
