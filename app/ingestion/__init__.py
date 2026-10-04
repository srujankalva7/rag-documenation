"""FastAPI documentation ingestion pipeline."""

from app.ingestion.models import Document, Section, Source
from app.ingestion.parser import FastAPIDocumentParser

__all__ = ["Document", "FastAPIDocumentParser", "Section", "Source"]
