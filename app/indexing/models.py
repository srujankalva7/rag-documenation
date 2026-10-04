"""Validated models used by the indexing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ChunkValidationError(ValueError):
    """Raised when chunk output cannot be safely indexed."""


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ChunkValidationError(f"{field_name} must be a non-empty string")
    return value


def _required_integer(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ChunkValidationError(f"{field_name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class IndexRecord:
    chunk_id: str
    content_hash: str
    document_id: str
    document_hash: str
    title: str
    category: str
    section: str
    section_index: int
    chunk_index: int
    section_chunk_index: int
    source_url: str
    text: str
    token_count: int

    @classmethod
    def from_dict(cls, value: Any) -> IndexRecord:
        if not isinstance(value, dict):
            raise ChunkValidationError("chunk must be an object")
        return cls(
            chunk_id=_required_string(value.get("chunk_id"), "chunk_id"),
            content_hash=_required_string(value.get("content_hash"), "content_hash"),
            document_id=_required_string(value.get("document_id"), "document_id"),
            document_hash=_required_string(value.get("document_hash"), "document_hash"),
            title=_required_string(value.get("title"), "title"),
            category=_required_string(value.get("category"), "category"),
            section=_required_string(value.get("section"), "section"),
            section_index=_required_integer(
                value.get("section_index"), "section_index"
            ),
            chunk_index=_required_integer(value.get("chunk_index"), "chunk_index"),
            section_chunk_index=_required_integer(
                value.get("section_chunk_index"), "section_chunk_index"
            ),
            source_url=_required_string(value.get("source_url"), "source_url"),
            text=_required_string(value.get("text"), "text"),
            token_count=_required_integer(value.get("token_count"), "token_count"),
        )


def records_from_chunked_document(value: Any) -> list[IndexRecord]:
    if not isinstance(value, dict):
        raise ChunkValidationError("chunked document must be an object")
    document_id = _required_string(value.get("document_id"), "document_id")
    chunks = value.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ChunkValidationError("chunks must be a non-empty array")
    records = [IndexRecord.from_dict(chunk) for chunk in chunks]
    if any(record.document_id != document_id for record in records):
        raise ChunkValidationError("chunk document_id does not match its document")
    if len({record.chunk_id for record in records}) != len(records):
        raise ChunkValidationError("chunk IDs must be unique within a document")
    return records
