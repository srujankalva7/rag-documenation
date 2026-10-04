import pytest

from app.indexing.models import ChunkValidationError, records_from_chunked_document


def make_chunk(chunk_id: str = "request-body:0") -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "content_hash": "content-hash",
        "document_id": "request-body",
        "document_hash": "document-hash",
        "title": "Request Body",
        "category": "tutorial",
        "section": "Request Body",
        "section_index": 0,
        "chunk_index": 0,
        "section_chunk_index": 0,
        "source_url": "https://fastapi.tiangolo.com/tutorial/body/",
        "text": "Use a Pydantic model to declare a request body.",
        "token_count": 12,
    }


def test_records_from_chunked_document_validates_and_loads_chunks() -> None:
    records = records_from_chunked_document(
        {"document_id": "request-body", "chunks": [make_chunk()]}
    )

    assert len(records) == 1
    assert records[0].chunk_id == "request-body:0"
    assert records[0].source_url.endswith("/tutorial/body/")


def test_records_rejects_mismatched_document_id() -> None:
    chunk = make_chunk()
    chunk["document_id"] = "other"

    with pytest.raises(ChunkValidationError, match="does not match"):
        records_from_chunked_document(
            {"document_id": "request-body", "chunks": [chunk]}
        )


def test_records_rejects_duplicate_chunk_ids() -> None:
    chunk = make_chunk()

    with pytest.raises(ChunkValidationError, match="must be unique"):
        records_from_chunked_document(
            {"document_id": "request-body", "chunks": [chunk, chunk.copy()]}
        )
