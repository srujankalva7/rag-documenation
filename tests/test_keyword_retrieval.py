from __future__ import annotations

from pathlib import Path

from app.indexing.models import IndexRecord
from app.indexing.store import SQLiteVectorIndex
from app.retrieval.keyword import KeywordRetriever, title_phrase_match_count, tokenize


def make_record(
    chunk_id: str,
    title: str,
    section: str,
    text: str,
    *,
    category: str = "tutorial",
) -> IndexRecord:
    return IndexRecord(
        chunk_id=chunk_id,
        content_hash=f"{chunk_id}-hash",
        document_id=chunk_id.split(":")[0],
        document_hash="document-hash",
        title=title,
        category=category,
        section=section,
        section_index=0,
        chunk_index=0,
        section_chunk_index=0,
        source_url=f"https://fastapi.tiangolo.com/{chunk_id}/",
        text=text,
        code_blocks=(),
        token_count=10,
    )


def seed_index(path: Path, records: list[IndexRecord]) -> SQLiteVectorIndex:
    store = SQLiteVectorIndex(path)
    with store.connect() as connection:
        store.insert_embeddings(
            connection,
            "fake-model",
            [(record.content_hash, [1.0, 0.0]) for record in records],
        )
        store.upsert_chunks(connection, records, "fake-model")
    return store


def test_tokenize_removes_question_stopwords() -> None:
    assert tokenize(
        "How do I create a request body in FastAPI?", remove_stopwords=True
    ) == ["create", "request", "body", "fastapi"]


def test_title_phrase_match_count_finds_request_body() -> None:
    assert (
        title_phrase_match_count(
            "How do I create a request body in FastAPI?", "Request Body"
        )
        == 1
    )


def test_keyword_search_prefers_exact_page_title(tmp_path: Path) -> None:
    request_body = make_record(
        "request-body:0",
        "Request Body",
        "Create your data model",
        "Declare a request body with a Pydantic model.",
    )
    multiple = make_record(
        "body-multiple:0",
        "Body - Multiple Parameters",
        "Mix Path, Query and body parameters",
        "Mix path, query, and request body declarations.",
    )
    first_steps = make_record(
        "first-steps:0",
        "First Steps",
        "Create a FastAPI instance",
        "Create the FastAPI application.",
    )
    store = seed_index(
        tmp_path / "vectors.sqlite3",
        [multiple, first_steps, request_body],
    )

    results = KeywordRetriever(store).search(
        "How do I create a request body in FastAPI?", top_k=3
    )

    assert results[0].chunk_id == "request-body:0"
    assert results[-1].chunk_id == "first-steps:0"


def test_keyword_search_filters_category(tmp_path: Path) -> None:
    tutorial = make_record(
        "request-body:0",
        "Request Body",
        "Request Body",
        "Request body tutorial.",
    )
    advanced = make_record(
        "advanced-body:0",
        "Advanced Request Body",
        "Custom Schema",
        "Request body schema.",
        category="advanced",
    )
    store = seed_index(tmp_path / "vectors.sqlite3", [tutorial, advanced])

    results = KeywordRetriever(store).search("request body", category="advanced")

    assert [result.chunk_id for result in results] == ["advanced-body:0"]
