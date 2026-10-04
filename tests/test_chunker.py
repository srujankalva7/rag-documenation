from __future__ import annotations

import json
from pathlib import Path

from app.chunking.chunker import DocumentChunker
from app.chunking.models import ChunkingConfig, InputDocument

FIXTURE = Path("tests/fixtures/normalized_document.json")


def load_document() -> InputDocument:
    return InputDocument.from_dict(json.loads(FIXTURE.read_text(encoding="utf-8")))


def test_chunker_preserves_metadata_code_and_citations() -> None:
    document = load_document()
    result = DocumentChunker(ChunkingConfig(max_tokens=100, overlap_tokens=10)).chunk(
        document
    )

    assert result.document_id == document.id
    assert result.document_hash == document.content_hash
    assert result.chunks[0].source_url.endswith("#create-your-data-model")
    assert result.chunks[0].code_blocks[0]["language"] == "python"
    assert "    price: float" in str(result.chunks[0].code_blocks[0]["code"])
    assert all(chunk.token_count > 0 for chunk in result.chunks)
    assert [chunk.chunk_index for chunk in result.chunks] == list(
        range(len(result.chunks))
    )


def test_chunking_is_deterministic() -> None:
    document = load_document()
    chunker = DocumentChunker(ChunkingConfig(max_tokens=40, overlap_tokens=8))

    assert chunker.chunk(document).to_dict() == chunker.chunk(document).to_dict()


def test_long_text_uses_overlap() -> None:
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    value["sections"][0]["content"] = " ".join(f"token-{index}" for index in range(100))
    value["sections"][0]["code_blocks"] = []
    document = InputDocument.from_dict(value)
    chunker = DocumentChunker(ChunkingConfig(max_tokens=30, overlap_tokens=5))

    chunks = chunker.chunk(document).chunks

    first_tokens = chunker.encoding.encode(chunks[0].text)
    second_tokens = chunker.encoding.encode(chunks[1].text)
    assert first_tokens[-5:] == second_tokens[:5]
    assert all(chunk.token_count <= 30 for chunk in chunks)


def test_oversized_code_block_is_never_split() -> None:
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    large_code = "\n".join(f"value_{index} = {index}" for index in range(80))
    value["sections"][0]["code_blocks"] = [{"language": "python", "code": large_code}]
    document = InputDocument.from_dict(value)
    result = DocumentChunker(ChunkingConfig(max_tokens=40, overlap_tokens=5)).chunk(
        document
    )

    code_chunks = [chunk for chunk in result.chunks if chunk.code_blocks]
    assert len(code_chunks) == 1
    assert code_chunks[0].code_blocks[0]["code"] == large_code
    assert code_chunks[0].token_count > 40
