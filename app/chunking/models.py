"""Input and output models for document chunking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class DocumentValidationError(ValueError):
    """Raised when a normalized document cannot be safely chunked."""


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DocumentValidationError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class InputCodeBlock:
    language: str | None
    code: str

    @classmethod
    def from_dict(cls, value: Any) -> InputCodeBlock:
        if not isinstance(value, dict):
            raise DocumentValidationError("code block must be an object")
        language = value.get("language")
        if language is not None and not isinstance(language, str):
            raise DocumentValidationError(
                "code block language must be a string or null"
            )
        return cls(
            language=language,
            code=_required_string(value.get("code"), "code block code"),
        )


@dataclass(frozen=True, slots=True)
class InputSection:
    heading: str
    heading_level: int
    anchor: str | None
    source_url: str
    content: str
    code_blocks: list[InputCodeBlock] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: Any) -> InputSection:
        if not isinstance(value, dict):
            raise DocumentValidationError("section must be an object")
        heading_level = value.get("heading_level")
        if not isinstance(heading_level, int) or not 1 <= heading_level <= 6:
            raise DocumentValidationError("heading_level must be between 1 and 6")
        anchor = value.get("anchor")
        if anchor is not None and not isinstance(anchor, str):
            raise DocumentValidationError("section anchor must be a string or null")
        content = value.get("content", "")
        if not isinstance(content, str):
            raise DocumentValidationError("section content must be a string")
        raw_blocks = value.get("code_blocks", [])
        if not isinstance(raw_blocks, list):
            raise DocumentValidationError("section code_blocks must be an array")
        code_blocks = [InputCodeBlock.from_dict(block) for block in raw_blocks]
        if not content.strip() and not code_blocks:
            raise DocumentValidationError("section must contain text or code")
        return cls(
            heading=_required_string(value.get("heading"), "section heading"),
            heading_level=heading_level,
            anchor=anchor,
            source_url=_required_string(value.get("source_url"), "section source_url"),
            content=content.strip(),
            code_blocks=code_blocks,
        )


@dataclass(frozen=True, slots=True)
class InputDocument:
    id: str
    title: str
    url: str
    category: str
    content_hash: str
    sections: list[InputSection]

    @classmethod
    def from_dict(cls, value: Any) -> InputDocument:
        if not isinstance(value, dict):
            raise DocumentValidationError("document must be an object")
        raw_sections = value.get("sections")
        if not isinstance(raw_sections, list) or not raw_sections:
            raise DocumentValidationError("document sections must be a non-empty array")
        return cls(
            id=_required_string(value.get("id"), "document id"),
            title=_required_string(value.get("title"), "document title"),
            url=_required_string(value.get("url"), "document url"),
            category=_required_string(value.get("category"), "document category"),
            content_hash=_required_string(
                value.get("content_hash"),
                "document content_hash",
            ),
            sections=[InputSection.from_dict(section) for section in raw_sections],
        )


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    max_tokens: int = 600
    overlap_tokens: int = 75
    encoding_name: str = "cl100k_base"

    def validate(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens cannot be negative")
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        if not self.encoding_name.strip():
            raise ValueError("encoding_name cannot be empty")


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
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
    code_blocks: list[dict[str, str | None]]
    token_count: int
    content_hash: str


@dataclass(frozen=True, slots=True)
class ChunkedDocument:
    document_id: str
    document_hash: str
    chunking: ChunkingConfig
    chunks: list[Chunk]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
