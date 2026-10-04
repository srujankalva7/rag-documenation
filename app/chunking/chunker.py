"""Split normalized documentation into deterministic retrieval chunks."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable

import tiktoken

from app.chunking.models import (
    Chunk,
    ChunkedDocument,
    ChunkingConfig,
    InputDocument,
    InputSection,
)

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
SLUG_CHARACTERS = re.compile(r"[^a-z0-9]+")


def _slugify(value: str) -> str:
    slug = SLUG_CHARACTERS.sub("-", value.casefold()).strip("-")
    return slug or "section"


def _content_hash(
    *,
    document_hash: str,
    source_url: str,
    text: str,
    code_blocks: list[dict[str, str | None]],
) -> str:
    serialized = json.dumps(
        {
            "document_hash": document_hash,
            "source_url": source_url,
            "text": text,
            "code_blocks": code_blocks,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class DocumentChunker:
    """Create section-aware chunks while keeping each code block intact."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()
        self.config.validate()
        self.encoding = tiktoken.get_encoding(self.config.encoding_name)

    def count_tokens(self, value: str) -> int:
        return len(self.encoding.encode(value))

    def _tail(self, value: str) -> str:
        if not value or self.config.overlap_tokens == 0:
            return ""
        tokens = self.encoding.encode(value)
        return self.encoding.decode(tokens[-self.config.overlap_tokens :]).strip()

    def _hard_split(self, value: str) -> list[str]:
        """Split an oversized sentence using token windows with overlap."""

        tokens = self.encoding.encode(value)
        stride = self.config.max_tokens - self.config.overlap_tokens
        pieces = []
        start = 0
        while start < len(tokens):
            token_window = tokens[start : start + self.config.max_tokens]
            pieces.append(self.encoding.decode(token_window).strip())
            if start + self.config.max_tokens >= len(tokens):
                break
            start += stride
        return [piece for piece in pieces if piece]

    def _paragraph_units(self, content: str) -> Iterable[str]:
        for paragraph in re.split(r"\n\s*\n", content):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if self.count_tokens(paragraph) <= self.config.max_tokens:
                yield paragraph
                continue

            sentences = [
                sentence.strip()
                for sentence in SENTENCE_BOUNDARY.split(paragraph)
                if sentence.strip()
            ]
            if len(sentences) == 1:
                yield from self._hard_split(paragraph)
                continue

            current: list[str] = []
            for sentence in sentences:
                if self.count_tokens(sentence) > self.config.max_tokens:
                    if current:
                        yield " ".join(current)
                        current = []
                    yield from self._hard_split(sentence)
                    continue
                candidate = " ".join([*current, sentence])
                if current and self.count_tokens(candidate) > self.config.max_tokens:
                    yield " ".join(current)
                    current = [sentence]
                else:
                    current.append(sentence)
            if current:
                yield " ".join(current)

    def _text_chunks(self, content: str) -> list[str]:
        units = list(self._paragraph_units(content))
        if not units:
            return []

        chunks: list[str] = []
        current = ""
        for unit in units:
            candidate = f"{current}\n\n{unit}".strip() if current else unit
            if current and self.count_tokens(candidate) > self.config.max_tokens:
                chunks.append(current)
                overlap = self._tail(current)
                with_overlap = f"{overlap}\n\n{unit}".strip()
                current = (
                    with_overlap
                    if self.count_tokens(with_overlap) <= self.config.max_tokens
                    else unit
                )
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    def _token_count(
        self,
        text: str,
        code_blocks: list[dict[str, str | None]],
    ) -> int:
        code = "\n\n".join(str(block["code"]) for block in code_blocks)
        return self.count_tokens(f"{text}\n\n{code}".strip())

    def _section_payloads(
        self,
        section: InputSection,
    ) -> list[tuple[str, list[dict[str, str | None]]]]:
        blocks = [
            {"language": block.language, "code": block.code}
            for block in section.code_blocks
        ]
        total = self._token_count(section.content, blocks)
        if total <= self.config.max_tokens:
            return [(section.content, blocks)]

        payloads = [(text, []) for text in self._text_chunks(section.content)]
        context = self._tail(section.content)
        for block in blocks:
            # Code is atomic. An unusually large code sample may exceed max_tokens,
            # which is preferable to corrupting it by splitting in the middle.
            block_context = context
            if self._token_count(block_context, [block]) > self.config.max_tokens:
                available = max(
                    self.config.max_tokens - self.count_tokens(str(block["code"])),
                    0,
                )
                if available:
                    context_tokens = self.encoding.encode(context)[-available:]
                    block_context = self.encoding.decode(context_tokens).strip()
                else:
                    block_context = ""
            payloads.append((block_context, [block]))
        return payloads

    def chunk(self, document: InputDocument) -> ChunkedDocument:
        chunks: list[Chunk] = []
        global_index = 0
        for section_index, section in enumerate(document.sections):
            for section_chunk_index, (text, code_blocks) in enumerate(
                self._section_payloads(section)
            ):
                chunk_id = (
                    f"{document.id}-{_slugify(section.heading)}-{global_index:04d}"
                )
                token_count = self._token_count(text, code_blocks)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document.id,
                        document_hash=document.content_hash,
                        title=document.title,
                        category=document.category,
                        section=section.heading,
                        section_index=section_index,
                        chunk_index=global_index,
                        section_chunk_index=section_chunk_index,
                        source_url=section.source_url,
                        text=text,
                        code_blocks=code_blocks,
                        token_count=token_count,
                        content_hash=_content_hash(
                            document_hash=document.content_hash,
                            source_url=section.source_url,
                            text=text,
                            code_blocks=code_blocks,
                        ),
                    )
                )
                global_index += 1
        if not chunks:
            raise ValueError(f"document {document.id!r} produced no chunks")
        return ChunkedDocument(
            document_id=document.id,
            document_hash=document.content_hash,
            chunking=self.config,
            chunks=chunks,
        )
