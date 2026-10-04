"""Read normalized documents, create chunks, and persist deterministic output."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from app.chunking.chunker import DocumentChunker
from app.chunking.models import InputDocument


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class ChunkingResult:
    document_id: str
    status: str
    chunk_count: int = 0
    error: str | None = None


@dataclass(slots=True)
class ChunkingSummary:
    results: list[ChunkingResult] = field(default_factory=list)

    @property
    def failed(self) -> int:
        return sum(result.status == "failed" for result in self.results)

    @property
    def chunk_count(self) -> int:
        return sum(result.chunk_count for result in self.results)

    def counts(self) -> dict[str, int]:
        statuses = {"created", "updated", "unchanged", "validated", "failed"}
        return {
            status: sum(result.status == status for result in self.results)
            for status in statuses
        }


@dataclass(slots=True)
class ChunkingPipeline:
    chunker: DocumentChunker
    documents_dir: Path
    output_dir: Path

    def _files(self, document_id: str | None) -> list[Path]:
        if document_id:
            path = self.documents_dir / f"{document_id}.json"
            if not path.exists():
                raise ValueError(f"unknown document id: {document_id}")
            return [path]
        files = sorted(self.documents_dir.glob("*.json"))
        if not files:
            raise ValueError(f"no normalized documents found in {self.documents_dir}")
        return files

    def run(
        self,
        *,
        document_id: str | None = None,
        dry_run: bool = False,
    ) -> ChunkingSummary:
        summary = ChunkingSummary()
        for path in self._files(document_id):
            fallback_id = path.stem
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                document = InputDocument.from_dict(value)
                chunked = self.chunker.chunk(document)
                serialized = _json_text(chunked.to_dict())
                output_path = self.output_dir / f"{document.id}.json"

                if dry_run:
                    status = "validated"
                elif output_path.exists():
                    status = (
                        "unchanged"
                        if output_path.read_text(encoding="utf-8") == serialized
                        else "updated"
                    )
                    if status == "updated":
                        _atomic_write(output_path, serialized)
                else:
                    _atomic_write(output_path, serialized)
                    status = "created"

                summary.results.append(
                    ChunkingResult(document.id, status, len(chunked.chunks))
                )
            except Exception as error:  # Keep processing other documents.
                summary.results.append(
                    ChunkingResult(
                        fallback_id,
                        "failed",
                        error=f"{type(error).__name__}: {error}",
                    )
                )
        return summary
