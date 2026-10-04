"""Atomic local storage and simple file-based document version history."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.ingestion.models import Document, Source

SaveStatus = Literal["created", "updated", "unchanged"]


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


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


@dataclass(slots=True)
class DocumentStorage:
    raw_dir: Path
    documents_dir: Path
    versions_dir: Path

    def raw_path(self, source: Source) -> Path:
        return self.raw_dir / f"{source.id}.html"

    def document_path(self, source: Source) -> Path:
        return self.documents_dir / f"{source.id}.json"

    def load_raw(self, source: Source) -> str:
        return self.raw_path(source).read_text(encoding="utf-8")

    def save_raw(self, source: Source, html: str) -> None:
        _atomic_write(self.raw_path(source), html)

    def save_document(self, source: Source, document: Document) -> SaveStatus:
        current_path = self.document_path(source)
        new_value = document.to_dict()

        if not current_path.exists():
            _atomic_write(current_path, _json_text(new_value))
            return "created"

        current_value = json.loads(current_path.read_text(encoding="utf-8"))
        previous_hash = current_value.get("content_hash")
        if previous_hash == document.content_hash:
            return "unchanged"

        if not isinstance(previous_hash, str) or not previous_hash:
            raise ValueError(f"stored document has no content hash: {current_path}")

        version_path = self.versions_dir / source.id / f"{previous_hash}.json"
        if not version_path.exists():
            _atomic_write(version_path, _json_text(current_value))
        _atomic_write(current_path, _json_text(new_value))
        return "updated"
