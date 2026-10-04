"""Orchestrate source validation, fetching, parsing, and local persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from app.ingestion.models import Source
from app.ingestion.parser import FastAPIDocumentParser
from app.ingestion.storage import DocumentStorage


class Fetcher(Protocol):
    def fetch(self, source: Source) -> str: ...


@dataclass(frozen=True, slots=True)
class IngestionResult:
    document_id: str
    status: str
    error: str | None = None


@dataclass(slots=True)
class IngestionSummary:
    results: list[IngestionResult] = field(default_factory=list)

    @property
    def failed(self) -> int:
        return sum(result.status == "failed" for result in self.results)

    def counts(self) -> dict[str, int]:
        statuses = {"created", "updated", "unchanged", "validated", "failed"}
        return {
            status: sum(result.status == status for result in self.results)
            for status in statuses
        }


def load_sources(manifest_path: Path) -> list[Source]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("source manifest must be a non-empty JSON array")

    sources = [Source.from_dict(item) for item in raw]
    ids = [source.id for source in sources]
    urls = [source.url for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("source manifest contains duplicate IDs")
    if len(urls) != len(set(urls)):
        raise ValueError("source manifest contains duplicate URLs")
    return sources


@dataclass(slots=True)
class IngestionPipeline:
    fetcher: Fetcher | None
    parser: FastAPIDocumentParser
    storage: DocumentStorage

    def run(
        self,
        sources: list[Source],
        *,
        document_id: str | None = None,
        use_local_html: bool = False,
        dry_run: bool = False,
    ) -> IngestionSummary:
        selected = [source for source in sources if source.id == document_id]
        if document_id is None:
            selected = sources
        elif not selected:
            raise ValueError(f"unknown document id: {document_id}")

        if not use_local_html and self.fetcher is None:
            raise ValueError("a fetcher is required unless --use-local-html is set")

        summary = IngestionSummary()
        for source in selected:
            try:
                if use_local_html:
                    html = self.storage.load_raw(source)
                else:
                    assert self.fetcher is not None
                    html = self.fetcher.fetch(source)
                    if not dry_run:
                        self.storage.save_raw(source, html)

                document = self.parser.parse(source, html)
                status = (
                    "validated"
                    if dry_run
                    else self.storage.save_document(source, document)
                )
                summary.results.append(IngestionResult(source.id, status))
            except Exception as error:  # Continue so one bad page does not stop a run.
                summary.results.append(
                    IngestionResult(
                        source.id,
                        "failed",
                        f"{type(error).__name__}: {error}",
                    )
                )
        return summary
