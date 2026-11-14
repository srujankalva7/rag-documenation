"""Orchestrate crawling, generic parsing, and versioned local storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.ingestion.crawler import CrawlResult
from app.ingestion.generic_parser import GenericDocumentParser
from app.ingestion.pipeline import IngestionResult, IngestionSummary
from app.ingestion.site import SiteConfig
from app.ingestion.storage import DocumentStorage


class Crawler(Protocol):
    def crawl(self, config: SiteConfig) -> CrawlResult: ...


@dataclass(frozen=True, slots=True)
class GenericSiteIngestionSummary:
    ingestion: IngestionSummary
    discovered: int
    skipped: int

    @property
    def failed(self) -> int:
        return self.ingestion.failed


@dataclass(slots=True)
class GenericSiteIngestionPipeline:
    crawler: Crawler
    parser: GenericDocumentParser
    storage: DocumentStorage

    def run(
        self,
        config: SiteConfig,
        *,
        dry_run: bool = False,
    ) -> GenericSiteIngestionSummary:
        crawl = self.crawler.crawl(config)
        summary = IngestionSummary(
            [
                IngestionResult(
                    document_id=failure.url,
                    status="failed",
                    error=failure.error,
                )
                for failure in crawl.failures
            ]
        )
        for page in crawl.pages:
            source = page.source
            try:
                document = self.parser.parse(source, page.html)
                if dry_run:
                    status = "validated"
                else:
                    self.storage.save_raw(source, page.html)
                    status = self.storage.save_document(source, document)
                summary.results.append(IngestionResult(source.id, status))
            except Exception as error:  # Keep other discovered pages progressing.
                summary.results.append(
                    IngestionResult(
                        source.id,
                        "failed",
                        f"{type(error).__name__}: {error}",
                    )
                )
        return GenericSiteIngestionSummary(
            ingestion=summary,
            discovered=len(crawl.pages),
            skipped=crawl.skipped,
        )
