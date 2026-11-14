"""Discover and ingest a public website from its homepage URL."""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import httpx

from app.ingestion.crawler import SiteCrawler
from app.ingestion.generic_parser import GenericDocumentParser
from app.ingestion.generic_pipeline import GenericSiteIngestionPipeline
from app.ingestion.models import SourceValidationError
from app.ingestion.site import SiteConfig, load_site_config
from app.ingestion.storage import DocumentStorage


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Public homepage URL to discover")
    source.add_argument("--config", type=Path, help="Existing site configuration JSON")
    parser.add_argument("--site-id")
    parser.add_argument("--allowed-domain", action="append", default=[])
    parser.add_argument("--include-path", action="append", default=[])
    parser.add_argument("--exclude-path", action="append")
    parser.add_argument("--sitemap-url")
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    parser.add_argument("--max-response-bytes", type=int, default=2_000_000)
    parser.add_argument("--data-dir", type=Path, default=Path("data/sites"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _config(arguments: Namespace) -> SiteConfig:
    if arguments.config:
        return load_site_config(arguments.config)
    return SiteConfig.create(
        arguments.url,
        site_id=arguments.site_id,
        allowed_domains=tuple(arguments.allowed_domain) or None,
        max_pages=arguments.max_pages,
        max_depth=arguments.max_depth,
        include_paths=tuple(arguments.include_path),
        exclude_paths=(
            tuple(arguments.exclude_path)
            if arguments.exclude_path is not None
            else None
        ),
        sitemap_url=arguments.sitemap_url,
    )


def run(arguments: Namespace) -> int:
    config = _config(arguments)
    site_dir = arguments.data_dir / config.site_id
    storage = DocumentStorage(
        raw_dir=site_dir / "raw",
        documents_dir=site_dir / "documents",
        versions_dir=site_dir / "versions",
    )
    with SiteCrawler(
        delay_seconds=arguments.delay_seconds,
        max_response_bytes=arguments.max_response_bytes,
    ) as crawler:
        summary = GenericSiteIngestionPipeline(
            crawler,
            GenericDocumentParser(),
            storage,
        ).run(config, dry_run=arguments.dry_run)

    if not arguments.dry_run:
        site_dir.mkdir(parents=True, exist_ok=True)
        (site_dir / "site.json").write_text(
            json.dumps(config.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    counts = summary.ingestion.counts()
    print(f"Site ingestion completed: {config.site_id}")
    print(f"Discovered: {summary.discovered}")
    print(f"Skipped:    {summary.skipped}")
    for status in ("created", "updated", "unchanged", "validated", "failed"):
        print(f"{status.title():<10} {counts[status]}")
    for result in summary.ingestion.results:
        if result.error:
            print(f"ERROR {result.document_id}: {result.error}")
    return 1 if summary.failed else 0


def main() -> int:
    parser = build_parser()
    try:
        return run(parser.parse_args())
    except (SourceValidationError, ValueError, httpx.HTTPError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
