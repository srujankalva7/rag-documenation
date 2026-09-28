"""Download and normalize approved FastAPI documentation pages."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

from app.ingestion.fetcher import FastAPIDocumentFetcher
from app.ingestion.parser import FastAPIDocumentParser
from app.ingestion.pipeline import IngestionPipeline, load_sources
from app.ingestion.storage import DocumentStorage


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/sources/fastapi_urls.json"),
    )
    parser.add_argument("--document-id", help="Process only one manifest entry")
    parser.add_argument(
        "--use-local-html",
        action="store_true",
        help="Parse data/raw HTML without making HTTP requests",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and parse without writing raw or normalized documents",
    )
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--documents-dir", type=Path, default=Path("data/documents"))
    parser.add_argument("--versions-dir", type=Path, default=Path("data/versions"))
    return parser


def run(arguments: Namespace) -> int:
    sources = load_sources(arguments.manifest)
    storage = DocumentStorage(
        raw_dir=arguments.raw_dir,
        documents_dir=arguments.documents_dir,
        versions_dir=arguments.versions_dir,
    )
    parser = FastAPIDocumentParser()

    if arguments.use_local_html:
        pipeline = IngestionPipeline(None, parser, storage)
        summary = pipeline.run(
            sources,
            document_id=arguments.document_id,
            use_local_html=True,
            dry_run=arguments.dry_run,
        )
    else:
        with FastAPIDocumentFetcher(delay_seconds=arguments.delay_seconds) as fetcher:
            pipeline = IngestionPipeline(fetcher, parser, storage)
            summary = pipeline.run(
                sources,
                document_id=arguments.document_id,
                dry_run=arguments.dry_run,
            )

    counts = summary.counts()
    print("FastAPI documentation ingestion completed")
    print(f"Total:     {len(summary.results)}")
    for status in ("created", "updated", "unchanged", "validated", "failed"):
        print(f"{status.title():<10} {counts[status]}")

    for result in summary.results:
        if result.error:
            print(f"ERROR {result.document_id}: {result.error}")
    return 1 if summary.failed else 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
