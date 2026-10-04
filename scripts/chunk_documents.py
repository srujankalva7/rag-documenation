"""Split normalized documentation into deterministic retrieval chunks."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

from app.chunking.chunker import DocumentChunker
from app.chunking.models import ChunkingConfig
from app.chunking.pipeline import ChunkingPipeline


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--documents-dir",
        type=Path,
        default=Path("data/documents"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument("--document-id", help="Chunk one normalized document")
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--overlap-tokens", type=int, default=75)
    parser.add_argument("--encoding", default="cl100k_base")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and chunk documents without writing output",
    )
    return parser


def run(arguments: Namespace) -> int:
    config = ChunkingConfig(
        max_tokens=arguments.max_tokens,
        overlap_tokens=arguments.overlap_tokens,
        encoding_name=arguments.encoding,
    )
    pipeline = ChunkingPipeline(
        chunker=DocumentChunker(config),
        documents_dir=arguments.documents_dir,
        output_dir=arguments.output_dir,
    )
    summary = pipeline.run(
        document_id=arguments.document_id,
        dry_run=arguments.dry_run,
    )
    counts = summary.counts()
    print("Documentation chunking completed")
    print(f"Documents: {len(summary.results)}")
    print(f"Chunks:    {summary.chunk_count}")
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
