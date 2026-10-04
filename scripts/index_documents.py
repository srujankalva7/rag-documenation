"""Embed retrieval chunks and synchronize the local vector index."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

from app.indexing.pipeline import IndexingPipeline
from app.indexing.providers import FastEmbedEmbeddingProvider
from app.indexing.store import SQLiteVectorIndex


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("data/index/vectors.sqlite3"),
    )
    parser.add_argument("--document-id", help="Index one chunked document")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and report changes without embedding or writing",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Clear the index before processing every chunk file",
    )
    return parser


def run(arguments: Namespace) -> int:
    if arguments.rebuild and arguments.document_id:
        raise ValueError("--rebuild cannot be combined with --document-id")
    pipeline = IndexingPipeline(
        provider=FastEmbedEmbeddingProvider(arguments.model),
        store=SQLiteVectorIndex(arguments.index_path),
        chunks_dir=arguments.chunks_dir,
        batch_size=arguments.batch_size,
    )
    summary = pipeline.run(
        document_id=arguments.document_id,
        dry_run=arguments.dry_run,
        rebuild=arguments.rebuild,
    )
    label = (
        "Indexing dry run completed" if summary.dry_run else "Vector indexing completed"
    )
    print(label)
    print(f"Documents: {summary.documents}")
    print(f"Chunks:    {summary.chunks}")
    print(f"Embedded:  {summary.embedded}")
    print(f"Unchanged: {summary.unchanged}")
    print(f"Removed:   {summary.removed}")
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
