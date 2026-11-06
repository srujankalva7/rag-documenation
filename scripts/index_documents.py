"""Embed retrieval chunks and synchronize the local vector index."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

from app.cache.store import SQLiteSemanticCache
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
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("data/cache/semantic_cache.sqlite3"),
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
    parser.add_argument(
        "--skip-cache-invalidation",
        action="store_true",
        help="Do not invalidate answers supported by changed documents",
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
    if summary.changed_documents:
        print("Changed documents: " + ", ".join(summary.changed_documents))
    invalidated = 0
    if (
        not summary.dry_run
        and not arguments.skip_cache_invalidation
        and summary.changed_documents
        and arguments.cache_path.is_file()
    ):
        cache = SQLiteSemanticCache(arguments.cache_path)
        entry_ids = cache.matching_entry_ids(
            document_ids=set(summary.changed_documents)
        )
        invalidated = cache.invalidate_entries(
            entry_ids,
            "supporting_document_reindexed",
        )
    print(f"Cache entries invalidated: {invalidated}")
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
