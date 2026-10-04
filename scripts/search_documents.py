"""Search the local FastAPI documentation vector index."""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

from app.indexing.providers import FastEmbedEmbeddingProvider
from app.indexing.store import SQLiteVectorIndex
from app.retrieval.models import RetrievalError
from app.retrieval.service import VectorRetriever


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Documentation question to search for")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--category", help="Limit results to one category")
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("data/index/vectors.sqlite3"),
    )
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def run(arguments: Namespace) -> int:
    results = VectorRetriever(
        provider=FastEmbedEmbeddingProvider(arguments.model),
        store=SQLiteVectorIndex(arguments.index_path),
    ).search(
        arguments.query,
        top_k=arguments.top_k,
        category=arguments.category,
    )
    if arguments.as_json:
        print(json.dumps([result.to_dict() for result in results], indent=2))
        return 0

    if not results:
        print("No matching chunks found.")
        return 0
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result.title} — {result.section}")
        print(f"   Score: {result.score:.4f}")
        print(f"   Source: {result.source_url}")
        print(f"   {result.content}")
        if rank != len(results):
            print()
    return 0


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        return run(arguments)
    except (RetrievalError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
