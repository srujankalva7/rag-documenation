"""Search the local FastAPI documentation vector index."""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

from app.indexing.providers import FastEmbedEmbeddingProvider
from app.indexing.store import SQLiteVectorIndex
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.keyword import KeywordRetriever
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
    parser.add_argument(
        "--mode",
        choices=("hybrid", "vector", "keyword"),
        default="hybrid",
    )
    parser.add_argument("--vector-weight", type=float, default=1.0)
    parser.add_argument("--keyword-weight", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--title-match-boost", type=float, default=0.01)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def run(arguments: Namespace) -> int:
    store = SQLiteVectorIndex(arguments.index_path)
    vector_retriever = VectorRetriever(
        provider=FastEmbedEmbeddingProvider(arguments.model),
        store=store,
    )
    if arguments.mode == "vector":
        results = vector_retriever.search(
            arguments.query,
            top_k=arguments.top_k,
            category=arguments.category,
        )
    elif arguments.mode == "keyword":
        results = KeywordRetriever(store).search(
            arguments.query,
            top_k=arguments.top_k,
            category=arguments.category,
        )
    else:
        results = HybridRetriever(
            vector_retriever=vector_retriever,
            keyword_retriever=KeywordRetriever(store),
            vector_weight=arguments.vector_weight,
            keyword_weight=arguments.keyword_weight,
            rrf_k=arguments.rrf_k,
            candidate_k=arguments.candidate_k,
            title_match_boost=arguments.title_match_boost,
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
        if arguments.mode == "hybrid":
            source = result.result
            print(f"{rank}. {source.title} — {source.section}")
            print(f"   Hybrid score: {result.score:.4f}")
            print(
                "   Ranks: "
                f"vector={result.vector_rank or '-'}, "
                f"keyword={result.keyword_rank or '-'}"
            )
        else:
            source = result
            print(f"{rank}. {source.title} — {source.section}")
            print(f"   Score: {source.score:.4f}")
        print(f"   Source: {source.source_url}")
        print(f"   {source.content}")
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
