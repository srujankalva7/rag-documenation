"""Preview or apply targeted semantic-cache invalidation."""

from __future__ import annotations

import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

from app.cache.store import SQLiteSemanticCache


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("data/cache/semantic_cache.sqlite3"),
    )
    parser.add_argument(
        "--document-id",
        action="append",
        default=[],
        help="Invalidate entries citing this document; repeat for multiple IDs",
    )
    parser.add_argument("--provider", help="Filter by generation provider")
    parser.add_argument("--model", help="Filter by generation model")
    parser.add_argument(
        "--older-than-days",
        type=float,
        help="Filter entries created more than this many days ago",
    )
    parser.add_argument(
        "--expired",
        action="store_true",
        help="Filter entries whose cache lifetime has expired",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Select every active entry; cannot be combined with filters",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply invalidation; otherwise only preview matching entries",
    )
    parser.add_argument("--reason", default="manual_invalidation")
    return parser


def run(arguments: Namespace, *, now: float | None = None) -> int:
    timestamp = time.time() if now is None else now
    filters_selected = bool(
        arguments.document_id
        or arguments.provider
        or arguments.model
        or arguments.older_than_days is not None
        or arguments.expired
    )
    if arguments.all and filters_selected:
        raise ValueError("--all cannot be combined with invalidation filters")
    if not arguments.all and not filters_selected:
        raise ValueError("select at least one filter or use --all")
    if arguments.older_than_days is not None and arguments.older_than_days <= 0:
        raise ValueError("--older-than-days must be positive")
    if not arguments.reason.strip():
        raise ValueError("--reason cannot be empty")

    if not arguments.cache_path.is_file():
        print(f"Cache not found: {arguments.cache_path}")
        return 0

    created_before = None
    if arguments.older_than_days is not None:
        created_before = timestamp - arguments.older_than_days * 86_400
    cache = SQLiteSemanticCache(arguments.cache_path)
    entry_ids = cache.matching_entry_ids(
        document_ids=set(arguments.document_id) or None,
        provider=arguments.provider,
        model=arguments.model,
        created_before=created_before,
        expired_before=timestamp if arguments.expired else None,
    )
    print(f"Matched entries: {len(entry_ids)}")
    if entry_ids:
        print("Entry IDs: " + ", ".join(str(value) for value in entry_ids))
    if not arguments.apply:
        print("Preview only; pass --apply to invalidate these entries")
        return 0

    invalidated = cache.invalidate_entries(entry_ids, arguments.reason, now=timestamp)
    print(f"Invalidated entries: {invalidated}")
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
