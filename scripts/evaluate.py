"""Run the versioned RAG evaluation dataset against a live API."""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from time import perf_counter

import httpx

from app.evaluation.dataset import load_cases
from app.evaluation.metrics import evaluate_response, summarize

DEFAULT_PRESETS = ("economy", "balanced", "accuracy")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/fastapi_questions.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/results/latest.json"),
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--preset",
        action="append",
        choices=DEFAULT_PRESETS,
        dest="presets",
        help="Preset to evaluate; repeat to compare multiple presets",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Allow cache reuse; use --repeat 2 or more to measure hit rates",
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser


def run(arguments: Namespace, *, client: httpx.Client | None = None) -> int:
    if arguments.repeat < 1:
        raise ValueError("--repeat must be positive")
    if arguments.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    cases = load_cases(arguments.dataset)
    presets = arguments.presets or list(DEFAULT_PRESETS)
    owns_client = client is None
    resolved_client = client or httpx.Client(timeout=arguments.timeout_seconds)
    results = []
    try:
        for preset in presets:
            for repetition in range(1, arguments.repeat + 1):
                for case in cases:
                    started = perf_counter()
                    response = resolved_client.post(
                        f"{arguments.api_url.rstrip('/')}/ask",
                        json={
                            "query": case.question,
                            "preset": preset,
                            "category": case.category,
                            "use_cache": arguments.use_cache,
                        },
                    )
                    wall_latency_ms = (perf_counter() - started) * 1000
                    response.raise_for_status()
                    results.append(
                        evaluate_response(
                            case,
                            preset,
                            repetition,
                            response.json(),
                            wall_latency_ms=wall_latency_ms,
                        )
                    )
    finally:
        if owns_client:
            resolved_client.close()

    summaries = [summarize(results, preset) for preset in presets]
    report = {
        "dataset": str(arguments.dataset),
        "cache_enabled": arguments.use_cache,
        "repeat": arguments.repeat,
        "summaries": [summary.to_dict() for summary in summaries],
        "results": [result.to_dict() for result in results],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for summary in summaries:
        print(f"{summary.preset}: {summary.questions} evaluations")
        print(f"  Hit@3: {summary.hit_rate_at_3:.1%}")
        print(f"  Hit@5: {summary.hit_rate_at_5:.1%}")
        print(f"  MRR: {summary.mean_reciprocal_rank:.3f}")
        print(f"  Citation accuracy: {summary.citation_accuracy:.1%}")
        print(f"  Average latency: {summary.average_server_latency_ms:.1f} ms")
        print(
            "  Estimated generation cost: "
            f"${summary.total_estimated_generation_cost_usd:.6f}"
        )
        print(
            "  Estimated avoided cost: "
            f"${summary.total_estimated_avoided_generation_cost_usd:.6f}"
        )
    print(f"Report: {arguments.output}")
    return 0


def main() -> int:
    parser = build_parser()
    try:
        return run(parser.parse_args())
    except (ValueError, httpx.HTTPError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
