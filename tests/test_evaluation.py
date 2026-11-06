from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import httpx
import pytest

from app.evaluation.dataset import load_cases
from app.evaluation.metrics import evaluate_response, summarize
from app.evaluation.models import EvaluationCase
from scripts.evaluate import run


def api_response(
    *,
    document_id: str = "request-body",
    cache_status: str = "miss",
    cost: float = 0.001,
) -> dict[str, object]:
    chunk_id = f"{document_id}:0"
    return {
        "supported": True,
        "retrieval_results": [{"document_id": document_id, "chunk_id": chunk_id}],
        "citations": [{"chunk_id": chunk_id}],
        "cache": {
            "status": cache_status,
            "avoided_generation_cost_usd": 0.001 if cache_status != "miss" else None,
        },
        "latency_ms": 12.5,
        "estimated_cost_usd": cost,
    }


def test_dataset_has_unique_reviewable_questions() -> None:
    dataset = Path("data/evaluation/fastapi_questions.json")
    cases = load_cases(dataset)
    sources = {
        item["id"]
        for item in json.loads(
            Path("data/sources/fastapi_urls.json").read_text(encoding="utf-8")
        )
    }

    assert len(cases) >= 30
    assert len({case.case_id for case in cases}) == len(cases)
    assert all(set(case.expected_document_ids) <= sources for case in cases)


def test_evaluate_response_scores_retrieval_citation_cache_and_cost() -> None:
    case = EvaluationCase(
        case_id="body",
        question="How do I declare a request body?",
        expected_document_ids=("request-body",),
    )

    result = evaluate_response(
        case,
        "economy",
        1,
        api_response(),
        wall_latency_ms=15.0,
    )

    assert result.relevant_rank == 1
    assert result.hit_at_3 is True
    assert result.hit_at_5 is True
    assert result.reciprocal_rank == 1.0
    assert result.citation_correct is True
    assert result.estimated_generation_cost_usd == 0.001
    assert result.estimated_avoided_generation_cost_usd == 0.0


def test_summary_keeps_actual_and_avoided_cost_separate() -> None:
    case = EvaluationCase("body", "body?", ("request-body",))
    miss = evaluate_response(
        case,
        "economy",
        1,
        api_response(),
        wall_latency_ms=15.0,
    )
    hit = evaluate_response(
        case,
        "economy",
        2,
        api_response(cache_status="exact_hit", cost=0.0),
        wall_latency_ms=2.0,
    )

    summary = summarize([miss, hit], "economy")

    assert summary.questions == 2
    assert summary.hit_rate_at_3 == 1.0
    assert summary.citation_accuracy == 1.0
    assert summary.exact_cache_hit_rate == 0.5
    assert summary.total_estimated_generation_cost_usd == 0.001
    assert summary.total_estimated_avoided_generation_cost_usd == 0.001


def test_evaluation_runner_exports_machine_readable_report(tmp_path: Path) -> None:
    dataset = tmp_path / "questions.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "body",
                    "question": "How do I declare a request body?",
                    "expected_document_ids": ["request-body"],
                    "category": "tutorial",
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/ask"
        assert json.loads(request.content)["preset"] == "economy"
        return httpx.Response(200, json=api_response())

    client = httpx.Client(transport=httpx.MockTransport(respond))
    arguments = Namespace(
        dataset=dataset,
        output=output,
        api_url="http://testserver",
        presets=["economy"],
        repeat=1,
        use_cache=False,
        timeout_seconds=10.0,
    )

    assert run(arguments, client=client) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summaries"][0]["hit_rate_at_3"] == 1.0
    assert report["results"][0]["case_id"] == "body"


def test_evaluation_rejects_invalid_repeat(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repeat must be positive"):
        run(
            Namespace(
                dataset=tmp_path / "missing.json",
                output=tmp_path / "report.json",
                api_url="http://testserver",
                presets=["economy"],
                repeat=0,
                use_cache=False,
                timeout_seconds=10.0,
            )
        )
