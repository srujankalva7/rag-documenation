"""Load and validate version-controlled evaluation questions."""

from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.models import EvaluationCase


def load_cases(path: Path) -> list[EvaluationCase]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("evaluation dataset must be a JSON array")
    cases = [EvaluationCase.from_dict(item) for item in value]
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("evaluation case ids must be unique")
    return cases
