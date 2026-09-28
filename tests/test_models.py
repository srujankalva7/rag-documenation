from __future__ import annotations

import pytest

from app.ingestion.models import Source, SourceValidationError


def test_source_accepts_official_fastapi_documentation() -> None:
    source = Source.from_dict(
        {
            "id": "request-body",
            "title": "Request Body",
            "url": "https://fastapi.tiangolo.com/tutorial/body/",
            "category": "tutorial",
        }
    )

    assert source.id == "request-body"


@pytest.mark.parametrize(
    "url",
    [
        "http://fastapi.tiangolo.com/tutorial/body/",
        "https://example.com/tutorial/body/",
        "https://fastapi.tiangolo.com.evil.example/tutorial/body/",
        "https://fastapi.tiangolo.com/tutorial/body/?download=true",
    ],
)
def test_source_rejects_unapproved_urls(url: str) -> None:
    with pytest.raises(SourceValidationError):
        Source.from_dict(
            {
                "id": "request-body",
                "title": "Request Body",
                "url": url,
                "category": "tutorial",
            }
        )
