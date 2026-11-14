from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.crawler import CrawledPage, CrawlResult
from app.ingestion.generic_parser import GenericDocumentParser
from app.ingestion.generic_pipeline import GenericSiteIngestionPipeline
from app.ingestion.models import SourceValidationError
from app.ingestion.site import SiteConfig, SiteSource, normalize_page_url
from app.ingestion.storage import DocumentStorage


def site_config(**overrides: object) -> SiteConfig:
    values = {
        "homepage_url": "https://docs.example.com/",
        "site_id": "example-docs",
        "max_pages": 10,
        "max_depth": 2,
    }
    values.update(overrides)
    return SiteConfig.create(**values)


def test_site_config_accepts_public_site_and_normalizes_url() -> None:
    config = site_config(homepage_url="HTTPS://DOCS.EXAMPLE.COM/start?ref=ad#top")

    assert config.homepage_url == "https://docs.example.com/start/"
    assert config.allowed_domains == ("docs.example.com",)
    assert config.allows_url("https://docs.example.com/guide/") is True
    assert config.allows_url("https://evil.example/guide/") is False
    assert normalize_page_url("https://docs.example.com/a#part") == (
        "https://docs.example.com/a/"
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/docs/",
        "http://127.0.0.1/docs/",
        "http://10.0.0.2/docs/",
        "file:///etc/passwd",
    ],
)
def test_site_config_rejects_non_public_targets(url: str) -> None:
    with pytest.raises(SourceValidationError):
        SiteConfig.create(url, site_id="unsafe")


def test_site_source_id_is_stable_for_same_page() -> None:
    config = site_config()

    first = SiteSource.from_url(config, "https://docs.example.com/guide/#intro")
    second = SiteSource.from_url(config, "https://docs.example.com/guide/?ref=home")

    assert first.id == second.id


def test_generic_parser_removes_chrome_and_extracts_sections() -> None:
    config = site_config()
    source = SiteSource.from_url(
        config,
        "https://docs.example.com/guide/",
        "Example Guide",
    )
    html = """
    <html><head><title>Example</title></head><body>
      <nav>Account Pricing Search</nav>
      <main>
        <h1 id="start">Getting Started</h1>
        <p>Install the package and create an application.</p>
        <pre><code class="language-python">app = Example()</code></pre>
        <h2 id="options">Options</h2>
        <ul><li>Fast mode</li><li>Safe mode</li></ul>
      </main>
      <footer>Copyright and navigation</footer>
    </body></html>
    """

    document = GenericDocumentParser().parse(source, html)

    assert [section.heading for section in document.sections] == [
        "Getting Started",
        "Options",
    ]
    assert document.sections[0].source_url.endswith("#start")
    assert document.sections[0].code_blocks[0].language == "python"
    assert "Account Pricing Search" not in str(document.to_dict())
    assert "Fast mode" in document.sections[1].content


class FakeCrawler:
    def __init__(self, page: CrawledPage) -> None:
        self.page = page

    def crawl(self, config: SiteConfig) -> CrawlResult:
        assert config.site_id == "example-docs"
        return CrawlResult(pages=[self.page])


def test_generic_pipeline_writes_existing_document_format(tmp_path: Path) -> None:
    config = site_config()
    source = SiteSource.from_url(config, config.homepage_url, "Example Docs")
    page = CrawledPage(
        source,
        "<html><main><h1>Example Docs</h1><p>Welcome to the docs.</p></main></html>",
        0,
    )
    storage = DocumentStorage(
        raw_dir=tmp_path / "raw",
        documents_dir=tmp_path / "documents",
        versions_dir=tmp_path / "versions",
    )
    pipeline = GenericSiteIngestionPipeline(
        FakeCrawler(page),
        GenericDocumentParser(),
        storage,
    )

    first = pipeline.run(config)
    second = pipeline.run(config)

    assert first.ingestion.results[0].status == "created"
    assert second.ingestion.results[0].status == "unchanged"
    assert (tmp_path / "documents" / f"{source.id}.json").exists()
