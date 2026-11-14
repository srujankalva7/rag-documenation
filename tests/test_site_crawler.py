from __future__ import annotations

import httpx

from app.ingestion.crawler import SiteCrawler
from app.ingestion.site import SiteConfig


def test_crawler_uses_sitemap_and_same_site_links_with_limits() -> None:
    pages = {
        "/robots.txt": (
            "text/plain",
            "User-agent: *\nAllow: /\nSitemap: https://docs.example.com/sitemap.xml",
        ),
        "/sitemap.xml": (
            "application/xml",
            """<?xml version="1.0"?>
            <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
              <url><loc>https://docs.example.com/guide/</loc></url>
              <url><loc>https://docs.example.com/api/</loc></url>
            </urlset>""",
        ),
        "/": (
            "text/html",
            """<html><main><h1>Example Docs</h1>
            <a href="/guide/">Guide</a>
            <a href="/login/">Login</a>
            <a href="https://other.example/page/">External</a>
            <a href="/manual.pdf">PDF</a></main></html>""",
        ),
        "/guide/": (
            "text/html",
            "<html><main><h1>Guide</h1><a href='/guide/deep/'>Deep</a></main></html>",
        ),
        "/api/": ("text/html", "<html><main><h1>API</h1></main></html>"),
        "/guide/deep/": (
            "text/html",
            "<html><main><h1>Deep Guide</h1></main></html>",
        ),
    }

    def respond(request: httpx.Request) -> httpx.Response:
        content_type, body = pages.get(
            request.url.path,
            ("text/plain", "not found"),
        )
        status = 200 if request.url.path in pages else 404
        return httpx.Response(
            status,
            text=body,
            headers={"content-type": content_type},
        )

    client = httpx.Client(transport=httpx.MockTransport(respond))
    config = SiteConfig.create(
        "https://docs.example.com/",
        site_id="example-docs",
        max_pages=3,
        max_depth=2,
    )
    crawler = SiteCrawler(delay_seconds=0, client=client)

    result = crawler.crawl(config)

    assert [page.source.title for page in result.pages] == [
        "Example Docs",
        "Guide",
        "API",
    ]
    assert all(
        page.source.url.startswith("https://docs.example.com/") for page in result.pages
    )
    assert not result.failures


def test_crawler_rejects_redirect_to_private_target() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/secret"})
        return httpx.Response(404, text="missing")

    crawler = SiteCrawler(
        delay_seconds=0,
        client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    config = SiteConfig.create(
        "https://docs.example.com/",
        site_id="example-docs",
        max_pages=1,
    )

    result = crawler.crawl(config)

    assert result.pages == []
    assert len(result.failures) == 1
    assert "public hostname" in result.failures[0].error


def test_include_path_does_not_block_sitemap_discovery() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.path == "/sitemap.xml":
            return httpx.Response(
                200,
                text=(
                    "<urlset><url><loc>"
                    "https://docs.example.com/guide/"
                    "</loc></url></urlset>"
                ),
                headers={"content-type": "application/xml"},
            )
        if request.url.path == "/guide/":
            return httpx.Response(
                200,
                text="<html><main><h1>Guide</h1></main></html>",
                headers={"content-type": "text/html"},
            )
        return httpx.Response(404)

    crawler = SiteCrawler(
        delay_seconds=0,
        client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    config = SiteConfig.create(
        "https://docs.example.com/",
        site_id="example-docs",
        include_paths=("/guide/",),
    )

    result = crawler.crawl(config)

    assert [page.source.title for page in result.pages] == ["Guide"]
