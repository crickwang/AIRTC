# AI-generated tests (Claude) for the ?v=<hash> cache-busting of /static/ assets in
# server.py, added 2026-09-06 after Cloudflare's per-edge cache kept serving an old
# main.js for hours after a deploy, even to hard-refreshing browsers.

import os

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import server


def test_rewrites_only_local_static_urls():
    html = (
        '<link href="/static/css/generate.css" rel="stylesheet">'
        '<link href="https://fonts.googleapis.com/css2?family=Inter" rel="stylesheet">'
        '<script src="/static/js/main.js"></script>'
        '<a href="/generate.html">go</a>'
    )
    out = server.version_static_urls(html, version="abc123")
    assert 'href="/static/css/generate.css?v=abc123"' in out
    assert 'src="/static/js/main.js?v=abc123"' in out
    # External and non-static URLs are left exactly as they were.
    assert 'href="https://fonts.googleapis.com/css2?family=Inter"' in out
    assert 'href="/generate.html"' in out


def test_version_changes_when_a_static_file_changes(tmp_path):
    # The hash must track content, not just names, or an edited main.js keeps its old URL.
    static = tmp_path / "static"
    (static / "js").mkdir(parents=True)
    (static / "js" / "main.js").write_text("v1")
    v1 = server._static_version(str(static))
    (static / "js" / "main.js").write_text("v2")
    v2 = server._static_version(str(static))
    assert v1 != v2
    assert len(v1) == 10
    # Deterministic for identical content, so every worker/process agrees on the URL.
    (static / "js" / "main.js").write_text("v1")
    assert server._static_version(str(static)) == v1


@pytest.mark.asyncio
async def test_served_page_references_versioned_assets():
    # The real handler, the real generate.html: every /static/ reference carries the
    # startup version, and the static route still serves the file with the query string.
    app = web.Application(middlewares=[server._static_no_cache])
    page = server.WebPage.__new__(server.WebPage)  # no args/DB/ICE setup needed for this
    app.router.add_get("/generate.html", page.generate)
    app.router.add_static("/static/", path=os.path.join(server.ROOT, "webpage", "static"))

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/generate.html")
        html = await resp.text()
        assert resp.status == 200
        assert f'src="/static/js/main.js?v={server.STATIC_VERSION}"' in html
        assert 'src="/static/js/main.js"' not in html

        asset = await client.get(f"/static/js/main.js?v={server.STATIC_VERSION}")
        assert asset.status == 200
        assert asset.headers["Cache-Control"] == "no-cache"
