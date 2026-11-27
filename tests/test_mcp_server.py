"""
End-to-end tests for mcp_server.py: spawn it as a real stdio subprocess and talk
to it with the MCP SDK client.

The tool-listing test needs nothing else. The search/list tests need a populated
vector_db/ and GOOGLE_API_KEY (dense/hybrid query embeddings), and skip otherwise.
"""
import asyncio
import json
import os
import sys

import chromadb
import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(ROOT, "mcp_server.py")
DB_PATH = os.path.join(ROOT, "vector_db")


def _db_ready() -> bool:
    try:
        return chromadb.PersistentClient(path=DB_PATH).get_collection("langchain").count() > 0
    except Exception:
        return False


needs_db = pytest.mark.skipif(not _db_ready(), reason="vector DB not populated")


async def _with_session(fn):
    # Launch from a different cwd to prove the server doesn't rely on it.
    params = StdioServerParameters(command=sys.executable, args=[SERVER], cwd=os.path.dirname(ROOT))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await fn(session)


def run(fn):
    return asyncio.run(_with_session(fn))


async def _call(session, name, args=None):
    res = await session.call_tool(name, args or {})
    return res


def _payload(res):
    """Tool results as Python data (structured content if present, else the JSON text)."""
    if res.structured_content is not None:
        sc = res.structured_content
        return sc["result"] if isinstance(sc, dict) and set(sc) == {"result"} else sc
    return [json.loads(c.text) for c in res.content]


def test_lists_every_tool_with_schemas():
    tools = run(lambda s: s.list_tools()).tools
    by_name = {t.name: t for t in tools}
    assert set(by_name) == {"search_documents", "list_documents", "get_page", "get_page_image"}

    search = by_name["search_documents"]
    assert search.description and "chunks" in search.description
    props = search.input_schema["properties"]
    assert set(props) == {"query", "strategy", "rerank", "k", "sources"}
    assert search.input_schema["required"] == ["query"]
    assert props["query"]["description"]
    assert by_name["list_documents"].description

    for name in ("get_page", "get_page_image"):
        tool = by_name[name]
        assert tool.description and "page" in tool.description.lower()
        assert set(tool.input_schema["properties"]) == {"source", "page"}
        assert sorted(tool.input_schema["required"]) == ["page", "source"]


@needs_db
def test_list_documents_returns_every_ingested_source():
    os.chdir(ROOT)
    from ingest import get_ingested_files

    res = run(lambda s: _call(s, "list_documents"))
    assert not res.is_error
    docs = _payload(res)
    assert {d["source"]: d["pages"] for d in docs} == get_ingested_files()


@needs_db
def test_search_returns_ranked_chunks_with_metadata():
    res = run(lambda s: _call(s, "search_documents", {"query": "safety precautions", "k": 3}))
    assert not res.is_error, res.content
    hits = _payload(res)
    assert [h["rank"] for h in hits] == [1, 2, 3]
    for h in hits:
        assert h["source"] and isinstance(h["page"], int) and h["content"]
        assert h["type"] in ("text", "image_summary")


@needs_db
def test_source_filter_restricts_results():
    async def go(s):
        docs = _payload(await _call(s, "list_documents"))
        target = docs[0]["source"]
        res = await _call(s, "search_documents", {"query": "overview", "k": 5, "sources": [target]})
        return target, res

    target, res = run(go)
    assert not res.is_error, res.content
    hits = _payload(res)
    assert hits and {h["source"] for h in hits} == {target}


@needs_db
def test_unknown_source_is_an_error():
    res = run(lambda s: _call(s, "search_documents", {"query": "x", "sources": ["nope.pdf"]}))
    assert res.is_error
    assert "nope.pdf" in res.content[0].text


# ── Page tools (need the source PDF in data/ too) ─────────────────────────────

def _page_source():
    """An ingested source whose PDF is in data/, or None."""
    os.chdir(ROOT)
    from ingest import get_ingested_files
    return next((s for s in get_ingested_files() if os.path.exists(os.path.join(ROOT, "data", s))), None)


needs_pdf = pytest.mark.skipif(not _db_ready() or _page_source() is None,
                               reason="no ingested source with its PDF in data/")


@needs_pdf
def test_get_page_returns_full_page_text():
    import fitz
    src = _page_source()
    res = run(lambda s: _call(s, "get_page", {"source": src, "page": 1}))
    assert not res.is_error, res.content
    page = _payload(res)
    with fitz.open(os.path.join(ROOT, "data", src)) as doc:
        expected = doc.load_page(0).get_text()
    assert page["source"] == src and page["page"] == 1
    assert page["text"] == expected
    assert "image_summary" in page


@needs_pdf
def test_get_page_image_returns_a_jpeg():
    import base64
    src = _page_source()
    res = run(lambda s: _call(s, "get_page_image", {"source": src, "page": 1}))
    assert not res.is_error, res.content
    images = [c for c in res.content if c.type == "image"]
    assert len(images) == 1 and images[0].mime_type == "image/jpeg"
    assert base64.b64decode(images[0].data)[:2] == b"\xff\xd8"   # JPEG magic
    assert any(c.type == "text" and src in c.text for c in res.content)


@needs_pdf
def test_page_out_of_range_is_an_error():
    src = _page_source()
    for tool in ("get_page", "get_page_image"):
        res = run(lambda s: _call(s, tool, {"source": src, "page": 99999}))
        assert res.is_error
        assert "99999" in res.content[0].text and "pages 1-" in res.content[0].text


@needs_db
def test_page_tools_reject_unknown_source():
    for tool in ("get_page", "get_page_image"):
        res = run(lambda s: _call(s, tool, {"source": "nope.pdf", "page": 1}))
        assert res.is_error
        assert "nope.pdf" in res.content[0].text and "Ingested sources" in res.content[0].text
