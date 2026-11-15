"""
mcp_server.py — CorpBrain's knowledge base as MCP tools, over stdio.

    python mcp_server.py

Tools:
    search_documents  ranked chunks from the shared Retriever, with source/page/type
    list_documents    every ingested source with its page count

stdout is the MCP protocol channel, so nothing in this process may print to it;
logging goes to stderr. The server chdirs to the repo root on start because the
vector DB paths are relative and MCP clients launch servers from anywhere.
"""
import contextlib
import os
import sys
from functools import lru_cache
from typing import Annotated, Literal

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

from dotenv import load_dotenv  # noqa: E402
from pydantic import Field  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))

from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

with contextlib.redirect_stdout(sys.stderr):  # keep import-time chatter off the protocol channel
    from ingest import get_ingested_files
    from retriever import Retriever

mcp = MCPServer(
    name="corpbrain",
    instructions=(
        "Search CorpBrain's ingested PDF knowledge base. Call list_documents to see "
        "which files exist, then search_documents to fetch relevant chunks. Cite "
        "source and page from the results."
    ),
)


@lru_cache(maxsize=32)
def _retriever(strategy: str, rerank: bool, top_n: int, sources: tuple[str, ...] | None) -> Retriever:
    with contextlib.redirect_stdout(sys.stderr):
        return Retriever(
            strategy=strategy,
            rerank=rerank,
            top_n=top_n,
            sources=list(sources) if sources else None,
        )


@mcp.tool()
def search_documents(
    query: Annotated[str, Field(description="Natural-language search query.")],
    strategy: Annotated[
        Literal["dense", "bm25", "hybrid"],
        Field(description="dense = embedding similarity, bm25 = keyword match, hybrid = both (best default)."),
    ] = "hybrid",
    rerank: Annotated[
        bool, Field(description="Rerank candidates with a local cross-encoder before taking the top k.")
    ] = True,
    k: Annotated[int, Field(ge=1, le=20, description="Number of chunks to return.")] = 5,
    sources: Annotated[
        list[str] | None,
        Field(description="Only search these files (exact names from list_documents). Omit to search everything."),
    ] = None,
) -> list[dict]:
    """Search the knowledge base and return the most relevant chunks, best first.

    Each result has rank, source (file name), page (1-based), type ("text" for
    page text, "image_summary" for a vision summary of the page's figures and
    diagrams) and content.
    """
    if sources:
        known = get_ingested_files()
        unknown = sorted(set(sources) - set(known))
        if unknown:
            raise ToolError(f"Unknown source(s) {unknown}. Ingested sources: {sorted(known)}")
        sources_key = tuple(sorted(set(sources)))
    else:
        sources_key = None

    r = _retriever(strategy, rerank, k, sources_key)
    with contextlib.redirect_stdout(sys.stderr):
        docs = r.retrieve(query)
    return [
        {
            "rank": i,
            "source": d.metadata.get("source"),
            "page": d.metadata.get("page"),
            "type": d.metadata.get("type"),
            "content": d.page_content,
        }
        for i, d in enumerate(docs, start=1)
    ]


@mcp.tool()
def list_documents() -> list[dict]:
    """List every ingested document with its page count. Use the names as `sources` in search_documents."""
    return [{"source": src, "pages": pages} for src, pages in get_ingested_files().items()]


if __name__ == "__main__":
    mcp.run("stdio")
