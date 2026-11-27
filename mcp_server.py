"""
mcp_server.py — CorpBrain's knowledge base as MCP tools, over stdio.

    python mcp_server.py

Tools:
    search_documents  ranked chunks from the shared Retriever, with source/page/type
    list_documents    every ingested source with its page count
    get_page          one page's full text (plus its vision summary, if any)
    get_page_image    one page rasterized as a JPEG, for diagrams and figures

stdout is the MCP protocol channel, so nothing in this process may print to it;
logging goes to stderr. The server chdirs to the repo root on start because the
vector DB paths are relative and MCP clients launch servers from anywhere.
"""
import contextlib
import io
import os
import sys
from functools import lru_cache
from typing import Annotated, Literal

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

from dotenv import load_dotenv  # noqa: E402
from pydantic import Field  # noqa: E402
from typing_extensions import TypedDict  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))

from mcp.server.mcpserver import Image, MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402
from mcp_types import TextContent  # noqa: E402

with contextlib.redirect_stdout(sys.stderr):  # keep import-time chatter off the protocol channel
    import fitz
    from PIL import Image as PILImage
    from corpus import DATA_PATH, IMAGE_OUT_PATH
    from ingest import get_ingested_files
    from retriever import Retriever

RENDER_DPI = 150          # Same as ingest's rasterization
MAX_IMAGE_SIDE = 1600     # Long side in px of images sent to the model
JPEG_QUALITY = 85

mcp = MCPServer(
    name="corpbrain",
    instructions=(
        "Search CorpBrain's ingested PDF knowledge base. Call list_documents to see "
        "which files exist, then search_documents to fetch relevant chunks. Cite "
        "source and page from the results. When a chunk is cut off, call get_page for the "
        "page's full text; when a result is an image_summary and the question is about a "
        "figure, diagram, chart or table, call get_page_image to see the page itself."
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


def _check_page(source: str, page: int) -> str:
    """Path to the source PDF, or a ToolError saying what's wrong with (source, page)."""
    known = get_ingested_files()
    if source not in known:
        raise ToolError(f"Unknown source {source!r}. Ingested sources: {sorted(known)}")
    path = os.path.join(DATA_PATH, source)
    if not os.path.exists(path):
        raise ToolError(f"{source} is ingested but its PDF is missing from {DATA_PATH}")
    with fitz.open(path) as doc:
        count = doc.page_count
    if not 1 <= page <= count:
        raise ToolError(f"{source} has pages 1-{count}; page {page} does not exist")
    return path


def _cached(source: str, page: int, suffix: str) -> str:
    """Ingest's per-page artifact path, e.g. extracted_images/x.pdf_page_3.png."""
    return os.path.join(IMAGE_OUT_PATH, f"{source}_page_{page}{suffix}")


def _to_jpeg(img: PILImage.Image) -> bytes:
    img = img.convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


SourceArg = Annotated[str, Field(description="Exact file name, as in search results or list_documents.")]
PageArg = Annotated[int, Field(ge=1, description="1-based page number, as in search results.")]


class PageText(TypedDict):
    source: str
    page: int
    text: str
    image_summary: str | None


@mcp.tool()
def get_page(source: SourceArg, page: PageArg) -> PageText:
    """Full text of one page. Use it when a search chunk is cut off or you need the
    surrounding text. Also returns image_summary, the vision summary of the page's
    figures, when ingest made one (else null)."""
    path = _check_page(source, page)
    with fitz.open(path) as doc:
        text = doc.load_page(page - 1).get_text()
    summary_path = _cached(source, page, "_summary.txt")
    summary = None
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            summary = f.read()
    return PageText(source=source, page=page, text=text, image_summary=summary)


@mcp.tool()
def get_page_image(source: SourceArg, page: PageArg) -> list[TextContent | Image]:
    """The page itself as an image. Use it when a result is an image_summary and the
    question is about a figure, diagram, chart or table, so you can read it directly."""
    path = _check_page(source, page)
    png = _cached(source, page, ".png")
    if os.path.exists(png):
        img = PILImage.open(png)
    else:
        with fitz.open(path) as doc:
            pix = doc.load_page(page - 1).get_pixmap(dpi=RENDER_DPI)
            img = PILImage.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return [
        TextContent(type="text", text=f"Page image of {source} p.{page}"),
        Image(data=_to_jpeg(img), format="jpeg"),
    ]


if __name__ == "__main__":
    mcp.run("stdio")
