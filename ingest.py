import os
import base64
import hashlib
import io
import json
import fitz
import chromadb
from PIL import Image
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma

load_dotenv()

DATA_PATH = "data/"
DB_PATH = "vector_db"
IMAGE_OUT_PATH = "extracted_images/"
CHROMA_COLLECTION = "langchain"
CONTEXT_CACHE_PATH = os.path.join(IMAGE_OUT_PATH, "chunk_contexts.json")


# ── Hashing ───────────────────────────────────────────────────────────────────

def compute_file_hash(file_bytes: bytes) -> str:
    """MD5 hash of file bytes — used as the deduplication key."""
    return hashlib.md5(file_bytes).hexdigest()


def _chunk_hash(text: str) -> str:
    """MD5 hash of chunk text — used as the context cache key."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ── Context cache ─────────────────────────────────────────────────────────────

def load_context_cache() -> dict:
    """Load the chunk context cache from disk."""
    if os.path.exists(CONTEXT_CACHE_PATH):
        with open(CONTEXT_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_context_cache(cache: dict):
    """Persist the chunk context cache to disk."""
    os.makedirs(IMAGE_OUT_PATH, exist_ok=True)
    with open(CONTEXT_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ── ChromaDB queries ──────────────────────────────────────────────────────────

def get_ingested_files(db_path: str = DB_PATH) -> dict:
    """
    Return {filename: page_count} for every document already in ChromaDB.
    Page count is derived from the highest page number seen in text chunks.
    """
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(CHROMA_COLLECTION)
        result = collection.get(include=["metadatas"])
        page_counts = {}
        for meta in result["metadatas"]:
            if meta and meta.get("type") == "text" and "source" in meta:
                source = meta["source"]
                page = meta.get("page", 0)
                if source not in page_counts or page > page_counts[source]:
                    page_counts[source] = page
        return dict(sorted(page_counts.items()))
    except Exception:
        return {}


def get_existing_hashes(db_path: str = DB_PATH) -> set:
    """
    Return the set of file hashes already stored in ChromaDB.
    Uses the raw chromadb client so no embedding API call is needed.
    """
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(CHROMA_COLLECTION)
        result = collection.get(include=["metadatas"])
        return {
            meta["file_hash"]
            for meta in result["metadatas"]
            if meta and "file_hash" in meta
        }
    except Exception:
        return set()


# ── Image helpers ─────────────────────────────────────────────────────────────

def _compress_image_for_api(image_path: str) -> tuple:
    """
    Compress a PNG to JPEG in-memory to stay under Claude's 5MB base64 limit.
    Returns (base64_string, media_type).
    """
    img = Image.open(image_path).convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded, "image/jpeg"


def summarize_page_image(image_path: str) -> str:
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
    encoded_string, media_type = _compress_image_for_api(image_path)

    prompt = (
        "You are an expert document analyst. Describe this page in thorough detail. "
        "Extract all visible text verbatim. If there are diagrams, charts, tables, or figures, "
        "explain what they depict, how components relate, and any labels, legends, or annotations. "
        "Capture all numbered steps, warnings, callouts, and part references exactly as written."
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": encoded_string,
                },
            },
        ]
    )

    print(f"Generating vision summary for {image_path}...")
    response = llm.invoke([message])
    return response.content


# ── Contextual chunking ───────────────────────────────────────────────────────

def generate_chunk_context(
    chunk_text: str,
    page_text: str,
    source: str,
    page_num: int,
    llm,
) -> str:
    """
    Call Claude Haiku to generate a single context sentence that situates
    this chunk within its page and document. Based on Anthropic's contextual
    retrieval technique.
    """
    prompt = (
        f"Document: {source}, Page {page_num}\n\n"
        f"Full page text:\n{page_text}\n\n"
        f"Chunk to contextualize:\n{chunk_text}\n\n"
        "Write a single sentence (max 30 words) situating this chunk within the "
        "document for search retrieval purposes. "
        "Output only the sentence, nothing else."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


# ── Core ingestion ────────────────────────────────────────────────────────────

def ingest_pdf_file(pdf_path: str, file_hash: str, progress_callback=None) -> int:
    """
    Ingest a single PDF into the existing ChromaDB incrementally.

    Pipeline:
      Phase 1 — per page: extract raw text, rasterize, vision summarize
      Phase 2 — chunk text documents only (image summaries kept whole)
      Phase 3 — contextual chunking: prepend a Haiku-generated context sentence
      Phase 4 — embed all chunks into ChromaDB

    progress_callback(current, total, status=None) is called to report progress.
    Returns the number of pages ingested.
    """
    os.makedirs(IMAGE_OUT_PATH, exist_ok=True)

    file = os.path.basename(pdf_path)
    text_documents = []
    image_documents = []
    raw_page_texts = {}  # {page_num: raw_text} — kept for context generation

    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

    # ── Phase 1: Page extraction and vision summarization ─────────────────────
    for page_num in range(total_pages):
        page = doc.load_page(page_num)

        raw_text = page.get_text()
        raw_page_texts[page_num + 1] = raw_text

        text_documents.append(Document(
            page_content=raw_text,
            metadata={
                "source": file,
                "page": page_num + 1,
                "type": "text",
                "file_hash": file_hash,
            },
        ))

        # Skip vision if the page has no embedded images AND has substantial text.
        # Rasterising a pure-text page wastes a Haiku call — PyMuPDF already
        # extracted the text more accurately via get_text(). Pages with embedded
        # images (charts, photos, infographics) still need vision even if they
        # also carry text, because the visual content won't appear in get_text().
        has_embedded_images = len(page.get_images(full=True)) > 0
        has_substantial_text = len(raw_text.strip()) > 100
        skip_vision = (not has_embedded_images) and has_substantial_text

        if skip_vision:
            print(f"  Skipping vision for page {page_num + 1} (text-only, no embedded images)")
            if progress_callback:
                progress_callback(page_num + 1, total_pages)
            continue

        # Rasterize page → PNG
        pix = page.get_pixmap(dpi=150)
        image_filename = f"{file}_page_{page_num + 1}.png"
        image_path = os.path.join(IMAGE_OUT_PATH, image_filename)
        pix.save(image_path)

        # Vision summary — use cached txt if available
        summary_path = image_path.replace(".png", "_summary.txt")
        if os.path.exists(summary_path):
            print(f"Using cached summary for {image_path}...")
            with open(summary_path, "r", encoding="utf-8") as f:
                page_summary = f.read()
        else:
            page_summary = summarize_page_image(image_path)
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(page_summary)

        # Option A: image summaries stored whole — never passed to text_splitter
        image_documents.append(Document(
            page_content=page_summary,
            metadata={
                "source": file,
                "page": page_num + 1,
                "image_path": image_path,
                "type": "image_summary",
                "file_hash": file_hash,
            },
        ))

        if progress_callback:
            progress_callback(page_num + 1, total_pages)

    doc.close()

    # ── Phase 2: Chunk text documents only (Option A) ─────────────────────────
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(text_documents)

    # ── Phase 3: Contextual chunking (Option C) ───────────────────────────────
    context_cache = load_context_cache()
    total_chunks = len(text_chunks)

    for idx, chunk in enumerate(text_chunks):
        chunk_key = _chunk_hash(chunk.page_content)
        page_num = chunk.metadata.get("page", 1)
        page_text = raw_page_texts.get(page_num, "")

        if chunk_key in context_cache:
            print(f"Using cached context for chunk {idx + 1}/{total_chunks}...")
            context = context_cache[chunk_key]
        else:
            context = generate_chunk_context(
                chunk.page_content, page_text, file, page_num, llm
            )
            context_cache[chunk_key] = context

        # Prepend context — the enriched text is what gets embedded
        chunk.page_content = f"{context}\n\n{chunk.page_content}"

        if progress_callback:
            progress_callback(
                idx + 1,
                total_chunks,
                f"Generating context {idx + 1} / {total_chunks}",
            )

    save_context_cache(context_cache)

    # ── Phase 4: Embed context-enriched text chunks + whole image docs ────────
    all_chunks = text_chunks + image_documents

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

    batch_size = 80
    for i in range(0, len(all_chunks), batch_size):
        vector_db.add_documents(all_chunks[i : i + batch_size])

    print(f"Ingested {total_pages} pages from {file}.")
    return total_pages


def create_multimodal_vector_db():
    """CLI entry point: scan data/ and ingest any PDFs not yet in ChromaDB."""
    print(f"Scanning {DATA_PATH} for new documents...")
    os.makedirs(IMAGE_OUT_PATH, exist_ok=True)

    existing_hashes = get_existing_hashes()
    new_files = 0

    for file in os.listdir(DATA_PATH):
        if not file.endswith(".pdf"):
            continue
        pdf_path = os.path.join(DATA_PATH, file)
        with open(pdf_path, "rb") as f:
            file_hash = compute_file_hash(f.read())

        if file_hash in existing_hashes:
            print(f"Skipping {file} — already indexed.")
            continue

        print(f"Ingesting {file}...")
        ingest_pdf_file(pdf_path, file_hash)
        new_files += 1

    if new_files == 0:
        print("No new documents to ingest.")
    else:
        print(f"Done. Ingested {new_files} new document(s).")


if __name__ == "__main__":
    create_multimodal_vector_db()
