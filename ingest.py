import os
import base64
import hashlib
import io
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


def compute_file_hash(file_bytes: bytes) -> str:
    """MD5 hash of file bytes — used as the deduplication key."""
    return hashlib.md5(file_bytes).hexdigest()


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


def _compress_image_for_api(image_path: str) -> tuple:
    """
    Compress a PNG to JPEG in-memory to stay under limit.
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


def ingest_pdf_file(pdf_path: str, file_hash: str, progress_callback=None) -> int:
    """
    Ingest a single PDF into the existing ChromaDB incrementally.
    progress_callback(current_page, total_pages) is called after each page is processed.
    Returns the number of pages ingested.
    """
    os.makedirs(IMAGE_OUT_PATH, exist_ok=True)

    file = os.path.basename(pdf_path)
    documents = []

    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    for page_num in range(total_pages):
        page = doc.load_page(page_num)

        # Raw text chunk
        raw_text = page.get_text()
        documents.append(Document(
            page_content=raw_text,
            metadata={
                "source": file,
                "page": page_num + 1,
                "type": "text",
                "file_hash": file_hash,
            },
        ))

        # Rasterize page → PNG
        pix = page.get_pixmap(dpi=150)
        image_filename = f"{file}_page_{page_num + 1}.png"
        image_path = os.path.join(IMAGE_OUT_PATH, image_filename)
        pix.save(image_path)

        # Vision summary — use cached txt if available to avoid redundant API calls
        summary_path = image_path.replace(".png", "_summary.txt")
        if os.path.exists(summary_path):
            print(f"Using cached summary for {image_path}...")
            with open(summary_path, "r", encoding="utf-8") as f:
                page_summary = f.read()
        else:
            page_summary = summarize_page_image(image_path)
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(page_summary)

        documents.append(Document(
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

    # Chunk and embed into existing ChromaDB
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

    batch_size = 80
    for i in range(0, len(chunks), batch_size):
        vector_db.add_documents(chunks[i : i + batch_size])

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
