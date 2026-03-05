# CorpBrain: Multimodal RAG Document Assistant

CorpBrain is a **Multimodal Retrieval-Augmented Generation (RAG)** application. Designed to handle dense technical documents (like motorcycle manuals), it goes beyond standard text retrieval by utilizing full-page rasterization and vision models to "see," summarize, and retrieve complex vector diagrams, charts, and mechanical line art.


<img width="624" height="645" alt="got" src="https://github.com/user-attachments/assets/8ab71e35-bbfa-4efd-b047-4fee5888924f" />
<img width="623" height="298" alt="got2" src="https://github.com/user-attachments/assets/d2047ea1-5128-46f5-90e8-0269d3987221" />
<img width="621" height="443" alt="got3" src="https://github.com/user-attachments/assets/25572d4d-ad18-4c15-a4a1-385916f42b18" />

## Key Features

* **Multimodal Image Summarization:** Uses `PyMuPDF` to take high-resolution screenshots of PDF pages and `gemini-2.5-flash-lite` to generate rich text summaries of visual diagrams.
* **Hybrid Dual-Extraction:** Extracts and embeds both the *raw text* (for exact keyword matching) and the *visual summaries* (for spatial/diagram comprehension) into the same database.
* **Ensemble Retrieval:** Combines Vector Search (ChromaDB) and Keyword Search (BM25) to ensure high-precision fetching of technical specifications and part numbers.
* **Dynamic Image Injection:** Automatically fetches the original local `.png` diagrams and injects them into the LLM's context window.
* **Visual Streamlit UI:** Renders the AI's text answer alongside the exact reference diagrams directly in the chat interface.

---

## Architecture Flow

1. **Ingestion (`ingest.py`):** * Reads PDFs from the `data/` folder.
   * Extracts raw text and saves it as standard chunks.
   * Rasterizes each page to a `.png` file, sends it to Gemini Vision for a summary, and saves the summary chunk with an `image_path` metadata tag.
   * Embeds all chunks into a local ChromaDB vector store.
2. **Retrieval (`rag.py`):**
   * Uses an `EnsembleRetriever` (Vector + BM25) to find the top 5 most relevant chunks.
   * Filters the retrieved chunks to prevent duplicate images, encodes the required `.png` files to Base64, and constructs a multimodal payload.
   * Sends the text context and images to `gemini-2.5-flash` to generate the final answer.
3. **Front-End (`app.py`):**
   * Manages conversation state and visually renders both the AI's response and the reference images.

---

## Installation & Setup

### 1. Prerequisites
* Python 3.9+
* A Google Gemini API Key

### 2. Clone the Repository
```
git clone https://github.com/qusydani/CorpBrain
cd CorpBrain
```
### 3. Install Dependencies
```
pip install -r requirements.txt
```
### 4. Add Google API Key (.env file)
```
GOOGLE_API_KEY=your_gemini_api_key_here
```
## Usage

1. **Add your documents:** Place your desired PDF files in the `data/` folder.
2. **Ingest and rasterize (`ingest.py`)**
   ```
   python ingest.py
   ```
3. **Launch the UI (`app.py`)**
   ```
   streamlit run app.py
   ```


