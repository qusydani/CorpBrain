# CorpBrain: Multimodal RAG Document Assistant

CorpBrain is a **production-grade Multimodal Retrieval-Augmented Generation (RAG)** system built for dense technical documents. It combines vision-based page summarization, contextual chunking, hybrid search, cross-encoder reranking, and a full RAGAS evaluation pipeline, all wrapped in an interactive Streamlit UI.


<img width="1919" height="911" alt="image" src="https://github.com/user-attachments/assets/3c8b6b32-6a8e-44d5-88d4-434561b6c8ac" />

>Main conversation page to ask questions about knowledge base

<img width="1919" height="912" alt="image" src="https://github.com/user-attachments/assets/83f30ca3-a151-4245-9f1c-3f483aa30d9b" />

>Explore vectors, query, retrieval and ranking in latent space

<img width="1919" height="914" alt="image" src="https://github.com/user-attachments/assets/06f26e3c-8bbc-4af7-8238-6bea39b3a7f9" />

>RAGAS Evaluation pipeline and insights analysis

[![Watch the demo](https://img.youtube.com/vi/R_5rqiOa0P4/0.jpg)](https://www.youtube.com/watch?v=R_5rqiOa0P4)

>Watch the Demo
---

## Key Features

### Ingestion
- **Multimodal Vision Summarization** — Rasterizes PDF pages with embedded images (charts, diagrams, schematics) and sends them to **Claude Haiku** for rich, verbatim-aware text summaries; pure text pages skip vision for efficiency.
- **Contextual Chunking** — Implements Anthropic's contextual retrieval technique: calls Claude Haiku to prepend a 1-sentence situating context to every chunk before embedding, boosting retrieval precision and recall.
- **Deduplication** — MD5 file hashing prevents re-ingesting the same document, chunk-level context caching avoids redundant LLM calls on re-ingestion.

### Retrieval
- **Hybrid Search** — `EnsembleRetriever` fuses ChromaDB vector search (semantic) with BM25 (keyword) to capture both conceptual intent and exact technical terminology.
- **Cross-Encoder Reranking** — `ms-marco-MiniLM-L-6-v2` jointly scores every (query, candidate) pair and keeps the top 5, filtering noise from initial retrieval.
- **Multimodal Context Injection** — Retrieved image-summary chunks trigger automatic base64 encoding of the original PNG diagram, injecting it directly into **Claude Sonnet**'s context window alongside the text.
- **Streaming Responses** — Answers stream token-by-token via LangChain's streaming interface.

### Explore (3D Vector Space Visualizer)
- UMAP reduces all chunk embeddings to 3D; Plotly renders an interactive scatter plot colored by source document.
- Submit any query to see the full retrieval pipeline animate in real time: retrieved candidates (orange), reranked top-5 (red diamonds), and the query point (purple) plotted in the same latent space.
- Keyword-based chunk browser for manual inspection of any document chunk.

### Eval (RAGAS Evaluation Pipeline)
- Generates a synthetic Q&A test set from your own documents using Claude Haiku.
- Benchmarks up to **12 configurations** (3 retrieval strategies × 2 reranking options × 2 chunking schemes) via RAGAS metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall, and Answer Correctness.
- Surfaces actionable **Key Findings** with comparative bar charts:
  - Contextual chunking consistently improves Context Precision & Recall across all strategies.
  - MS MARCO cross-encoder reranking degrades Faithfulness on formal documents (domain mismatch).
  - Hybrid search leads all strategies on Answer Relevancy.

---

## Architecture

```
PDF(s)
  │
  ▼
[ ingest.py ]
  ├─ Phase 1: PyMuPDF text extraction + page rasterization (150 dpi)
  │           Claude Haiku vision → image_summary chunks (skips text-only pages)
  ├─ Phase 2: RecursiveCharacterTextSplitter (1000 / 200 overlap) on text docs
  ├─ Phase 3: Claude Haiku contextual chunking → prepend context sentence per chunk
  └─ Phase 4: Batch embed (Gemini embedding-001) → ChromaDB

Query
  │
  ▼
[ rag.py — MultimodalRAGChain ]
  ├─ Hybrid retrieval: ChromaDB (k=10) + BM25 (k=10) → 20 candidates
  ├─ Cross-encoder reranking → top 5
  ├─ Build multimodal payload: text context + base64 PNGs for image chunks
  └─ Claude Sonnet stream → streamed answer + source attribution

[ app.py — Streamlit ]
  ├─ Chat:    Q&A with streaming, source list, diagram display
  ├─ Explore: UMAP 3D + retrieval pipeline overlay + chunk browser
  └─ Eval:    Test set generation, baseline DB creation, RAGAS benchmark
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Embeddings | Google Gemini `embedding-001` |
| Vision / Contextual Chunking | Claude Haiku (`claude-haiku-4-5`) |
| Answer Generation | Claude Sonnet (`claude-sonnet-4-6`) |
| Eval LLM | Gemini 2.5 Flash |
| Orchestrastion Glue | LangChain |
| Vector Store | ChromaDB |
| Keyword Search | BM25 (rank-bm25) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| PDF Processing | PyMuPDF (fitz) |
| Evaluation | RAGAS |
| Dimensionality Reduction | UMAP |
| Visualization | Plotly |
| UI | Streamlit |

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- A Google Gemini API Key
- An Anthropic API Key

### 1. Clone the Repository
```bash
git clone https://github.com/qusydani/CorpBrain
cd CorpBrain
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys (`.env`)
```
GOOGLE_API_KEY=your_gemini_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 4. Run the App
```bash
streamlit run app.py
```

Upload PDFs via the sidebar, click **Ingest Documents**, then ask questions in the **Chat** tab. Use **Explore** to visualize your knowledge base in 3D, and **Eval** to benchmark retrieval configurations.
