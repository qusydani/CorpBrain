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
- **Multimodal Vision Summarization** — Rasterizes PDF pages with embedded images or vector-drawn charts and sends them to **Claude Haiku** (or Gemini Flash with `--llm gemini`) for rich, verbatim-aware text summaries; pure text pages skip vision for efficiency.
- **Contextual Chunking** — Implements Anthropic's contextual retrieval technique: calls Claude Haiku to prepend a 1-sentence situating context to every chunk before embedding, boosting retrieval precision and recall.
- **Deduplication** — MD5 file hashing prevents re-ingesting the same document, chunk-level context caching avoids redundant LLM calls on re-ingestion.

### Retrieval
- **Hybrid Search** — `EnsembleRetriever` fuses ChromaDB vector search (semantic) with BM25 (keyword) to capture both conceptual intent and exact technical terminology.
- **Cross-Encoder Reranking** — `ms-marco-MiniLM-L-6-v2` jointly scores every (query, candidate) pair and keeps the top 5, filtering noise from initial retrieval.
- **Multimodal Context Injection** — Retrieved image-summary chunks trigger automatic base64 encoding of the original PNG diagram, injecting it directly into **Claude Sonnet**'s context window alongside the text.
- **Streaming Responses** — Answers stream token-by-token via LangChain's streaming interface.

### Agent Mode
- **RAG / Agent toggle** in Chat. Agent mode runs a LangGraph agent on **Gemini 2.5 Flash** that decides which tools to call, and how often, before answering.
- **Router and planner** — a router call classifies each question. Simple ones go straight to the tool-calling loop. Complex ones (comparisons, multi-part, cross-document) get a structured plan of up to 4 sub-questions, whose searches run in parallel. A grader then marks any thin results, those sub-questions are searched once more with a rewritten query, and a synthesizer writes one cited answer. The plan appears in Chat before the answer streams.

```mermaid
flowchart LR
    S([question]) --> R[router]
    R -- simple --> M[model]
    M <--> T[tools]
    M --> E([answer])
    R -- complex --> P[planner]
    P -- no plan --> M
    P --> Q1[search 1] & Q2[search 2] & Qn[search N]
    Q1 & Q2 & Qn --> G[grade]
    G -- thin --> RS[re-search]
    RS --> SY[synthesize]
    G -- ok --> SY
    SY --> E
```

- **Tools over MCP** — the agent reaches retrieval only through `mcp_server.py` over stdio MCP, the same server any MCP client can use.
- **Live tool calls** — each call and its result appear as they happen; the answer streams with inline `[source p.N]` citations, a list of the cited pages, and token usage and cost.
- **Hard caps** — at most 6 tool-enabled model calls in the loop and 8 tool calls overall. When the loop hits a cap it makes one final call without tools so it still answers, and planned searches beyond the budget are skipped. Tool errors go back to the model instead of crashing the run.

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
| Tracing | Langfuse v4 (self-hosted, Docker Compose) |

---

## Installation & Setup

### Prerequisites
- Python 3.11+ (or just Docker, see "Run it with Docker" below)
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

#### Or run it with Docker

```bash
cp .env.example .env          # then fill in GOOGLE_API_KEY / ANTHROPIC_API_KEY
docker compose up --build     # app at http://localhost:8501
```

One image holds the Streamlit app, the agent and the MCP server. The agent starts `mcp_server.py` over stdio inside the same container, so Agent mode works with no extra setup. The image uses CPU-only torch and has the cross-encoder reranker built in, so nothing downloads at startup. `data/`, `vector_db/`, `vector_db_baseline/`, `vector_db_fixture/`, `extracted_images/` and `eval_data/` are mounted from the repo folder, so the app reads your existing index and anything you ingest or evaluate in the UI is saved to your disk. API keys come from `.env`. None of it is baked into the image.

- **With tracing:** `docker compose --profile observability up --build` also starts Langfuse (see section 8), and the app waits for it to be healthy so agent traces show up at http://localhost:3000. Inside the compose network the app reaches Langfuse at `langfuse-web:3000`, so you don't need to change `LANGFUSE_BASE_URL` in `.env`. The project keys default to the same values on both sides, so `.env` needs no `LANGFUSE_*` entries unless you override them.
- **Public corpus only:** `CORPBRAIN_CORPUS=fixture docker compose up` runs on the committed fixture PDFs (build the index first, see section 7).
- **Stop:** `docker compose down` (add `--profile observability` to stop Langfuse as well).

### 5. MCP Server (optional)

`mcp_server.py` exposes the knowledge base to any MCP client over stdio:

| Tool | What it returns |
|---|---|
| `search_documents(query, strategy, rerank, k, sources)` | Top `k` chunks (best first) with `source`, `page`, `type` (`text` / `image_summary`) and `content`. `sources` restricts the search to specific files. |
| `list_documents()` | Every ingested file with its page count. |

Try it with the MCP Inspector:
```bash
npx @modelcontextprotocol/inspector .venv/Scripts/python mcp_server.py
```

**Claude Desktop:** add this to `claude_desktop_config.json` (Settings → Developer → Edit Config), using absolute paths, then restart Claude Desktop:
```json
{
  "mcpServers": {
    "corpbrain": {
      "command": "C:\path\to\CorpBrain\.venv\Scripts\python.exe",
      "args": ["C:\path\to\CorpBrain\mcp_server.py"]
    }
  }
}
```
On macOS/Linux use `.venv/bin/python`. The server reads `GOOGLE_API_KEY` from the repo's `.env`, so nothing else needs configuring.

### 6. Agent from the CLI (optional)

```bash
python agent.py "Compare Enterprise Singapore's internationalisation support with the FY2026 budget's measures"
```

Tool calls and stats go to stderr; the streamed answer and its sources go to stdout.

### 7. Fixture corpus (public PDFs, used by CI)

`data/` holds personal documents and is never committed. For CI there is a small public corpus in `tests/fixtures/`:

- `corpus/pdfs/`: the FY2026 Singapore Budget Statement (82 pages) and a journal article on global energy demand (11 pages).
- `corpus/cache/`: Gemini Flash outputs from ingestion (one vision summary, plus `chunk_contexts.json` with a context sentence for each chunk). They're committed because LLM output differs from run to run. Rebuilding from them makes no LLM calls and gives the same chunks every time.
- `test_set.json`: 10 frozen questions (6 single-hop, 4 multi-hop: comparison, aggregation, multi-part). There are no cross-document questions because the two PDFs are unrelated, and generated cross-document pairs came out speculative.

`CORPBRAIN_CORPUS=fixture` points ingestion, retrieval, the MCP server and the eval at the fixture corpus. Its index goes in `vector_db_fixture/` and its results go in `eval_data/eval_results_fixture.json`. In fixture mode ingestion uses Gemini by default, so you only need `GOOGLE_API_KEY`.

```bash
# bash
CORPBRAIN_CORPUS=fixture python ingest.py            # build the fixture index
CORPBRAIN_CORPUS=fixture python eval.py run --agent  # eval on the fixture set
# PowerShell
$env:CORPBRAIN_CORPUS="fixture"; python ingest.py
```

Ingestion cost for the fixture corpus (Gemini 2.5 Flash list prices, embeddings estimated at 4 characters per token):

| Build | LLM calls | LLM cost | Embeddings (176 chunks) | Total |
|---|---|---|---|---|
| From scratch (no cache) | 176 | $0.060 | $0.007 | **~$0.067** |
| From the committed cache | 0 | $0 | $0.007 | **~$0.007** |

`python ingest.py --llm gemini|anthropic` picks the vision and context model for either corpus. The default corpus still uses Claude Haiku.

### 8. Tracing with Langfuse (optional)

Self-hosted [Langfuse](https://langfuse.com) v4 records every agent question as one trace: each Gemini call is a generation with its tokens, $ cost and latency, and each MCP tool call (`search_documents`, `get_page`, `get_page_image`, `list_documents`) is a tool span with its arguments, result and latency. Requires Docker and Python 3.11+ (for the `langfuse` SDK).

1. Add the project keys to `.env`. On first boot the server creates a `corpbrain` project with exactly these keys:
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-corpbrain-local
   LANGFUSE_SECRET_KEY=sk-lf-corpbrain-local
   LANGFUSE_BASE_URL=http://localhost:3000
   ```
2. Start the stack (web, worker, Postgres, ClickHouse, Redis, MinIO):
   ```bash
   docker compose --profile observability up -d
   ```
3. Open http://localhost:3000 and sign in as `admin@corpbrain.local` / `corpbrain-admin`.
4. Start the app or run `python agent.py "..."`. Traces appear under **Tracing**; chats from one browser session share a session id.

Tracing turns on only when the keys are set and the server answers its health check when the app starts. Otherwise the app runs as usual with tracing silently off, so start Langfuse before the app, or restart the app after starting Langfuse. The first time tracing turns on, the app registers the Gemini 2.5 Flash price from `pricing.py` in the project (**Settings → Models**), so Langfuse's costs match the eval's. The admin password, keys and every database secret have local-only defaults. Override the `LANGFUSE_*` variables listed in `docker-compose.yml` before exposing the stack beyond your machine. To stop it: `docker compose --profile observability down` (add `-v` to delete trace data).

### 9. CI eval gate

`.github/workflows/eval-gate.yml` runs on every pull request that touches Python code, `requirements.txt` or `tests/fixtures/`, and can also be started by hand from the Actions tab. It runs the unit tests (no API calls), rebuilds the fixture index from the committed cache, then runs the gate:

```bash
CORPBRAIN_CORPUS=fixture python eval.py gate
```

The gate answers the 10 fixture questions with two arms, the **Agent** and the best plain-RAG config (**Hybrid+Rerank+Ctx**), and grades each answer with the Gemini Flash accuracy judge. RAGAS is skipped. Each arm's accuracy is compared with the committed baseline in `tests/fixtures/eval_baseline.json`, and the job fails if any arm scores below *baseline − tolerance*. It also fails if `tests/fixtures/test_set.json` changed since the baseline was recorded, because the old scores would no longer be comparable. Scores, per-type accuracy, cost and latency go to the job summary, and the full per-question results are uploaded as the `gate-results` artifact.

**Tolerance.** Per arm: max(2 × the run-to-run standard deviation, one question), rounded up to a whole number of questions. On 10 questions accuracy moves in steps of 0.1, so a tolerance below one question would fail on noise. Measured on 2026-11-27:

| Arm | Baseline runs | Mean | Std | Tolerance | Fails below |
|---|---|---|---|---|---|
| Hybrid+Rerank+Ctx | 0.9, 0.9, 0.9 | 0.90 | 0.00 | 0.10 | 0.80 |
| Agent | 1.0, 1.0, 1.0 | 1.00 | 0.00 | 0.10 | 0.90 |

The next check run scored Hybrid+Rerank+Ctx at 1.0, so the runs aren't noise-free. The one-question floor absorbs that: a std of 0.05 over the 4 runs still gives a 0.10 tolerance. With a deliberate regression (retrieval `TOP_N` cut from 5 to 1), the gate failed: Hybrid+Rerank+Ctx dropped to 0.60 (multi-hop 0.25). The agent held at 1.00, because it searches several times per question.

**Cost.** About **$0.09–0.10 per gate run** (4 runs: $0.085, $0.094, $0.086, $0.099), plus ~$0.007 of embeddings to rebuild the index. About 78% is the agent and almost all of it is answer generation (Flash thinking tokens). The accuracy judge costs about $0.003. A run takes about 6 minutes after dependencies are installed.

**Setup (once):**
1. **Secret:** GitHub repo → Settings → Secrets and variables → Actions → *New repository secret*: `GOOGLE_API_KEY`. No Anthropic key is needed. Pull requests from forks don't receive secrets, so the gate fails on them with a clear error. Run those by hand after review.
2. **Make it block merges:** Settings → Branches (or Rules → Rulesets) → add a rule for `main` → *Require status checks to pass before merging* → add the `eval-gate` check. The check name appears after the workflow has run once. Without this the gate only reports and doesn't block anything.

**Updating the baseline.** Update it only when a change is meant to move the scores (a better prompt, a new tool, an edited fixture test set), not to make a red gate go green:

```bash
CORPBRAIN_CORPUS=fixture python eval.py gate --update-baseline --runs 3   # ~$0.27
```

This reruns both arms 3 times and rewrites `tests/fixtures/eval_baseline.json` with the runs, mean, std, tolerance and test-set hash. Commit it in the same PR as the change that caused it, so the diff shows the score change for review. PowerShell: `$env:CORPBRAIN_CORPUS="fixture"; python eval.py gate --update-baseline --runs 3`.
