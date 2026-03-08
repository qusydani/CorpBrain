"""
eval.py — CorpBrain RAG Evaluation Pipeline

CLI usage:
    python eval.py generate [--n 25]        # Generate synthetic test set
    python eval.py baseline                 # Create baseline DB (no contextual chunking)
    python eval.py run [--strategies dense bm25 hybrid]
                       [--no-baseline]      # Skip baseline comparison
                       [--no-reranking]     # Skip no-rerank configs

Requires: pip install ragas>=0.2.0
"""

import os
import json
import random
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import chromadb
import fitz
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

from ingest import DB_PATH, CHROMA_COLLECTION, IMAGE_OUT_PATH, DATA_PATH

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────

BASELINE_DB_PATH = "vector_db_baseline"
EVAL_DATA_PATH   = "eval_data"
TEST_SET_PATH    = os.path.join(EVAL_DATA_PATH, "test_set.json")
RESULTS_PATH     = os.path.join(EVAL_DATA_PATH, "eval_results.json")

# Gemini 2.5 Flash for all eval LLM calls — measures retrieval quality, not LLM capability
EVAL_LLM_ID = "gemini-2.5-flash"

METRIC_DISPLAY = {
    "faithfulness":                           "Faithfulness",
    "answer_relevancy":                       "Ans. Relevancy",
    "llm_context_precision_with_reference":   "Ctx. Precision",
    "llm_context_recall":                     "Ctx. Recall",
    "context_precision":                      "Ctx. Precision",
    "context_recall":                         "Ctx. Recall",
    "answer_correctness":                     "Ans. Correctness",
}


# ── Config dataclass ───────────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    strategy:   str   # "dense" | "bm25" | "hybrid"
    reranking:  bool
    contextual: bool  # True = vector_db (contextual), False = vector_db_baseline

    @property
    def label(self) -> str:
        s = {"dense": "Dense", "bm25": "BM25", "hybrid": "Hybrid"}[self.strategy]
        r = "+Rerank" if self.reranking else ""
        c = "+Ctx"    if self.contextual else "+Base"
        return f"{s}{r}{c}"

    @property
    def db_path(self) -> str:
        return DB_PATH if self.contextual else BASELINE_DB_PATH


# ── Overall score ─────────────────────────────────────────────────────────────

# Weighted to reflect that this is a retrieval-system evaluation.
# Retrieval metrics (Precision + Recall) carry 50% of the weight combined;
# answer-quality metrics share the remaining 50%.
_METRIC_WEIGHTS = {
    # RAGAS 0.2.x names
    "llm_context_precision_with_reference": 0.25,
    "llm_context_recall":                   0.25,
    "faithfulness":                         0.20,
    "answer_relevancy":                     0.15,
    "answer_correctness":                   0.15,
    # RAGAS <0.2 fallback names
    "context_precision":                    0.25,
    "context_recall":                       0.25,
}


def _overall_score(aggregate: dict) -> float:
    """
    Compute a weighted overall score from a RAGAS aggregate dict.
    Keys not listed in _METRIC_WEIGHTS fall back to equal weight among
    the unrecognised metrics so the function never silently drops data.
    Returns a value in [0, 1] rounded to 4 decimal places.
    """
    known, unknown = {}, {}
    for k, v in aggregate.items():
        if k in _METRIC_WEIGHTS:
            known[k] = v
        else:
            unknown[k] = v

    total_weight = sum(_METRIC_WEIGHTS[k] for k in known)
    score = sum(_METRIC_WEIGHTS[k] * v for k, v in known.items())

    if unknown:
        # Distribute remaining weight equally across unrecognised metrics
        leftover = max(0.0, 1.0 - total_weight) / len(unknown)
        score += sum(v * leftover for v in unknown.values())
        total_weight += leftover * len(unknown)

    if total_weight == 0:
        return 0.0
    return round(score / total_weight, 4)


def backfill_overall_scores() -> bool:
    """
    Add/recalculate overall_score for every config in an existing
    eval_results.json without re-running any evaluation.
    Returns True if the file was updated, False if not found.
    """
    data = load_results()
    if not data:
        return False
    for cfg in data.get("configs", []):
        cfg["overall_score"] = _overall_score(cfg["aggregate"])
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return True


# ── Persistence helpers ────────────────────────────────────────────────────────

def load_test_set() -> Optional[list]:
    if os.path.exists(TEST_SET_PATH):
        with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_test_set(test_set: list):
    os.makedirs(EVAL_DATA_PATH, exist_ok=True)
    with open(TEST_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)


def load_results() -> Optional[dict]:
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def baseline_exists() -> bool:
    try:
        client = chromadb.PersistentClient(path=BASELINE_DB_PATH)
        col = client.get_collection(CHROMA_COLLECTION)
        return col.count() > 0
    except Exception:
        return False


# ── Test set generation ────────────────────────────────────────────────────────

def generate_test_set(n: int = 25, progress_callback=None) -> list:
    """
    Sample n text chunks from ChromaDB. For each, prompt Claude Haiku to produce
    a question + ground-truth answer pair.

    Returns a list of dicts:
        {"question", "ground_truth", "source", "page", "chunk_text"}
    """
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(CHROMA_COLLECTION)
    result = collection.get(include=["documents", "metadatas"])

    text_items = [
        (txt, meta)
        for txt, meta in zip(result["documents"], result["metadatas"])
        if meta and meta.get("type") == "text" and len((txt or "").strip()) > 100
    ]

    if not text_items:
        raise ValueError("No text chunks found in ChromaDB. Ingest a document first.")

    sampled = random.sample(text_items, min(n, len(text_items)))
    llm = ChatGoogleGenerativeAI(model=EVAL_LLM_ID, temperature=0.3)
    test_set = []

    for i, (txt, meta) in enumerate(sampled):
        if progress_callback:
            progress_callback(
                f"Generating Q{i + 1}/{len(sampled)} — "
                f"{meta.get('source', '')} p.{meta.get('page', '?')}",
                (i + 1) / len(sampled),
            )

        prompt = (
            "You are building an evaluation dataset for a Retrieval-Augmented Generation system.\n\n"
            f"Document: {meta.get('source', 'Unknown')}\n"
            f"Page: {meta.get('page', '?')}\n\n"
            "Chunk:\n"
            f"{txt}\n\n"
            "Generate exactly ONE specific, factual question whose answer is contained "
            "entirely within this chunk, and the expected correct answer (2-3 sentences).\n"
            "Return ONLY valid JSON, no markdown fences, no extra text:\n"
            '{"question": "...", "ground_truth": "..."}'
        )

        try:
            raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
            # Strip markdown fences if Haiku wrapped the JSON
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            qa = json.loads(raw.strip())
            test_set.append({
                "question":     qa["question"],
                "ground_truth": qa["ground_truth"],
                "source":       meta.get("source", "Unknown"),
                "page":         meta.get("page", 0),
                "chunk_text":   txt,
            })
        except Exception as e:
            print(f"  Skipping chunk {i + 1}: {e}")
            continue

    return test_set


# ── Baseline collection ────────────────────────────────────────────────────────

def create_baseline_collection(progress_callback=None):
    """
    Re-ingest all source PDFs into vector_db_baseline/ WITHOUT Phase 3
    (contextual chunking). Uses cached vision summaries so no Haiku calls needed.
    Only embedding (Gemini) is re-run.
    """
    # Collect source filenames + file hashes from the main DB
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(CHROMA_COLLECTION)
    result = collection.get(include=["metadatas"])

    source_hashes: dict[str, str] = {}
    for meta in result["metadatas"]:
        if meta and "source" in meta and "file_hash" in meta:
            source_hashes[meta["source"]] = meta["file_hash"]

    # Discover what's already in the baseline to allow incremental runs
    already_done: set[str] = set()
    if baseline_exists():
        b_client = chromadb.PersistentClient(path=BASELINE_DB_PATH)
        b_col    = b_client.get_collection(CHROMA_COLLECTION)
        b_result = b_col.get(include=["metadatas"])
        already_done = {m["source"] for m in b_result["metadatas"] if m and "source" in m}

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_db       = Chroma(persist_directory=BASELINE_DB_PATH, embedding_function=embedding_model)
    text_splitter   = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    to_process = [s for s in source_hashes if s not in already_done]
    total = len(to_process)

    for i, source in enumerate(to_process):
        pdf_path = os.path.join(DATA_PATH, source)
        if not os.path.exists(pdf_path):
            print(f"  PDF not found at {pdf_path} — skipping.")
            continue

        if progress_callback:
            progress_callback(
                f"Baseline: ingesting {source} ({i + 1}/{total})",
                (i + 1) / total,
            )

        file_hash = source_hashes[source]
        doc = fitz.open(pdf_path)
        text_documents  = []
        image_documents = []

        for page_num in range(doc.page_count):
            page     = doc.load_page(page_num)
            raw_text = page.get_text()

            text_documents.append(Document(
                page_content=raw_text,
                metadata={
                    "source":    source,
                    "page":      page_num + 1,
                    "type":      "text",
                    "file_hash": file_hash,
                },
            ))

            # Reuse cached vision summaries — no Haiku calls needed
            image_filename = f"{source}_page_{page_num + 1}.png"
            image_path     = os.path.join(IMAGE_OUT_PATH, image_filename)
            summary_path   = image_path.replace(".png", "_summary.txt")
            if os.path.exists(summary_path):
                with open(summary_path, "r", encoding="utf-8") as f:
                    page_summary = f.read()
                image_documents.append(Document(
                    page_content=page_summary,
                    metadata={
                        "source":     source,
                        "page":       page_num + 1,
                        "image_path": image_path,
                        "type":       "image_summary",
                        "file_hash":  file_hash,
                    },
                ))

        doc.close()

        # Phase 2 only — chunk text, NO context generation
        text_chunks = text_splitter.split_documents(text_documents)
        all_chunks  = text_chunks + image_documents

        batch_size = 80
        for j in range(0, len(all_chunks), batch_size):
            vector_db.add_documents(all_chunks[j : j + batch_size])

    print("Baseline collection ready.")


# ── Retriever builder ──────────────────────────────────────────────────────────

def _build_retriever(strategy: str, db_path: str, embedding_model):
    """Build a LangChain retriever (no reranking) for the given strategy and DB."""
    vector_db = Chroma(persist_directory=db_path, embedding_function=embedding_model)

    if strategy == "dense":
        return vector_db.as_retriever(search_kwargs={"k": 10})

    db_data   = vector_db.get()
    bm25_docs = [
        Document(page_content=txt, metadata=meta)
        for txt, meta in zip(db_data["documents"], db_data["metadatas"])
    ]

    if strategy == "bm25":
        ret   = BM25Retriever.from_documents(bm25_docs)
        ret.k = 10
        return ret

    # hybrid
    vector_ret    = vector_db.as_retriever(search_kwargs={"k": 10})
    bm25_ret      = BM25Retriever.from_documents(bm25_docs)
    bm25_ret.k    = 10
    return EnsembleRetriever(
        retrievers=[vector_ret, bm25_ret],
        weights=[0.5, 0.5],
    )


# ── RAGAS scoring ──────────────────────────────────────────────────────────────

def _run_ragas(samples: list, ragas_llm, ragas_emb) -> tuple:
    """
    Run all 5 RAGAS metrics on collected samples.
    Returns (aggregate_scores: dict, per_question_scores: list[dict]).

    RunConfig explanation
    ---------------------
    RAGAS fires all (metric × question) scoring jobs concurrently by default
    (125 jobs for 25 questions × 5 metrics). This saturates Anthropic's rate
    limits and causes TimeoutError on many jobs, which silently produce NaN
    and skew aggregate means.

    max_workers=4  — Haiku's Tier 1 rate limit is 50 RPM. At ~7s/job, 4 workers
                     ≈ 34 RPM — safely under the cap. 8 workers ≈ 68 RPM which
                     exceeds the limit and causes the TimeoutErrors.
    timeout=120    — 2-minute per-job deadline (Gemini Flash is fast; 60s default
                     is too tight when the API is under load)
    max_retries=3  — retry timed-out or rate-limited jobs before giving up
    """
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics import Faithfulness, AnswerRelevancy, AnswerCorrectness
    from ragas.run_config import RunConfig

    # Handle metric renames between RAGAS versions
    try:
        from ragas.metrics import LLMContextPrecisionWithReference as CtxPrec
        from ragas.metrics import LLMContextRecall as CtxRecall
    except ImportError:
        from ragas.metrics import ContextPrecision as CtxPrec   # type: ignore
        from ragas.metrics import ContextRecall as CtxRecall    # type: ignore

    dataset = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["contexts"],
            reference=s["ground_truth"],
        )
        for s in samples
    ])

    run_cfg = RunConfig(max_workers=4, timeout=120, max_retries=3)

    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), CtxPrec(), CtxRecall(), AnswerCorrectness()],
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=run_cfg,
    )

    df         = result.to_pandas()
    score_cols = [c for c in df.columns if c not in
                  ("user_input", "response", "retrieved_contexts", "reference")]

    # Warn if any NaNs remain after retries — means some calls still failed
    nan_counts = df[score_cols].isna().sum()
    if nan_counts.any():
        print(f"  WARNING: NaN scores remain after retries — {nan_counts[nan_counts > 0].to_dict()}")
        print("  These questions are excluded from the mean. Consider re-running.")

    aggregate  = {col: round(float(df[col].mean()), 4) for col in score_cols}
    per_q      = (
        df[["user_input"] + score_cols]
        .rename(columns={"user_input": "question"})
        .to_dict(orient="records")
    )
    return aggregate, per_q


# ── Single config run ──────────────────────────────────────────────────────────

def _make_sub_progress(config_idx: int, total_configs: int, parent_cb):
    """Factory to convert per-config (msg, frac) into overall (msg, frac)."""
    def sub_cb(msg: str, frac: float):
        if parent_cb:
            overall = (config_idx + frac) / total_configs
            parent_cb(msg, overall)
    return sub_cb


def run_config(
    test_set:   list,
    config:     EvalConfig,
    reranker:   CrossEncoder,
    llm:        ChatGoogleGenerativeAI,
    ragas_llm,
    ragas_emb,
    embedding_model,
    progress_callback=None,
) -> dict:
    """
    Evaluate one EvalConfig.
    Returns {"aggregate": {metric: score}, "per_question": [{"question", metric...}]}
    """
    retriever = _build_retriever(config.strategy, config.db_path, embedding_model)
    samples   = []

    for i, item in enumerate(test_set):
        question     = item["question"]
        ground_truth = item["ground_truth"]

        if progress_callback:
            progress_callback(
                f"[{config.label}] Q{i + 1}/{len(test_set)}: {question[:70]}",
                (i + 0.5) / len(test_set),
            )

        # Retrieve candidates
        candidates = retriever.invoke(question)

        # Optionally rerank
        if config.reranking and reranker:
            pairs  = [(question, doc.page_content) for doc in candidates]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            docs   = [doc for _, doc in ranked[:5]]
        else:
            docs = candidates[:5]

        contexts = [doc.page_content for doc in docs]

        # Generate answer (text-only for reproducibility across configs)
        context_text = "\n\n---\n\n".join(contexts)
        answer_prompt = (
            "Answer the question using only the provided context. "
            "Be concise and accurate.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}"
        )
        answer = llm.invoke([HumanMessage(content=answer_prompt)]).content

        samples.append({
            "question":     question,
            "ground_truth": ground_truth,
            "answer":       answer,
            "contexts":     contexts,
        })

    # RAGAS scoring (batch — one evaluate() call per config)
    if progress_callback:
        progress_callback(f"[{config.label}] Scoring with RAGAS...", 0.95)

    aggregate, per_q = _run_ragas(samples, ragas_llm, ragas_emb)
    return {"aggregate": aggregate, "per_question": per_q}


# ── Full evaluation ────────────────────────────────────────────────────────────

def run_full_evaluation(
    configs:           list,
    test_set:          list,
    progress_callback=None,
) -> dict:
    """
    Run evaluation across all configs. Saves results to eval_data/eval_results.json.

    Checkpointing: results are written to disk after every completed config so a
    crash mid-run doesn't lose all progress. Already-completed config labels are
    detected on resume and skipped — only the remaining configs are re-run and
    merged back into the existing file.

    Returns the full results dict.
    """
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    os.makedirs(EVAL_DATA_PATH, exist_ok=True)

    # ── Resume detection ──────────────────────────────────────────────────────
    # Load any previously checkpointed results so we can skip finished configs.
    existing = load_results()
    if existing:
        done_labels = {r["label"] for r in existing.get("configs", [])}
        pending     = [c for c in configs if c.label not in done_labels]
        results     = list(existing.get("configs", []))
        if done_labels & {c.label for c in configs}:
            skipped = sorted(done_labels & {c.label for c in configs})
            print(f"  Resuming — skipping {len(skipped)} already-completed config(s): {skipped}")
    else:
        pending = configs
        results = []

    if not pending:
        print("  All requested configs already completed. Delete eval_results.json to re-run.")
        return existing

    # ── Load shared components once ───────────────────────────────────────────
    # Flash for cheap generation tasks (answer gen); Haiku for RAGAS scoring —
    # Haiku follows structured generation instructions reliably (fixes the
    # "1 generation instead of 3" Answer Relevancy issue that Gemini Flash had)
    # and is ~3.75× cheaper than Sonnet ($0.80/$4 vs $3/$15 per MTok in/out).
    # Full 25q × 12 config run costs ~$8 with Haiku vs ~$36 with Sonnet.
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    llm             = ChatGoogleGenerativeAI(model=EVAL_LLM_ID, temperature=0)
    reranker        = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ragas_llm       = LangchainLLMWrapper(ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0))
    ragas_emb       = LangchainEmbeddingsWrapper(embedding_model)

    total = len(configs)
    for i, config in enumerate(pending):
        overall_idx = total - len(pending) + i  # position in the full config list
        if progress_callback:
            progress_callback(
                f"Starting config {overall_idx + 1}/{total}: {config.label}",
                overall_idx / total,
            )

        config_result = run_config(
            test_set, config, reranker, llm, ragas_llm, ragas_emb, embedding_model,
            progress_callback=_make_sub_progress(overall_idx, total, progress_callback),
        )
        results.append({
            "config":        asdict(config),
            "label":         config.label,
            "overall_score": _overall_score(config_result["aggregate"]),
            **config_result,
        })

        # ── Checkpoint after every config ─────────────────────────────────────
        checkpoint = {
            "timestamp":   datetime.now().isoformat(),
            "n_questions": len(test_set),
            "configs":     results,
        }
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

        if progress_callback:
            progress_callback(
                f"Checkpoint saved ({len(results)}/{total} configs done).",
                (overall_idx + 1) / total,
            )

    if progress_callback:
        progress_callback("Done — all results saved.", 1.0)

    return checkpoint


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CorpBrain RAG Evaluation Pipeline")
    sub    = parser.add_subparsers(dest="command")

    gen_p = sub.add_parser("generate", help="Generate synthetic test set")
    gen_p.add_argument("--n", type=int, default=25, help="Number of questions (default: 25)")

    sub.add_parser("baseline", help="Create baseline DB (no contextual chunking)")

    run_p = sub.add_parser("run", help="Run evaluation across configs")
    run_p.add_argument("--strategies", nargs="+", default=["dense", "bm25", "hybrid"],
                       choices=["dense", "bm25", "hybrid"])
    run_p.add_argument("--no-baseline",  action="store_true", help="Skip baseline configs")
    run_p.add_argument("--no-reranking", action="store_true", help="Skip no-reranking configs")

    args = parser.parse_args()

    def cli_cb(msg: str, frac: float):
        print(f"  [{frac * 100:5.1f}%] {msg}")

    if args.command == "generate":
        print(f"Generating {args.n} test questions...")
        ts = generate_test_set(n=args.n, progress_callback=cli_cb)
        save_test_set(ts)
        print(f"Saved {len(ts)} questions to {TEST_SET_PATH}")

    elif args.command == "baseline":
        print("Creating baseline collection (no contextual chunking)...")
        create_baseline_collection(progress_callback=cli_cb)
        print("Done.")

    elif args.command == "run":
        ts = load_test_set()
        if not ts:
            print(f"No test set found. Run 'python eval.py generate' first.")
            raise SystemExit(1)

        reranking_opts  = [True, False] if not args.no_reranking  else [True]
        contextual_opts = [True]
        if not args.no_baseline:
            if baseline_exists():
                contextual_opts.append(False)
            else:
                print("Baseline DB not found — run 'python eval.py baseline' first. Skipping baseline configs.")

        configs = [
            EvalConfig(strategy=s, reranking=r, contextual=c)
            for s in args.strategies
            for r in reranking_opts
            for c in contextual_opts
        ]

        print(f"Running {len(configs)} configs × {len(ts)} questions...\n")
        results = run_full_evaluation(configs, ts, progress_callback=cli_cb)

        print(f"\n{'Config':<25} | " + " | ".join(
            METRIC_DISPLAY.get(k, k) for k in next(iter(results["configs"]))["aggregate"]
        ))
        print("-" * 90)
        for r in results["configs"]:
            scores_str = " | ".join(f"{v:.3f}" for v in r["aggregate"].values())
            print(f"{r['label']:<25} | {scores_str}")

        print(f"\nFull results → {RESULTS_PATH}")

    else:
        parser.print_help()
