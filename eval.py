"""
eval.py — CorpBrain RAG Evaluation Pipeline

CLI usage:
    python eval.py generate [--n 25]        # Generate synthetic (single-hop) test set
    python eval.py generate-multihop [--n 15] [--db contextual|baseline] [--seed S]
                                            # Multi-hop candidates → eval_data/multihop_candidates.json
    python eval.py merge-multihop           # Label test set + append reviewed candidates
    python eval.py baseline                 # Create baseline DB (no contextual chunking)
    python eval.py run [--strategies dense bm25 hybrid]
                       [--no-baseline]      # Skip baseline comparison
                       [--no-reranking]     # Skip no-rerank configs
                       [--limit N]          # Only the first N questions
                       [--out PATH]         # Results file (default eval_data/eval_results.json)

Requires: pip install ragas>=0.2.0
"""

import os
import json
import random
import argparse
import statistics
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import chromadb
import fitz
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest import DB_PATH, CHROMA_COLLECTION, IMAGE_OUT_PATH, DATA_PATH
from pricing import FLASH_MODEL_ID, cost as _cost, usage as _usage
from retriever import BASELINE_DB_PATH, Retriever, get_embedding_model

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────

EVAL_DATA_PATH   = "eval_data"
TEST_SET_PATH    = os.path.join(EVAL_DATA_PATH, "test_set.json")
RESULTS_PATH     = os.path.join(EVAL_DATA_PATH, "eval_results.json")

# Gemini 2.5 Flash for all eval LLM calls — measures retrieval quality, not LLM capability
EVAL_LLM_ID = FLASH_MODEL_ID


class UsageTracker(BaseCallbackHandler):
    """Sums token usage across every chat-model call it is attached to."""

    def __init__(self):
        self.input_tokens  = 0
        self.output_tokens = 0
        self.calls         = 0

    def on_llm_end(self, response, **kwargs):
        for gens in response.generations:
            for g in gens:
                msg = getattr(g, "message", None)
                if msg is not None:
                    i, o = _usage(msg)
                    self.input_tokens  += i
                    self.output_tokens += o
                    self.calls         += 1

METRIC_DISPLAY = {
    "faithfulness":                           "Faithfulness",
    "answer_relevancy":                       "Ans. Relevancy",
    "llm_context_precision_with_reference":   "Ctx. Precision",
    "llm_context_recall":                     "Ctx. Recall",
    "context_precision":                      "Ctx. Precision",
    "context_recall":                         "Ctx. Recall",
    "answer_correctness":                     "Ans. Correctness",
    "accuracy":                               "Accuracy",
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
    # Headline metric shown on its own; kept out of the overall score so scores
    # stay comparable with runs made before accuracy existed.
    "accuracy":                             0.0,
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


def load_results(path: str = RESULTS_PATH) -> Optional[dict]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
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
    Sample n text chunks from ChromaDB. For each, prompt Gemini Flash to produce
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
            # Strip markdown fences if the model wrapped the JSON
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


# ── Multi-hop test set ─────────────────────────────────────────────────────────

MULTIHOP_CANDIDATES_PATH = os.path.join(EVAL_DATA_PATH, "multihop_candidates.json")

# Question kinds rotated across the generated set. "group" is how many chunks
# feed the question; cross-document always pairs chunks from different PDFs.
HOP_KINDS = {
    "comparison": {
        "group": 2, "cross_doc": False,
        "instruction": "a COMPARISON question that contrasts a figure, policy or "
                       "attribute described in one excerpt with one described in the other",
    },
    "aggregation": {
        "group": 3, "cross_doc": False,
        "instruction": "an AGGREGATION question whose answer combines facts from every "
                       "excerpt (e.g. a total, a count, or a list drawn from all of them)",
    },
    "multi-part": {
        "group": 2, "cross_doc": False,
        "instruction": "a MULTI-PART question with one sub-question answered by each excerpt, "
                       "joined naturally into a single question",
    },
    "cross-document": {
        "group": 2, "cross_doc": True,
        "instruction": "a CROSS-DOCUMENT question that links the two documents, e.g. how a "
                       "measure in one relates to an initiative or result in the other",
    },
}


def _load_text_chunks(db_path: str) -> list:
    """Text chunks (>100 chars) with their stored embeddings: [{text, source, page, emb}]."""
    import numpy as np

    collection = chromadb.PersistentClient(path=db_path).get_collection(CHROMA_COLLECTION)
    result = collection.get(include=["documents", "metadatas", "embeddings"])
    chunks = []
    for txt, meta, emb in zip(result["documents"], result["metadatas"], result["embeddings"]):
        if meta and meta.get("type") == "text" and len((txt or "").strip()) > 100:
            chunks.append({
                "text":   txt,
                "source": meta.get("source", "Unknown"),
                "page":   meta.get("page", 0),
                "emb":    np.asarray(emb, dtype=float),
            })
    return chunks


def sample_chunk_groups(chunks: list, kinds: list, rng: random.Random, top_k: int = 5) -> list:
    """
    Pick one group of related chunks per entry in `kinds`.
    Returns [(kind, [chunk indices])]. Anchors are never reused.

    Related = highest cosine similarity of the stored embeddings, so no API calls.
    Partners must sit on a different page (same-doc kinds) or in a different
    document (cross-document); one is drawn at random from the top_k most similar
    so repeated runs don't always produce the same pairs. Cross-document falls
    back to same-document when the corpus has only one source.
    """
    import numpy as np

    embs  = np.stack([c["emb"] for c in chunks])
    embs  = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    sims  = embs @ embs.T
    multi_source = len({c["source"] for c in chunks}) > 1

    anchors = list(range(len(chunks)))
    rng.shuffle(anchors)
    used_anchors, groups = set(), []

    for kind in kinds:
        spec = HOP_KINDS[kind]
        cross = spec["cross_doc"] and multi_source
        while anchors:
            a = anchors.pop()
            if a in used_anchors:
                continue
            group = [a]
            for _ in range(spec["group"] - 1):
                taken = {(chunks[i]["source"], chunks[i]["page"]) for i in group}
                eligible = [
                    j for j in np.argsort(-sims[a])
                    if j not in group
                    and (chunks[j]["source"], chunks[j]["page"]) not in taken
                    and ((chunks[j]["source"] != chunks[a]["source"]) if cross
                         else (chunks[j]["source"] == chunks[a]["source"]))
                ][:top_k]
                if not eligible:
                    break
                group.append(int(rng.choice(eligible)))
            if len(group) == spec["group"]:
                used_anchors.add(a)
                groups.append((kind, group))
                break
    return groups


def _multihop_prompt(kind: str, group: list) -> str:
    excerpts = "\n\n".join(
        f"[Excerpt {i + 1}] Document: {c['source']}, Page {c['page']}\n{c['text']}"
        for i, c in enumerate(group)
    )
    return (
        "You are building a MULTI-HOP evaluation dataset for a Retrieval-Augmented "
        "Generation system. A multi-hop question can only be answered by combining "
        "information from several separate passages.\n\n"
        f"{excerpts}\n\n"
        f"Write {HOP_KINDS[kind]['instruction']}.\n"
        "Rules:\n"
        "- Answering must REQUIRE facts from every excerpt; no single excerpt is enough.\n"
        "- The question must stand alone: never mention 'excerpt', 'passage', 'chunk' "
        "or page numbers.\n"
        "- Ask about specific facts, not opinions. Never invent a link between the excerpts "
        "that they don't state; the answer must not speculate ('may', 'could', 'suggests').\n"
        "- The ground truth answer (2-4 sentences) must be fully supported by the excerpts.\n"
        "If the excerpts don't support a natural question like this, skip instead of forcing one.\n"
        "Return ONLY valid JSON, no markdown fences, no extra text:\n"
        '{"question": "...", "ground_truth": "..."}  or  {"skip": true}'
    )


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def generate_multihop_test_set(
    n: int = 15, db_path: str = DB_PATH, seed: Optional[int] = None, progress_callback=None,
) -> tuple:
    """
    Generate n multi-hop questions from groups of related chunks (see
    sample_chunk_groups), rotating through HOP_KINDS. Groups the model skips are
    replaced, up to 3n attempts in total.

    Returns (questions, cost) where each question is
        {"question", "ground_truth", "type": "multi-hop", "hop_kind",
         "sources": [{"source", "page"}], "chunk_texts": [...]}
    and cost = {"usd", "input_tokens", "output_tokens", "calls", "skipped"}.
    """
    chunks = _load_text_chunks(db_path)
    if len(chunks) < 3:
        raise ValueError(f"Not enough text chunks in {db_path}. Ingest documents first.")

    rng    = random.Random(seed)
    kinds  = [list(HOP_KINDS)[i % len(HOP_KINDS)] for i in range(3 * n)]
    groups = sample_chunk_groups(chunks, kinds, rng)

    # Thinking off: short structured output, and thinking tokens bill as output.
    llm   = ChatGoogleGenerativeAI(model=EVAL_LLM_ID, temperature=0.3, thinking_budget=0)
    wanted = {k: kinds[:n].count(k) for k in HOP_KINDS}  # target mix for the first n
    questions, in_tok, out_tok, calls, skipped = [], 0, 0, 0, 0

    for kind, idxs in groups:
        if len(questions) >= n:
            break
        if wanted[kind] <= 0 and any(v > 0 for v in wanted.values()):
            continue  # keep the kind mix balanced while other kinds still need questions
        group = [chunks[i] for i in idxs]
        if progress_callback:
            progress_callback(
                f"Multi-hop Q{len(questions) + 1}/{n} ({kind}) — "
                + ", ".join(f"{c['source']} p.{c['page']}" for c in group),
                len(questions) / n,
            )
        try:
            msg = llm.invoke([HumanMessage(content=_multihop_prompt(kind, group))])
            i_t, o_t = _usage(msg)
            in_tok, out_tok, calls = in_tok + i_t, out_tok + o_t, calls + 1
            qa = _parse_json(msg.content if isinstance(msg.content, str) else str(msg.content))
        except Exception as e:
            print(f"  Skipping group ({kind}): {e}")
            skipped += 1
            continue
        if qa.get("skip") or not qa.get("question") or not qa.get("ground_truth"):
            skipped += 1
            continue
        wanted[kind] -= 1
        questions.append({
            "question":     qa["question"],
            "ground_truth": qa["ground_truth"],
            "type":         "multi-hop",
            "hop_kind":     kind,
            "sources":      [{"source": c["source"], "page": c["page"]} for c in group],
            "chunk_texts":  [c["text"] for c in group],
        })

    cost = {
        "usd": round(_cost(in_tok, out_tok), 6), "input_tokens": in_tok,
        "output_tokens": out_tok, "calls": calls, "skipped": skipped,
    }
    return questions, cost


def label_single_hop(item: dict) -> dict:
    """Return a copy of a legacy single-chunk question with type + sources filled in."""
    out = dict(item)
    out.setdefault("type", "single-hop")
    if "sources" not in out and "source" in out:
        out["sources"] = [{"source": out["source"], "page": out.get("page", 0)}]
    return out


def merge_multihop(test_set: list, candidates: list) -> list:
    """
    Label existing questions single-hop and append the multi-hop candidates.
    Questions already present (same text) are not added twice, so re-running is safe.
    """
    merged = [label_single_hop(q) if q.get("type") != "multi-hop" else dict(q) for q in test_set]
    seen = {q["question"] for q in merged}
    for c in candidates:
        if c["question"] not in seen:
            merged.append({**c, "type": "multi-hop"})
            seen.add(c["question"])
    return merged


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

# ── RAGAS scoring ──────────────────────────────────────────────────────────────

def _run_ragas(samples: list, ragas_llm, ragas_emb) -> tuple:
    """
    Run all 5 RAGAS metrics on collected samples.
    Returns (aggregate_scores: dict, per_question_scores: list[dict], usage: UsageTracker).

    AnswerRelevancy(strictness=1): the default strictness=3 asks the judge for
    3 generations in one call (n=3). Gemini returns only 1, which produced NaN /
    degraded scores. One generation per question is all we need.

    RunConfig explanation
    ---------------------
    RAGAS fires all (metric × question) scoring jobs concurrently by default.
    Unbounded concurrency trips rate limits and produces TimeoutError → NaN.

    max_workers=4  — keeps the request rate comfortably under Gemini's per-minute cap
    timeout=120    — 2-minute per-job deadline (60s default is too tight under load)
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
    usage   = UsageTracker()

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(), AnswerRelevancy(strictness=1),
            CtxPrec(), CtxRecall(), AnswerCorrectness(),
        ],
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=run_cfg,
        callbacks=[usage],
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
    return aggregate, per_q, usage


# ── Binary answer accuracy ─────────────────────────────────────────────────────

def _judge_accuracy(judge_llm, question: str, ground_truth: str, answer: str) -> tuple:
    """
    One LLM-judge call: is the answer correct w.r.t. the ground truth?
    Returns (correct, input_tokens, output_tokens) where correct is 1, 0,
    or None if the judge's output couldn't be parsed.
    """
    prompt = (
        "You are grading an answer produced by a question-answering system.\n\n"
        f"Question: {question}\n\n"
        f"Reference answer: {ground_truth}\n\n"
        f"System answer: {answer}\n\n"
        "Is the system answer correct? It is correct if it states the key facts of the "
        "reference answer without contradicting it; extra detail and different wording "
        "are fine. A refusal or 'not in the context' answer is incorrect.\n"
        'Return ONLY valid JSON, no markdown fences: {"correct": true} or {"correct": false}'
    )
    msg = judge_llm.invoke([HumanMessage(content=prompt)])
    in_t, out_t = _usage(msg)
    raw = (msg.content if isinstance(msg.content, str) else str(msg.content)).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        correct = int(bool(json.loads(raw.strip())["correct"]))
    except Exception:
        low = raw.lower().replace(" ", "")
        correct = 1 if '"correct":true' in low else 0 if '"correct":false' in low else None
    return correct, in_t, out_t


# ── Per-type aggregation ───────────────────────────────────────────────────────

QUESTION_TYPES = ("single-hop", "multi-hop")


def _aggregate_by_type(per_q: list, metric_keys: list) -> dict:
    """
    {type: {metric: mean, ..., "n": count, "overall_score": weighted}} over per-question rows.
    Rows without a "type" count as single-hop. NaN / None scores are excluded
    from each metric's mean, matching how the overall aggregate is computed.
    """
    by_type = {}
    for t in QUESTION_TYPES:
        rows = [r for r in per_q if r.get("type", "single-hop") == t]
        if not rows:
            continue
        agg = {}
        for k in metric_keys:
            vals = [r[k] for r in rows
                    if isinstance(r.get(k), (int, float)) and r[k] == r[k]]  # r[k]==r[k] drops NaN
            agg[k] = round(sum(vals) / len(vals), 4) if vals else float("nan")
        by_type[t] = {**agg, "n": len(rows),
                      "overall_score": _overall_score({k: v for k, v in agg.items() if v == v})}
    return by_type


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
    llm:        ChatGoogleGenerativeAI,
    judge_llm:  ChatGoogleGenerativeAI,
    ragas_llm,
    ragas_emb,
    progress_callback=None,
) -> dict:
    """
    Evaluate one EvalConfig.
    Returns {"aggregate", "per_question", "cost", "latency"}.

    Latency is per question: retrieval + answer generation (what a user waits for).
    Cost covers every LLM call: answer generation, accuracy judge and RAGAS judge.
    RAGAS usage is only observable per config, so it's split evenly across questions.
    """
    retriever = Retriever(
        strategy=config.strategy, rerank=config.reranking, contextual=config.contextual,
    )
    samples   = []

    for i, item in enumerate(test_set):
        question     = item["question"]
        ground_truth = item["ground_truth"]

        if progress_callback:
            progress_callback(
                f"[{config.label}] Q{i + 1}/{len(test_set)}: {question[:70]}",
                (i + 0.5) / len(test_set),
            )

        # Retrieve top 5 (reranked, or first 5 candidates when reranking is off)
        t0       = time.perf_counter()
        docs     = retriever.retrieve(question)
        contexts = [doc.page_content for doc in docs]

        # Generate answer (text-only for reproducibility across configs)
        context_text = "\n\n---\n\n".join(contexts)
        answer_prompt = (
            "Answer the question using only the provided context. "
            "Be concise and accurate.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}"
        )
        answer_msg = llm.invoke([HumanMessage(content=answer_prompt)])
        latency    = time.perf_counter() - t0
        answer     = answer_msg.content
        ans_in, ans_out = _usage(answer_msg)

        correct, acc_in, acc_out = _judge_accuracy(judge_llm, question, ground_truth, answer)

        samples.append({
            "question":     question,
            "type":         item.get("type", "single-hop"),
            "ground_truth": ground_truth,
            "answer":       answer,
            "contexts":     contexts,
            "latency_s":    round(latency, 3),
            "accuracy":     correct,
            "tokens": {
                "answer_in": ans_in, "answer_out": ans_out,
                "accuracy_judge_in": acc_in, "accuracy_judge_out": acc_out,
            },
        })

    # RAGAS scoring (batch — one evaluate() call per config)
    if progress_callback:
        progress_callback(f"[{config.label}] Scoring with RAGAS...", 0.95)

    aggregate, per_q, ragas_usage = _run_ragas(samples, ragas_llm, ragas_emb)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    graded = [s["accuracy"] for s in samples if s["accuracy"] is not None]
    if len(graded) < len(samples):
        print(f"  WARNING: accuracy judge output unparseable for "
              f"{len(samples) - len(graded)} question(s); excluded from the mean.")
    aggregate["accuracy"] = round(sum(graded) / len(graded), 4) if graded else float("nan")

    # ── Cost & latency ────────────────────────────────────────────────────────
    n = len(samples)
    ragas_in_q  = ragas_usage.input_tokens  / n
    ragas_out_q = ragas_usage.output_tokens / n
    for s, row in zip(samples, per_q):
        t = s["tokens"]
        t["ragas_judge_in"]  = round(ragas_in_q)
        t["ragas_judge_out"] = round(ragas_out_q)
        in_tok  = t["answer_in"]  + t["accuracy_judge_in"]  + ragas_in_q
        out_tok = t["answer_out"] + t["accuracy_judge_out"] + ragas_out_q
        row["type"]      = s["type"]
        row["accuracy"]  = s["accuracy"]
        row["latency_s"] = s["latency_s"]
        row["cost_usd"]  = round(_cost(in_tok, out_tok), 6)
        row["tokens"]    = t
        row["answer"]    = s["answer"]

    answer_cost = sum(_cost(s["tokens"]["answer_in"], s["tokens"]["answer_out"]) for s in samples)
    acc_cost    = sum(_cost(s["tokens"]["accuracy_judge_in"], s["tokens"]["accuracy_judge_out"])
                      for s in samples)
    ragas_cost  = _cost(ragas_usage.input_tokens, ragas_usage.output_tokens)
    total_cost  = answer_cost + acc_cost + ragas_cost
    latencies   = [s["latency_s"] for s in samples]

    cost = {
        "total_usd":               round(total_cost, 6),
        "per_question_usd":        round(total_cost / n, 6),
        "answer_per_question_usd": round(answer_cost / n, 6),
        "breakdown_usd": {
            "answer":         round(answer_cost, 6),
            "accuracy_judge": round(acc_cost, 6),
            "ragas_judge":    round(ragas_cost, 6),
        },
        "tokens": {
            "input":  sum(s["tokens"]["answer_in"] + s["tokens"]["accuracy_judge_in"] for s in samples)
                      + ragas_usage.input_tokens,
            "output": sum(s["tokens"]["answer_out"] + s["tokens"]["accuracy_judge_out"] for s in samples)
                      + ragas_usage.output_tokens,
        },
    }
    latency_summary = {
        "avg_s": round(statistics.mean(latencies), 3),
        "p50_s": round(statistics.median(latencies), 3),
        "max_s": round(max(latencies), 3),
    }
    return {
        "aggregate":         aggregate,
        "aggregate_by_type": _aggregate_by_type(per_q, list(aggregate)),
        "per_question":      per_q,
        "cost":              cost,
        "latency":           latency_summary,
    }


# ── Full evaluation ────────────────────────────────────────────────────────────

def run_full_evaluation(
    configs:           list,
    test_set:          list,
    progress_callback=None,
    results_path:      str = RESULTS_PATH,
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
    existing = load_results(results_path)
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
    # Gemini Flash for everything. Judges run with thinking disabled
    # (thinking_budget=0): grading doesn't need it, and thinking tokens bill as
    # output, which would dominate the cost. Answer generation keeps the defaults
    # so the system under test is unchanged.
    # Retrieval components (embeddings, reranker) are shared via retriever.py.
    llm             = ChatGoogleGenerativeAI(model=EVAL_LLM_ID, temperature=0)
    judge_llm       = ChatGoogleGenerativeAI(model=EVAL_LLM_ID, temperature=0, thinking_budget=0)
    ragas_llm       = LangchainLLMWrapper(judge_llm)
    ragas_emb       = LangchainEmbeddingsWrapper(get_embedding_model())

    total = len(configs)
    for i, config in enumerate(pending):
        overall_idx = total - len(pending) + i  # position in the full config list
        if progress_callback:
            progress_callback(
                f"Starting config {overall_idx + 1}/{total}: {config.label}",
                overall_idx / total,
            )

        config_result = run_config(
            test_set, config, llm, judge_llm, ragas_llm, ragas_emb,
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
        with open(results_path, "w", encoding="utf-8") as f:
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

    mh_p = sub.add_parser("generate-multihop",
                          help=f"Generate multi-hop candidates into {MULTIHOP_CANDIDATES_PATH} for review")
    mh_p.add_argument("--n", type=int, default=15, help="Number of questions (default: 15)")
    mh_p.add_argument("--db", choices=["contextual", "baseline"], default="contextual",
                      help="Vector DB to sample chunks from (default: contextual)")
    mh_p.add_argument("--seed", type=int, default=None, help="Random seed for chunk sampling")

    sub.add_parser("merge-multihop",
                   help="Label test set single-hop and append the reviewed multi-hop candidates")

    sub.add_parser("baseline", help="Create baseline DB (no contextual chunking)")

    run_p = sub.add_parser("run", help="Run evaluation across configs")
    run_p.add_argument("--strategies", nargs="+", default=["dense", "bm25", "hybrid"],
                       choices=["dense", "bm25", "hybrid"])
    run_p.add_argument("--no-baseline",  action="store_true", help="Skip baseline configs")
    run_p.add_argument("--no-reranking", action="store_true", help="Skip no-reranking configs")
    run_p.add_argument("--limit", type=int, default=None, help="Only evaluate the first N questions")
    run_p.add_argument("--out", default=RESULTS_PATH, help=f"Results file (default: {RESULTS_PATH})")

    args = parser.parse_args()

    def cli_cb(msg: str, frac: float):
        print(f"  [{frac * 100:5.1f}%] {msg}")

    if args.command == "generate":
        print(f"Generating {args.n} test questions...")
        ts = generate_test_set(n=args.n, progress_callback=cli_cb)
        save_test_set(ts)
        print(f"Saved {len(ts)} questions to {TEST_SET_PATH}")

    elif args.command == "generate-multihop":
        db_path = DB_PATH if args.db == "contextual" else BASELINE_DB_PATH
        print(f"Generating {args.n} multi-hop questions from {db_path}...")
        qs, cost = generate_multihop_test_set(
            n=args.n, db_path=db_path, seed=args.seed, progress_callback=cli_cb,
        )
        os.makedirs(EVAL_DATA_PATH, exist_ok=True)
        with open(MULTIHOP_CANDIDATES_PATH, "w", encoding="utf-8") as f:
            json.dump(qs, f, ensure_ascii=False, indent=2)
        kinds = {k: sum(q["hop_kind"] == k for q in qs) for k in HOP_KINDS}
        print(f"Saved {len(qs)} candidates to {MULTIHOP_CANDIDATES_PATH} — {kinds}")
        print(f"Generation cost: ${cost['usd']:.5f} "
              f"({cost['calls']} calls, {cost['input_tokens']} in / {cost['output_tokens']} out tokens, "
              f"{cost['skipped']} groups skipped)")
        print("Review/edit the candidates, then run 'python eval.py merge-multihop'.")

    elif args.command == "merge-multihop":
        ts = load_test_set() or []
        if not os.path.exists(MULTIHOP_CANDIDATES_PATH):
            print(f"No candidates at {MULTIHOP_CANDIDATES_PATH}. Run 'generate-multihop' first.")
            raise SystemExit(1)
        with open(MULTIHOP_CANDIDATES_PATH, "r", encoding="utf-8") as f:
            candidates = json.load(f)
        if ts:
            backup = TEST_SET_PATH.replace(".json", f".backup-{datetime.now():%Y%m%d-%H%M%S}.json")
            with open(backup, "w", encoding="utf-8") as f:
                json.dump(ts, f, ensure_ascii=False, indent=2)
            print(f"Backed up current test set to {backup}")
        merged = merge_multihop(ts, candidates)
        save_test_set(merged)
        counts = {t: sum(q["type"] == t for q in merged) for t in QUESTION_TYPES}
        print(f"Saved {len(merged)} questions to {TEST_SET_PATH} — {counts}")

    elif args.command == "baseline":
        print("Creating baseline collection (no contextual chunking)...")
        create_baseline_collection(progress_callback=cli_cb)
        print("Done.")

    elif args.command == "run":
        ts = load_test_set()
        if not ts:
            print(f"No test set found. Run 'python eval.py generate' first.")
            raise SystemExit(1)
        if args.limit:
            ts = ts[: args.limit]

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
        results = run_full_evaluation(configs, ts, progress_callback=cli_cb, results_path=args.out)

        print(f"\n{'Config':<25} | " + " | ".join(
            METRIC_DISPLAY.get(k, k) for k in next(iter(results["configs"]))["aggregate"]
        ) + " | $/question | Avg latency")
        print("-" * 120)
        for r in results["configs"]:
            scores_str = " | ".join(f"{v:.3f}" for v in r["aggregate"].values())
            cpq = r.get("cost", {}).get("per_question_usd")
            lat = r.get("latency", {}).get("avg_s")
            extra = f" | ${cpq:.5f} | {lat:.2f}s" if cpq is not None and lat is not None else " | — | —"
            print(f"{r['label']:<25} | {scores_str}{extra}")

        by_type_cfgs = [r for r in results["configs"] if r.get("aggregate_by_type")]
        if by_type_cfgs:
            print(f"\n{'Config':<25} | " + " | ".join(
                f"{t} acc (n) | {t} overall" for t in QUESTION_TYPES))
            print("-" * 100)
            for r in by_type_cfgs:
                cells = []
                for t in QUESTION_TYPES:
                    bt = r["aggregate_by_type"].get(t)
                    cells.append(f"{bt.get('accuracy', float('nan')):.3f} ({bt['n']}) | "
                                 f"{bt['overall_score']:.3f}" if bt else "— | —")
                print(f"{r['label']:<25} | " + " | ".join(cells))

        total_usd = sum(r.get("cost", {}).get("total_usd", 0) for r in results["configs"])
        print(f"\nTotal LLM cost recorded in this file: ${total_usd:.5f}")
        print(f"Full results → {args.out}")

    else:
        parser.print_help()
