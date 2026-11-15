"""
Smoke tests: the shared Retriever returns the same results as the old,
duplicated code paths in rag.py and eval.py.

tests/fixtures/retrieval_golden.json was captured from the pre-refactor code
(hashes of page_content, plus source/page, in rank order). These tests hit the
local vector DBs and call the Gemini embedding API for the dense/hybrid query
embeddings, so they need a populated vector_db/ (and vector_db_baseline/ for the
baseline configs) and GOOGLE_API_KEY.
"""
import hashlib
import json
import os
from functools import lru_cache

import chromadb
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)  # DB paths are relative to the repo root

from retriever import BASELINE_DB_PATH, DB_PATH, Retriever  # noqa: E402

GOLDEN_PATH = os.path.join(ROOT, "tests", "fixtures", "retrieval_golden.json")
with open(GOLDEN_PATH, encoding="utf-8") as f:
    GOLDEN = json.load(f)


def _fp(doc):
    sha = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()[:16]
    return {"sha": sha, "source": doc.metadata.get("source"), "page": doc.metadata.get("page")}


def _db_ready(path: str) -> bool:
    try:
        return chromadb.PersistentClient(path=path).get_collection("langchain").count() > 0
    except Exception:
        return False


@lru_cache(maxsize=None)
def _retriever(strategy: str, rerank: bool, contextual: bool) -> Retriever:
    return Retriever(strategy=strategy, rerank=rerank, contextual=contextual)


def _parse(key: str):
    strategy, rerank, contextual = key.split("|")
    return strategy, rerank.split("=")[1] == "True", contextual.split("=")[1] == "True"


EVAL_CASES = [
    pytest.param(key, q, id=f"{key}::{q[:30]}")
    for key, per_q in GOLDEN["eval"].items()
    for q in per_q
]


@pytest.mark.parametrize("key,query", EVAL_CASES)
def test_matches_old_eval_path(key, query):
    strategy, rerank, contextual = _parse(key)
    if not _db_ready(DB_PATH if contextual else BASELINE_DB_PATH):
        pytest.skip("vector DB not populated")
    expected = GOLDEN["eval"][key][query]

    cands, top = _retriever(strategy, rerank, contextual).retrieve_with_stages(query)

    assert [_fp(d) for d in cands] == expected["candidates"]
    assert [_fp(d) for d in top] == expected["top"]


@pytest.mark.parametrize("query", GOLDEN["queries"])
def test_matches_old_chat_and_explore_path(query):
    if not _db_ready(DB_PATH):
        pytest.skip("vector DB not populated")
    expected = GOLDEN["chain"][query]
    r = _retriever("hybrid", True, True)

    cands, top = r.retrieve_with_stages(query)          # Explore
    assert [_fp(d) for d in cands] == expected["candidates"]
    assert [_fp(d) for d in top] == expected["top"]
    assert [_fp(d) for d in r.retrieve(query)] == expected["chat"]  # Chat


def test_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        Retriever(strategy="nope")
