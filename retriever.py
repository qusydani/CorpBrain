"""
retriever.py — the one retrieval pipeline shared by Chat, Explore and the eval.

    Retriever(strategy="hybrid", rerank=True, contextual=True)

Stage 1 fetches candidates with dense, BM25 or hybrid search (k=10 per
retriever). Stage 2 reranks them with a local cross-encoder and keeps the top 5.
With rerank=False it keeps the first 5 candidates instead.

Pass sources=[...] to restrict both stages to chunks from those files.
"""
import sys
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

from corpus import BASELINE_DB_PATH, DB_PATH

load_dotenv()  # GOOGLE_API_KEY for query embeddings

STRATEGIES = ("dense", "bm25", "hybrid")

EMBEDDING_MODEL_ID = "models/gemini-embedding-001"
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATE_K = 10   # Results per underlying retriever (hybrid returns up to 2×)
TOP_N = 5          # Docs kept after reranking


@lru_cache(maxsize=1)
def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """Shared embedding model — the same one used to build the vector DBs."""
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_ID)


@lru_cache(maxsize=1)
def get_reranker():
    """Shared cross-encoder, loaded once per process (local, no API cost)."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_MODEL_ID)


class Retriever:
    def __init__(
        self,
        strategy: str = "hybrid",
        rerank: bool = True,
        contextual: bool = True,
        k: int = CANDIDATE_K,
        top_n: int = TOP_N,
        sources: list[str] | None = None,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy {strategy!r}; expected one of {STRATEGIES}")
        self.strategy = strategy
        self.rerank = rerank
        self.contextual = contextual
        self.top_n = top_n
        self.sources = list(sources) if sources else None
        self.db_path = DB_PATH if contextual else BASELINE_DB_PATH
        self._base = self._build(strategy, k)

    def _build(self, strategy: str, k: int):
        vector_db = Chroma(persist_directory=self.db_path, embedding_function=get_embedding_model())
        search_kwargs = {"k": k}
        if self.sources:
            search_kwargs["filter"] = {"source": {"$in": self.sources}}
        dense = vector_db.as_retriever(search_kwargs=search_kwargs)
        if strategy == "dense":
            return dense

        db_data = vector_db.get(where=search_kwargs.get("filter"))
        docs = [
            Document(page_content=txt, metadata=meta)
            for txt, meta in zip(db_data["documents"], db_data["metadatas"])
        ]
        if not docs:
            # BM25 can't index an empty corpus — fall back to dense until something is ingested.
            # stderr, not stdout: stdout is the protocol channel for the MCP server.
            print(f"No documents in {self.db_path} yet — using vector retriever only.", file=sys.stderr)
            return dense

        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = k
        if strategy == "bm25":
            return bm25
        return EnsembleRetriever(retrievers=[dense, bm25], weights=[0.5, 0.5])

    def candidates(self, query: str) -> list[Document]:
        """Stage 1: all candidates from the configured search strategy."""
        return self._base.invoke(query)

    def select(self, query: str, candidates: list[Document]) -> list[Document]:
        """Stage 2: reranked top N, or the first N candidates when reranking is off."""
        if not self.rerank:
            return candidates[: self.top_n]
        scores = get_reranker().predict([(query, doc.page_content) for doc in candidates])
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[: self.top_n]]

    def retrieve_with_stages(self, query: str) -> tuple[list[Document], list[Document]]:
        """Returns (candidates, top) so callers can show both stages."""
        cands = self.candidates(query)
        return cands, self.select(query, cands)

    def retrieve(self, query: str) -> list[Document]:
        return self.retrieve_with_stages(query)[1]
