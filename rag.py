import os
import base64
import io
from PIL import Image
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

load_dotenv()

DB_PATH = "vector_db"
MEMORY_TURNS = 3   # Number of prior conversation turns to include
RERANK_TOP_N = 5   # Docs to keep after reranking and pass to Claude


class MultimodalRAGChain:
    def __init__(self):
        # 1. Setup Retrievers
        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vector_db = Chroma(persist_directory=DB_PATH, embedding_function=self.embedding_model)
        vector_retriever = vector_db.as_retriever(search_kwargs={"k": 10})

        print("Building Keyword Index...")
        db_data = vector_db.get()
        docs_for_bm25 = [
            Document(page_content=txt, metadata=meta)
            for txt, meta in zip(db_data["documents"], db_data["metadatas"])
        ]

        if docs_for_bm25:
            bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
            bm25_retriever.k = 10
            self.retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.5, 0.5],
            )
        else:
            print("No documents in DB yet — using vector retriever only.")
            self.retriever = vector_retriever

        # 2. Setup Reranker (local, no API cost)
        print("Loading reranker...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # 3. Setup Vision-Capable LLM
        self.llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

    def _encode_image(self, image_path: str) -> tuple:
        """Compress PNG to JPEG in-memory and return (base64_string, media_type)."""
        img = Image.open(image_path).convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return encoded, "image/jpeg"

    def _build_message(self, user_query, docs, chat_history):
        """Build the multimodal HumanMessage from retrieved docs and chat history."""
        text_context = ""
        image_blocks = []
        seen_images = set()

        for doc in docs:
            text_context += f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}\n\n"

            if doc.metadata.get("type") == "image_summary" and "image_path" in doc.metadata:
                img_path = doc.metadata["image_path"]
                if img_path not in seen_images and os.path.exists(img_path):
                    seen_images.add(img_path)
                    encoded_img, media_type = self._encode_image(img_path)
                    image_blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": encoded_img,
                            },
                        }
                    )

        # Format the last N turns of conversation history
        history_text = ""
        if chat_history:
            history_text = "CONVERSATION HISTORY:\n"
            for msg in chat_history[-(MEMORY_TURNS * 2):]:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n"
            history_text += "\n"

        system_prompt = (
            "You are an expert assistant for question-answering tasks. "
            "Use the retrieved text context and any attached images to answer the question. "
            "If you don't know the answer based on the provided context, say so.\n\n"
            f"{history_text}"
            f"TEXT CONTEXT:\n{text_context}"
        )

        message_content = [
            {"type": "text", "text": system_prompt},
            {"type": "text", "text": f"Question: {user_query}"},
        ]
        message_content.extend(image_blocks)

        return HumanMessage(content=message_content)

    def _rerank(self, query: str, docs: list) -> list:
        """
        Score every (query, doc) pair jointly with the cross-encoder.
        Returns the top RERANK_TOP_N docs sorted by relevance score.
        """
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:RERANK_TOP_N]]

    def embed_query(self, query: str) -> list:
        """Embed a query string into a vector using the same model as the DB."""
        return self.embedding_model.embed_query(query)

    def retrieve_with_stages(self, user_query: str) -> tuple:
        """
        Returns (candidates, reranked) separately for visualization.
        candidates — all 20 hybrid retrieval results
        reranked   — top 5 after cross-encoder reranking
        """
        candidates = self.retriever.invoke(user_query)
        reranked = self._rerank(user_query, candidates)
        return candidates, reranked

    def retrieve(self, user_query):
        """Stage 1: hybrid retrieval. Stage 2: cross-encoder reranking."""
        candidates = self.retriever.invoke(user_query)  # 20 candidates
        return self._rerank(user_query, candidates)      # top 5

    def stream(self, query_dict):
        """
        Returns (docs, answer_generator).
        Retrieval is done eagerly so the caller can show sources immediately,
        the answer is streamed token-by-token.
        """
        user_query = query_dict["input"]
        chat_history = query_dict.get("chat_history", [])

        docs = self.retrieve(user_query)
        message = self._build_message(user_query, docs, chat_history)

        def _generator():
            for chunk in self.llm.stream([message]):
                if isinstance(chunk.content, str):
                    yield chunk.content
                elif isinstance(chunk.content, list):
                    for block in chunk.content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            yield block.get("text", "")

        return docs, _generator()

    def invoke(self, query_dict):
        """Non-streaming invoke kept for compatibility."""
        user_query = query_dict["input"]
        chat_history = query_dict.get("chat_history", [])

        docs = self.retrieve(user_query)
        message = self._build_message(user_query, docs, chat_history)
        response = self.llm.invoke([message])
        return {"answer": response.content, "context": docs}


def get_hybrid_chain():
    return MultimodalRAGChain()
