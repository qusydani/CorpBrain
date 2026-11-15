import os
import base64
import io
from PIL import Image
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from retriever import Retriever, get_embedding_model, get_reranker

load_dotenv()

MEMORY_TURNS = 3   # Number of prior conversation turns to include


class MultimodalRAGChain:
    def __init__(self):
        # 1. Hybrid retrieval + cross-encoder reranking on the contextual DB
        print("Building Keyword Index...")
        self.retriever = Retriever(strategy="hybrid", rerank=True, contextual=True)

        # 2. Load the reranker now rather than on the first query
        print("Loading reranker...")
        get_reranker()

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

    def embed_query(self, query: str) -> list:
        """Embed a query string into a vector using the same model as the DB."""
        return get_embedding_model().embed_query(query)

    def retrieve_with_stages(self, user_query: str) -> tuple:
        """
        Returns (candidates, reranked) separately for visualization.
        candidates — all hybrid retrieval results (up to 20)
        reranked   — top 5 after cross-encoder reranking
        """
        return self.retriever.retrieve_with_stages(user_query)

    def retrieve(self, user_query):
        """Stage 1: hybrid retrieval. Stage 2: cross-encoder reranking (top 5)."""
        return self.retriever.retrieve(user_query)

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
