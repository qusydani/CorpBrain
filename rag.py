import os
import base64
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

load_dotenv()

DB_PATH = "vector_db"

class MultimodalRAGChain:
    def __init__(self):
        # 1. Setup Retrievers
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
        vector_retriever = vector_db.as_retriever(search_kwargs={"k": 5})
        
        print("Building Keyword Index...")
        db_data = vector_db.get()
        docs_for_bm25 = [
            Document(page_content=txt, metadata=meta) 
            for txt, meta in zip(db_data['documents'], db_data['metadatas'])
        ]
        bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
        bm25_retriever.k = 5

        self.retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
        
        # 2. Setup Vision-Capable LLM
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    def _encode_image(self, image_path):
        # Translates the local image into Base64 for the API payload.
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def invoke(self, query_dict):
        user_query = query_dict["input"]
        
        # 3. Retrieve documents (text chunks and image summaries)
        docs = self.retriever.invoke(user_query)
        
        text_context = ""
        image_blocks = []
        
        # 4. Separate text and images
        for doc in docs:
            text_context += f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}\n\n"
            
            # If the database pulled an image summary, grab the actual image file
            if doc.metadata.get("type") == "image_summary" and "image_path" in doc.metadata:
                img_path = doc.metadata["image_path"]
                if os.path.exists(img_path):
                    encoded_img = self._encode_image(img_path)
                    image_blocks.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_img}"}}
                    )

        # 5. Construct the Multimodal Prompt
        system_prompt = (
            "You are an expert assistant for question-answering tasks. "
            "Use the following pieces of retrieved text context and any attached images to answer the question. "
            "If you don't know the answer, say that you don't know.\n\n"
            f"TEXT CONTEXT:\n{text_context}"
        )
        
        # Combine text prompt and image blocks into a single payload
        message_content = [{"type": "text", "text": system_prompt}, {"type": "text", "text": f"Question: {user_query}"}]
        message_content.extend(image_blocks)
        
        message = HumanMessage(content=message_content)
        
        # 6. Generate Answer
        response = self.llm.invoke([message])
        
        # Return exact format expected by app.py
        return {"answer": response.content, "context": docs}

def get_hybrid_chain():
    return MultimodalRAGChain()