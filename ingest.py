import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_PATH = "data/"
DB_PATH = "vector_db"

def create_vector_db():
    print(f"Loading documents from {DATA_PATH}...")
    
    loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", r"(?<=\. )", " ", ""] 
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Initialize an empty Vector DB first
    print("Initializing Vector Store...")
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

    # The Rate-Limit Workaround to process 80 at a time.
    batch_size = 80
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        print(f"Processing batch {i} to {i + len(batch)} out of {len(chunks)}...")
        
        # Add this specific batch to the database
        vector_db.add_documents(batch)
        
        #  Sleep for 60 seconds to reset limit
        if i + batch_size < len(chunks):
            print("Sleeping for 60 seconds...")
            time.sleep(60)

    print("Vector Store Created Successfully!")
    return vector_db, chunks

if __name__ == "__main__":
    vector_db, chunks = create_vector_db()