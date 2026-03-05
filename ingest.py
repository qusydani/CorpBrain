import os
import time
import base64
import fitz
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma

load_dotenv()

DATA_PATH = "data/"
DB_PATH = "vector_db"
IMAGE_OUT_PATH = "extracted_images/"

def summarize_page_image(image_path):
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    prompt = (
        "You are an expert document analyst. Describe this page in thorough detail. "
        "Extract all visible text verbatim. If there are diagrams, charts, tables, or figures, "
        "explain what they depict, how components relate, and any labels, legends, or annotations. "
        "Capture all numbered steps, warnings, callouts, and part references exactly as written."
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": encoded_string,
                },
            },
        ]
    )
    
    print(f"Generating vision summary for {image_path}...")
    response = llm.invoke([message])
    return response.content

def create_multimodal_vector_db():
    print(f"Rasterizing and summarizing documents from {DATA_PATH}...")
    
    # Ensure our image output directory exists
    os.makedirs(IMAGE_OUT_PATH, exist_ok=True)
    
    documents = []
    
    # 1. Extraction & Vision Summarization Loop
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(DATA_PATH, file)
            doc = fitz.open(pdf_path) # Open the PDF with PyMuPDF
            
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                
                # Extract raw text
                raw_text = page.get_text()
                # Save the raw text as a standard document chunk
                documents.append(Document(
                    page_content=raw_text, 
                    metadata={"source": file, "page": page_num + 1, "type": "text"}
                ))

                # Render page to a high-res image
                pix = page.get_pixmap(dpi=150) 
                image_filename = f"{file}_page_{page_num + 1}.png"
                image_path = os.path.join(IMAGE_OUT_PATH, image_filename)
                
                # Save the image locally so the UI can display later
                pix.save(image_path) 
                
                # Send the image to Gemini for a text summary
                page_summary = summarize_page_image(image_path)
                
                # Create a LangChain Document out of the image summary
                # Attach the image path in the metadata so we can fetch later
                metadata = {
                    "source": file, 
                    "page": page_num + 1, 
                    "image_path": image_path,
                    "type": "image_summary"
                }
                documents.append(Document(page_content=page_summary, metadata=metadata))
                
            doc.close()

    print(f"Generated {len(documents)} page summaries.")

    # 2. Standard Chunking (We chunk the generated summaries)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)

    # 3. Vector Embedding
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
        #if i + batch_size < len(chunks):
        #    print("Sleeping for 60 seconds...")
        #    time.sleep(60)

    print("Multimodal Vector Store Created Successfully!")
    return vector_db, chunks

if __name__ == "__main__":
    vector_db, chunks = create_multimodal_vector_db()