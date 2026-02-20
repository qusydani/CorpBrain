import streamlit as st
import os
from rag import get_hybrid_chain

st.set_page_config(page_title="CorpBrain RAG")
st.title("CorpBrain: Multimodal Docs Assistant")

# Initialize chain only once
if "chain" not in st.session_state:
    with st.spinner("Initializing Multimodal Hybrid Search Engine..."):
        st.session_state.chain = get_hybrid_chain()
    st.success("System Ready!")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chats and images
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "images" in message and message["images"]:
            for img_path in message["images"]:
                if os.path.exists(img_path):
                    st.image(img_path, caption=os.path.basename(img_path))

# User Input
if prompt := st.chat_input("Ask a question about the uploaded documents..."):
    # Append user prompt to history
    st.session_state.messages.append({"role": "user", "content": prompt, "images": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and analyzing images..."):
            response = st.session_state.chain.invoke({"input": prompt})
            answer = response["answer"]
            docs = response["context"]
            
            # Separate text sources from image paths based on metadata
            text_sources = list(set([doc.metadata.get('source', 'Unknown') for doc in docs if doc.metadata.get('type') != 'image_summary']))
            image_paths = list(set([doc.metadata.get('image_path') for doc in docs if doc.metadata.get('type') == 'image_summary']))
            
            # 1. Print Text Answer
            full_response = f"{answer}\n\n**Text Sources:**\n" + "\n".join([f"- {s}" for s in text_sources])
            st.markdown(full_response)
            
            # 2. Render Retrieved Images
            valid_images = []
            if image_paths:
                st.markdown("**Referenced Diagrams:**")
                for img_path in image_paths:
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, caption=os.path.basename(img_path))
                        valid_images.append(img_path)
            
    # Save the response and the image file paths to session state
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "images": valid_images
    })