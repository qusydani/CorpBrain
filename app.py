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

    # Build chat history for memory (exclude image paths — text only)
    chat_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]  # exclude the message we just appended
    ]

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context..."):
            docs, answer_stream = st.session_state.chain.stream({
                "input": prompt,
                "chat_history": chat_history,
            })

        # Separate text sources from image paths
        text_sources = list(set([
            doc.metadata.get("source", "Unknown")
            for doc in docs
            if doc.metadata.get("type") != "image_summary"
        ]))
        image_paths = list(set([
            doc.metadata.get("image_path")
            for doc in docs
            if doc.metadata.get("type") == "image_summary"
        ]))

        # 1. Stream the answer
        answer = st.write_stream(answer_stream)

        # 2. Show text sources
        sources_md = "\n\n**Text Sources:**\n" + "\n".join([f"- {s}" for s in text_sources])
        st.markdown(sources_md)
        full_response = answer + sources_md

        # 3. Render retrieved images
        valid_images = []
        if image_paths:
            st.markdown("**Referenced Diagrams:**")
            for img_path in image_paths:
                if img_path and os.path.exists(img_path):
                    st.image(img_path, caption=os.path.basename(img_path))
                    valid_images.append(img_path)

    # Save to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "images": valid_images,
    })
