import streamlit as st
import os
import time
from rag import get_hybrid_chain
from ingest import compute_file_hash, get_existing_hashes, ingest_pdf_file

DATA_PATH = "data/"

st.set_page_config(page_title="CorpBrain RAG", layout="wide")
st.title("CorpBrain: Multimodal Docs Assistant")

# Initialize chain only once
if "chain" not in st.session_state:
    with st.spinner("Initializing Multimodal Hybrid Search Engine..."):
        st.session_state.chain = get_hybrid_chain()
    st.success("System Ready!")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Uploader key — incrementing this string resets the file_uploader widget
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = "uploader_0"

# ── Sidebar: Document Upload ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Document Management")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        key=st.session_state.uploader_key,
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) selected")

        if st.button("Ingest Documents", use_container_width=True):
            os.makedirs(DATA_PATH, exist_ok=True)
            existing_hashes = get_existing_hashes()
            any_new = False

            # One placeholder container per file — lets us overwrite or erase it
            placeholders = {f.name: st.empty() for f in uploaded_files}

            # Show every file as pending before we start
            for f in uploaded_files:
                placeholders[f.name].info(f"Pending: {f.name}")

            for f in uploaded_files:
                file_bytes = f.read()
                file_hash = compute_file_hash(file_bytes)

                if file_hash in existing_hashes:
                    # Flag it and leave it visible so the user knows
                    placeholders[f.name].warning(f"Already indexed: {f.name}")
                    continue

                any_new = True

                # Save to data/
                save_path = os.path.join(DATA_PATH, f.name)
                with open(save_path, "wb") as out:
                    out.write(file_bytes)

                # Replace the placeholder with a progress container
                with placeholders[f.name].container():
                    st.caption(f"Ingesting: {f.name}")
                    progress_bar = st.progress(0, text="Starting...")

                def make_callback(bar):
                    def callback(current, total):
                        bar.progress(current / total, text=f"Page {current} / {total}")
                    return callback

                ingest_pdf_file(save_path, file_hash, progress_callback=make_callback(progress_bar))

                # Done — remove this file from the page
                placeholders[f.name].empty()

            if any_new:
                with st.spinner("Rebuilding search index..."):
                    st.session_state.chain = get_hybrid_chain()
                st.success("Ingestion complete.")

            # Reset the uploader widget to clear selected files
            st.session_state.uploader_key = f"uploader_{int(time.time())}"
            time.sleep(1)
            st.rerun()

# ── Chat UI ───────────────────────────────────────────────────────────────────

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "images" in message and message["images"]:
            for img_path in message["images"]:
                if os.path.exists(img_path):
                    st.image(img_path, caption=os.path.basename(img_path))

# User input
if prompt := st.chat_input("Ask a question about the uploaded documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "images": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build chat history for memory (text only, exclude current message)
    chat_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

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

        # Stream the answer
        answer = st.write_stream(answer_stream)

        # Show sources
        sources_md = "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in text_sources])
        st.markdown(sources_md)
        full_response = answer + sources_md

        # Render retrieved images
        valid_images = []
        if image_paths:
            st.markdown("**Referenced Diagrams:**")
            for img_path in image_paths:
                if img_path and os.path.exists(img_path):
                    st.image(img_path, caption=os.path.basename(img_path))
                    valid_images.append(img_path)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "images": valid_images,
    })
