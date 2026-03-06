import streamlit as st
import os
import time
import numpy as np
import chromadb
import plotly.graph_objects as go
import plotly.colors as pc
from umap import UMAP
from rag import get_hybrid_chain
from ingest import (
    DB_PATH,
    CHROMA_COLLECTION,
    compute_file_hash,
    get_existing_hashes,
    get_ingested_files,
    ingest_pdf_file,
)

DATA_PATH = "data/"
UMAP_CACHE_KEYS = ["umap_coords", "umap_reducer", "umap_texts", "umap_metadata"]


# ── UMAP helpers ──────────────────────────────────────────────────────────────

def compute_umap():
    """
    Fetch all embeddings from ChromaDB, fit UMAP, and cache results in
    session state. Called once per session (or after new docs are ingested).
    """
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(CHROMA_COLLECTION)
    result = collection.get(include=["embeddings", "documents", "metadatas"])

    embeddings = np.array(result["embeddings"])
    reducer = UMAP(n_components=3, random_state=42, n_neighbors=10, min_dist=0.3, spread=1.5)
    coords = reducer.fit_transform(embeddings)

    st.session_state["umap_coords"]   = coords
    st.session_state["umap_reducer"]  = reducer
    st.session_state["umap_texts"]    = result["documents"]
    st.session_state["umap_metadata"] = result["metadatas"]


def build_figure(coords, texts, metadata, candidates=None, reranked=None, query_coord=None):
    """
    Build the Plotly 3D scatter figure.
    - All chunks plotted, colored by source document.
    - Text chunks → circles. Image summary chunks → squares.
    - candidates  → orange open-circle overlay.
    - reranked    → red diamond overlay.
    - query_coord → purple cross with label.
    """
    sources = sorted(set(m.get("source", "Unknown") for m in metadata))
    palette = pc.qualitative.Plotly
    color_map = {s: palette[i % len(palette)] for i, s in enumerate(sources)}

    # Dim all points when a query is active so highlights stand out
    base_opacity = 0.35 if (candidates or reranked) else 0.75

    fig = go.Figure()

    # ── Base layer: one trace per source × chunk type ─────────────────────────
    for source in sources:
        for chunk_type, symbol, size in [("text", "circle", 4), ("image_summary", "square", 5)]:
            idx = [
                i for i, m in enumerate(metadata)
                if m.get("source") == source and m.get("type") == chunk_type
            ]
            if not idx:
                continue

            hover = [
                f"<b>{metadata[i].get('source', 'Unknown')}</b> — "
                f"Page {metadata[i].get('page', '?')}<br>"
                f"Type: {metadata[i].get('type', 'unknown')}<br>"
                f"{texts[i][:150].replace(chr(10), ' ')}..."
                for i in idx
            ]

            label = source if chunk_type == "text" else f"{source} (image)"
            fig.add_trace(go.Scatter3d(
                x=coords[idx, 0],
                y=coords[idx, 1],
                z=coords[idx, 2],
                mode="markers",
                marker=dict(
                    color=color_map[source],
                    size=size,
                    opacity=base_opacity,
                    symbol=symbol,
                ),
                name=label,
                text=hover,
                hoverinfo="text",
                legendgroup=source,
            ))

    # ── Highlight: retrieved candidates (orange open circle) ──────────────────
    if candidates:
        candidate_set = {d.page_content for d in candidates}
        c_idx = [i for i, t in enumerate(texts) if t in candidate_set]
        if c_idx:
            c_hover = [
                f"<b>[Retrieved] {metadata[i].get('source', 'Unknown')}</b> — "
                f"Page {metadata[i].get('page', '?')}<br>"
                f"Type: {metadata[i].get('type', 'unknown')}<br>"
                f"{texts[i][:150].replace(chr(10), ' ')}..."
                for i in c_idx
            ]
            fig.add_trace(go.Scatter3d(
                x=coords[c_idx, 0],
                y=coords[c_idx, 1],
                z=coords[c_idx, 2],
                mode="markers",
                marker=dict(
                    color="orange",
                    size=12,
                    symbol="circle-open",
                ),
                name=f"Retrieved ({len(candidates)})",
                text=c_hover,
                hoverinfo="text",
            ))

    # ── Highlight: reranked top 5 (red diamond) ────────────────────────────────
    if reranked:
        reranked_set = {d.page_content for d in reranked}
        r_idx = [i for i, t in enumerate(texts) if t in reranked_set]
        if r_idx:
            r_hover = [
                f"<b>[Reranked] {metadata[i].get('source', 'Unknown')}</b> — "
                f"Page {metadata[i].get('page', '?')}<br>"
                f"Type: {metadata[i].get('type', 'unknown')}<br>"
                f"{texts[i][:150].replace(chr(10), ' ')}..."
                for i in r_idx
            ]
            fig.add_trace(go.Scatter3d(
                x=coords[r_idx, 0],
                y=coords[r_idx, 1],
                z=coords[r_idx, 2],
                mode="markers",
                marker=dict(color="red", size=10, symbol="diamond"),
                name=f"Reranked Top {len(reranked)}",
                text=r_hover,
                hoverinfo="text",
            ))

    # ── Query point (purple cross) ─────────────────────────────────────────────
    if query_coord is not None:
        fig.add_trace(go.Scatter3d(
            x=[query_coord[0]],
            y=[query_coord[1]],
            z=[query_coord[2]],
            mode="markers+text",
            marker=dict(color="purple", size=10, symbol="cross"),
            text=["Query"],
            textposition="top center",
            name="Query",
        ))

    fig.update_layout(
        title="Knowledge Base — Vector Space (UMAP 3D)",
        scene=dict(
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            zaxis_title="UMAP Dimension 3",
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        ),
        height=700,
        legend=dict(itemsizing="constant"),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


# ── App bootstrap ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="CorpBrain RAG", layout="wide")
st.title("CorpBrain: Multimodal Docs Assistant")

if "chain" not in st.session_state:
    with st.spinner("Initializing Multimodal Hybrid Search Engine..."):
        st.session_state.chain = get_hybrid_chain()
    st.success("System Ready!")

if "messages" not in st.session_state:
    st.session_state.messages = []

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

            placeholders = {f.name: st.empty() for f in uploaded_files}
            for f in uploaded_files:
                placeholders[f.name].info(f"Pending: {f.name}")

            for f in uploaded_files:
                file_bytes = f.read()
                file_hash = compute_file_hash(file_bytes)

                if file_hash in existing_hashes:
                    placeholders[f.name].warning(f"Already indexed: {f.name}")
                    continue

                any_new = True

                save_path = os.path.join(DATA_PATH, f.name)
                with open(save_path, "wb") as out:
                    out.write(file_bytes)

                with placeholders[f.name].container():
                    st.caption(f"Ingesting: {f.name}")
                    progress_bar = st.progress(0, text="Starting...")

                def make_callback(bar):
                    def callback(current, total, status=None):
                        text = status if status else f"Page {current} / {total}"
                        bar.progress(current / total, text=text)
                    return callback

                ingest_pdf_file(save_path, file_hash, progress_callback=make_callback(progress_bar))
                placeholders[f.name].empty()

            if any_new:
                with st.spinner("Rebuilding search index..."):
                    st.session_state.chain = get_hybrid_chain()

                # Invalidate UMAP cache — DB has changed
                for key in UMAP_CACHE_KEYS:
                    st.session_state.pop(key, None)

                st.success("Ingestion complete.")

            st.session_state.uploader_key = f"uploader_{int(time.time())}"
            time.sleep(1)
            st.rerun()

    st.divider()
    st.subheader("Knowledge Base")
    ingested = get_ingested_files()
    if ingested:
        for filename, pages in ingested.items():
            st.caption(f"📄 {filename} — {pages} pages")
    else:
        st.caption("No documents ingested yet.")


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_chat, tab_explore = st.tabs(["Chat", "Explore"])


# ── Chat Tab ──────────────────────────────────────────────────────────────────

with tab_chat:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "images" in message and message["images"]:
                for img_path in message["images"]:
                    if os.path.exists(img_path):
                        st.image(img_path, caption=os.path.basename(img_path))

    if prompt := st.chat_input("Ask a question about the uploaded documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt, "images": []})
        with st.chat_message("user"):
            st.markdown(prompt)

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

            answer = st.write_stream(answer_stream)

            sources_md = "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in text_sources])
            st.markdown(sources_md)
            full_response = answer + sources_md

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


# ── Explore Tab ───────────────────────────────────────────────────────────────

with tab_explore:
    st.subheader("Vector Space Explorer")

    ingested = get_ingested_files()
    if not ingested:
        st.info("No documents ingested yet. Upload and ingest PDFs first.")
    else:
        col_btn, col_info = st.columns([1, 3])

        with col_btn:
            if st.button("Compute Visualization", use_container_width=True):
                with st.spinner("Fetching embeddings and running UMAP..."):
                    compute_umap()
                st.success("Done!")

        with col_info:
            if "umap_coords" in st.session_state:
                n = len(st.session_state["umap_coords"])
                st.caption(f"{n} chunks mapped. Hover over points to preview content.")
            else:
                st.caption("Click 'Compute Visualization' to map the knowledge base in 3D.")

        if "umap_coords" in st.session_state:
            coords   = st.session_state["umap_coords"]
            texts    = st.session_state["umap_texts"]
            metadata = st.session_state["umap_metadata"]

            # ── Query input ───────────────────────────────────────────────────
            explore_query = st.text_input(
                "Type a query to highlight the retrieval pipeline:",
                placeholder="e.g. how do I adjust the brake cable?",
                key="explore_query",
            )

            candidates  = None
            reranked    = None
            query_coord = None

            if explore_query:
                with st.spinner("Retrieving and reranking..."):
                    candidates, reranked = st.session_state.chain.retrieve_with_stages(explore_query)
                    query_vector = st.session_state.chain.embed_query(explore_query)
                    reducer = st.session_state["umap_reducer"]
                    query_2d = reducer.transform([query_vector])
                    query_coord = query_2d[0]

                col_a, col_b, col_c = st.columns(3)
                col_a.caption(f"🟠 Retrieved candidates: {len(candidates)}")
                col_b.caption(f"🔴 Reranked top: {len(reranked)}")
                col_c.caption("🟣 Query point")

            # ── Plot ──────────────────────────────────────────────────────────
            fig = build_figure(coords, texts, metadata, candidates, reranked, query_coord)
            st.plotly_chart(fig, use_container_width=True)
