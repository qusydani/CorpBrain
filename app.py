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
from eval import (
    EvalConfig,
    METRIC_DISPLAY,
    RESULTS_PATH,
    baseline_exists,
    create_baseline_collection,
    generate_test_set,
    load_results,
    load_test_set,
    run_full_evaluation,
    save_test_set,
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
                customdata=list(idx),
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
                customdata=list(c_idx),
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
                customdata=list(r_idx),
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
            customdata=[-1],
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
        clickmode="event+select",
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
                st.session_state.pop("selected_chunk_idx", None)

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


# ── Navigation ────────────────────────────────────────────────────────────────

active_view = st.segmented_control(
    "Navigation", ["Chat", "Explore", "Eval"], default="Chat", key="active_view",
    label_visibility="collapsed",
)


# ── Chat Tab ──────────────────────────────────────────────────────────────────

if not active_view or active_view == "Chat":
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

elif active_view == "Explore":
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
                st.toast("Visualization computed!")

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
                col_a.caption(f"Retrieved candidates: {len(candidates)}")
                col_b.caption(f"Reranked top: {len(reranked)}")
                col_c.caption("Query point")

            # ── Plot ──────────────────────────────────────────────────────────
            fig = build_figure(coords, texts, metadata, candidates, reranked, query_coord)
            st.plotly_chart(fig, use_container_width=True)

            # ── Searchable chunk browser ──────────────────────────────────────
            st.divider()
            search_query = st.text_input(
                "Search chunks by keyword:",
                placeholder="e.g. brake, adjustment, cable...",
                key="chunk_search",
            )

            if search_query:
                # Find matching chunks
                matching = [
                    i for i in range(len(texts))
                    if (search_query.lower() in texts[i].lower() or
                        search_query.lower() in metadata[i].get("source", "").lower())
                ]

                if matching:
                    # Display matching chunks as selectbox
                    chunk_opts = [
                        f"{metadata[i].get('source', 'Unknown')} (Page {metadata[i].get('page', '?')}) — "
                        f"{texts[i][:60].replace(chr(10), ' ')}..."
                        for i in matching
                    ]
                    selected_opt = st.selectbox("Matching chunks:", chunk_opts, key="chunk_select")
                    sel_idx = matching[chunk_opts.index(selected_opt)]

                    # Show full details
                    meta_sel = metadata[sel_idx]
                    chunk_text = texts[sel_idx]
                    with st.container(border=True):
                        st.markdown(
                            f"**{meta_sel.get('source', 'Unknown')}** — "
                            f"Page {meta_sel.get('page', '?')} · `{meta_sel.get('type', 'unknown')}`"
                        )
                        st.markdown(chunk_text)
                        img_path = meta_sel.get("image_path")
                        if img_path and os.path.exists(img_path):
                            st.image(img_path, caption=os.path.basename(img_path))
                else:
                    st.info(f"No chunks match '{search_query}'.")


# ── Eval Tab ───────────────────────────────────────────────────────────────────

elif active_view == "Eval":
    st.subheader("RAG Evaluation Pipeline")

    ingested = get_ingested_files()
    if not ingested:
        st.info("No documents ingested yet. Upload and ingest PDFs first.")
    else:
        # ── Step 1: Test Set ──────────────────────────────────────────────────
        st.markdown("### Step 1 — Test Set")

        test_set = load_test_set()
        col_gen, col_gen_info = st.columns([1, 3])

        with col_gen:
            n_questions = st.number_input(
                "Questions to generate", min_value=5, max_value=50, value=25, step=5,
                key="eval_n_questions",
            )
            if st.button("Generate Test Set", use_container_width=True, key="btn_generate"):
                progress_placeholder = st.empty()
                prog_bar = progress_placeholder.progress(0, text="Starting...")

                def gen_cb(msg: str, frac: float):
                    prog_bar.progress(frac, text=msg)

                with st.spinner("Generating Q&A pairs with Claude Haiku..."):
                    test_set = generate_test_set(n=int(n_questions), progress_callback=gen_cb)
                    save_test_set(test_set)

                progress_placeholder.empty()
                st.toast(f"Test set ready: {len(test_set)} questions saved.")
                st.rerun()

        with col_gen_info:
            if test_set:
                st.success(f"{len(test_set)} questions loaded from `eval_data/test_set.json`")
                with st.expander("Preview test set"):
                    for i, item in enumerate(test_set[:5]):
                        st.markdown(
                            f"**Q{i + 1}** ({item.get('source', '?')} p.{item.get('page', '?')}): "
                            f"{item['question']}"
                        )
                        st.caption(f"A: {item['ground_truth'][:150]}...")
                    if len(test_set) > 5:
                        st.caption(f"… and {len(test_set) - 5} more.")
            else:
                st.info("No test set found. Generate one to proceed.")

        st.divider()

        # ── Step 2: Baseline Collection ───────────────────────────────────────
        st.markdown("### Step 2 — Baseline Collection (optional)")
        st.caption(
            "Creates a second ChromaDB without contextual chunking so you can "
            "compare retrieval quality with vs. without context augmentation."
        )

        col_base, col_base_info = st.columns([1, 3])
        with col_base:
            if st.button("Create Baseline DB", use_container_width=True, key="btn_baseline"):
                base_placeholder = st.empty()
                base_bar = base_placeholder.progress(0, text="Starting...")

                def base_cb(msg: str, frac: float):
                    base_bar.progress(frac, text=msg)

                with st.spinner("Re-ingesting PDFs without contextual chunking..."):
                    create_baseline_collection(progress_callback=base_cb)

                base_placeholder.empty()
                st.toast("Baseline collection created.")
                st.rerun()

        with col_base_info:
            if baseline_exists():
                st.success("Baseline DB exists — contextual vs. baseline comparison available.")
            else:
                st.warning("No baseline DB yet. Evaluation will only run contextual configs.")

        st.divider()

        # ── Step 3: Run Evaluation ────────────────────────────────────────────
        st.markdown("### Step 3 — Run Evaluation")

        # Initialise checkbox defaults once per session (only if key not yet set)
        _check_defaults = {
            "eval_dense": True, "eval_bm25": True, "eval_hybrid": True,
            "eval_rerank": True, "eval_no_rerank": True,
            "eval_ctx": True, "eval_base": baseline_exists(),
        }
        for _k, _v in _check_defaults.items():
            if _k not in st.session_state:
                st.session_state[_k] = _v

        # Preset buttons — write to session_state keys before checkboxes render
        pre_col1, pre_col2, pre_col3 = st.columns([1, 1, 4])
        if pre_col1.button("Quick (2 configs)", use_container_width=True, key="preset_quick"):
            # Hybrid+Rerank only — your production setup vs. no-rerank variant
            st.session_state["eval_dense"]     = False
            st.session_state["eval_bm25"]      = False
            st.session_state["eval_hybrid"]    = True
            st.session_state["eval_rerank"]    = True
            st.session_state["eval_no_rerank"] = False
            st.session_state["eval_ctx"]       = True
            st.session_state["eval_base"]      = False
        if pre_col2.button("Full (12 configs)", use_container_width=True, key="preset_full"):
            st.session_state["eval_dense"]     = True
            st.session_state["eval_bm25"]      = True
            st.session_state["eval_hybrid"]    = True
            st.session_state["eval_rerank"]    = True
            st.session_state["eval_no_rerank"] = True
            st.session_state["eval_ctx"]       = True
            st.session_state["eval_base"]      = baseline_exists()

        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            st.caption("Retrieval strategies")
            use_dense  = st.checkbox("Dense (semantic)", key="eval_dense")
            use_bm25   = st.checkbox("BM25 (keyword)",  key="eval_bm25")
            use_hybrid = st.checkbox("Hybrid",           key="eval_hybrid")
        with col_cfg2:
            st.caption("Reranking")
            use_rerank    = st.checkbox("With reranking",    key="eval_rerank")
            use_no_rerank = st.checkbox("Without reranking", key="eval_no_rerank")
        with col_cfg3:
            st.caption("Chunking")
            use_contextual = st.checkbox("Contextual chunks", key="eval_ctx")
            use_baseline   = st.checkbox(
                "Baseline chunks",
                disabled=not baseline_exists(),
                key="eval_base",
            )

        strategies     = [s for s, flag in [("dense", use_dense), ("bm25", use_bm25), ("hybrid", use_hybrid)] if flag]
        reranking_opts = [v for v, flag in [(True, use_rerank), (False, use_no_rerank)] if flag]
        ctx_opts       = [v for v, flag in [(True, use_contextual), (False, use_baseline)] if flag]

        configs = [
            EvalConfig(strategy=s, reranking=r, contextual=c)
            for s in strategies
            for r in reranking_opts
            for c in ctx_opts
        ]

        n_cfg = len(configs)
        n_q   = len(test_set) if test_set else 0
        est_calls = n_cfg * n_q * 2  # retrieve + answer per question, RAGAS on top
        st.caption(
            f"**{n_cfg} configs × {n_q} questions** — ~{est_calls} LLM calls "
            f"(answer generation) + RAGAS scoring passes."
        )

        run_disabled = not test_set or not strategies or not reranking_opts or not ctx_opts
        if st.button(
            "Run Evaluation",
            use_container_width=True,
            disabled=run_disabled,
            type="primary",
            key="btn_run_eval",
        ):
            eval_placeholder = st.empty()
            eval_bar = eval_placeholder.progress(0, text="Starting evaluation...")

            def eval_cb(msg: str, frac: float):
                eval_bar.progress(frac, text=msg)

            with st.spinner(f"Evaluating {n_cfg} configs..."):
                run_full_evaluation(configs, test_set, progress_callback=eval_cb)

            eval_placeholder.empty()
            st.toast("Evaluation complete!")
            st.rerun()

        st.divider()

        # ── Results ───────────────────────────────────────────────────────────
        results = load_results()
        if results:
            st.markdown("### Results")
            st.caption(
                f"Evaluated {results['n_questions']} questions · "
                f"Run at {results['timestamp'][:19].replace('T', ' ')}"
            )

            # Build a flat table
            rows = []
            metric_keys = None
            for cfg_result in results["configs"]:
                agg = cfg_result["aggregate"]
                if metric_keys is None:
                    metric_keys = list(agg.keys())
                row = {"Config": cfg_result["label"]}
                row.update({METRIC_DISPLAY.get(k, k): round(v, 3) for k, v in agg.items()})
                rows.append(row)

            import pandas as pd
            df = pd.DataFrame(rows).set_index("Config")
            st.dataframe(df.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)

            # Bar chart per metric
            if metric_keys:
                metric_labels = [METRIC_DISPLAY.get(k, k) for k in metric_keys]
                selected_metric = st.selectbox(
                    "Chart metric:", metric_labels, key="eval_chart_metric"
                )
                raw_key = next(
                    (k for k in metric_keys if METRIC_DISPLAY.get(k, k) == selected_metric),
                    metric_keys[0],
                )

                bar_fig = go.Figure(go.Bar(
                    x=[r["Config"] for r in rows],
                    y=[r["aggregate"][raw_key] for r in results["configs"]],
                    marker_color="steelblue",
                    text=[f"{r['aggregate'][raw_key]:.3f}" for r in results["configs"]],
                    textposition="outside",
                ))
                bar_fig.update_layout(
                    title=f"{selected_metric} by Configuration",
                    yaxis=dict(range=[0, 1.05], title=selected_metric),
                    xaxis_title="Config",
                    height=450,
                    margin=dict(l=40, r=40, t=60, b=120),
                    xaxis=dict(tickangle=-30),
                )
                st.plotly_chart(bar_fig, use_container_width=True)

            # Per-question breakdown
            with st.expander("Per-question breakdown"):
                cfg_labels = [r["label"] for r in results["configs"]]
                sel_cfg = st.selectbox("Config:", cfg_labels, key="eval_perq_cfg")
                cfg_data = next(r for r in results["configs"] if r["label"] == sel_cfg)
                per_q_df = pd.DataFrame(cfg_data["per_question"])
                per_q_df.columns = [
                    METRIC_DISPLAY.get(c, c) if c != "question" else "Question"
                    for c in per_q_df.columns
                ]
                st.dataframe(per_q_df, use_container_width=True)
        else:
            st.info("No results yet. Run evaluation above to see scores.")
