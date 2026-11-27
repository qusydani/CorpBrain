# CorpBrain app image: Streamlit UI + agent + MCP server (spawned by the agent
# over stdio from this same interpreter, so no separate MCP container).
# Documents, vector DBs and extracted images are mounted at runtime (see
# docker-compose.yml); API keys come from the environment.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# CPU-only torch first, so sentence-transformers doesn't pull the CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the cross-encoder reranker into the image: no download at startup.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
ENV HF_HUB_OFFLINE=1

COPY *.py ./
# Public fixture corpus (PDFs + LLM cache), for CORPBRAIN_CORPUS=fixture.
COPY tests/fixtures/ tests/fixtures/

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
