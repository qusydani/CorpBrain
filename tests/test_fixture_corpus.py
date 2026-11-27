"""
Tests for the public fixture corpus (tests/fixtures/) and the corpus switch.
No API calls: model classes are constructed with dummy keys and never invoked.
"""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import corpus  # noqa: E402

FIXTURES = os.path.join(ROOT, "tests", "fixtures")
PDF_DIR = os.path.join(FIXTURES, "corpus", "pdfs")
CACHE_DIR = os.path.join(FIXTURES, "corpus", "cache")
TEST_SET = os.path.join(FIXTURES, "test_set.json")

# The only PDFs allowed in the committed fixture corpus. Adding one here is a
# deliberate decision that the document is public and redistributable.
PUBLIC_FIXTURE_PDFS = {
    "fy2026_budget_statement.pdf",
    "Global_Energy_Demand_For_Different_Energy_Sources.pdf",
}


# ── Corpus switch ─────────────────────────────────────────────────────────────

def test_default_corpus_when_unset(monkeypatch):
    monkeypatch.delenv(corpus.CORPUS_ENV, raising=False)
    assert corpus.corpus_name() == "default"
    assert corpus.paths()["data"] == "data/"


def test_fixture_corpus_paths(monkeypatch):
    monkeypatch.setenv(corpus.CORPUS_ENV, "Fixture")
    p = corpus.paths()
    assert os.path.normpath(p["data"]) == os.path.join("tests", "fixtures", "corpus", "pdfs")
    assert p["db"] == "vector_db_fixture"
    assert p["ingest_llm"] == "gemini"
    # Nothing in the fixture corpus may point at the personal corpus folders.
    assert not any(str(v).startswith(("data", "vector_db" + os.sep, "extracted_images")) for v in p.values())


def test_unknown_corpus_rejected(monkeypatch):
    monkeypatch.setenv(corpus.CORPUS_ENV, "prod")
    with pytest.raises(ValueError):
        corpus.corpus_name()


def test_switch_reaches_every_module():
    """One env var moves ingest, retriever and eval together (fresh process: paths bind at import)."""
    code = (
        "import ingest, retriever, eval as ev, corpus;"
        "print(ingest.DB_PATH, retriever.DB_PATH, ev.DB_PATH, ev.TEST_SET_PATH, ev.RESULTS_PATH)"
    )
    env = {**os.environ, corpus.CORPUS_ENV: "fixture"}
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    db1, db2, db3, test_set, results = out.stdout.split()[-5:]
    assert db1 == db2 == db3 == "vector_db_fixture"
    assert os.path.normpath(test_set) == os.path.join("tests", "fixtures", "test_set.json")
    assert results.endswith("eval_results_fixture.json")


def test_mcp_server_inherits_environment(monkeypatch):
    """The mcp library forwards only a few system vars by default; the agent must pass everything."""
    from agent import MCPToolbox
    monkeypatch.setenv(corpus.CORPUS_ENV, "fixture")
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-for-test")
    env = MCPToolbox()._params.env
    assert env[corpus.CORPUS_ENV] == "fixture"
    assert env["GOOGLE_API_KEY"] == "dummy-for-test"


# ── Ingest LLM flag ───────────────────────────────────────────────────────────

def test_gemini_ingest_llm(monkeypatch):
    from langchain_google_genai import ChatGoogleGenerativeAI
    import ingest
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-for-test")
    llm = ingest.get_ingest_llm("gemini")
    assert isinstance(llm, ChatGoogleGenerativeAI)
    assert llm.model.endswith(ingest.FLASH_MODEL_ID)


def test_anthropic_ingest_llm(monkeypatch):
    from langchain_anthropic import ChatAnthropic
    import ingest
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-for-test")
    assert isinstance(ingest.get_ingest_llm("anthropic"), ChatAnthropic)


def test_unknown_ingest_llm():
    import ingest
    with pytest.raises(ValueError):
        ingest.get_ingest_llm("gpt")


def test_vision_message_formats(tmp_path):
    from PIL import Image
    import ingest
    img = tmp_path / "page.png"
    Image.new("RGB", (8, 8), "white").save(img)

    gemini = ingest.vision_message(str(img), "gemini").content[1]
    assert gemini["type"] == "image_url"
    assert gemini["image_url"].startswith("data:image/jpeg;base64,")

    claude = ingest.vision_message(str(img), "anthropic").content[1]
    assert claude["type"] == "image" and claude["source"]["type"] == "base64"


def test_usage_recording():
    from langchain_core.messages import AIMessage
    import ingest
    usage = ingest.new_usage()
    msg = AIMessage(content="x", usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13})
    ingest._record(usage, msg)
    ingest._record(usage, msg)
    assert (usage["calls"], usage["input_tokens"], usage["output_tokens"]) == (2, 20, 6)
    assert "Total" in ingest.cost_report(usage, "gemini")


def test_text_from_content_blocks():
    from langchain_core.messages import AIMessage
    import ingest
    assert ingest._text(AIMessage(content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])) == "ab"


# ── Committed fixture files ───────────────────────────────────────────────────

def test_fixture_pdfs_are_public_only():
    found = set(os.listdir(PDF_DIR))
    assert found == PUBLIC_FIXTURE_PDFS


def test_fixture_cache_only_covers_public_pdfs():
    """Cached vision summaries are named <pdf>_page_<n>_summary.txt."""
    for name in os.listdir(CACHE_DIR):
        if name == "chunk_contexts.json" or name.endswith(".png"):
            continue
        assert name.endswith("_summary.txt"), name
        assert name.split("_page_")[0] in PUBLIC_FIXTURE_PDFS, name


def test_nothing_from_personal_corpus_committed():
    """No committable PDF under tests/fixtures/ (anywhere, not just pdfs/) may be off the allowlist."""
    tracked = subprocess.run(
        ["git", "ls-files", "--others", "--cached", "--exclude-standard", "tests/fixtures"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    for path in tracked:
        if path.lower().endswith(".pdf"):
            assert os.path.basename(path) in PUBLIC_FIXTURE_PDFS, path


def test_frozen_test_set_shape():
    with open(TEST_SET, encoding="utf-8") as f:
        ts = json.load(f)
    assert 8 <= len(ts) <= 15
    types = {q["type"] for q in ts}
    assert types == {"single-hop", "multi-hop"}
    for q in ts:
        assert q["question"] and q["ground_truth"]
        for s in q["sources"]:
            assert s["source"] in PUBLIC_FIXTURE_PDFS, s
