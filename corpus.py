"""
corpus.py — which document corpus this process works on, and where its files live.

    CORPBRAIN_CORPUS=default   (unset)  the personal corpus in data/ — never committed
    CORPBRAIN_CORPUS=fixture            the small public corpus committed under
                                        tests/fixtures/corpus/, used by CI

Every module takes its paths from here, so one environment variable switches
ingestion, retrieval, the MCP server and the eval together. The agent passes its
environment on to the MCP server subprocess, so the server follows the same switch.

The fixture cache folder (vision summaries + chunk contexts) is committed: those
are LLM outputs that differ run to run, so rebuilding the fixture index from them
needs only embedding calls and gives the same chunks every time.
"""
import os

CORPUS_ENV = "CORPBRAIN_CORPUS"
CORPORA = ("default", "fixture")

FIXTURE_ROOT = os.path.join("tests", "fixtures")

_PATHS = {
    "default": {
        "data":          "data/",
        "db":            "vector_db",
        "baseline_db":   "vector_db_baseline",
        "cache":         "extracted_images/",
        "test_set":      os.path.join("eval_data", "test_set.json"),
        "ingest_llm":    "anthropic",
    },
    "fixture": {
        "data":          os.path.join(FIXTURE_ROOT, "corpus", "pdfs") + os.sep,
        "db":            "vector_db_fixture",
        "baseline_db":   "vector_db_fixture_baseline",
        "cache":         os.path.join(FIXTURE_ROOT, "corpus", "cache") + os.sep,
        "test_set":      os.path.join(FIXTURE_ROOT, "test_set.json"),
        "ingest_llm":    "gemini",
    },
}


def corpus_name() -> str:
    name = os.environ.get(CORPUS_ENV, "default").strip().lower() or "default"
    if name not in CORPORA:
        raise ValueError(f"{CORPUS_ENV}={name!r}; expected one of {CORPORA}")
    return name


def paths(name: str | None = None) -> dict:
    return dict(_PATHS[name or corpus_name()])


_P = paths()
CORPUS            = corpus_name()
DATA_PATH         = _P["data"]
DB_PATH           = _P["db"]
BASELINE_DB_PATH  = _P["baseline_db"]
IMAGE_OUT_PATH    = _P["cache"]
TEST_SET_PATH     = _P["test_set"]
DEFAULT_INGEST_LLM = _P["ingest_llm"]
