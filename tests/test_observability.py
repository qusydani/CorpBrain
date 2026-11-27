"""
Tests for observability.py. No Langfuse server and no paid calls: tracing-off
paths run with keys unset or the server unreachable; the tracing-on path
exports spans to memory and runs the agent with a scripted model and a fake
toolbox.
"""
import asyncio
import os
import sys
import time

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import observability  # noqa: E402
from agent import MCPToolbox, run_agent  # noqa: E402
from test_agent import ai, call, model_of  # noqa: E402

_client = observability.client   # The cached original; some tests monkeypatch it
KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL", "LANGFUSE_HOST",
        "LANGFUSE_TRACING_ENABLED")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    _client.cache_clear()
    yield
    _client.cache_clear()


class FakeToolbox(MCPToolbox):
    """The real call() wrapper (tool spans) over canned results instead of MCP."""

    def __init__(self):
        super().__init__()
        self.specs = [{"type": "function", "function": {"name": n, "description": "", "parameters": {}}}
                      for n in ("search_documents", "list_documents")]

    async def _call(self, name, args):
        if args.get("query") == "boom":
            return "Tool search_documents failed: boom", True, None
        hit = [{"source": "a.pdf", "page": 1, "type": "text", "content": "alpha"}]
        return '[{"source": "a.pdf"}]', False, hit


def run(model, toolbox=None, **kw):
    async def go():
        return [ev async for ev in run_agent("q", toolbox=toolbox or FakeToolbox(), llm=model, **kw)]
    return asyncio.run(go())


def tool_loop_model():
    return model_of([
        ai(tool_calls=[call("search_documents", {"query": "alpha"}, "c1"),
                       call("search_documents", {"query": "boom"}, "c2")]),
        ai("Alpha [a.pdf p.1]."),
    ])


# ── Tracing off ───────────────────────────────────────────────────────────────

def test_no_keys_means_no_client_and_no_op_helpers():
    assert observability.client() is None
    with observability.agent_trace("q") as t:
        assert t.callbacks == []
        t.update(output="x")
    with observability.tool_span("search_documents", {}) as s:
        s.update(output="x")
    observability.flush()


def test_unreachable_server_disables_tracing_quickly(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:9")   # Nothing listens here
    start = time.monotonic()
    assert observability.client() is None
    assert time.monotonic() - start < observability.HEALTH_TIMEOUT_S + 1


def test_explicitly_disabled_tracing_skips_the_health_check(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    monkeypatch.setattr(observability, "_reachable", lambda url: pytest.fail("health-checked"))
    assert observability.client() is None


def test_agent_runs_normally_with_tracing_off():
    final = run(tool_loop_model())[-1]
    assert final.answer == "Alpha [a.pdf p.1]."
    assert final.tool_calls == 2


# ── Tracing on (spans exported to memory) ─────────────────────────────────────

@pytest.fixture
def spans(monkeypatch):
    from langfuse import Langfuse
    exporter = InMemorySpanExporter()
    lf = Langfuse(public_key="pk-lf-memtest", secret_key="sk-lf-memtest",
                  base_url="http://127.0.0.1:9", span_exporter=exporter)
    monkeypatch.setattr(observability, "client", lambda: lf)
    yield lambda: (lf.flush(), exporter.get_finished_spans())[1]
    lf.shutdown()


def _type(span):
    return span.attributes.get("langfuse.observation.type")


def test_one_trace_with_generations_and_tool_spans(spans):
    model = tool_loop_model()
    run(model, session_id="chat-test")
    got = spans()

    assert len({s.context.trace_id for s in got}) == 1
    root = next(s for s in got if s.parent is None)
    assert root.name == observability.TRACE_NAME
    assert _type(root) == "agent"

    tools = [s for s in got if _type(s) == "tool"]
    assert [s.name for s in tools] == ["search_documents", "search_documents"]
    assert [s.attributes.get("langfuse.observation.level") for s in tools] == ["DEFAULT", "ERROR"]
    # One generation per chat-model call (the scripted router bypasses the chat model)
    assert len([s for s in got if _type(s) == "generation"]) == model.calls
    assert any(s.attributes.get("session.id") == "chat-test"
               or s.attributes.get("langfuse.trace.session_id") == "chat-test" for s in got)


# ── Model price registration ──────────────────────────────────────────────────

def test_model_definition_uses_pricing_and_matches_the_model_name():
    import re
    from pricing import FLASH_MODEL_ID, PRICE_IN_PER_TOKEN, PRICE_OUT_PER_TOKEN
    d = observability.model_definition()
    prices = d["pricingTiers"][0]["prices"]
    assert prices == {"input": PRICE_IN_PER_TOKEN, "output": PRICE_OUT_PER_TOKEN,
                      "output_reasoning": PRICE_OUT_PER_TOKEN}
    assert re.match(d["matchPattern"].replace("(?i)", ""), FLASH_MODEL_ID)
    assert re.match(d["matchPattern"].replace("(?i)", ""), f"models/{FLASH_MODEL_ID}")
    assert not re.match(d["matchPattern"].replace("(?i)", ""), f"{FLASH_MODEL_ID}-lite")


def _fake_api(pages):
    posted = []

    def api(url, path, body=None):
        if body is not None:
            posted.append(body)
            return {}
        n = int(path.rsplit("page=", 1)[1])
        return {"data": pages[n - 1], "meta": {"totalPages": len(pages)}}
    return api, posted


def test_price_is_registered_when_missing(monkeypatch):
    api, posted = _fake_api([[{"modelName": "gemini-2.5-flash", "scope": "LANGFUSE"}]])
    monkeypatch.setattr(observability, "_api", api)
    observability.ensure_model_price("http://x")
    assert posted == [observability.model_definition()]


def test_price_already_in_the_project_is_not_registered_again(monkeypatch):
    from pricing import FLASH_MODEL_ID
    api, posted = _fake_api([[{"modelName": "other", "scope": "LANGFUSE"}],
                             [{"modelName": FLASH_MODEL_ID, "scope": "PROJECT"}]])
    monkeypatch.setattr(observability, "_api", api)
    observability.ensure_model_price("http://x")
    assert posted == []


def test_price_registration_failure_never_raises(monkeypatch):
    def broken(*a, **k):
        raise OSError("down")
    monkeypatch.setattr(observability, "_api", broken)
    observability.ensure_model_price("http://x")
