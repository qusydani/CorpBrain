"""
observability.py — optional Langfuse tracing for the agent.

    with agent_trace(question, session_id) as t:     # one trace per question
        graph.astream(..., {"callbacks": t.callbacks})  # LLM calls → generations
        t.update(output=answer)
    with tool_span(name, args) as span:              # an MCP tool call → tool span
        span.update(output=text)

Tracing is on only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set,
the langfuse package imports, and the server answers its health check
(LANGFUSE_BASE_URL, default http://localhost:3000, the docker-compose
`observability` profile). Otherwise every helper is a silent no-op, so the app
runs the same without Langfuse. The decision is made once per process: start
Langfuse before the app, or restart the app after starting it. When tracing
turns on, the Gemini Flash price from pricing.py is registered in the project
(once) so Langfuse can put a $ cost on every generation.
"""
import base64
import functools
import json
import logging
import os
import re
import urllib.request
from contextlib import contextmanager
from typing import Any, Iterator

from pricing import FLASH_MODEL_ID, PRICE_IN_PER_TOKEN, PRICE_OUT_PER_TOKEN

DEFAULT_BASE_URL = "http://localhost:3000"
TRACE_NAME = "corpbrain-agent"
HEALTH_TIMEOUT_S = 2

log = logging.getLogger(__name__)


class _NoSpan:
    """Stands in for a Langfuse span when tracing is off."""
    callbacks: list = []

    def update(self, **_: Any) -> None:
        pass


_NO_SPAN = _NoSpan()


class _Trace:
    """A live root span plus the LangChain callbacks that report under it."""

    def __init__(self, span, callbacks: list):
        self.span, self.callbacks = span, callbacks

    def update(self, **kwargs: Any) -> None:
        self.span.update(**kwargs)


def base_url() -> str:
    return (os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or DEFAULT_BASE_URL).rstrip("/")


def _reachable(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/public/health", timeout=HEALTH_TIMEOUT_S) as r:
            return r.status == 200
    except Exception:
        return False


@functools.cache
def client():
    """The Langfuse client, or None when tracing is off. Never raises."""
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None
    if os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() == "false":
        return None
    try:
        from langfuse import Langfuse
    except ImportError:
        log.info("Langfuse keys are set but the langfuse package is not installed; tracing off.")
        return None
    url = base_url()
    if not _reachable(url):
        log.warning("Langfuse not reachable at %s; tracing off for this process.", url)
        return None
    try:
        lf = Langfuse(base_url=url)
    except Exception as e:
        log.warning("Langfuse client failed to start (%s); tracing off.", e)
        return None
    ensure_model_price(url)
    return lf


def _api(url: str, path: str, body: dict | None = None):
    token = base64.b64encode(
        f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()).decode()
    req = urllib.request.Request(
        f"{url}/api/public/{path}", method="POST" if body is not None else "GET",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Basic {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=HEALTH_TIMEOUT_S * 3) as r:
        return json.load(r)


def model_definition() -> dict:
    """Langfuse model price for Gemini Flash, from pricing.py so traces and the eval agree.
    Reasoning (thinking) tokens bill as output; Langfuse reports them separately."""
    out = PRICE_OUT_PER_TOKEN
    return {
        "modelName": FLASH_MODEL_ID,
        "matchPattern": rf"(?i)^(models/)?{re.escape(FLASH_MODEL_ID)}$",
        "unit": "TOKENS",
        "pricingTiers": [{
            "name": "Standard", "isDefault": True, "priority": 0, "conditions": [],
            "prices": {"input": PRICE_IN_PER_TOKEN, "output": out, "output_reasoning": out},
        }],
    }


def ensure_model_price(url: str) -> None:
    """Register the Flash price in the project once (newer Langfuse servers ship
    one built in; this pins it to ours either way). Best effort, never raises."""
    try:
        n = pages = 1
        while n <= pages:
            page = _api(url, f"models?limit=100&page={n}")
            if any(m.get("modelName") == FLASH_MODEL_ID and m.get("scope") == "PROJECT"
                   for m in page.get("data", [])):
                return
            pages, n = page.get("meta", {}).get("totalPages", 1), n + 1
        _api(url, "models", model_definition())
        log.info("Registered %s pricing in Langfuse.", FLASH_MODEL_ID)
    except Exception as e:
        log.warning("Could not register %s pricing in Langfuse (%s); costs may be missing.",
                    FLASH_MODEL_ID, e)


def enabled() -> bool:
    return client() is not None


@contextmanager
def agent_trace(question: str, session_id: str | None = None) -> Iterator[Any]:
    """
    The root span of one agent run. Yields an object with `.callbacks` (pass in
    the LangGraph config) and `.update(output=..., metadata=...)`.
    """
    lf = client()
    if lf is None:
        yield _NO_SPAN
        return
    from langfuse import propagate_attributes
    from langfuse.langchain import CallbackHandler

    with lf.start_as_current_observation(as_type="agent", name=TRACE_NAME, input={"question": question}) as span, \
            propagate_attributes(trace_name=TRACE_NAME, session_id=session_id, tags=["agent"]):
        yield _Trace(span, [CallbackHandler()])


@contextmanager
def tool_span(name: str, args: dict) -> Iterator[Any]:
    """A tool span under the current trace. Yields something with `.update(...)`."""
    lf = client()
    if lf is None:
        yield _NO_SPAN
        return
    with lf.start_as_current_observation(as_type="tool", name=name, input=args) as span:
        yield span


def flush() -> None:
    """Send buffered spans now (short-lived scripts; long-lived apps flush in the background)."""
    lf = client()
    if lf is not None:
        lf.flush()
