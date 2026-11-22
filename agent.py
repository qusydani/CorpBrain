"""
agent.py — a tool-calling agent over CorpBrain's MCP server.

    python agent.py "What does the handbook say about PPE?"

A small LangGraph loop on Gemini Flash: the model either calls tools or answers.
Tools come from mcp_server.py over stdio MCP, never by importing retrieval.

    run_agent(question, history, toolbox=...)   async stream of events
    AgentRunner().ask(question, history)        the same stream, synchronously,
                                                over one long-lived MCP session

Events, in order: ToolCall / ToolResult pairs as tools run, Token chunks of the
answer, then one Final with the answer, cited sources and token usage. The UI
and the eval both consume this stream.

Caps: at most MAX_ITERATIONS tool-enabled model calls and MAX_TOOL_CALLS tool
executions. Hitting either triggers one last model call without tools, so the
agent still answers from what it has; Final.stop_reason records which cap hit.
"""
import asyncio
import json
import operator
import os
import queue
import re
import sys
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Annotated, Any, AsyncIterator, Iterator, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage, message_chunk_to_message,
)
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from pricing import FLASH_MODEL_ID, cost, usage

ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))

SERVER_SCRIPT = os.path.join(ROOT, "mcp_server.py")
MAX_ITERATIONS = 6
MAX_TOOL_CALLS = 8
HISTORY_TURNS = 3   # Prior user/assistant turns passed to the model

SYSTEM_PROMPT = (
    "You answer questions about the user's PDF knowledge base using the tools. "
    "Always search before answering; never answer from general knowledge. "
    "Call list_documents when you need exact file names, e.g. to restrict search_documents "
    "with `sources`. If the question has several parts, search for each part.\n\n"
    "Cite every factual claim inline as [<source> p.<page>] using the source and page "
    "fields of the search results, e.g. [handbook.pdf p.12]; separate several with "
    "semicolons, e.g. [a.pdf p.3; b.pdf p.7]. If the documents don't contain the answer, say so."
)
FORCE_ANSWER_PROMPT = (
    "Tool budget exhausted. Answer now using only the search results above, with "
    "citations. If they are not enough, say what is missing."
)


# ── Events ────────────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class ToolResult:
    id: str
    name: str
    content: str          # Text the model sees (JSON for successful calls)
    is_error: bool


@dataclass
class Token:
    text: str


@dataclass
class Final:
    answer: str
    sources: list         # [{"source", "page"}], cited ones in citation order
    usage: dict           # {"input_tokens", "output_tokens", "usd"}
    iterations: int       # Model calls, including a forced final one
    tool_calls: int       # Tool executions (skipped-over-cap calls excluded)
    stop_reason: str      # "answered" | "iteration_limit" | "tool_call_limit"
    retrieved: list = field(default_factory=list)   # Every chunk search_documents returned


Event = ToolCall | ToolResult | Token | Final


# ── MCP toolbox ───────────────────────────────────────────────────────────────

class MCPToolbox:
    """
    An open stdio MCP session to mcp_server.py, exposing its tools as
    OpenAI-style function specs (for bind_tools) plus an async call().
    """

    def __init__(self, server_script: str = SERVER_SCRIPT):
        self._params = StdioServerParameters(command=sys.executable, args=[server_script], cwd=ROOT)
        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self.specs: list[dict] = []

    async def __aenter__(self):
        read, write = await self._stack.enter_async_context(stdio_client(self._params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        tools = (await self.session.list_tools()).tools
        self.specs = [
            {"type": "function", "function": {
                "name": t.name, "description": t.description or "", "parameters": t.input_schema,
            }}
            for t in tools
        ]
        return self

    async def __aexit__(self, *exc):
        await self._stack.aclose()

    @property
    def names(self) -> set[str]:
        return {s["function"]["name"] for s in self.specs}

    async def call(self, name: str, args: dict) -> tuple[str, bool, Any]:
        """(text for the model, is_error, parsed payload or None). Never raises."""
        if name not in self.names:
            return f"Unknown tool {name!r}. Available tools: {sorted(self.names)}", True, None
        try:
            res = await self.session.call_tool(name, args)
        except Exception as e:  # transport failure, server crash, bad args
            return f"Tool {name} failed: {type(e).__name__}: {e}", True, None
        text = "\n".join(getattr(c, "text", "") for c in res.content)
        if res.is_error:
            return text or f"Tool {name} returned an error.", True, None
        payload = res.structured_content
        if isinstance(payload, dict) and set(payload) == {"result"}:
            payload = payload["result"]
        return (json.dumps(payload) if payload is not None else text), False, payload


# ── Citations ─────────────────────────────────────────────────────────────────

_BRACKET = re.compile(r"\[([^\[\]]+)\]")
_CITE = re.compile(r"^\s*(.+?)\s*,?\s+p(?:age|\.)?\s*(\d+)\s*$", re.IGNORECASE)


def match_sources(answer: str, chunks: list[dict]) -> list[dict]:
    """
    Sources for the answer: retrieved (source, page) pairs the answer cites as
    [file p.N], in first-citation order. Falls back to every retrieved pair
    when nothing cited matches.
    """
    retrieved = list(dict.fromkeys((c["source"], c["page"]) for c in chunks))
    known = set(retrieved)
    cited = []
    for group in _BRACKET.findall(answer):
        for part in group.split(";"):
            m = _CITE.match(part)
            if m and (key := (m.group(1), int(m.group(2)))) in known and key not in cited:
                cited.append(key)
    return [{"source": s, "page": p} for s, p in (cited or retrieved)]


# ── Graph ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    iterations: int
    tool_calls: int
    stop_reason: str
    input_tokens: int
    output_tokens: int
    retrieved: Annotated[list[dict], operator.add]


def _text(content) -> str:
    """Plain text of a message's content. Gemini (with tools bound) streams a first
    chunk of content blocks then plain strings, and merging leaves both kinds in
    one list, so read bare strings as well as text blocks."""
    if isinstance(content, str):
        return content
    return "".join(
        b if isinstance(b, str) else b.get("text", "")
        for b in content
        if isinstance(b, str) or (isinstance(b, dict) and b.get("type") == "text")
    )


def build_graph(llm, toolbox: MCPToolbox, max_iterations: int, max_tool_calls: int):
    llm_with_tools = llm.bind_tools(toolbox.specs)

    async def model(state: AgentState) -> dict:
        write = get_stream_writer()
        if state["tool_calls"] >= max_tool_calls:
            stop = "tool_call_limit"
        elif state["iterations"] >= max_iterations:
            stop = "iteration_limit"
        else:
            stop = "answered"
        forced = stop != "answered"
        runnable = llm if forced else llm_with_tools
        messages = state["messages"]
        if forced:
            messages = messages + [HumanMessage(FORCE_ANSWER_PROMPT)]

        full = None
        async for chunk in runnable.astream(messages):
            if text := _text(chunk.content):
                write(Token(text))
            full = chunk if full is None else full + chunk
        msg = message_chunk_to_message(full) if full is not None else AIMessage("")
        if forced and msg.tool_calls:   # No tools in the final call; drop stray calls
            msg = AIMessage(content=msg.content, usage_metadata=msg.usage_metadata)

        in_tok, out_tok = usage(msg)
        return {
            "messages": [msg],
            "iterations": state["iterations"] + 1,
            "stop_reason": stop,
            "input_tokens": state["input_tokens"] + in_tok,
            "output_tokens": state["output_tokens"] + out_tok,
        }

    async def tools(state: AgentState) -> dict:
        write = get_stream_writer()
        used = state["tool_calls"]
        out, retrieved = [], []
        for tc in state["messages"][-1].tool_calls:
            write(ToolCall(tc["id"], tc["name"], tc["args"]))
            if used >= max_tool_calls:
                text, is_error, payload = f"Skipped: tool-call limit ({max_tool_calls}) reached.", True, None
            else:
                used += 1
                text, is_error, payload = await toolbox.call(tc["name"], tc["args"])
            write(ToolResult(tc["id"], tc["name"], text, is_error))
            if tc["name"] == "search_documents" and isinstance(payload, list):
                retrieved += payload
            out.append(ToolMessage(text, tool_call_id=tc["id"], name=tc["name"],
                                   status="error" if is_error else "success"))
        return {"messages": out, "tool_calls": used, "retrieved": retrieved}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if state["stop_reason"] == "answered" and last.tool_calls else END

    g = StateGraph(AgentState)
    g.add_node("model", model)
    g.add_node("tools", tools)
    g.add_edge(START, "model")
    g.add_conditional_edges("model", route, ["tools", END])
    g.add_edge("tools", "model")
    return g.compile()


def default_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=FLASH_MODEL_ID, temperature=0)


def _history_messages(history: list[dict] | None) -> list[AnyMessage]:
    msgs = []
    for m in (history or [])[-(HISTORY_TURNS * 2):]:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        msgs.append(cls(m["content"]))
    return msgs


async def run_agent(
    question: str,
    history: list[dict] | None = None,
    *,
    toolbox: MCPToolbox,
    llm=None,
    max_iterations: int = MAX_ITERATIONS,
    max_tool_calls: int = MAX_TOOL_CALLS,
) -> AsyncIterator[Event]:
    """Answer one question, yielding ToolCall/ToolResult/Token events then a Final."""
    graph = build_graph(llm or default_llm(), toolbox, max_iterations, max_tool_calls)
    state = {
        "messages": [SystemMessage(SYSTEM_PROMPT), *_history_messages(history), HumanMessage(question)],
        "iterations": 0, "tool_calls": 0, "stop_reason": "answered",
        "input_tokens": 0, "output_tokens": 0, "retrieved": [],
    }
    # Each model call (at most max_iterations + 1) and tool step is one superstep.
    config = {"recursion_limit": 2 * max_iterations + 4}

    final_state = state
    async for mode, data in graph.astream(state, config, stream_mode=["custom", "values"]):
        if mode == "custom":
            yield data
        else:
            final_state = data

    answer = _text(final_state["messages"][-1].content)
    in_tok, out_tok = final_state["input_tokens"], final_state["output_tokens"]
    yield Final(
        answer=answer,
        sources=match_sources(answer, final_state["retrieved"]),
        usage={"input_tokens": in_tok, "output_tokens": out_tok, "usd": round(cost(in_tok, out_tok), 6)},
        iterations=final_state["iterations"],
        tool_calls=final_state["tool_calls"],
        stop_reason=final_state["stop_reason"],
        retrieved=final_state["retrieved"],
    )


# ── Sync runner (Streamlit, tests) ────────────────────────────────────────────

_DONE = object()


class AgentRunner:
    """
    Owns one MCP server subprocess and event loop on a background thread, so
    sync callers (Streamlit) pay the server's startup cost once, not per question.
    """

    def __init__(self, server_script: str = SERVER_SCRIPT):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="agent-mcp")
        self._thread.start()
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._error: BaseException | None = None
        self.toolbox: MCPToolbox | None = None
        self._owner = asyncio.run_coroutine_threadsafe(self._serve(server_script), self._loop)
        self._ready.wait()
        if self._error:
            self.close()
            raise RuntimeError("Could not start the MCP server") from self._error

    async def _serve(self, server_script: str):
        # The session is entered and exited in this one task (anyio requires it);
        # questions run as other tasks on the same loop and share the session.
        self._stop = asyncio.Event()
        try:
            async with MCPToolbox(server_script) as tb:
                self.toolbox = tb
                self._ready.set()
                await self._stop.wait()
        except BaseException as e:
            self._error = e
            self._ready.set()

    def ask(self, question: str, history: list[dict] | None = None, **kwargs) -> Iterator[Event]:
        """Synchronous run_agent: yields events as the background loop produces them."""
        q: queue.Queue = queue.Queue()

        async def pump():
            try:
                async for ev in run_agent(question, history, toolbox=self.toolbox, **kwargs):
                    q.put(ev)
            except BaseException as e:
                q.put(e)
            finally:
                q.put(_DONE)

        asyncio.run_coroutine_threadsafe(pump(), self._loop)
        while (item := q.get()) is not _DONE:
            if isinstance(item, BaseException):
                raise item
            yield item

    def close(self):
        if self._stop is not None and not self._owner.done():
            self._loop.call_soon_threadsafe(self._stop.set)
            try:
                self._owner.result(timeout=10)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


# ── CLI ───────────────────────────────────────────────────────────────────────

async def _cli(question: str):
    async with MCPToolbox() as tb:
        async for ev in run_agent(question, toolbox=tb):
            if isinstance(ev, ToolCall):
                print(f"\n→ {ev.name}({json.dumps(ev.args)})", file=sys.stderr)
            elif isinstance(ev, ToolResult):
                tag = "ERROR" if ev.is_error else "ok"
                preview = ev.content.replace("\n", " ")[:160]
                print(f"  ← [{tag}] {preview}", file=sys.stderr)
            elif isinstance(ev, Token):
                print(ev.text, end="", flush=True)
            elif isinstance(ev, Final):
                print("\n\nSources:")
                for s in ev.sources:
                    print(f"  - {s['source']} p.{s['page']}")
                u = ev.usage
                print(f"\n{ev.iterations} model calls · {ev.tool_calls} tool calls · "
                      f"{u['input_tokens']} in / {u['output_tokens']} out tokens · "
                      f"${u['usd']:.4f} · stop: {ev.stop_reason}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('usage: python agent.py "your question"')
    asyncio.run(_cli(" ".join(sys.argv[1:])))
