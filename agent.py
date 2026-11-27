"""
agent.py — a tool-calling agent over CorpBrain's MCP server.

    python agent.py "What does the handbook say about PPE?"

A LangGraph agent on Gemini Flash. Tools come from mcp_server.py over stdio
MCP, never by importing retrieval.

    run_agent(question, history, toolbox=...)   async stream of events
    AgentRunner().ask(question, history)        the same stream, synchronously,
                                                over one long-lived MCP session

Graph:

    START → router ─simple──→ model ⇄ tools ─────────────────────────────→ END
                  └complex─→ planner ─→ search ×N ─→ grade ─ok───→ synthesize → END
                                (parallel)   └thin──→ research ×M ─┘
                               (no plan) ──→ model

    router      one structured Flash call: is this simple or complex?
    model/tools the tool-calling loop; the model searches until it can answer
    planner     lists documents, then splits the question into ≤ MAX_SUB_QUESTIONS
                sub-questions, each with a search query (and optional file filter)
    search      one search_documents call per sub-question, all in parallel
    grade       one structured call marks each sub-question's results ok or thin
    research    one retry per thin sub-question: a rewritten query, no file filter
    synthesize  fetches page images for up to MAX_SYNTH_IMAGES image_summary pages
                in the findings, then one call without tools writes the cited
                answer from all findings and images

Page images: get_page_image returns an image block. Gemini doesn't reliably take
images inside a tool message, so each image goes to it as a HumanMessage right
after the tool results. Every image the agent looked at ends up in Final.images.

Events, in order: one Plan once the route (and for complex questions the
sub-questions) is known, ToolCall / ToolResult pairs as tools run, Token
chunks of the answer, then one Final with the answer, cited sources and token
usage. The UI and the eval both consume this stream.

Caps: at most MAX_ITERATIONS tool-enabled model calls in the loop and
MAX_TOOL_CALLS tool executions overall. In the loop, hitting either triggers
one last model call without tools, so the agent still answers from what it
has; Final.stop_reason records which cap hit. On the complex path, searches
beyond the tool budget are not run.
"""
import asyncio
import base64
import json
import operator
import os
import queue
import re
import sys
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Annotated, Any, AsyncIterator, Iterator, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage, message_chunk_to_message,
)
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from pydantic import BaseModel, Field

from observability import agent_trace, flush as flush_traces, tool_span
from pricing import FLASH_MODEL_ID, cost, usage

ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))

SERVER_SCRIPT = os.path.join(ROOT, "mcp_server.py")
MAX_ITERATIONS = 6
MAX_TOOL_CALLS = 8
HISTORY_TURNS = 3   # Prior user/assistant turns passed to the model
MAX_SUB_QUESTIONS = 4
GRADE_SNIPPET_CHARS = 500   # Per chunk, in the grader's view of the results
MAX_SYNTH_IMAGES = 2        # Page images the complex path attaches for synthesis

CITATION_RULES = (
    "Cite every factual claim inline as [<source> p.<page>] using the source and page "
    "fields of the search results, e.g. [handbook.pdf p.12]; separate several with "
    "semicolons, e.g. [a.pdf p.3; b.pdf p.7]. If the documents don't contain the answer, say so."
)

SYSTEM_PROMPT = (
    "You answer questions about the user's PDF knowledge base using the tools. "
    "Always search before answering; never answer from general knowledge. "
    "Call list_documents when you need exact file names, e.g. to restrict search_documents "
    "with `sources`. If the question has several parts, search for each part. "
    "When a result's type is image_summary and the question is about a figure, diagram, "
    "chart or table, call get_page_image to look at the page itself. When a chunk is "
    "cut off mid-thought, call get_page for the page's full text.\n\n"
    + CITATION_RULES
)
FORCE_ANSWER_PROMPT = (
    "Tool budget exhausted. Answer now using only the search results above, with "
    "citations. If they are not enough, say what is missing."
)
ROUTER_PROMPT = (
    "Classify the user's latest question about their PDF knowledge base.\n"
    "simple: one fact or one topic, answerable with one or two searches.\n"
    "complex: needs several separate lookups combined — comparisons, several entities, "
    "documents or time periods, multi-part questions, or reasoning across sources.\n"
    "Use the conversation to resolve follow-ups. Give a one-sentence reason."
)
PLANNER_PROMPT = (
    "Split the user's latest question into at most {n} self-contained sub-questions "
    "that together answer it. Each needs its own search: give a concise keyword-rich "
    "search query, and set `sources` to exact file names only when the sub-question "
    "clearly targets specific documents (otherwise leave it empty).\n\n"
    "Available documents:\n{documents}"
)
GRADER_PROMPT = (
    "For each sub-question, decide whether its search results contain the information "
    "needed to answer it. Mark sufficient=false when the results are missing, off-topic "
    "or errored, and then give new_query: a differently worded search query more likely "
    "to find it. Return one verdict per sub-question id."
)
SYNTH_PROMPT = (
    "Answer the user's question using only the research findings below, which are "
    "search results for sub-questions of it. Combine them into one coherent answer; "
    "don't answer from general knowledge.\n\n" + CITATION_RULES
)


# ── Structured outputs ────────────────────────────────────────────────────────

class Route(BaseModel):
    complexity: Literal["simple", "complex"]
    reason: str = ""


class SubQuestion(BaseModel):
    question: str = Field(description="A self-contained sub-question.")
    query: str = Field(description="Search query for it.")
    sources: list[str] | None = Field(default=None, description="Exact file names to restrict to, or empty.")


class SearchPlan(BaseModel):
    sub_questions: list[SubQuestion]


class Verdict(BaseModel):
    id: int
    sufficient: bool
    new_query: str | None = None


class Coverage(BaseModel):
    verdicts: list[Verdict]


# ── Events ────────────────────────────────────────────────────────────────────

@dataclass
class Plan:
    complexity: str       # "simple" | "complex"
    reason: str
    sub_questions: list = field(default_factory=list)   # [{"id", "question", "query", "sources"}]


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
    iterations: int       # Every model call: router, planner, grader, loop, synthesizer
    tool_calls: int       # Tool executions (skipped-over-cap calls excluded)
    stop_reason: str      # "answered" | "iteration_limit" | "tool_call_limit"
    retrieved: list = field(default_factory=list)   # Every chunk search_documents returned
    route: str = "simple"                           # Path taken: "simple" | "complex"
    plan: Plan | None = None
    images: list = field(default_factory=list)      # [{"source", "page", "mime_type", "data": bytes}]


Event = Plan | ToolCall | ToolResult | Token | Final


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
        """(text for the model, is_error, parsed payload or None). Never raises.

        A result with image blocks has payload {"text", "images": [{"mime_type",
        "data" (base64)}]}, where text is whatever text blocks came with them.
        Each call is a Langfuse tool span when tracing is on (text only, no image data)."""
        with tool_span(name, args) as span:
            text, is_error, payload = await self._call(name, args)
            span.update(output=text, level="ERROR" if is_error else "DEFAULT",
                        status_message=text[:200] if is_error else None)
        return text, is_error, payload

    async def _call(self, name: str, args: dict) -> tuple[str, bool, Any]:
        if name not in self.names:
            return f"Unknown tool {name!r}. Available tools: {sorted(self.names)}", True, None
        try:
            res = await self.session.call_tool(name, args)
        except Exception as e:  # transport failure, server crash, bad args
            return f"Tool {name} failed: {type(e).__name__}: {e}", True, None
        text = "\n".join(getattr(c, "text", "") for c in res.content)
        if res.is_error:
            return text or f"Tool {name} returned an error.", True, None
        images = [{"mime_type": c.mime_type, "data": c.data}
                  for c in res.content if getattr(c, "type", None) == "image"]
        if images:
            return text or f"{len(images)} image(s)", False, {"text": text, "images": images}
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


# ── Page images ───────────────────────────────────────────────────────────────

def _page_images(args: dict, payload: Any) -> list[dict]:
    """A get_page_image payload as [{"source", "page", "mime_type", "data" (base64)}]."""
    if not isinstance(payload, dict):
        return []
    return [{"source": args.get("source"), "page": args.get("page"), **img}
            for img in payload.get("images", [])]


def _image_message(images: list[dict]) -> HumanMessage:
    """Page images as one user message, so Gemini sees them as image input."""
    content = []
    for img in images:
        content.append({"type": "text", "text": f"Page image of {img['source']} p.{img['page']}:"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:{img['mime_type']};base64,{img['data']}"}})
    return HumanMessage(content=content)


def _final_images(images: list[dict]) -> list[dict]:
    """One decoded image per (source, page), in the order the agent looked."""
    out = {}
    for img in images:
        out.setdefault((img["source"], img["page"]), {
            "source": img["source"], "page": img["page"], "mime_type": img["mime_type"],
            "data": base64.b64decode(img["data"]),
        })
    return list(out.values())


# ── Graph ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Counters are summed deltas, since parallel search nodes update them at once.
    messages: Annotated[list[AnyMessage], add_messages]
    iterations: Annotated[int, operator.add]      # Every model call
    loop_calls: Annotated[int, operator.add]      # Model calls in the tool loop (the iteration cap)
    tool_calls: Annotated[int, operator.add]
    stop_reason: str
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]
    retrieved: Annotated[list[dict], operator.add]
    route: str
    plan: Plan | None
    findings: Annotated[list[dict], operator.add]  # {"id", "question", "query", "attempt", "results", "error"}
    retries: list[dict]                            # Sub-questions to search again, with new queries
    images: Annotated[list[dict], operator.add]    # Page images seen: {"source", "page", "mime_type", "data"}


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


def _conversation(state: AgentState) -> list[AnyMessage]:
    """History and the question, without the loop's system prompt."""
    return [m for m in state["messages"] if not isinstance(m, SystemMessage)]


def _format_results(results: list[dict], snippet: int | None = None) -> str:
    lines = []
    for r in results:
        text = r["content"] if snippet is None else r["content"][:snippet]
        lines.append(f"[{r['source']} p.{r['page']}] ({r['type']}) {text}")
    return "\n".join(lines) or "(no results)"


def _merged_findings(findings: list[dict]) -> list[dict]:
    """One entry per sub-question: every attempt's results, deduplicated."""
    by_id: dict[int, dict] = {}
    for f in sorted(findings, key=lambda f: (f["id"], f["attempt"])):
        entry = by_id.setdefault(f["id"], {"id": f["id"], "question": f["question"],
                                           "results": [], "errors": []})
        seen = {(r["source"], r["page"], r["content"]) for r in entry["results"]}
        entry["results"] += [r for r in f["results"] if (r["source"], r["page"], r["content"]) not in seen]
        if f["error"]:
            entry["errors"].append(f["error"])
    return list(by_id.values())


def build_graph(llm, toolbox: MCPToolbox, max_iterations: int, max_tool_calls: int):
    llm_with_tools = llm.bind_tools(toolbox.specs)

    async def structured(schema, messages) -> tuple[Any, dict]:
        """One structured-output call: (parsed or None, counter updates)."""
        out = await llm.with_structured_output(schema, include_raw=True).ainvoke(messages)
        in_tok, out_tok = usage(out.get("raw"))
        return out.get("parsed"), {"iterations": 1, "input_tokens": in_tok, "output_tokens": out_tok}

    # ── Router ──

    async def router(state: AgentState) -> dict:
        parsed, counts = await structured(Route, [SystemMessage(ROUTER_PROMPT), *_conversation(state)])
        route = parsed.complexity if parsed else "simple"
        update = {**counts, "route": route}
        if route == "simple":
            plan = Plan("simple", parsed.reason if parsed else "")
            get_stream_writer()(plan)
            update["plan"] = plan
        return update

    def after_router(state: AgentState) -> str:
        return "planner" if state["route"] == "complex" else "model"

    # ── Complex path ──

    async def planner(state: AgentState) -> dict:
        write = get_stream_writer()
        write(ToolCall("plan-docs", "list_documents", {}))
        text, is_error, _ = await toolbox.call("list_documents", {})
        write(ToolResult("plan-docs", "list_documents", text, is_error))

        prompt = PLANNER_PROMPT.format(n=MAX_SUB_QUESTIONS, documents=text)
        parsed, counts = await structured(SearchPlan, [SystemMessage(prompt), *_conversation(state)])
        budget = max(0, max_tool_calls - 1)   # list_documents used one
        subs = (parsed.sub_questions if parsed else [])[:min(MAX_SUB_QUESTIONS, budget)]
        update = {**counts, "tool_calls": 1}
        if not subs:   # No usable plan: fall back to the tool loop
            plan = Plan("simple", "Planner returned no sub-questions; searching directly.")
            update["route"] = "simple"
        else:
            plan = Plan("complex", "", [
                {"id": i, "question": s.question, "query": s.query, "sources": s.sources or None}
                for i, s in enumerate(subs, start=1)
            ])
        write(plan)
        update["plan"] = plan
        return update

    def after_planner(state: AgentState):
        if state["route"] != "complex":
            return "model"
        return [Send("search", {"sub": s, "attempt": 1}) for s in state["plan"].sub_questions]

    async def search(task: dict) -> dict:
        """One sub-question's search. Runs as a Send task, so it sees only `task`."""
        write = get_stream_writer()
        sub, attempt = task["sub"], task["attempt"]
        args = {"query": sub["query"]}
        if sub.get("sources"):
            args["sources"] = sub["sources"]
        call_id = f"sub{sub['id']}-try{attempt}"
        write(ToolCall(call_id, "search_documents", args))
        text, is_error, payload = await toolbox.call("search_documents", args)
        write(ToolResult(call_id, "search_documents", text, is_error))
        results = payload if isinstance(payload, list) else []
        return {
            "tool_calls": 1,
            "retrieved": results,
            "findings": [{"id": sub["id"], "question": sub["question"], "query": sub["query"],
                          "attempt": attempt, "results": results, "error": text if is_error else None}],
        }

    async def grade(state: AgentState) -> dict:
        blocks = []
        for f in _merged_findings(state["findings"]):
            errors = "".join(f"\nSearch error: {e}" for e in f["errors"])
            blocks.append(f"## Sub-question {f['id']}: {f['question']}{errors}\n"
                          f"{_format_results(f['results'], GRADE_SNIPPET_CHARS)}")
        parsed, counts = await structured(Coverage, [SystemMessage(GRADER_PROMPT), HumanMessage("\n\n".join(blocks))])
        thin = {v.id: v.new_query for v in (parsed.verdicts if parsed else [])
                if not v.sufficient and v.new_query}
        # Retry with the new query and no file filter, since a too-narrow filter is a common cause.
        retries = [{**s, "query": thin[s["id"]], "sources": None}
                   for s in state["plan"].sub_questions if s["id"] in thin]
        return {**counts, "retries": retries[:max(0, max_tool_calls - state["tool_calls"])]}

    def after_grade(state: AgentState):
        if not state["retries"]:
            return "synthesize"
        return [Send("research", {"sub": s, "attempt": 2}) for s in state["retries"]]

    async def fetch_image(call_id: str, source: str, page: int) -> list[dict]:
        write = get_stream_writer()
        args = {"source": source, "page": page}
        write(ToolCall(call_id, "get_page_image", args))
        text, is_error, payload = await toolbox.call("get_page_image", args)
        write(ToolResult(call_id, "get_page_image", text, is_error))
        return [] if is_error else _page_images(args, payload)

    async def synthesize(state: AgentState) -> dict:
        write = get_stream_writer()
        findings = _merged_findings(state["findings"])
        blocks = [f"## Sub-question {f['id']}: {f['question']}\n{_format_results(f['results'])}"
                  for f in findings]

        # Look at the pages behind the image summaries found, within the tool budget.
        pages = list(dict.fromkeys((r["source"], r["page"]) for f in findings for r in f["results"]
                                   if r.get("type") == "image_summary"))
        pages = pages[:min(MAX_SYNTH_IMAGES, max(0, max_tool_calls - state["tool_calls"]))]
        fetched = await asyncio.gather(*(fetch_image(f"synth-img{i}", src, pg)
                                         for i, (src, pg) in enumerate(pages, start=1)))
        images = [img for imgs in fetched for img in imgs]

        conversation = _conversation(state)
        question = conversation[-1].content
        messages = [SystemMessage(SYNTH_PROMPT), *conversation[:-1],
                    HumanMessage(f"Question: {question}\n\nResearch findings:\n\n" + "\n\n".join(blocks))]
        if images:
            messages.append(_image_message(images))
        full = None
        async for chunk in llm.astream(messages):
            if text := _text(chunk.content):
                write(Token(text))
            full = chunk if full is None else full + chunk
        msg = message_chunk_to_message(full) if full is not None else AIMessage("")
        if msg.tool_calls:
            msg = AIMessage(content=msg.content, usage_metadata=msg.usage_metadata)
        in_tok, out_tok = usage(msg)
        return {"messages": [msg], "iterations": 1, "stop_reason": "answered",
                "input_tokens": in_tok, "output_tokens": out_tok,
                "tool_calls": len(pages), "images": images}

    # ── Tool loop (simple path) ──

    async def model(state: AgentState) -> dict:
        write = get_stream_writer()
        if state["tool_calls"] >= max_tool_calls:
            stop = "tool_call_limit"
        elif state["loop_calls"] >= max_iterations:
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
            "iterations": 1,
            "loop_calls": 1,
            "stop_reason": stop,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        }

    async def tools(state: AgentState) -> dict:
        write = get_stream_writer()
        start = used = state["tool_calls"]
        out, retrieved, images = [], [], []
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
            new = [] if is_error else _page_images(tc["args"], payload)
            if new:
                images += new
                text = f"{text} (attached below)"
            out.append(ToolMessage(text, tool_call_id=tc["id"], name=tc["name"],
                                   status="error" if is_error else "success"))
        # Images go after every tool message: Gemini wants all responses to a turn's calls together.
        if images:
            out.append(_image_message(images))
        return {"messages": out, "tool_calls": used - start, "retrieved": retrieved, "images": images}

    def after_model(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if state["stop_reason"] == "answered" and last.tool_calls else END

    g = StateGraph(AgentState)
    g.add_node("router", router)
    g.add_node("planner", planner)
    g.add_node("search", search)
    g.add_node("grade", grade)
    g.add_node("research", search)
    g.add_node("synthesize", synthesize)
    g.add_node("model", model)
    g.add_node("tools", tools)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", after_router, ["planner", "model"])
    g.add_conditional_edges("planner", after_planner, ["search", "model"])
    g.add_edge("search", "grade")
    g.add_conditional_edges("grade", after_grade, ["research", "synthesize"])
    g.add_edge("research", "synthesize")
    g.add_edge("synthesize", END)
    g.add_conditional_edges("model", after_model, ["tools", END])
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
    session_id: str | None = None,
) -> AsyncIterator[Event]:
    """Answer one question, yielding Plan, ToolCall/ToolResult, Token events, then a Final.

    With Langfuse on (see observability.py) the run is one trace: LLM calls as
    generations, MCP tool calls as tool spans, grouped by session_id if given."""
    graph = build_graph(llm or default_llm(), toolbox, max_iterations, max_tool_calls)
    state = {
        "messages": [SystemMessage(SYSTEM_PROMPT), *_history_messages(history), HumanMessage(question)],
        "iterations": 0, "loop_calls": 0, "tool_calls": 0, "stop_reason": "answered",
        "input_tokens": 0, "output_tokens": 0, "retrieved": [],
        "route": "simple", "plan": None, "findings": [], "retries": [], "images": [],
    }
    # Supersteps: router, planner, then the loop's model calls (at most
    # max_iterations + 1) and tool steps; the complex path needs only six.
    config = {"recursion_limit": 2 * max_iterations + 6}

    with agent_trace(question, session_id) as trace:
        config["callbacks"] = trace.callbacks
        final_state = state
        async for mode, data in graph.astream(state, config, stream_mode=["custom", "values"]):
            if mode == "custom":
                yield data
            else:
                final_state = data

        answer = _text(final_state["messages"][-1].content)
        in_tok, out_tok = final_state["input_tokens"], final_state["output_tokens"]
        trace.update(output=answer, metadata={
            "route": final_state["route"], "stop_reason": final_state["stop_reason"],
            "tool_calls": str(final_state["tool_calls"]), "iterations": str(final_state["iterations"]),
        })
    yield Final(
        answer=answer,
        sources=match_sources(answer, final_state["retrieved"]),
        usage={"input_tokens": in_tok, "output_tokens": out_tok, "usd": round(cost(in_tok, out_tok), 6)},
        iterations=final_state["iterations"],
        tool_calls=final_state["tool_calls"],
        stop_reason=final_state["stop_reason"],
        retrieved=final_state["retrieved"],
        route=final_state["route"],
        plan=final_state["plan"],
        images=_final_images(final_state["images"]),
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
            if isinstance(ev, Plan):
                print(f"[{ev.complexity}] {ev.reason}", file=sys.stderr)
                for s in ev.sub_questions:
                    print(f"  {s['id']}. {s['question']}  (search: {s['query']!r})", file=sys.stderr)
            elif isinstance(ev, ToolCall):
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
                for img in ev.images:
                    print(f"  [viewed page image] {img['source']} p.{img['page']}")
                u = ev.usage
                print(f"\n{ev.route} route · {ev.iterations} model calls · {ev.tool_calls} tool calls · "
                      f"{u['input_tokens']} in / {u['output_tokens']} out tokens · "
                      f"${u['usd']:.4f} · stop: {ev.stop_reason}", file=sys.stderr)
    flush_traces()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('usage: python agent.py "your question"')
    asyncio.run(_cli(" ".join(sys.argv[1:])))
