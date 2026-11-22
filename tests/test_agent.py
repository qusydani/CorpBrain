"""
Tests for agent.py. No paid API calls: a scripted chat model stands in for
Gemini, while the tools are the real mcp_server.py reached over stdio MCP
(one server for the whole module, via AgentRunner's background session).
The parallelism test uses a fake toolbox so it can count overlapping calls.
"""
import asyncio
import os
import sys

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import (  # noqa: E402
    AgentRunner, Final, Plan, Token, ToolCall, ToolResult, match_sources, run_agent,
)

STRUCTURED_TOKENS = (1, 1)   # Usage reported by each scripted structured call
SIMPLE = {"complexity": "simple", "reason": "one fact"}
COMPLEX = {"complexity": "complex", "reason": "compares two things"}


class ScriptedModel(BaseChatModel):
    """
    Replays a fixed list of AIMessages, one per call; the last one repeats.
    Structured-output calls (router, planner, grader) replay `structured`,
    keyed by schema name, the same way; the router defaults to "simple" and
    a None entry means the model's output failed to parse.
    """

    script: list
    structured: dict = {}
    calls: int = 0
    structured_calls: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        # Same object, so tool-bound and forced-final calls share one script position.
        names = {t["function"]["name"] for t in tools}
        assert {"search_documents", "list_documents"} <= names, names
        return self

    def with_structured_output(self, schema, include_raw=False, **kwargs):
        assert include_raw   # The agent needs raw messages for token usage

        def respond(messages):
            name = schema.__name__
            n = self.structured_calls.count(name)
            self.structured_calls.append(name)
            replies = self.structured.get(name) or ([SIMPLE] if name == "Route" else [None])
            data = replies[min(n, len(replies) - 1)]
            parsed = schema(**data) if data is not None else None
            return {"raw": ai(tokens=STRUCTURED_TOKENS), "parsed": parsed, "parsing_error": None}

        return RunnableLambda(respond)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self.script[min(self.calls, len(self.script) - 1)]
        object.__setattr__(self, "calls", self.calls + 1)
        return ChatResult(generations=[ChatGeneration(message=msg)])


def model_of(script, structured=None):
    # Fresh lists per model: pydantic copies mutable defaults, but be explicit.
    return ScriptedModel(script=script, structured=structured or {}, structured_calls=[])


def call(name, args=None, id="c1"):
    return {"name": name, "args": args or {}, "id": id, "type": "tool_call"}


def ai(text="", tool_calls=None, tokens=(10, 5)):
    return AIMessage(
        content=text,
        tool_calls=tool_calls or [],
        usage_metadata={"input_tokens": tokens[0], "output_tokens": tokens[1],
                        "total_tokens": sum(tokens)},
    )


@pytest.fixture(scope="module")
def runner():
    r = AgentRunner()
    yield r
    r.close()


def ask(runner, model, question="q", **kw):
    return list(runner.ask(question, llm=model, **kw))


# ── match_sources (pure) ──────────────────────────────────────────────────────

CHUNKS = [
    {"source": "a.pdf", "page": 3, "type": "text", "content": "alpha"},
    {"source": "b.pdf", "page": 7, "type": "text", "content": "beta"},
    {"source": "a.pdf", "page": 3, "type": "image_summary", "content": "fig"},
]


def test_match_sources_keeps_cited_chunks_in_citation_order():
    answer = "Beta [b.pdf p.7]. Alpha [a.pdf p.3; b.pdf p.7]. Ghost [c.pdf p.1]."
    assert match_sources(answer, CHUNKS) == [
        {"source": "b.pdf", "page": 7},
        {"source": "a.pdf", "page": 3},
    ]


def test_match_sources_falls_back_to_all_retrieved_when_nothing_cited():
    assert match_sources("No citations here.", CHUNKS) == [
        {"source": "a.pdf", "page": 3},
        {"source": "b.pdf", "page": 7},
    ]


# ── Tool loop (simple path) over real MCP ─────────────────────────────────────

def test_events_flow_plan_tool_call_result_tokens_final(runner):
    model = model_of([
        ai(tool_calls=[call("list_documents")]),
        ai("Here is the answer."),
    ])
    events = ask(runner, model)

    kinds = [type(e) for e in events]
    assert kinds[:3] == [Plan, ToolCall, ToolResult] and kinds[-1] is Final
    assert all(k is Token for k in kinds[3:-1]) and kinds[3:-1]

    plan, tc, tr, final = events[0], events[1], events[2], events[-1]
    assert plan.complexity == "simple" and plan.sub_questions == []
    assert tc.name == "list_documents" and tr.id == tc.id and not tr.is_error
    assert "".join(e.text for e in events if isinstance(e, Token)) == "Here is the answer."
    assert final.answer == "Here is the answer."
    assert final.route == "simple" and final.plan == plan
    # Router + 2 loop calls; usage includes the router's tokens
    assert final.iterations == 3 and final.tool_calls == 1 and final.stop_reason == "answered"
    assert final.usage["input_tokens"] == 21 and final.usage["output_tokens"] == 11
    assert final.usage["usd"] > 0


class ChunkedModel(ScriptedModel):
    """Streams the final answer the way Gemini does with tools bound: a first
    chunk of content blocks (carrying a thought signature), then plain strings."""

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk
        parts = [
            [{"type": "text", "text": "Grants rose ", "extras": {"signature": "abc"}}],
            "to 70% ",
            "[b.pdf p.7].",
        ]
        for p in parts:
            yield ChatGenerationChunk(message=AIMessageChunk(content=p))


def test_final_answer_keeps_every_streamed_chunk(runner):
    events = ask(runner, ChunkedModel(script=[ai()], structured={}, structured_calls=[]))
    streamed = "".join(e.text for e in events if isinstance(e, Token))
    assert streamed == "Grants rose to 70% [b.pdf p.7]."
    assert events[-1].answer == streamed


def test_tool_error_surfaces_as_error_result_and_agent_recovers(runner):
    model = model_of([
        ai(tool_calls=[call("search_documents", {"query": "x", "sources": ["nope.pdf"]})]),
        ai(tool_calls=[call("no_such_tool", id="c2")]),
        ai("Sorry, I couldn't find that."),
    ])
    events = ask(runner, model)
    results = [e for e in events if isinstance(e, ToolResult)]
    assert [r.is_error for r in results] == [True, True]
    assert "nope.pdf" in results[0].content
    assert "no_such_tool" in results[1].content
    assert events[-1].answer == "Sorry, I couldn't find that."


def test_tool_call_cap_forces_a_final_answer(runner):
    model = model_of([
        ai(tool_calls=[call("list_documents", id="a"), call("list_documents", id="b")]),
        ai(tool_calls=[call("list_documents", id="c"), call("list_documents", id="d")]),
        ai("Final after budget."),
    ])
    events = ask(runner, model, max_tool_calls=3, max_iterations=10)
    results = [e for e in events if isinstance(e, ToolResult)]
    final = events[-1]
    assert final.tool_calls == 3                       # 4th call skipped, not executed
    assert results[-1].is_error and "limit" in results[-1].content
    assert final.stop_reason == "tool_call_limit"
    assert final.answer == "Final after budget."


def test_iteration_cap_forces_a_final_answer(runner):
    model = model_of([
        ai(tool_calls=[call("list_documents", id="a")]),
        ai(tool_calls=[call("list_documents", id="b")]),
        ai("Wrapped up."),
    ])
    events = ask(runner, model, max_iterations=2, max_tool_calls=10)
    final = events[-1]
    assert final.stop_reason == "iteration_limit"
    assert final.iterations == 4                       # router + 2 tool rounds + 1 forced answer
    assert final.answer == "Wrapped up."


# ── Planning path ─────────────────────────────────────────────────────────────

def plan_of(*queries):
    return {"sub_questions": [{"question": f"What about {q}?", "query": q} for q in queries]}


def all_ok(n):
    return {"verdicts": [{"id": i, "sufficient": True} for i in range(1, n + 1)]}


def test_simple_question_skips_the_planner(runner):
    model = model_of([ai("Direct answer.")])
    events = ask(runner, model)
    assert model.structured_calls == ["Route"]
    assert not any(isinstance(e, ToolCall) for e in events)
    assert events[-1].route == "simple" and events[-1].answer == "Direct answer."


def test_complex_question_plans_searches_and_synthesizes(runner):
    model = model_of(
        [ai("Combined answer.")],
        {"Route": [COMPLEX], "SearchPlan": [plan_of("grants", "budget")], "Coverage": [all_ok(2)]},
    )
    events = ask(runner, model, question="Compare grants with the budget")
    plan_idx = next(i for i, e in enumerate(events) if isinstance(e, Plan))
    searches = [i for i, e in enumerate(events) if isinstance(e, ToolCall) and e.name == "search_documents"]
    first_token = next(i for i, e in enumerate(events) if isinstance(e, Token))

    plan = events[plan_idx]
    assert plan.complexity == "complex"
    assert [s["query"] for s in plan.sub_questions] == ["grants", "budget"]
    assert len(searches) == 2 and plan_idx < min(searches) and max(searches) < first_token
    assert sum(isinstance(e, Plan) for e in events) == 1

    final = events[-1]
    assert final.route == "complex" and final.answer == "Combined answer."
    assert final.tool_calls == 3                        # list_documents + 2 searches
    assert final.iterations == 4                        # router, planner, grader, synthesizer
    assert final.retrieved                              # Real search results reached the state
    assert model.structured_calls == ["Route", "SearchPlan", "Coverage"]


def test_thin_results_trigger_exactly_one_re_search(runner):
    model = model_of(
        [ai("Answer after retry.")],
        {
            "Route": [COMPLEX],
            "SearchPlan": [{"sub_questions": [
                {"question": "Q1?", "query": "grants"},
                {"question": "Q2?", "query": "budget", "sources": ["nope.pdf"]},   # errors
            ]}],
            "Coverage": [{"verdicts": [{"id": 1, "sufficient": True},
                                       {"id": 2, "sufficient": False, "new_query": "budget measures"}]}],
        },
    )
    events = ask(runner, model)
    calls = [e for e in events if isinstance(e, ToolCall) and e.name == "search_documents"]
    assert sorted(c.id for c in calls[:2]) == ["sub1-try1", "sub2-try1"]
    assert len(calls) == 3
    retry = calls[2]
    assert retry.id == "sub2-try2" and retry.args == {"query": "budget measures"}   # filter dropped
    assert model.structured_calls.count("Coverage") == 1
    assert events[-1].answer == "Answer after retry."


def test_unusable_plan_falls_back_to_the_tool_loop(runner):
    model = model_of([ai("Loop answer.")], {"Route": [COMPLEX], "SearchPlan": [None]})
    events = ask(runner, model)
    final = events[-1]
    assert final.route == "simple" and final.answer == "Loop answer."
    assert final.plan.complexity == "simple"


def test_searches_beyond_the_tool_budget_are_not_run(runner):
    model = model_of(
        [ai("Budgeted.")],
        {"Route": [COMPLEX], "SearchPlan": [plan_of("a", "b", "c", "d")],
         "Coverage": [{"verdicts": [{"id": 1, "sufficient": False, "new_query": "a2"}]}]},
    )
    events = ask(runner, model, max_tool_calls=3)
    final = events[-1]
    assert len(final.plan.sub_questions) == 2           # list_documents + 2 searches = budget
    assert final.tool_calls == 3                        # no retry left in the budget


class SlowToolbox:
    """Fake MCP toolbox: each search sleeps, recording how many overlap."""

    specs = [{"type": "function", "function": {"name": n, "description": "", "parameters": {}}}
             for n in ("search_documents", "list_documents")]

    def __init__(self):
        self.in_flight = self.peak = 0

    async def call(self, name, args):
        if name == "list_documents":
            return "[]", False, []
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(0.2)
        self.in_flight -= 1
        hit = {"rank": 1, "source": "a.pdf", "page": 1, "type": "text", "content": args["query"]}
        return "[]", False, [hit]


def test_sub_question_searches_run_in_parallel():
    model = model_of(
        [ai("Done [a.pdf p.1].")],
        {"Route": [COMPLEX], "SearchPlan": [plan_of("a", "b", "c")], "Coverage": [all_ok(3)]},
    )
    toolbox = SlowToolbox()

    async def run():
        return [ev async for ev in run_agent("q", toolbox=toolbox, llm=model)]

    events = asyncio.run(run())
    assert toolbox.peak == 3
    assert events[-1].sources == [{"source": "a.pdf", "page": 1}]
