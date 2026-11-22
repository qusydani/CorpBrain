"""
Tests for agent.py. No paid API calls: a scripted chat model stands in for
Gemini, while the tools are the real mcp_server.py reached over stdio MCP
(one server for the whole module, via AgentRunner's background session).
"""
import os
import sys

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import (  # noqa: E402
    AgentRunner, Final, Token, ToolCall, ToolResult, match_sources,
)


class ScriptedModel(BaseChatModel):
    """Replays a fixed list of AIMessages, one per call; the last one repeats."""

    script: list
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        # Same object, so tool-bound and forced-final calls share one script position.
        names = {t["function"]["name"] for t in tools}
        assert {"search_documents", "list_documents"} <= names, names
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self.script[min(self.calls, len(self.script) - 1)]
        object.__setattr__(self, "calls", self.calls + 1)
        return ChatResult(generations=[ChatGeneration(message=msg)])


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


# ── Agent loop over real MCP ──────────────────────────────────────────────────

def test_events_flow_tool_call_result_tokens_final(runner):
    model = ScriptedModel(script=[
        ai(tool_calls=[call("list_documents")]),
        ai("Here is the answer."),
    ])
    events = ask(runner, model)

    kinds = [type(e) for e in events]
    assert kinds[0] is ToolCall and kinds[1] is ToolResult and kinds[-1] is Final
    assert all(k is Token for k in kinds[2:-1]) and kinds[2:-1]

    tc, tr, final = events[0], events[1], events[-1]
    assert tc.name == "list_documents" and tr.id == tc.id and not tr.is_error
    assert "".join(e.text for e in events if isinstance(e, Token)) == "Here is the answer."
    assert final.answer == "Here is the answer."
    assert final.iterations == 2 and final.tool_calls == 1 and final.stop_reason == "answered"
    assert final.usage["input_tokens"] == 20 and final.usage["output_tokens"] == 10
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
    events = ask(runner, ChunkedModel(script=[ai()]))
    streamed = "".join(e.text for e in events if isinstance(e, Token))
    assert streamed == "Grants rose to 70% [b.pdf p.7]."
    assert events[-1].answer == streamed


def test_tool_error_surfaces_as_error_result_and_agent_recovers(runner):
    model = ScriptedModel(script=[
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
    model = ScriptedModel(script=[
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
    model = ScriptedModel(script=[
        ai(tool_calls=[call("list_documents", id="a")]),
        ai(tool_calls=[call("list_documents", id="b")]),
        ai("Wrapped up."),
    ])
    events = ask(runner, model, max_iterations=2, max_tool_calls=10)
    final = events[-1]
    assert final.stop_reason == "iteration_limit"
    assert final.iterations == 3                       # 2 tool rounds + 1 forced answer
    assert final.answer == "Wrapped up."
