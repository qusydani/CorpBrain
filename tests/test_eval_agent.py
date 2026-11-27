"""
Offline tests for the agent eval arm: how agent runs become eval samples, the
balanced dev subset, and checkpoint/resume. A fake runner stands in for
agent.AgentRunner; RAGAS, the judge and the LLMs are stubbed. No API calls.
"""
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval as ev  # noqa: E402
from agent import Final, Token, ToolCall  # noqa: E402


def chunk(content, source="a.pdf", page=1):
    return {"content": content, "source": source, "page": page, "type": "text"}


def final(answer="ans", retrieved=(), tokens=(100, 20), route="complex"):
    return Final(answer=answer, sources=[], usage={"input_tokens": tokens[0], "output_tokens": tokens[1],
                                                   "usd": 0.0},
                 iterations=4, tool_calls=3, stop_reason="answered",
                 retrieved=list(retrieved), route=route)


class FakeRunner:
    """Replays one scripted event list (or exception) per question, in order."""

    def __init__(self, *scripts):
        self.scripts = list(scripts)
        self.calls = []

    def ask(self, question, history=None, **kwargs):
        self.calls.append((question, kwargs))
        script = self.scripts.pop(0)
        if isinstance(script, Exception):
            raise script
        yield from script


Q = [{"question": "q1", "ground_truth": "g1", "type": "single-hop"},
     {"question": "q2", "ground_truth": "g2", "type": "multi-hop"}]


def test_agent_sample_contexts_are_the_union_of_every_search():
    # Two sub-query searches overlapping on chunk "b": contexts keep each chunk once, in order.
    retrieved = [chunk("a"), chunk("b"), chunk("b"), chunk("c", "b.pdf", 2)]
    runner = FakeRunner([ToolCall("1", "search_documents", {}), Token("ans"), final(retrieved=retrieved)])
    [s] = ev._agent_samples(Q[:1], ev.AgentConfig(), runner)
    assert s["contexts"] == ["a", "b", "c"]
    assert s["answer"] == "ans"


def test_agent_sample_tokens_come_from_the_agents_usage():
    runner = FakeRunner([final(tokens=(1234, 56))])
    [s] = ev._agent_samples(Q[:1], ev.AgentConfig(), runner)
    assert s["tokens"] == {"answer_in": 1234, "answer_out": 56}
    assert s["agent"] == {"route": "complex", "tool_calls": 3, "iterations": 4,
                          "stop_reason": "answered", "error": None}
    assert s["latency_s"] >= 0


def test_agent_caps_and_session_are_passed_to_the_runner():
    runner = FakeRunner([final()])
    ev._agent_samples(Q[:1], ev.AgentConfig(max_iterations=2, max_tool_calls=3), runner, session_id="s")
    assert runner.calls == [("q1", {"max_iterations": 2, "max_tool_calls": 3, "session_id": "s"})]


def test_a_failed_agent_run_is_recorded_and_the_rest_still_run():
    runner = FakeRunner(RuntimeError("MCP died"), [final(answer="ok")])
    s1, s2 = ev._agent_samples(Q, ev.AgentConfig(), runner)
    assert s1["answer"] == "" and s1["contexts"] == []
    assert s1["agent"]["error"] == "RuntimeError: MCP died"
    assert s1["tokens"] == {"answer_in": 0, "answer_out": 0}
    assert s2["answer"] == "ok" and s2["agent"]["error"] is None


def test_dev_subset_is_balanced_across_types():
    ts = [{"question": f"s{i}"} for i in range(5)] + \
         [{"question": f"m{i}", "type": "multi-hop"} for i in range(5)]
    assert [q["question"] for q in ev.dev_subset(ts, 5)] == ["s0", "s1", "s2", "m0", "m1"]


def test_dev_subset_tops_up_when_one_type_runs_short():
    ts = [{"question": "s0"}] + [{"question": f"m{i}", "type": "multi-hop"} for i in range(5)]
    assert [q["question"] for q in ev.dev_subset(ts, 4)] == ["s0", "m0", "m1", "m2"]
    assert len(ev.dev_subset(ts[:1], 4)) == 1


# ── run_full_evaluation with everything external stubbed ─────────────────────

@pytest.fixture
def stubbed(monkeypatch):
    import ragas.embeddings
    import ragas.llms

    monkeypatch.setattr(ev, "ChatGoogleGenerativeAI", lambda **kw: object())
    monkeypatch.setattr(ev, "get_embedding_model", lambda: object())
    monkeypatch.setattr(ragas.llms, "LangchainLLMWrapper", lambda x: x)
    monkeypatch.setattr(ragas.embeddings, "LangchainEmbeddingsWrapper", lambda x: x)
    judged, timeouts = [], []

    def judge(llm, question, ground_truth, answer):
        judged.append(question)
        return 1, 10, 2

    def ragas(samples, llm, emb, timeout=ev.RAGAS_TIMEOUT):
        timeouts.append(timeout)
        rows = [{"question": s["question"], "faithfulness": 0.5} for s in samples]
        return {"faithfulness": 0.5}, rows, SimpleNamespace(input_tokens=40, output_tokens=8)

    monkeypatch.setattr(ev, "_judge_accuracy", judge)
    monkeypatch.setattr(ev, "_run_ragas", ragas)
    return SimpleNamespace(judged=judged, timeouts=timeouts)


def test_agent_arm_result_has_metrics_cost_latency_and_types(tmp_path, stubbed):
    out = tmp_path / "r.json"
    runner = FakeRunner([final(tokens=(1000, 100))], RuntimeError("boom"))
    res = ev.run_full_evaluation([ev.AgentConfig()], Q, results_path=str(out), agent_runner=runner)

    [cfg] = res["configs"]
    assert cfg["label"] == "Agent" and cfg["config"]["kind"] == "agent"
    assert cfg["aggregate"] == {"faithfulness": 0.5, "accuracy": 0.5}   # failed run scores 0
    assert stubbed.judged == ["q1"]                                      # failed run isn't judged
    assert stubbed.timeouts == [ev.AGENT_RAGAS_TIMEOUT]                  # room for its larger contexts
    assert set(cfg["aggregate_by_type"]) == {"single-hop", "multi-hop"}
    assert cfg["cost"]["breakdown_usd"]["answer"] == round(ev._cost(1000, 100), 6)
    assert cfg["cost"]["answer_per_question_usd"] == round(ev._cost(1000, 100) / 2, 6)
    assert set(cfg["latency"]) == {"avg_s", "p50_s", "max_s"}
    row1, row2 = cfg["per_question"]
    assert row1["route"] == "complex" and row1["error"] is None
    assert row2["error"] == "RuntimeError: boom" and row2["accuracy"] == 0
    assert json.loads(out.read_text(encoding="utf-8"))["configs"][0]["label"] == "Agent"


def test_resume_skips_a_finished_agent_arm(tmp_path, stubbed):
    out = tmp_path / "r.json"
    ev.run_full_evaluation([ev.AgentConfig()], Q, results_path=str(out),
                           agent_runner=FakeRunner([final()], [final()]))
    untouched = FakeRunner()   # Any ask() would fail: no scripts left
    res = ev.run_full_evaluation([ev.AgentConfig()], Q, results_path=str(out), agent_runner=untouched)
    assert untouched.calls == []
    assert [c["label"] for c in res["configs"]] == ["Agent"]
