"""
Offline tests for the CI eval gate (gate.py): tolerance, baseline building,
comparison and the job summary, plus accuracy-only scoring in eval.py.
No API calls.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gate  # noqa: E402

TEST_SET = [{"question": f"q{i}", "ground_truth": "a",
             "type": "single-hop" if i < 6 else "multi-hop"} for i in range(10)]


def arm(acc, single=None, multi=None, cost=0.01):
    return {
        "aggregate": {"accuracy": acc},
        "aggregate_by_type": {"single-hop": {"accuracy": single if single is not None else acc, "n": 6},
                              "multi-hop":  {"accuracy": multi if multi is not None else acc, "n": 4}},
        "cost": {"total_usd": cost},
        "latency": {"avg_s": 2.0},
    }


def baseline(agent=0.8, rag=0.7, tol=0.1, test_set=TEST_SET):
    return {
        "test_set_sha256": gate.hash_test_set(test_set),
        "n_questions": len(test_set),
        "arms": {"Agent": {"accuracy": agent, "tolerance": tol},
                 "Hybrid+Rerank+Ctx": {"accuracy": rag, "tolerance": tol}},
    }


# ── Tolerance ─────────────────────────────────────────────────────────────────

def test_tolerance_floor_is_one_question():
    assert gate.tolerance([0.8, 0.8, 0.8], 10) == 0.1


def test_tolerance_two_std_rounded_up_to_whole_questions():
    # stdev([0.7, 0.8, 0.9]) = 0.1 → 2σ = 0.2 → exactly 2 questions
    assert gate.tolerance([0.7, 0.8, 0.9], 10) == 0.2
    # stdev([0.7, 0.8, 0.8]) ≈ 0.0577 → 2σ ≈ 0.115 → rounds up to 0.2
    assert gate.tolerance([0.7, 0.8, 0.8], 10) == 0.2


def test_tolerance_single_run_uses_floor():
    assert gate.tolerance([0.6], 20) == 0.05


# ── Hash ──────────────────────────────────────────────────────────────────────

def test_hash_ignores_key_order_but_not_content():
    reordered = [{k: q[k] for k in reversed(list(q))} for q in TEST_SET]
    assert gate.hash_test_set(reordered) == gate.hash_test_set(TEST_SET)
    edited = [dict(q) for q in TEST_SET]
    edited[0]["ground_truth"] = "b"
    assert gate.hash_test_set(edited) != gate.hash_test_set(TEST_SET)


# ── Baseline building ─────────────────────────────────────────────────────────

def test_build_baseline_records_runs_mean_std_tolerance_and_cost():
    runs = [
        {"Agent": arm(0.8, cost=0.02), "Hybrid+Rerank+Ctx": arm(0.7, cost=0.01)},
        {"Agent": arm(0.9, cost=0.03), "Hybrid+Rerank+Ctx": arm(0.7, cost=0.01)},
        {"Agent": arm(0.7, cost=0.02), "Hybrid+Rerank+Ctx": arm(0.7, cost=0.01)},
    ]
    b = gate.build_baseline(runs, TEST_SET)
    a = b["arms"]["Agent"]
    assert a["runs"] == [0.8, 0.9, 0.7]
    assert a["accuracy"] == 0.8
    assert a["std"] == 0.1
    assert a["tolerance"] == 0.2
    assert b["arms"]["Hybrid+Rerank+Ctx"]["tolerance"] == 0.1
    assert set(a["by_type"]) == {"single-hop", "multi-hop"}
    assert b["cost_per_run_usd"]["runs"] == [0.03, 0.04, 0.03]
    assert b["n_questions"] == 10
    assert b["test_set_sha256"] == gate.hash_test_set(TEST_SET)


# ── Comparison ────────────────────────────────────────────────────────────────

def test_passes_at_baseline_and_within_tolerance():
    passed, rows, problems = gate.compare(
        {"Agent": arm(0.7), "Hybrid+Rerank+Ctx": arm(0.7)}, baseline(), TEST_SET)
    assert passed and not problems
    assert [r["passed"] for r in rows] == [True, True]
    assert rows[0]["delta"] == pytest.approx(-0.1)


def test_fails_when_one_arm_drops_beyond_tolerance():
    passed, rows, _ = gate.compare(
        {"Agent": arm(0.6), "Hybrid+Rerank+Ctx": arm(0.9)}, baseline(), TEST_SET)
    assert not passed
    assert {r["label"]: r["passed"] for r in rows} == {"Agent": False, "Hybrid+Rerank+Ctx": True}


def test_threshold_with_non_step_mean():
    # mean 0.8333 - 0.1 = 0.7333: 0.8 passes, 0.7 fails
    b = baseline(agent=0.8333)
    assert gate.compare({"Agent": arm(0.8), "Hybrid+Rerank+Ctx": arm(0.7)}, b, TEST_SET)[0]
    assert not gate.compare({"Agent": arm(0.7), "Hybrid+Rerank+Ctx": arm(0.7)}, b, TEST_SET)[0]


def test_nan_accuracy_fails():
    passed, rows, _ = gate.compare(
        {"Agent": arm(float("nan")), "Hybrid+Rerank+Ctx": arm(0.7)}, baseline(), TEST_SET)
    assert not passed and not rows[0]["passed"]


def test_changed_test_set_fails_as_stale_baseline():
    edited = TEST_SET[:-1]
    passed, rows, problems = gate.compare(
        {"Agent": arm(0.9), "Hybrid+Rerank+Ctx": arm(0.9)}, baseline(), edited)
    assert not passed
    assert all(r["passed"] for r in rows)
    assert "test set changed" in problems[0]


def test_missing_arm_fails():
    passed, _, problems = gate.compare({"Agent": arm(0.9)}, baseline(), TEST_SET)
    assert not passed
    assert "Hybrid+Rerank+Ctx" in problems[0]


# ── Summary ───────────────────────────────────────────────────────────────────

def test_summary_markdown_shows_scores_and_verdict():
    passed, rows, problems = gate.compare(
        {"Agent": arm(0.6, single=0.5, multi=0.75), "Hybrid+Rerank+Ctx": arm(0.7)}, baseline(), TEST_SET)
    md = gate.summary_markdown(passed, rows, problems, 10)
    assert md.startswith("## Eval gate: FAILED")
    assert "| Agent | 0.60 | 0.80 | -0.20 | 0.70 | 0.50 | 0.75 |" in md
    assert "**FAIL**" in md
    assert "Total LLM cost of this run: $0.0200" in md


# ── main() guards ─────────────────────────────────────────────────────────────

def test_main_refuses_non_fixture_corpus(monkeypatch):
    monkeypatch.setattr(gate, "CORPUS", "default")
    assert gate.main() == 2


def test_main_writes_job_summary_and_exit_code(monkeypatch, tmp_path):
    import eval as ev
    monkeypatch.setattr(gate, "CORPUS", "fixture")
    monkeypatch.setattr(ev, "load_test_set", lambda: TEST_SET)
    monkeypatch.setattr(gate, "GATE_RESULTS_PATH", str(tmp_path / "gate_results.json"))
    monkeypatch.setattr(gate, "run_arms",
                        lambda ts, cb=None: {"Agent": arm(0.5), "Hybrid+Rerank+Ctx": arm(0.7)})
    bpath = tmp_path / "baseline.json"
    bpath.write_text(json.dumps(baseline()), encoding="utf-8")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    assert gate.main(baseline_path=str(bpath)) == 1
    assert "Eval gate: FAILED" in summary.read_text(encoding="utf-8")
    assert json.loads((tmp_path / "gate_results.json").read_text())["passed"] is False


# ── Accuracy-only scoring ─────────────────────────────────────────────────────

def test_score_samples_without_ragas(monkeypatch):
    import eval as ev
    from types import SimpleNamespace

    def boom(*a, **k):
        raise AssertionError("RAGAS must not run")
    monkeypatch.setattr(ev, "_run_ragas", boom)
    monkeypatch.setattr(ev, "_judge_accuracy", lambda llm, q, gt, a: (1 if a == "right" else 0, 10, 2))
    samples = [
        {"question": "q1", "type": "single-hop", "ground_truth": "g", "answer": "right",
         "contexts": [], "latency_s": 1.0, "tokens": {"answer_in": 100, "answer_out": 10}},
        {"question": "q2", "type": "multi-hop", "ground_truth": "g", "answer": "wrong",
         "contexts": [], "latency_s": 3.0, "tokens": {"answer_in": 100, "answer_out": 10}},
    ]
    out = ev._score_samples(samples, "X", SimpleNamespace(), None, None, ragas=False)
    assert out["aggregate"] == {"accuracy": 0.5}
    assert out["aggregate_by_type"]["single-hop"]["accuracy"] == 1
    assert out["aggregate_by_type"]["multi-hop"]["accuracy"] == 0
    assert out["cost"]["breakdown_usd"]["ragas_judge"] == 0
    assert out["cost"]["total_usd"] > 0
