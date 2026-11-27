"""
gate.py — the CI eval gate: fail a PR when answer accuracy drops below the
committed baseline.

    CORPBRAIN_CORPUS=fixture python eval.py gate                              # check
    CORPBRAIN_CORPUS=fixture python eval.py gate --update-baseline --runs 3   # re-baseline

Two arms run on the frozen fixture test set: the agent and the best plain-RAG
config (GATE_RAG_CONFIG). Only the binary accuracy judge scores them. RAGAS is
skipped, which keeps a run to a few cents and a few minutes.

Baseline (tests/fixtures/eval_baseline.json, committed)
    Each arm's accuracy from several runs: every run, the mean, the standard
    deviation and the tolerance. It also holds a hash of the test set, so editing
    the questions without re-baselining fails the gate instead of comparing
    against numbers from different questions.

Tolerance, per arm
    max(2 × run-to-run standard deviation, one question), rounded up to a whole
    number of questions. Accuracy on n questions moves in steps of 1/n, so a
    smaller tolerance would fail on noise. The gate fails an arm when
    accuracy < baseline mean − tolerance.
"""
import hashlib
import json
import math
import os
import statistics
from datetime import datetime

from corpus import CORPUS, FIXTURE_ROOT, TEST_SET_PATH

BASELINE_PATH = os.path.join(FIXTURE_ROOT, "eval_baseline.json")
GATE_RESULTS_PATH = os.path.join("eval_data", "gate_results.json")

# Best plain-RAG config on the full-corpus sweep (highest overall score).
GATE_RAG_CONFIG = {"strategy": "hybrid", "reranking": True, "contextual": True}

TOLERANCE_STD_MULTIPLIER = 2


# ── Pure logic (unit tested, no API calls) ────────────────────────────────────

def hash_test_set(test_set: list) -> str:
    """sha256 of the test set's canonical JSON, so line endings and key order don't matter."""
    canon = json.dumps(test_set, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def tolerance(runs: list, n_questions: int) -> float:
    """max(2 × sample std of `runs`, 1/n), rounded up to a multiple of 1/n."""
    std = statistics.stdev(runs) if len(runs) > 1 else 0.0
    raw = max(TOLERANCE_STD_MULTIPLIER * std, 1 / n_questions)
    steps = math.ceil(raw * n_questions - 1e-9)
    return round(steps / n_questions, 4)


def build_baseline(run_results: list, test_set: list) -> dict:
    """
    run_results: one {label: arm_result} per run, arm_result as returned by
    run_arms (needs aggregate.accuracy, aggregate_by_type, cost.total_usd).
    """
    n = len(test_set)
    labels = list(run_results[0])
    arms = {}
    for label in labels:
        accs = [r[label]["aggregate"]["accuracy"] for r in run_results]
        by_type = {}
        for t in run_results[0][label].get("aggregate_by_type", {}):
            vals = [r[label]["aggregate_by_type"][t]["accuracy"] for r in run_results]
            by_type[t] = round(statistics.mean(vals), 4)
        arms[label] = {
            "accuracy":  round(statistics.mean(accs), 4),
            "runs":      accs,
            "std":       round(statistics.stdev(accs), 4) if len(accs) > 1 else 0.0,
            "tolerance": tolerance(accs, n),
            "by_type":   by_type,
        }
    run_costs = [round(sum(arm["cost"]["total_usd"] for arm in r.values()), 6) for r in run_results]
    return {
        "created":         datetime.now().isoformat(timespec="seconds"),
        "test_set":        TEST_SET_PATH.replace(os.sep, "/"),
        "test_set_sha256": hash_test_set(test_set),
        "n_questions":     n,
        "tolerance_rule":  f"max({TOLERANCE_STD_MULTIPLIER} x stdev of runs, 1/n), "
                           "rounded up to a multiple of 1/n",
        "arms":            arms,
        "cost_per_run_usd": {"mean": round(statistics.mean(run_costs), 6), "runs": run_costs},
    }


def compare(results: dict, baseline: dict, test_set: list) -> tuple:
    """
    (passed, rows, problems). rows: one dict per baseline arm with accuracy,
    baseline, delta, threshold, passed. problems: reasons the gate fails that
    aren't a score (stale baseline, missing arm).
    """
    problems = []
    if baseline.get("test_set_sha256") != hash_test_set(test_set):
        problems.append(
            "The fixture test set changed since the baseline was recorded. "
            "Re-baseline: python eval.py gate --update-baseline --runs 3")

    rows = []
    for label, b in baseline["arms"].items():
        r = results.get(label)
        if r is None:
            problems.append(f"Arm {label!r} is in the baseline but wasn't run.")
            continue
        acc = r["aggregate"]["accuracy"]
        threshold = round(b["accuracy"] - b["tolerance"], 4)
        ok = acc == acc and acc >= threshold - 1e-9   # acc == acc rejects NaN
        rows.append({
            "label":     label,
            "accuracy":  acc,
            "baseline":  b["accuracy"],
            "delta":     round(acc - b["accuracy"], 4) if acc == acc else float("nan"),
            "tolerance": b["tolerance"],
            "threshold": threshold,
            "by_type":   {t: v.get("accuracy") for t, v in r.get("aggregate_by_type", {}).items()},
            "cost_usd":  r.get("cost", {}).get("total_usd"),
            "latency_s": r.get("latency", {}).get("avg_s"),
            "passed":    ok,
        })
    passed = not problems and all(row["passed"] for row in rows)
    return passed, rows, problems


def summary_markdown(passed: bool, rows: list, problems: list, n_questions: int) -> str:
    """Markdown for the GitHub job summary (and the console)."""
    lines = [
        f"## Eval gate: {'PASSED' if passed else 'FAILED'}",
        "",
        f"Accuracy on the {n_questions}-question fixture set (Gemini Flash judge). "
        "An arm fails below baseline - tolerance.",
        "",
        "| Arm | Accuracy | Baseline | Change | Fails below | Single-hop | Multi-hop | Cost | Avg latency | Result |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        bt = r["by_type"]
        fmt = lambda v: "-" if v is None or v != v else f"{v:.2f}"  # noqa: E731
        lines.append(
            f"| {r['label']} | {fmt(r['accuracy'])} | {r['baseline']:.2f} | {r['delta']:+.2f} | "
            f"{r['threshold']:.2f} | {fmt(bt.get('single-hop'))} | {fmt(bt.get('multi-hop'))} | "
            f"${(r['cost_usd'] or 0):.4f} | {(r['latency_s'] or 0):.1f}s | "
            f"{'pass' if r['passed'] else '**FAIL**'} |")
    total = sum(r["cost_usd"] or 0 for r in rows)
    lines += ["", f"Total LLM cost of this run: ${total:.4f}"]
    for p in problems:
        lines += ["", f"**{p}**"]
    return "\n".join(lines) + "\n"


def load_baseline(path: str = BASELINE_PATH):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Running the arms (API calls) ──────────────────────────────────────────────

def run_arms(test_set: list, progress_callback=None) -> dict:
    """Answer and accuracy-judge the test set with both gate arms. {label: arm_result}."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    import eval as ev
    from agent import AgentRunner
    from observability import flush as flush_traces

    llm       = ChatGoogleGenerativeAI(model=ev.EVAL_LLM_ID, temperature=0)
    judge_llm = ChatGoogleGenerativeAI(model=ev.EVAL_LLM_ID, temperature=0, thinking_budget=0)

    rag_cfg = ev.EvalConfig(**GATE_RAG_CONFIG)
    results = {}
    samples = ev._rag_samples(test_set, rag_cfg, llm, progress_callback)
    results[rag_cfg.label] = ev._score_samples(
        samples, rag_cfg.label, judge_llm, None, None, progress_callback, ragas=False)

    agent_cfg = ev.AgentConfig()
    runner = AgentRunner()
    try:
        session_id = f"gate-{datetime.now():%Y%m%d-%H%M%S}"
        samples = ev._agent_samples(test_set, agent_cfg, runner, session_id, progress_callback)
    finally:
        runner.close()
        flush_traces()
    results[agent_cfg.label] = ev._score_samples(
        samples, agent_cfg.label, judge_llm, None, None, progress_callback, ragas=False)
    return results


def main(update_baseline: bool = False, runs: int = 3, baseline_path: str = BASELINE_PATH,
         progress_callback=None) -> int:
    """CLI entry (python eval.py gate). Returns the process exit code."""
    import eval as ev

    if CORPUS != "fixture":
        print("The gate runs on the fixture corpus only. Set CORPBRAIN_CORPUS=fixture.")
        return 2
    test_set = ev.load_test_set()
    if not test_set:
        print(f"No test set at {TEST_SET_PATH}.")
        return 2
    os.makedirs(os.path.dirname(GATE_RESULTS_PATH), exist_ok=True)

    if update_baseline:
        run_results = []
        for i in range(runs):
            print(f"\nBaseline run {i + 1}/{runs}")
            run_results.append(run_arms(test_set, progress_callback))
            accs = {k: v["aggregate"]["accuracy"] for k, v in run_results[-1].items()}
            print(f"  accuracy: {accs}")
        baseline = build_baseline(run_results, test_set)
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(baseline, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"\nSaved baseline to {baseline_path}")
        for label, a in baseline["arms"].items():
            print(f"  {label:<20} mean {a['accuracy']:.3f}  runs {a['runs']}  "
                  f"std {a['std']:.3f}  tolerance {a['tolerance']:.2f}")
        print(f"  cost per run: ${baseline['cost_per_run_usd']['mean']:.4f} "
              f"(runs {baseline['cost_per_run_usd']['runs']})")
        return 0

    baseline = load_baseline(baseline_path)
    if baseline is None:
        print(f"No baseline at {baseline_path}. Create one: python eval.py gate --update-baseline --runs 3")
        return 2

    results = run_arms(test_set, progress_callback)
    passed, rows, problems = compare(results, baseline, test_set)
    md = summary_markdown(passed, rows, problems, len(test_set))
    print("\n" + md)

    with open(GATE_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(timespec="seconds"), "passed": passed,
                   "rows": rows, "problems": problems, "arms": results},
                  f, ensure_ascii=False, indent=2)
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(md)
    return 0 if passed else 1
