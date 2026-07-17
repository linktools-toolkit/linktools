#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comparison aggregation + baseline-vs-candidate delta tests (section 25.4)."""

from linktools.ai.evaluation.comparison import aggregate, compare
from linktools.ai.evaluation.models import EvalResult


def _result(scores=None, metrics=None, error=False, case="c") -> EvalResult:
    return EvalResult(
        id=case,
        eval_run_id="r",
        case_id=case,
        scores=scores or {},
        metrics=metrics or {},
        error_type="Boom" if error else None,
        error_message="x" if error else None,
    )


def test_aggregate_empty_returns_empty() -> None:
    assert aggregate(()) == {}


def test_aggregate_score_avg_pass_rate_and_error_rate() -> None:
    results = [
        _result({"q": 0.9}, case="c1"),
        _result({"q": 0.4}, case="c2"),
        _result(error=True, case="c3"),
    ]
    agg = aggregate(results)
    assert agg["q_avg"] == (0.9 + 0.4) / 2  # only the scored cases
    assert agg["q_pass_rate"] == 0.5  # 0.9 passes 0.5, 0.4 does not
    assert agg["error_rate"] == 1 / 3


def test_aggregate_latency_percentiles_and_usage() -> None:
    results = [
        _result(
            metrics={"latency_seconds": 1.0, "total_tokens": 100,
                     "total_cost": 0.5, "safety_refusal": 0},
            case="c1",
        ),
        _result(
            metrics={"latency_seconds": 2.0, "total_tokens": 200,
                     "total_cost": 1.5, "safety_refusal": 1},
            case="c2",
        ),
        _result(
            metrics={"latency_seconds": 3.0, "total_tokens": 300,
                     "total_cost": 1.0, "safety_refusal": 0},
            case="c3",
        ),
    ]
    agg = aggregate(results)
    assert agg["p50_latency_seconds"] == 2.0
    assert agg["p95_latency_seconds"] >= 2.8
    assert agg["avg_tokens"] == 200.0
    assert agg["avg_cost"] == 1.0
    assert agg["safety_refusal_rate"] == 1 / 3
    assert agg["avg_retry_count"] == 0.0


def test_compare_inverts_lower_is_better() -> None:
    baseline = [
        _result({"q": 0.6}, metrics={"latency_seconds": 5.0, "total_tokens": 500}, case="b1"),
        _result({"q": 0.6}, metrics={"latency_seconds": 5.0, "total_tokens": 500}, case="b2"),
    ]
    candidate = [
        _result({"q": 0.8}, metrics={"latency_seconds": 2.0, "total_tokens": 300}, case="c1"),
        _result({"q": 0.8}, metrics={"latency_seconds": 2.0, "total_tokens": 300}, case="c2"),
    ]
    delta = compare(baseline, candidate)
    # Higher score -> positive delta (candidate better).
    assert delta["q_avg"] > 0
    # Lower latency / tokens -> positive delta after inversion.
    assert delta["p50_latency_seconds"] > 0
    assert delta["avg_tokens"] > 0


def test_skill_baseline_vs_candidate_compare_end_to_end() -> None:
    """Section 24.2 Skill baseline/candidate: run the host agent WITHOUT the
    skill (baseline) vs WITH it (candidate) and compare -- the delta shows the
    candidate scoring higher. Exercises EvalSuite.baseline_target + compare()."""
    import asyncio

    from linktools.ai.evaluation import EvalCase, EvalRunner, EvalSuite, EvalTarget
    from linktools.ai.evaluation.evaluators.exact import ExactMatchEvaluator
    from linktools.ai.evaluation.models import EvalExecution

    class _SkillAwareExecutor:
        async def execute(self, target, case):
            expected = case.metadata.get("expected")
            # Without the skill the agent gets it wrong; with it, right.
            output = expected if target.config.get("use_skill") else f"wrong:{expected}"
            return EvalExecution(case_id=case.id, run_id=None, output=output)

    cases = {
        "c1": EvalCase(id="c1", input_artifact_id="in", metadata={"expected": "answer"}),
        "c2": EvalCase(id="c2", input_artifact_id="in", metadata={"expected": "result"}),
    }
    runner = EvalRunner(_SkillAwareExecutor(), {"exact_match": ExactMatchEvaluator()})
    candidate = EvalTarget(kind="agent", id="host", config={"use_skill": True})
    baseline = EvalTarget(kind="agent", id="host", config={"use_skill": False})

    def suite(target):
        return EvalSuite(
            id="s1",
            name="skill-ab",
            version="1",
            target=target,
            case_ids=("c1", "c2"),
            evaluator_names=("exact_match",),
            baseline_target=baseline,
        )

    async def run():
        cand = await runner.run_suite(suite(candidate), cases)
        base = await runner.run_suite(suite(baseline), cases)
        delta = compare(base, cand)
        # Candidate (skill loaded) passes exact-match; baseline fails -> positive delta.
        assert delta["exact_match_avg"] > 0
        assert delta["exact_match_pass_rate"] > 0

    asyncio.run(run())


def test_compare_does_not_invert_evaluator_whose_name_contains_cost() -> None:
    # A business evaluator named "cost_efficiency" is higher-is-better; although
    # its name contains "cost", it must NOT be treated as a lower-is-better
    # metric (only the exact known aggregate keys invert).
    baseline = [_result({"cost_efficiency": 0.5}, case="c1")]
    candidate = [_result({"cost_efficiency": 0.9}, case="c1")]  # candidate better
    delta = compare(baseline, candidate)
    assert delta["cost_efficiency_avg"] > 0  # positive => candidate better


def test_compare_inverts_known_lower_is_better_keys() -> None:
    # The known aggregate avg_cost IS lower-is-better: candidate cheaper should
    # produce a positive (better) delta after inversion.
    baseline = [_result(metrics={"total_cost": 1.0}, case="c1")]
    candidate = [_result(metrics={"total_cost": 0.4}, case="c1")]  # candidate cheaper
    delta = compare(baseline, candidate)
    assert delta["avg_cost"] > 0  # inverted: cheaper => positive
