#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EvalRunner direct mode"""

import asyncio

import pytest

from linktools.ai.evaluation import (
    EvalCase,
    EvalRunner,
    EvalSuite,
    EvalTarget,
)
from linktools.ai.evaluation.evaluators.exact import ExactMatchEvaluator


class _EchoExecutor:
    """Returns the case's input as the output (for exact-match testing)."""

    async def execute(self, target, case):
        from linktools.ai.evaluation.models import EvalExecution

        return EvalExecution(
            case_id=case.id, run_id=None, output=case.metadata.get("expected")
        )


class _FailingExecutor:
    async def execute(self, target, case):
        raise RuntimeError("boom")


def test_eval_runner_scores_exact_match() -> None:
    suite = EvalSuite(
        id="s1",
        name="test",
        version="1",
        target=EvalTarget(kind="agent", id="a1"),
        case_ids=("c1", "c2"),
        evaluator_names=("exact_match",),
    )
    cases = {
        "c1": EvalCase(
            id="c1", input_artifact_id="in1", metadata={"expected": "hello"}
        ),
        "c2": EvalCase(
            id="c2", input_artifact_id="in2", metadata={"expected": "world"}
        ),
    }
    runner = EvalRunner(_EchoExecutor(), {"exact_match": ExactMatchEvaluator()})

    async def run():
        results = await runner.run_suite(suite, cases)
        assert len(results) == 2
        assert results[0].scores["exact_match"] == 1.0
        assert results[1].scores["exact_match"] == 1.0

    asyncio.run(run())


def test_failing_case_does_not_abort_suite() -> None:
    suite = EvalSuite(
        id="s1",
        name="test",
        version="1",
        target=EvalTarget(kind="agent", id="a1"),
        case_ids=("c1", "c2"),
        evaluator_names=("exact_match",),
    )
    cases = {
        "c1": EvalCase(id="c1", input_artifact_id="in1"),
        "c2": EvalCase(id="c2", input_artifact_id="in2"),
    }
    runner = EvalRunner(_FailingExecutor(), {"exact_match": ExactMatchEvaluator()})

    async def run():
        results = await runner.run_suite(suite, cases)
        assert len(results) == 2
        assert results[0].error_type == "RuntimeError"
        assert results[1].error_type == "RuntimeError"

    asyncio.run(run())


class _CrashingEvaluator:
    @property
    def name(self) -> str:
        return "crash"

    async def evaluate(self, case, execution):
        raise RuntimeError("evaluator blew up")


def test_failing_evaluator_does_not_abort_suite() -> None:
    suite = EvalSuite(
        id="s1",
        name="test",
        version="1",
        target=EvalTarget(kind="agent", id="a1"),
        case_ids=("c1",),
        evaluator_names=("exact_match", "crash"),
    )
    cases = {
        "c1": EvalCase(id="c1", input_artifact_id="in1", metadata={"expected": "x"})
    }
    runner = EvalRunner(
        _EchoExecutor(),
        {"exact_match": ExactMatchEvaluator(), "crash": _CrashingEvaluator()},
    )

    async def run():
        results = await runner.run_suite(suite, cases)
        assert len(results) == 1
        assert results[0].scores["exact_match"] == 1.0
        # The crashing evaluator scored 0.0, not aborted the suite.
        assert results[0].scores["crash"] == 0.0

    asyncio.run(run())


def test_missing_evaluator_is_silently_skipped() -> None:
    suite = EvalSuite(
        id="s1",
        name="test",
        version="1",
        target=EvalTarget(kind="agent", id="a1"),
        case_ids=("c1",),
        evaluator_names=("exact_match", "nonexistent"),
    )
    cases = {
        "c1": EvalCase(id="c1", input_artifact_id="in1", metadata={"expected": "x"})
    }
    runner = EvalRunner(_EchoExecutor(), {"exact_match": ExactMatchEvaluator()})

    async def run():
        results = await runner.run_suite(suite, cases)
        assert "exact_match" in results[0].scores
        assert "nonexistent" not in results[0].scores

    asyncio.run(run())


# ---- Schema / Trajectory / Usage evaluators ----


def test_schema_evaluator_matches() -> None:
    from linktools.ai.evaluation.evaluators.schema import SchemaEvaluator
    from linktools.ai.evaluation.models import EvalExecution

    ev = SchemaEvaluator()
    case = EvalCase(
        id="c", input_artifact_id="i", metadata={"schema": {"required": ["name"]}}
    )

    async def run():
        ok = await ev.evaluate(
            case, EvalExecution(case_id="c", run_id=None, output={"name": "x"})
        )
        bad = await ev.evaluate(
            case, EvalExecution(case_id="c", run_id=None, output={"no": "x"})
        )
        assert ok.score == 1.0 and bad.score == 0.0

    asyncio.run(run())


def test_trajectory_evaluator() -> None:
    from linktools.ai.evaluation.evaluators.trajectory import TrajectoryEvaluator
    from linktools.ai.evaluation.models import EvalExecution

    ev = TrajectoryEvaluator()
    case = EvalCase(
        id="c",
        input_artifact_id="i",
        metadata={
            "trajectory": {
                "required_actions": ["search"],
                "forbidden_actions": ["delete"],
            }
        },
    )

    async def run():
        ok = await ev.evaluate(
            case,
            EvalExecution(
                case_id="c",
                run_id=None,
                output={"actions": ["search", "read"], "total_calls": 2},
            ),
        )
        bad = await ev.evaluate(
            case,
            EvalExecution(
                case_id="c",
                run_id=None,
                output={"actions": ["read", "delete"], "total_calls": 2},
            ),
        )
        assert ok.score == 1.0
        assert bad.score == 0.0

    asyncio.run(run())


def test_usage_evaluator() -> None:
    from linktools.ai.evaluation.evaluators.usage import UsageEvaluator
    from linktools.ai.evaluation.models import EvalExecution

    ev = UsageEvaluator()
    case = EvalCase(
        id="c",
        input_artifact_id="i",
        metadata={"usage_limits": {"max_tokens": 1000}},
    )

    async def run():
        ok = await ev.evaluate(
            case,
            EvalExecution(
                case_id="c",
                run_id=None,
                output={"total_tokens": 500, "total_cost": "0.01"},
            ),
        )
        bad = await ev.evaluate(
            case,
            EvalExecution(
                case_id="c",
                run_id=None,
                output={"total_tokens": 2000, "total_cost": "0.05"},
            ),
        )
        assert ok.score == 1.0 and bad.score == 0.0

    asyncio.run(run())


# ---- RunSnapshot + Comparison ----


def test_run_snapshot_is_frozen() -> None:
    import dataclasses

    from linktools.ai.evaluation.snapshot import RunSnapshot

    snap = RunSnapshot(
        run_id="r1",
        run_record_artifact_id="a1",
        run_definition_artifact_id="a2",
        input_artifact_id="a3",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.run_id = "other"


def test_comparison_baseline_vs_candidate() -> None:
    from linktools.ai.evaluation.comparison import compare
    from linktools.ai.evaluation.models import EvalResult

    baseline = [
        EvalResult(id="b1", eval_run_id="b", case_id="c1", scores={"exact_match": 1.0}),
        EvalResult(id="b2", eval_run_id="b", case_id="c2", scores={"exact_match": 0.0}),
    ]
    candidate = [
        EvalResult(id="k1", eval_run_id="k", case_id="c1", scores={"exact_match": 1.0}),
        EvalResult(id="k2", eval_run_id="k", case_id="c2", scores={"exact_match": 1.0}),
    ]
    delta = compare(baseline, candidate)
    assert delta["exact_match_avg"] > 0  # candidate improved


def test_eval_runner_persists_lifecycle_and_results() -> None:
    """When a store is wired, the runner persists the EvalRun lifecycle and the
    per-case results."""
    from linktools.ai.evaluation.models import EvalRunStatus
    from linktools.ai.evaluation.store import InMemoryEvalStore

    store = InMemoryEvalStore()
    runner = EvalRunner(
        _EchoExecutor(), {"exact_match": ExactMatchEvaluator()}, store=store
    )
    suite = EvalSuite(
        id="s1",
        name="t",
        version="1",
        target=EvalTarget(kind="agent", id="a1"),
        case_ids=("c1", "c2"),
        evaluator_names=("exact_match",),
    )
    cases = {
        "c1": EvalCase(id="c1", input_artifact_id="in1", metadata={"expected": "hi"}),
        "c2": EvalCase(id="c2", input_artifact_id="in2", metadata={"expected": "yo"}),
    }

    async def run():
        results = await runner.run_suite(suite, cases)
        assert len(results) == 2
        run_id = results[0].eval_run_id
        run = await store.get_run(run_id)
        assert run is not None
        assert run.status == EvalRunStatus.SUCCEEDED
        assert run.finished_at is not None
        stored = await store.list_results(run_id)
        assert len(stored) == 2
        assert {r.case_id for r in stored} == {"c1", "c2"}

    asyncio.run(run())


def test_eval_runner_marks_run_failed_on_case_error() -> None:
    """A case whose execution errors transitions the EvalRun to FAILED."""
    from linktools.ai.evaluation.models import EvalRunStatus
    from linktools.ai.evaluation.store import InMemoryEvalStore

    store = InMemoryEvalStore()
    runner = EvalRunner(_FailingExecutor(), {}, store=store)
    suite = EvalSuite(
        id="s1",
        name="t",
        version="1",
        target=EvalTarget(kind="agent", id="a1"),
        case_ids=("c1",),
        evaluator_names=(),
    )
    cases = {"c1": EvalCase(id="c1", input_artifact_id="in1")}

    async def run():
        results = await runner.run_suite(suite, cases)
        run = await store.get_run(results[0].eval_run_id)
        assert run is not None
        assert run.status == EvalRunStatus.FAILED

    asyncio.run(run())


def test_in_memory_eval_store_rejects_duplicate_ids() -> None:
    """The store treats run/result ids as immutable: a duplicate create/append
    raises rather than silently overwriting."""
    from datetime import datetime, timezone

    from linktools.ai.evaluation.models import (
        EvalResult,
        EvalRun,
        EvalRunStatus,
        EvalTarget,
    )
    from linktools.ai.evaluation.store import (
        EvalResultConflictError,
        EvalRunNotFoundError,
        InMemoryEvalStore,
    )

    store = InMemoryEvalStore()

    async def run():
        now = datetime.now(timezone.utc)
        eval_run = EvalRun(
            id="er1",
            suite_id="s1",
            target=EvalTarget(kind="agent", id="a1"),
            status=EvalRunStatus.RUNNING,
            baseline_target=None,
            created_at=now,
            started_at=now,
        )
        await store.create_run(eval_run)
        with pytest.raises(EvalResultConflictError):
            await store.create_run(eval_run)
        await store.transition_run("er1", status=EvalRunStatus.SUCCEEDED, finished_at=now)
        with pytest.raises(EvalRunNotFoundError):
            await store.transition_run("missing", status=EvalRunStatus.SUCCEEDED)
        result = EvalResult(id="er1-c1", eval_run_id="er1", case_id="c1")
        await store.append_result(result)
        with pytest.raises(EvalResultConflictError):
            await store.append_result(result)
        assert len(await store.list_results("er1")) == 1

    asyncio.run(run())


def test_eval_runner_threads_usage_into_result_metrics() -> None:
    """The executor's captured model_usage (tokens / cost) lands on the result
    metrics so aggregate() can compute avg tokens / avg cost."""
    from linktools.ai.evaluation.models import EvalExecution

    class _UsageExecutor:
        async def execute(self, target, case):
            return EvalExecution(
                case_id=case.id,
                run_id=None,
                output="ok",
                model_usage={"total_tokens": 250, "total_cost": 1.25},
            )

    runner = EvalRunner(_UsageExecutor(), {})
    suite = EvalSuite(
        id="s1",
        name="t",
        version="1",
        target=EvalTarget(kind="agent", id="a1"),
        case_ids=("c1",),
        evaluator_names=(),
    )
    cases = {"c1": EvalCase(id="c1", input_artifact_id="in1")}

    async def run():
        results = await runner.run_suite(suite, cases)
        assert results[0].metrics["total_tokens"] == 250
        assert results[0].metrics["total_cost"] == 1.25

    asyncio.run(run())


def test_target_error_returned_not_raised_marks_result_and_run_failed() -> None:
    """A target that returns EvalExecution(error=...) (instead of raising) still
    records the error on the result and transitions the run to FAILED."""
    from linktools.ai.evaluation.models import EvalExecution, EvalRunStatus
    from linktools.ai.evaluation.store import InMemoryEvalStore

    class _ErrorExecutor:
        async def execute(self, target, case):
            return EvalExecution(
                case_id=case.id, run_id="r", output=None, error="RuntimeError"
            )

    store = InMemoryEvalStore()
    runner = EvalRunner(_ErrorExecutor(), {}, store=store)
    suite = EvalSuite(
        id="s1",
        name="t",
        version="1",
        target=EvalTarget(kind="agent", id="a1"),
        case_ids=("c1",),
        evaluator_names=(),
    )
    cases = {"c1": EvalCase(id="c1", input_artifact_id="in1")}

    async def run():
        results = await runner.run_suite(suite, cases)
        assert results[0].error_type == "RuntimeError"
        run_head = await store.get_run(results[0].eval_run_id)
        assert run_head.status == EvalRunStatus.FAILED

    asyncio.run(run())
