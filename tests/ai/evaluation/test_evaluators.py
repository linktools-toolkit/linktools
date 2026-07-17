#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Built-in evaluator tests: UsageEvaluator reads model_usage; DelegationEvaluator
reads the snapshot."""

import asyncio

from linktools.ai.evaluation.evaluators.delegation import DelegationEvaluator
from linktools.ai.evaluation.evaluators.usage import UsageEvaluator
from linktools.ai.evaluation.models import EvalCase, EvalExecution
from linktools.ai.evaluation.snapshot import RunSnapshot


def _execution(model_usage=None, output=None) -> EvalExecution:
    return EvalExecution(
        case_id="c1",
        run_id="r1",
        output=output,
        model_usage=model_usage or {},
    )


def _snapshot(events=1) -> RunSnapshot:
    return RunSnapshot(
        run_id="r1",
        run_record_artifact_id="rr",
        run_definition_artifact_id="rd",
        input_artifact_id="in",
        event_artifact_ids=tuple(f"evt-{i}" for i in range(events)),
    )


def test_usage_evaluator_reads_captured_model_usage() -> None:
    case = EvalCase(
        id="c1",
        input_artifact_id="in",
        metadata={"usage_limits": {"max_tokens": 100}},
    )
    over = _execution(model_usage={"total_tokens": 250})

    async def run():
        score = await UsageEvaluator().evaluate(case, over)
        assert score.score == 0.0
        assert "tokens" in score.details["over_budget"]

    asyncio.run(run())


def test_usage_evaluator_passes_under_budget() -> None:
    case = EvalCase(
        id="c1",
        input_artifact_id="in",
        metadata={"usage_limits": {"max_tokens": 100}},
    )
    ok = _execution(model_usage={"total_tokens": 50})

    async def run():
        assert (await UsageEvaluator().evaluate(case, ok)).score == 1.0

    asyncio.run(run())


def test_delegation_evaluator_scores_from_snapshot() -> None:
    deleg = DelegationEvaluator()
    expects = EvalCase(
        id="c1", input_artifact_id="in", metadata={"expects_delegation": True}
    )
    no_expect = EvalCase(id="c2", input_artifact_id="in")

    async def run():
        # Expected + events present -> pass.
        s1 = await deleg.evaluate(expects, _execution(), snapshot=_snapshot(1))
        assert s1.score == 1.0
        # Expected + no events -> fail.
        s2 = await deleg.evaluate(expects, _execution(), snapshot=_snapshot(0))
        assert s2.score == 0.0
        # No expectation declared -> neutral pass.
        s3 = await deleg.evaluate(no_expect, _execution(), snapshot=None)
        assert s3.score == 1.0

    asyncio.run(run())
