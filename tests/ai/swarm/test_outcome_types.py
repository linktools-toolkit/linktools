#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SwarmExecutionOutcome: the discriminated union SwarmEngine.execute() will
return. Locks the type contract ahead of the SwarmEngine cutover: construction
works, the three variants are mutually exclusive, and isinstance dispatch
distinguishes them (so an invalid combination is not constructible)."""

from linktools.ai.run.models import RunErrorInfo, RunResult
from linktools.ai.session.models import MessageRole, NewSessionMessage
from linktools.ai.swarm.models import (
    SwarmCheckpoint,
    SwarmCompleted,
    SwarmExecutionOutcome,
    SwarmFailed,
    SwarmPaused,
    SwarmUsage,
)


def _msg(content: str) -> NewSessionMessage:
    return NewSessionMessage(role=MessageRole.ASSISTANT, content=content, run_id="r")


def test_swarm_completed_carries_result_messages_usage():
    result = RunResult(output="aggregate")
    outcome: SwarmExecutionOutcome = SwarmCompleted(
        result=result,
        aggregate_messages=(_msg("a"), _msg("b")),
        usage=SwarmUsage(input_tokens=10, output_tokens=5),
    )
    assert isinstance(outcome, SwarmCompleted)
    assert outcome.result.output == "aggregate"
    assert len(outcome.aggregate_messages) == 2
    assert outcome.usage.input_tokens == 10


def test_swarm_paused_carries_checkpoint():
    checkpoint = SwarmCheckpoint(
        completed_task_ids=("t1",),
        failed_task_ids=(),
        pending_task_ids=("t2",),
        active_task_ids=(),
    )
    outcome: SwarmExecutionOutcome = SwarmPaused(checkpoint=checkpoint)
    assert isinstance(outcome, SwarmPaused)
    assert outcome.checkpoint.pending_task_ids == ("t2",)


def test_swarm_failed_carries_redacted_error():
    outcome: SwarmExecutionOutcome = SwarmFailed(
        error=RunErrorInfo(error_type="SwarmError", message="boom")
    )
    assert isinstance(outcome, SwarmFailed)
    assert outcome.error.error_type == "SwarmError"


def test_outcome_variants_are_mutually_exclusive_by_isinstance():
    completed = SwarmCompleted(
        result=RunResult(output="x"),
        aggregate_messages=(),
        usage=SwarmUsage(),
    )
    paused = SwarmPaused(checkpoint=SwarmCheckpoint((), (), (), ()))
    failed = SwarmFailed(error=RunErrorInfo(error_type="E", message="m"))
    # Each variant is distinguished by isinstance; no overlap.
    for outcome in (completed, paused, failed):
        matches = [
            cls for cls in (SwarmCompleted, SwarmPaused, SwarmFailed)
            if isinstance(outcome, cls)
        ]
        assert matches == [type(outcome)]
