#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step projection and local stream behavior."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.core import ExecutionDeltaType, ExecutionLineageKind, ExecutionStatus
from linktools.ai.runtime import (
    ExecutionDelta,
    LiveExecutionEventBroker,
    RuntimeDomain,
    RuntimeState,
)
from linktools.ai.runtime.state import ExecutionRecord
from linktools.ai.spec import AgentSpec
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord, StepEvent


def _run() -> RunRecord:
    return RunRecord(
        run_id="run",
        conversation_id="conversation",
        parent_run_id=None,
        agent_name="agent",
        metadata={},
        started_at=datetime.now(timezone.utc),
    )


def _binding_snapshot() -> AgentBindingSnapshot:
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent"),
        model={"route_id": "default", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        binding_digest="a" * 64,
    )


def _execution() -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    return ExecutionRecord(
        execution_id="execution",
        tenant_id="tenant",
        session_id=None,
        binding_digest="a" * 64,
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=ExecutionStatus.STARTED,
        revision=0,
        event_sequence=0,
        agent_run_sequence=1,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        mode="run",
        planning=False,
        thinking=False,
        binding=_binding_snapshot(),
    )


@pytest.mark.asyncio
async def test_step_events_wait_for_a_safe_snapshot(tmp_path: Path) -> None:
    state = RuntimeState.filesystem(tmp_path / "runtime")
    await state.initialize(namespace="step-io", tenant_id="tenant")
    try:
        await state.execution.executions.create_with_history_head(_execution())
        run = _run()
        await state.steps.register_run(run)
        now = datetime.now(timezone.utc)
        for index, kind in enumerate(
            ("model_request_started", "model_request_completed", "tool_call_started"),
            1,
        ):
            await state.steps.append_event(
                StepEvent(
                    run_id=run.run_id,
                    kind=kind,
                    step_index=index,
                    timestamp=now,
                    conversation_id=run.conversation_id,
                    agent_name=run.agent_name,
                )
            )

        execution = state.steps.read_store(RuntimeDomain.EXECUTION)
        recovery = state.steps.read_store(RuntimeDomain.RECOVERY)
        assert await execution.list_events(run_id=run.run_id) == []
        assert await recovery.get_run(run_id=run.run_id) is None

        snapshot = ContinuableSnapshot(
            run_id=run.run_id,
            step_index=3,
            messages=[
                ModelRequest(
                    parts=[UserPromptPart(content="hello")],
                    conversation_id=run.conversation_id,
                )
            ],
            conversation_id=run.conversation_id,
            parent_run_id=None,
            agent_name=run.agent_name,
            timestamp=now,
        )
        await state.steps.save_snapshot(snapshot)
        assert await execution.list_events(run_id=run.run_id) == []

        await state.steps.flush_execution_projection(
            run.run_id,
            execution_id="execution",
        )

        assert len(await execution.list_events(run_id=run.run_id)) == 3
        assert await execution.latest_snapshot(run_id=run.run_id) == snapshot
        assert await recovery.get_run(run_id=run.run_id) == run
        assert await recovery.latest_snapshot(run_id=run.run_id) == snapshot
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_prepared_local_stream_survives_fast_completion() -> None:
    broker = LiveExecutionEventBroker()
    broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 0)
    broker.publish(
        ExecutionDelta(
            "execution",
            ExecutionDeltaType.ASSISTANT_TEXT_DELTA,
            "fast",
        )
    )
    broker.complete("execution")

    subscription = broker.subscribe("execution")
    assert (await subscription.__anext__()).content == "fast"
    with pytest.raises(StopAsyncIteration):
        await subscription.__anext__()
