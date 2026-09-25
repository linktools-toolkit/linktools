#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step projection and local stream behavior."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.core import ExecutionDeltaType, ExecutionLineageKind, ExecutionStatus
from linktools.ai.runtime import RuntimeDomain, RuntimeState
from linktools.ai.runtime._event import ExecutionDelta, LiveExecutionEventBroker
from linktools.ai.runtime.state._contracts import ExecutionRecord
from linktools.ai.spec import AgentSpec
from pydantic_ai.messages import ModelRequest, UserPromptPart
from linktools.ai.runtime.state._step_contracts import (
    ContinuableSnapshot,
    RunRecord,
    StepEvent,
)
from ._runtime_test_helpers import execution_owner_fields


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
        agent_spec=AgentSpec("agent"),
        model_contract={"route_id": "default", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={"type": "object", "properties": {"text": {"type": "string"}}},
    )


def _execution() -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    return ExecutionRecord(
        execution_id="execution",
        session_id=None,
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
        **execution_owner_fields(),
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
            transcript_message_count_before=0,
        )
        await state.steps.save_snapshot(snapshot)
        assert await execution.list_events(run_id=run.run_id) == []

        await state.steps.flush_execution_projection(
            run.run_id,
            execution_id="execution",
        )

        assert len(await execution.list_events(run_id=run.run_id)) == 3
        latest = await execution.latest_snapshot(run_id=run.run_id)
        assert latest is not None
        assert latest.transcript_message_count_before is None
        assert latest.messages == snapshot.messages
        assert await recovery.get_run(run_id=run.run_id) == run
        recovery_latest = await recovery.latest_snapshot(run_id=run.run_id)
        assert recovery_latest is not None
        assert recovery_latest.transcript_message_count_before is None
        assert recovery_latest.messages == snapshot.messages
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_step_run_reuses_durable_recovery_identity(tmp_path: Path) -> None:
    state = RuntimeState.filesystem(tmp_path / "runtime")
    await state.initialize(namespace="step-run-recovery", tenant_id="tenant")
    try:
        recovery = state.steps.read_store(RuntimeDomain.RECOVERY)
        durable = RunRecord(
            run_id="run",
            conversation_id="conversation",
            parent_run_id=None,
            agent_name="agent",
            metadata={"scope": "recovery"},
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        await recovery.register_run(durable)
        candidate = RunRecord(
            run_id="run",
            conversation_id="conversation",
            parent_run_id=None,
            agent_name="agent",
            metadata={"scope": "recovery"},
            started_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )

        await state.steps.register_run(candidate)

        assert await state.steps.get_run(run_id="run") == durable
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
