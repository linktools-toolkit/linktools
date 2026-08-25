#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conformance coverage for staged step projection and local streaming."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.agent._output import bind_output
from linktools.ai.core import ExecutionDeltaType, ExecutionLineageKind, ExecutionStatus
from linktools.ai.runtime import RuntimeDomain, RuntimeState
from linktools.ai.runtime._event import ExecutionDelta, LiveExecutionEventBroker
from linktools.ai.runtime.state import (
    ExecutionHistoryHeadRecord,
    ExecutionHistoryState,
    ExecutionRecord,
)
from linktools.ai.runtime.state._filesystem import FilesystemStateStore
from linktools.ai.runtime.state._store import StateTransaction
from linktools.ai.spec import AgentSpec
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    RunRecord,
    StepEvent,
    ToolEffectRecord,
)


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
    output = bind_output()
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", 1, "default"),
        agent_digest="b" * 64,
        output_schema_id=output.schema_id,
        output_schema_revision=output.schema_revision,
        output_schema_fingerprint=output.schema_fingerprint,
        local_runtime_capability_descriptors=(),
        binding_digest="a" * 64,
        global_runtime_capability_descriptors=(),
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
        planning=False,
        thinking=False,
        binding=_binding_snapshot(),
    )


@pytest.mark.asyncio
async def test_step_events_wait_for_a_safe_snapshot(tmp_path: Path) -> None:
    state = RuntimeState.filesystem(tmp_path / "runtime")
    await state.initialize(namespace="step-io", tenant_id="tenant")
    try:
        await state.execution.executions.create(_execution())
        await state.execution.executions.state_store.mutate(
            lambda transaction: state.execution.executions.insert_history_head_in_transaction(
                transaction,
                ExecutionHistoryHeadRecord(
                    "execution",
                    "tenant",
                    ExecutionHistoryState.OPEN,
                    0,
                    None,
                ),
            )
        )
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
async def test_first_recovery_effect_uses_one_filesystem_generation(tmp_path: Path) -> None:
    state = RuntimeState.filesystem(tmp_path / "runtime")
    await state.initialize(namespace="effect-io", tenant_id="tenant")
    try:
        run = _run()
        await state.steps.register_run(run)
        now = datetime.now(timezone.utc)
        await state.steps.record_tool_effect(
            ToolEffectRecord(
                tool_call_id="call",
                tool_name="tool",
                run_id=run.run_id,
                status="started",
                started_at=now,
                ended_at=None,
                idempotency_key="idempotency",
                effect_summary=None,
            )
        )
        generation = next((path for path in (tmp_path / "runtime" / "recovery").rglob("generation")), None)
        assert generation is not None
        assert generation.read_text(encoding="utf-8") == "1"
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_filesystem_speculative_callback_runs_once(tmp_path: Path) -> None:
    store = FilesystemStateStore(
        tmp_path / "state",
        namespace="mutation-io",
        tenant_id="tenant",
        runtime_domain="execution",
    )
    await store.initialize()
    calls = 0

    async def callback(transaction: StateTransaction) -> int:
        nonlocal calls
        calls += 1
        return await transaction.get_sequence(b"x" * 32)

    try:
        assert await store.mutate(callback) == 0
        assert calls == 1
    finally:
        await store.close()


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
