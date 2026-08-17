#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution history projection for claimed and materialized attempts."""

from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord, StepEvent

from linktools.ai.adapter import StepExecutionHistoryReader
from linktools.ai.core import (
    ExecutionLineageKind,
    ExecutionStatus,
    HmacCursorSigner,
    step_conversation_id,
    step_run_id,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime.state import ExecutionRecord, RuntimeDomain


def _record(status: ExecutionStatus, sequence: int) -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    return ExecutionRecord(
        execution_id="execution",
        tenant_id="tenant",
        session_id=None,
        binding_digest="binding",
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=status,
        revision=0,
        event_sequence=0,
        agent_run_sequence=sequence,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
    )


async def _materialize_attempt(state: RuntimeState, sequence: int, prompt: str) -> None:
    run_id = step_run_id(
        namespace="history",
        tenant_id="tenant",
        execution_id="execution",
        segment_sequence=sequence,
    )
    conversation_id = step_conversation_id(
        namespace="history",
        tenant_id="tenant",
        execution_id="execution",
    )
    now = datetime.now(timezone.utc)
    await state.steps.register_run(
        RunRecord(
            run_id=run_id,
            conversation_id=conversation_id,
            parent_run_id=None,
            agent_name="default",
            metadata={
                "segment_sequence": str(sequence),
                "agent_name": "default",
            },
            started_at=now,
        )
    )
    await state.steps.append_event(
        StepEvent(
            run_id=run_id,
            kind="model_request_started",
            step_index=1,
            timestamp=now,
            conversation_id=conversation_id,
            agent_name="default",
        )
    )
    await state.steps.append_event(
        StepEvent(
            run_id=run_id,
            kind="model_request_completed",
            step_index=2,
            timestamp=now,
            conversation_id=conversation_id,
            agent_name="default",
        )
    )
    await state.steps.save_snapshot(
        ContinuableSnapshot(
            run_id=run_id,
            step_index=2,
            messages=[
                ModelRequest(
                    parts=[UserPromptPart(content=prompt)],
                    conversation_id=conversation_id,
                ),
                ModelResponse(
                    parts=[TextPart(content="response")],
                    conversation_id=conversation_id,
                ),
            ],
            conversation_id=conversation_id,
            parent_run_id=None,
            agent_name="default",
            timestamp=now,
        )
    )


def _reader(state: RuntimeState) -> StepExecutionHistoryReader:
    return StepExecutionHistoryReader(
        namespace="history",
        executions=state.execution.executions,
        store=state.steps.read_store(RuntimeDomain.EXECUTION),
        cursor_signer=HmacCursorSigner("history", b"history-key"),
    )


@pytest.mark.asyncio
async def test_failed_claimed_attempt_without_run_is_skipped() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="history", tenant_id="tenant")
    try:
        await state.execution.executions.create(_record(ExecutionStatus.FAILED, 1))
        reader = _reader(state)

        history = await reader.history("execution", tenant_id="tenant", cursor=None, limit=200)
        trace = await reader.trace("execution", tenant_id="tenant", cursor=None, limit=200)
        transcript = await reader.transcript("execution", tenant_id="tenant", cursor=None, limit=200)

        assert history.items == ()
        assert trace.items == ()
        assert transcript.items == ()
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_history_skips_missing_non_final_attempt() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="history", tenant_id="tenant")
    try:
        await state.execution.executions.create(_record(ExecutionStatus.FAILED, 2))
        await _materialize_attempt(state, 2, "attempt-2")
        await state.retention.release_execution_handoff("execution", tenant_id="tenant")
        reader = _reader(state)

        history = await reader.history("execution", tenant_id="tenant", cursor=None, limit=200)
        trace = await reader.trace("execution", tenant_id="tenant", cursor=None, limit=200)
        transcript = await reader.transcript("execution", tenant_id="tenant", cursor=None, limit=200)

        assert [item.content for item in history.items] == ["attempt-2", "response"]
        assert [item.payload["segment_sequence"] for item in trace.items] == [2, 2]
        assert [item.text for item in transcript.items] == ["attempt-2", "response"]
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", ("history", "trace", "transcript"))
async def test_successful_execution_requires_final_history_evidence(method_name: str) -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="history", tenant_id="tenant")
    try:
        await state.execution.executions.create(_record(ExecutionStatus.SUCCEEDED, 1))
        reader = _reader(state)
        method = getattr(reader, method_name)

        with pytest.raises(AIError) as error:
            await method("execution", tenant_id="tenant", cursor=None, limit=200)

        assert error.value.code is ErrorCode.EXECUTION_HISTORY_UNAVAILABLE
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_successful_history_preserves_user_prompt_and_projects_all_views() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="history", tenant_id="tenant")
    try:
        prompt = '  {"question":"你好\\nworld","x":1}  '
        await state.execution.executions.create(_record(ExecutionStatus.SUCCEEDED, 1))
        await _materialize_attempt(state, 1, prompt)
        await state.retention.release_execution_handoff("execution", tenant_id="tenant")
        reader = _reader(state)

        history = await reader.history("execution", tenant_id="tenant", cursor=None, limit=200)
        trace = await reader.trace("execution", tenant_id="tenant", cursor=None, limit=200)
        transcript = await reader.transcript("execution", tenant_id="tenant", cursor=None, limit=200)

        assert [item.content for item in history.items] == [prompt, "response"]
        assert [item.payload["segment_sequence"] for item in trace.items] == [1, 1]
        assert [item.text for item in transcript.items] == [prompt, "response"]
    finally:
        await state.close()
