#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime mutation transaction and restart evidence."""

import asyncio
import hashlib
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path

import pytest
from linktools.ai.persistence import Persistence
from linktools.ai.agent import AgentBindingContract
from linktools.ai.agent._output import bind_output
from linktools.ai.core import (
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    ToolOperationStatus,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceDomain, PersistenceStorage
from linktools.ai.runtime._tool import ToolOperationRecord
from linktools.ai.persistence.contracts import ExecutionRecord, ToolOperationAdmission
from linktools.ai.persistence.commands import StateCommands
from linktools.ai.spec import AgentSpec
from ._runtime_test_helpers import execution_owner_fields


def _binding() -> AgentBindingContract:
    output = bind_output()
    return AgentBindingContract(
        agent_spec=AgentSpec("agent", model="default", tool_retries=3, output_retries=3),
        model_contract={"route_id": "default", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
    )


def _tool_record() -> ToolOperationRecord:
    now = datetime.now(timezone.utc)
    return ToolOperationRecord(
        tool_operation_id="operation",
        execution_id="execution",
        agent_run_id="run",
        tool_call_id="call",
        tool_name="tool",
        arguments_digest="b" * 64,
        binding_digest="c" * 64,
        replay_safe=False,
        status=ToolOperationStatus.PENDING,
        owner=None,
        fence=0,
        error_code=None,
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_configured_runtime_mutation_requires_active_transaction() -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="mutation-guard", tenant_id="tenant")
    try:
        with pytest.raises(RuntimeError, match="storage mutation outside transaction"):
            state.execution.executions._mark_changed()
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_effect_unknown_commits_before_error_and_survives_reopen(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    state = Persistence(PersistenceStorage.from_root(root, exclusive_writer=True))
    await state.initialize(namespace="effect-unknown", tenant_id="tenant")
    try:
        from .test_step_projection import _execution

        execution = replace(_execution(), revision=1, producer_generation=1, producer_claim_id="worker")
        await state.execution.executions.create_execution(execution)
        commands = StateCommands(state.execution.executions, namespace="effect-unknown",
            events=state.execution.events, tools=state.checkpoint.tools, background_tasks=set())
        record = _tool_record()
        request = ToolOperationAdmission(
            execution_id=record.execution_id, tool_operation_id=record.tool_operation_id,
            agent_run_id=record.agent_run_id, recovery_agent_run_id=None, tool_call_id=record.tool_call_id,
             tool_name=record.tool_name,
            arguments_digest=record.arguments_digest, binding_digest=record.binding_digest,
            replay_safe=False, owner="worker", producer_generation=1,
        )
        await commands.commit_tool_admission(request)
        successor = await state.execution.executions.claim_next_agent_run("execution", tenant_id="tenant",
            expected_revision=1, expected_agent_run_seq=1, producer_claim_id="worker-2")
        await commands.reconcile_tool_turn_takeover("operation", execution_id="execution",
            producer_generation=successor.producer_generation, producer_claim_id=successor.producer_claim_id)
        with pytest.raises(AIError) as error:
            await commands.commit_tool_admission(replace(request,
                owner=successor.producer_claim_id, producer_generation=successor.producer_generation))
        assert error.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
    finally:
        await state.close()

    reopened = Persistence(PersistenceStorage.from_root(root, exclusive_writer=True))
    await reopened.initialize(namespace="effect-unknown", tenant_id="tenant")
    try:
        record = await reopened.checkpoint.tools.get_operation(
            "operation",
            tenant_id="tenant",
        )
        assert record is not None and record.status is ToolOperationStatus.EFFECT_UNKNOWN
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_nested_event_mutation_persists_after_restart(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    state = Persistence(PersistenceStorage.from_root(root, exclusive_writer=True))
    await state.initialize(namespace="nested-event", tenant_id="tenant")
    now = datetime.now(timezone.utc)
    execution = ExecutionRecord(
        execution_id="execution",
        session_id=None,
        parent_execution_id=None,
        root_execution_id="execution",
        previous_execution_id=None,
        fork_base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=ExecutionStatus.STARTED,
        revision=0,
        event_seq=0,
        agent_run_seq=0,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        mode="run",
        planning=False,
        thinking=False,
        binding=_binding(),
        **execution_owner_fields(),
    )
    try:
        await state.execution.executions.create(execution)
        event = await state.execution.events.append_expected(
            "execution",
            tenant_id="tenant",
            expected_event_seq=0,
            event_type=ExecutionEventType.EXECUTION_STARTED,
            payload={},
        )
        assert event.event_seq == 1
    finally:
        await state.close()

    reopened = Persistence(PersistenceStorage.from_root(root, exclusive_writer=True))
    await reopened.initialize(namespace="nested-event", tenant_id="tenant")
    try:
        events = await reopened.execution.events.list(
            "execution",
            tenant_id="tenant",
            after_event_seq=0,
            limit=10,
        )
        assert tuple(item.event_type for item in events.items) == (
            ExecutionEventType.EXECUTION_STARTED,
        )
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_blob_stream_consumption_does_not_hold_runtime_transaction_lock() -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="blob-stream", tenant_id="tenant")
    started = asyncio.Event()
    release = asyncio.Event()
    data = b"stream"
    digest = hashlib.sha256(data).hexdigest()
    store = state.object_store(PersistenceDomain.EXECUTION)

    async def chunks() -> AsyncIterator[bytes]:
        started.set()
        await release.wait()
        yield data

    async def inline_chunks() -> AsyncIterator[bytes]:
        yield b"inline"

    try:
        stream = asyncio.create_task(
            store.put(
                "v1/stream",
                chunks(),
                expected_size=len(data),
                expected_digest=digest,
            )
        )
        await started.wait()
        inline = asyncio.create_task(
            store.put(
                "v1/inline",
                inline_chunks(),
                expected_size=6,
                expected_digest=hashlib.sha256(b"inline").hexdigest(),
            )
        )
        await asyncio.sleep(0)
        assert inline.done()
        release.set()
        await inline
        await stream
    finally:
        release.set()
        await state.close()
