#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filesystem Runtime mutation transaction evidence."""

import asyncio
import hashlib
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path

import pytest
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.agent._output import bind_output
from linktools.ai.core import (
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    ToolOperationStatus,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeDomain, RuntimeState
from linktools.ai.runtime._tool import ToolOperationRecord
from linktools.ai.runtime.state._contracts import ExecutionRecord
from linktools.ai.spec import AgentSpec


def _binding() -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", model="default"),
        model={"route_id": "default", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
        binding_digest="a" * 64,
    )


def _tool_record() -> ToolOperationRecord:
    now = datetime.now(timezone.utc)
    return ToolOperationRecord(
        tool_operation_id="operation",
        tenant_id="tenant",
        step_run_id="run",
        tool_call_id="call",
        idempotency_key_digest="a" * 64,
        tool_name="tool",
        arguments_digest="b" * 64,
        binding_digest="c" * 64,
        replay_safe=False,
        status=ToolOperationStatus.PENDING,
        owner=None,
        fence=0,
        lease_expires_at=None,
        error_code=None,
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_configured_runtime_mutation_requires_active_transaction() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="mutation-guard", tenant_id="tenant")
    try:
        with pytest.raises(RuntimeError, match="storage mutation outside transaction"):
            state.execution.executions._mark_changed()
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_effect_unknown_commits_before_error_and_survives_reopen(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    state = RuntimeState.filesystem(root)
    await state.initialize(namespace="effect-unknown", tenant_id="tenant")
    try:
        await state.recovery.tools.reserve(_tool_record())
        await state.recovery.tools.claim(
            "operation",
            tenant_id="tenant",
            owner="worker",
            lease_seconds=1,
        )
        await asyncio.sleep(1.05)
        with pytest.raises(AIError) as error:
            await state.recovery.tools.claim(
                "operation",
                tenant_id="tenant",
                owner="worker-2",
                lease_seconds=1,
            )
        assert error.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
    finally:
        await state.close()

    reopened = RuntimeState.filesystem(root)
    await reopened.initialize(namespace="effect-unknown", tenant_id="tenant")
    try:
        record = await reopened.recovery.tools.get_operation(
            "operation",
            tenant_id="tenant",
        )
        assert record is not None and record.status is ToolOperationStatus.EFFECT_UNKNOWN
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_nested_event_mutation_persists_after_restart(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    state = RuntimeState.filesystem(root)
    await state.initialize(namespace="nested-event", tenant_id="tenant")
    now = datetime.now(timezone.utc)
    execution = ExecutionRecord(
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
        agent_run_sequence=0,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        mode="run",
        planning=False,
        thinking=False,
        binding=_binding(),
    )
    try:
        await state.execution.executions.create(execution)
        event = await state.execution.events.append(
            "execution",
            tenant_id="tenant",
            expected_sequence=0,
            event_type=ExecutionEventType.EXECUTION_STARTED,
            payload={},
        )
        assert event.sequence == 1
    finally:
        await state.close()

    reopened = RuntimeState.filesystem(root)
    await reopened.initialize(namespace="nested-event", tenant_id="tenant")
    try:
        events = await reopened.execution.events.list(
            "execution",
            tenant_id="tenant",
            after_sequence=0,
            limit=10,
        )
        assert tuple(item.event_type for item in events.items) == (
            ExecutionEventType.EXECUTION_STARTED,
        )
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_blob_stream_consumption_does_not_hold_runtime_transaction_lock() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="blob-stream", tenant_id="tenant")
    started = asyncio.Event()
    release = asyncio.Event()
    data = b"stream"
    digest = hashlib.sha256(data).hexdigest()
    store = state.object_store(RuntimeDomain.EXECUTION)

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
