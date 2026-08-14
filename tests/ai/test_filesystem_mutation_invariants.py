#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filesystem Runtime mutation transaction evidence."""

import asyncio
import hashlib
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path

import pytest
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._memory import build_filesystem_runtime, build_in_memory_runtime
from linktools.ai.core import (
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    ToolOperationStatus,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ExecutionRecord, ToolOperationRecord


def _tool_record() -> ToolOperationRecord:
    now = datetime.now(timezone.utc)
    return ToolOperationRecord(
        "operation",
        "tenant",
        "run",
        "call",
        "a" * 64,
        "tool",
        "arguments",
        "binding",
        False,
        ToolOperationStatus.PENDING,
        None,
        0,
        None,
        None,
        None,
        now,
        now,
    )


@pytest.mark.asyncio
async def test_configured_runtime_mutation_requires_active_transaction() -> None:
    runtime = build_in_memory_runtime(namespace="mutation-guard")
    await runtime.initialize()
    try:
        with pytest.raises(RuntimeError, match="storage mutation outside transaction"):
            runtime.components[0]._mark_changed()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_effect_unknown_commits_before_error_and_survives_reopen(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    runtime = build_filesystem_runtime(str(runtime_root), namespace="effect-unknown", persist=RuntimeDomain.RECOVERY)
    await runtime.initialize()
    try:
        await runtime.persistence.recovery.tools.reserve(_tool_record())
        await runtime.persistence.recovery.tools.claim("operation", tenant_id="tenant", owner="worker", lease_seconds=1)
        await asyncio.sleep(1.05)
        with pytest.raises(AIError) as error:
            await runtime.persistence.recovery.tools.claim("operation", tenant_id="tenant", owner="worker-2", lease_seconds=1)
        assert error.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
    finally:
        await runtime.close()

    reopened = build_filesystem_runtime(str(runtime_root), namespace="effect-unknown", persist=RuntimeDomain.RECOVERY)
    await reopened.initialize()
    try:
        record = await reopened.persistence.recovery.tools.get_operation("operation", tenant_id="tenant")
        assert record is not None and record.status is ToolOperationStatus.EFFECT_UNKNOWN
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_effect_unknown_storage_failure_wins_over_business_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = build_filesystem_runtime(str(tmp_path), namespace="effect-failure", persist=RuntimeDomain.RECOVERY)
    await runtime.initialize()
    try:
        await runtime.persistence.recovery.tools.reserve(_tool_record())
        await runtime.persistence.recovery.tools.claim("operation", tenant_id="tenant", owner="worker", lease_seconds=1)
        await asyncio.sleep(1.05)

        def fail_flush(_: RuntimeDomain) -> None:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

        monkeypatch.setattr(runtime, "_flush_domain", fail_flush)
        with pytest.raises(AIError) as error:
            await runtime.persistence.recovery.tools.claim("operation", tenant_id="tenant", owner="worker-2", lease_seconds=1)
        assert error.value.code is ErrorCode.STORAGE_UNAVAILABLE
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_nested_event_mutation_flushes_execution_domain_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = build_filesystem_runtime(str(tmp_path), namespace="nested-event", persist=RuntimeDomain.EXECUTION)
    await runtime.initialize()
    now = datetime.now(timezone.utc)
    execution = ExecutionRecord(
        "execution",
        "tenant",
        None,
        "binding",
        None,
        "execution",
        None,
        None,
        ExecutionLineageKind.RUN,
        ExecutionStatus.STARTED,
        0,
        0,
        0,
        None,
        {},
        now,
        now,
    )
    try:
        await runtime.persistence.execution.executions.create(execution)
        flushes: list[RuntimeDomain] = []
        original_flush = runtime._flush_domain

        def record_flush(domain: RuntimeDomain) -> None:
            flushes.append(domain)
            original_flush(domain)

        monkeypatch.setattr(runtime, "_flush_domain", record_flush)
        event = await runtime.persistence.execution.events.append(
            "execution",
            tenant_id="tenant",
            expected_sequence=0,
            event_type=ExecutionEventType.EXECUTION_STARTED,
            payload={},
        )
        assert event.sequence == 1
        assert flushes == [RuntimeDomain.EXECUTION]
    finally:
        await runtime.close()

    reopened = build_filesystem_runtime(str(tmp_path), namespace="nested-event", persist=RuntimeDomain.EXECUTION)
    await reopened.initialize()
    try:
        events = await reopened.persistence.execution.events.list("execution", tenant_id="tenant", after_sequence=0, limit=10)
        assert tuple(item.event_type for item in events.items) == (ExecutionEventType.EXECUTION_STARTED,)
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_blob_stream_consumption_does_not_hold_runtime_transaction_lock() -> None:
    runtime = build_in_memory_runtime(namespace="blob-stream")
    await runtime.initialize()
    started = asyncio.Event()
    release = asyncio.Event()
    data = b"stream"
    digest = hashlib.sha256(data).hexdigest()
    store = runtime.persistence.object_store(RuntimeDomain.EXECUTION)

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
                expected_size=len(b"inline"),
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
        await runtime.close()
