#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused evidence for the post-review task, SQL, and result fixes."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from linktools.ai import RuntimePersistenceConfig
from linktools.ai.adapter import build_in_memory_runtime
from linktools.ai.core import (
    ExecutionLineageKind,
    ExecutionStatus,
    Principal,
    ResourceKind,
    ResourceRef,
    StopReason,
    TaskStatus,
    TenantAuthorizationPolicy,
    Page,
    idempotency_key_hash,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import (
    DefaultExecutionService,
    ExecutionRecord,
    ResultRecord,
)
from linktools.ai.task import (
    CancelGraphRequest,
    DefaultTaskService,
    TaskGraph,
    TaskGraphRequest,
    TaskNode,
)

from tests.ai.persistence.helper import open_sql_resources


class _Launcher:
    def __init__(self) -> None:
        self.cancel_entered = asyncio.Event()
        self.cancel_release = asyncio.Event()
        self.cancel_calls = 0

    async def start(self, request: TaskGraphRequest) -> None:
        del request

    async def cancel(self, graph_id: str, request: CancelGraphRequest) -> None:
        del graph_id, request
        self.cancel_calls += 1
        self.cancel_entered.set()
        await self.cancel_release.wait()


@pytest.mark.asyncio
async def test_task_cancel_claim_replay_and_caller_cancellation() -> None:
    runtime = build_in_memory_runtime(namespace="review-fix-cancel")
    await runtime.initialize()
    launcher = _Launcher()
    service = DefaultTaskService(runtime.persistence, TenantAuthorizationPolicy(), launcher)
    principal = Principal("owner", "tenant")
    request = TaskGraphRequest(TaskGraph("graph", (TaskNode("task"),)), principal, "run-key")
    try:
        await service.run_graph(request)
        cancel_request = CancelGraphRequest(principal, "cancel-key")
        cancellation = asyncio.create_task(service.cancel_graph("graph", cancel_request))
        await launcher.cancel_entered.wait()
        cancellation.cancel()
        await asyncio.sleep(0)
        cancellation.cancel()
        launcher.cancel_release.set()
        with pytest.raises(asyncio.CancelledError):
            await cancellation

        operation = await runtime.persistence.operations.get(idempotency_key_hash("cancel-key"), tenant_id="tenant")
        assert operation is not None and operation.status.value == "SUCCEEDED"
        view = await service.cancel_graph("graph", cancel_request)
        assert view.status is TaskStatus.CANCELLED
        assert launcher.cancel_calls == 1
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_sql_task_lease_mutations_share_database_time_contract(tmp_path: Path) -> None:
    config = RuntimePersistenceConfig.sqlite(str(tmp_path / "runtime.db"), namespace="review-fix-sql", deployment_id="test")
    async with open_sql_resources(config) as resources:
        await resources.domain.tasks.create_plan(TaskGraph("graph", (TaskNode("task"),)), tenant_id="tenant")
        lease = await resources.domain.tasks.claim("graph", "task", tenant_id="tenant", owner="owner", lease_seconds=1)
        renewed = await resources.domain.tasks.renew(lease, tenant_id="tenant", lease_seconds=2)
        assert renewed.lease_expires_at > lease.lease_expires_at
        terminal = await resources.domain.tasks.complete(renewed, tenant_id="tenant", execution_id=None, result_digest="a" * 64)
        assert terminal.status is TaskStatus.SUCCEEDED
        with pytest.raises(AIError) as error:
            await resources.domain.tasks.renew(renewed, tenant_id="tenant", lease_seconds=1)
        assert error.value.code is ErrorCode.TASK_FENCE_STALE


class _ExecutionRepository:
    def __init__(self, record: ExecutionRecord) -> None:
        self.record = record

    async def get_header(self, execution_id: str, *, tenant_id: str) -> ResourceRef | None:
        if execution_id != self.record.execution_id or tenant_id != self.record.tenant_id:
            return None
        return ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id)

    async def get(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        if execution_id != self.record.execution_id or tenant_id != self.record.tenant_id:
            return None
        return self.record


class _ResultRepository:
    def __init__(self, record: ResultRecord) -> None:
        self.record = record

    async def get(self, execution_id: str, *, tenant_id: str) -> ResultRecord | None:
        if execution_id != self.record.execution_id or tenant_id != self.record.tenant_id:
            return None
        return self.record


class _ExecutionPersistence:
    def __init__(self, execution: ExecutionRecord, result: ResultRecord) -> None:
        self.executions = _ExecutionRepository(execution)
        self.results = _ResultRepository(result)


class _History:
    async def trace(self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int) -> "Page[object]":
        del execution_id, tenant_id, cursor, limit
        return Page((), None)

    async def transcript(self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int) -> "Page[object]":
        del execution_id, tenant_id, cursor, limit
        return Page((), None)

    async def history(self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int) -> "Page[object]":
        del execution_id, tenant_id, cursor, limit
        return Page((), None)


@pytest.mark.asyncio
async def test_failed_execution_result_requires_empty_payload_identity() -> None:
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
        ExecutionStatus.FAILED,
        1,
        0,
        0,
        None,
        None,
        "TASK_NODE_FAILED",
        {},
        now,
        now,
    )
    result = ResultRecord("execution", "tenant", ExecutionStatus.FAILED, "none", 1, "none", None, None, StopReason.ERROR, 0, 0, 0, now)
    persistence = _ExecutionPersistence(execution, result)
    service = DefaultExecutionService(persistence, TenantAuthorizationPolicy(), history_reader=_History())
    principal = Principal("owner", "tenant")
    response = await service.result("execution", principal=principal)
    assert response.status is ExecutionStatus.FAILED

    persistence.executions.record = replace(execution, result_ref="ref", result_digest="a" * 64)
    persistence.results.record = replace(result, payload_ref="ref", payload_digest="a" * 64)
    with pytest.raises(AIError) as error:
        await service.result("execution", principal=principal)
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
