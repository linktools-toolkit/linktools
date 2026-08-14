#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused evidence for the post-review task, SQL, and result fixes."""

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai import RuntimeDomain, RuntimeStorage, RuntimeStoragePlan, RuntimeStorageRoute
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
from linktools.ai.storage import resolve_dialect
from linktools.ai.task import (
    CancelGraphRequest,
    DefaultTaskService,
    TaskGraph,
    TaskGraphRequest,
    TaskGraphView,
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
    service = DefaultTaskService(runtime.persistence.task, TenantAuthorizationPolicy(), launcher)
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

        operation = await runtime.persistence.task.operations.get(idempotency_key_hash("cancel-key"), tenant_id="tenant")
        assert operation is not None and operation.status.value == "SUCCEEDED"
        view = await service.cancel_graph("graph", cancel_request)
        assert view.status is TaskStatus.CANCELLED
        assert launcher.cancel_calls == 1
    finally:
        await runtime.close()


@pytest.mark.parametrize(
    "finalizer_error",
    [AIError(ErrorCode.STORAGE_UNAVAILABLE), RuntimeError("finalizer failed")],
    ids=["ai-error", "ordinary-error"],
)
@pytest.mark.parametrize("caller_cancelled", [False, True], ids=["complete", "cancelled"])
@pytest.mark.asyncio
async def test_task_cancel_finalizer_outcome_precedence(
    finalizer_error: BaseException,
    caller_cancelled: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = build_in_memory_runtime(namespace="review-fix-cancel-outcome")
    await runtime.initialize()
    service = DefaultTaskService(runtime.persistence.task, TenantAuthorizationPolicy(), _Launcher())
    principal = Principal("owner", "tenant")
    try:
        await service.run_graph(TaskGraphRequest(TaskGraph("graph", (TaskNode("task"),)), principal, "run-key"))
        entered = asyncio.Event()
        release = asyncio.Event()

        async def failing_finalizer(
            graph_id: str,
            request: CancelGraphRequest,
            operation_id: str,
            request_digest: str,
        ) -> TaskGraphView:
            del graph_id, request, operation_id, request_digest
            entered.set()
            await release.wait()
            raise finalizer_error

        monkeypatch.setattr(service, "_cancel_finalizer", failing_finalizer)
        request = CancelGraphRequest(principal, "cancel-key")
        if not caller_cancelled:
            release.set()
            with pytest.raises(type(finalizer_error)) as error:
                await service.cancel_graph("graph", request)
            assert error.value is finalizer_error
            return

        cancellation = asyncio.create_task(service.cancel_graph("graph", request))
        await entered.wait()
        cancellation.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await cancellation
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_sql_database_now_uses_utc_expression_for_mysql() -> None:
    value = datetime(2026, 8, 12, 12, 0, 0)
    session = SimpleNamespace(
        bind=SimpleNamespace(dialect=SimpleNamespace(name="mysql")),
        scalar=None,
    )
    session.get_bind = lambda: session.bind

    async def scalar(statement: object) -> datetime:
        session.statement = statement
        return value

    session.scalar = scalar
    result = await resolve_dialect(session).database_now(session)
    assert result == value.replace(tzinfo=timezone.utc)
    assert "utc_timestamp" in str(session.statement).lower()


@pytest.mark.asyncio
async def test_sql_database_now_converts_postgresql_offset_to_utc() -> None:
    value = datetime(2026, 8, 12, 20, 0, 0, tzinfo=timezone(timedelta(hours=8)))
    session = SimpleNamespace(
        bind=SimpleNamespace(dialect=SimpleNamespace(name="postgresql")),
        scalar=None,
    )
    session.get_bind = lambda: session.bind

    async def scalar(statement: object) -> datetime:
        session.statement = statement
        return value

    session.scalar = scalar
    result = await resolve_dialect(session).database_now(session)
    assert result == datetime(2026, 8, 12, 12, 0, 0, tzinfo=timezone.utc)
    assert "now()" in str(session.statement).lower()


@pytest.mark.asyncio
async def test_sqlite_database_now_preserves_fractional_seconds() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            values = []
            for _ in range(3):
                values.append(await resolve_dialect(session).database_now(session))
                await asyncio.sleep(0.005)
            assert all(value.tzinfo is not None for value in values)
            assert any(value.microsecond >= 1_000 for value in values)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_sql_task_lease_mutations_use_aware_expiry(tmp_path: Path) -> None:
    storage = RuntimeStorage.sqlite(
        str(tmp_path / "runtime.db"),
        plan=RuntimeStoragePlan({RuntimeDomain.TASK: RuntimeStorageRoute.durable()}),
    )
    async with open_sql_resources(storage, namespace="review-fix-sql") as resources:
        await resources.domain.task.tasks.create_graph(TaskGraph("graph", (TaskNode("task"),)), tenant_id="review-fix-sql")
        plan = await resources.domain.task.tasks.reconcile_graph("graph", tenant_id="review-fix-sql")
        assert plan.status is TaskStatus.READY
        lease = await resources.domain.task.tasks.claim("graph", "task", tenant_id="review-fix-sql", owner="owner", lease_seconds=1)
        now = datetime.now(timezone.utc)
        assert now < lease.lease_expires_at <= now + timedelta(seconds=2)
        renewed = await resources.domain.task.tasks.renew(lease, tenant_id="review-fix-sql", lease_seconds=2)
        assert renewed.lease_expires_at > lease.lease_expires_at
        terminal = await resources.domain.task.tasks.complete(renewed, tenant_id="review-fix-sql", execution_id=None, result_digest="a" * 64)
        assert terminal.status is TaskStatus.SUCCEEDED
        with pytest.raises(AIError) as error:
            await resources.domain.task.tasks.renew(renewed, tenant_id="review-fix-sql", lease_seconds=1)
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
        execution_repository = _ExecutionRepository(execution)
        result_repository = _ResultRepository(result)
        self.execution = SimpleNamespace(
            executions=execution_repository,
            results=result_repository,
            idempotency=None,
            blobs=None,
            get_result=result_repository.get,
        )


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
        "TASK_NODE_FAILED",
        {},
        now,
        now,
    )
    result = ResultRecord("execution", "tenant", None, None, None, None, StopReason.ERROR, 0, 0, 0, now)
    persistence = _ExecutionPersistence(execution, result)
    service = DefaultExecutionService(persistence, TenantAuthorizationPolicy(), history_reader=_History())
    principal = Principal("owner", "tenant")
    response = await service.result("execution", principal=principal)
    assert response.status is ExecutionStatus.FAILED

    persistence.execution.results.record = replace(
        result,
        output_schema_id="text",
        output_schema_revision=1,
        output_schema_fingerprint="digest",
        object_ref=object(),
    )
    with pytest.raises(AIError) as error:
        await service.result("execution", principal=principal)
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
