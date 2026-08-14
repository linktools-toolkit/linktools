#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Public RuntimeResources composition checks for in-memory, filesystem and SQLite."""

import asyncio
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from linktools.ai import RuntimeDomain, RuntimeStorage, RuntimeStoragePlan, RuntimeStorageRoute
from linktools.ai.adapter import (
    FilesystemStepArchive,
    RuntimeStepPersistence,
    build_runtime_sql_metadata,
    build_filesystem_runtime,
    build_in_memory_runtime,
    open_runtime_persistence,
)
from linktools.ai.core import (
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    IdempotencyStatus,
    OperationKind,
    OperationStatus,
    ResourceKind,
    SessionStatus,
    StopReason,
    idempotency_key_hash,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.runtime import (
    ExecutionRecord,
    ExecutionStartClaim,
    ExecutionTerminalCommit,
    IdempotencyRecord,
    IdempotencyTerminalUpdate,
    MemoryRecord,
    OperationLedgerInput,
    ResultRecord,
    SessionRecord,
)
from linktools.ai.storage import ObjectRef, provision_sql, validate_sql
from pydantic_ai_harness.step_persistence import InMemoryStepStore
from sqlalchemy.ext.asyncio import create_async_engine

from tests.ai.persistence.helper import RuntimeResources, open_sql_resources


@pytest.mark.asyncio
async def test_in_memory_resources_owns_harness_store() -> None:
    runtime = build_in_memory_runtime(namespace="memory")
    await runtime.initialize()
    try:
        resources = RuntimeResources("memory", "memory", runtime.persistence, InMemoryStepStore())
        assert isinstance(resources.steps, InMemoryStepStore)
        assert resources.namespace == "memory"
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_filesystem_resources_uses_a_durable_step_archive(tmp_path: Path) -> None:
    runtime = build_filesystem_runtime(str(tmp_path), namespace="project")
    steps = FilesystemStepArchive(
        tmp_path,
        namespace="project",
        tenant_id="tenant",
        runtime_domain=RuntimeDomain.CONVERSATION,
    )
    await runtime.initialize()
    await steps.initialize()
    try:
        resources = RuntimeResources("filesystem", "project", runtime.persistence, steps)
        assert isinstance(resources.steps, FilesystemStepArchive)
        runtime_root = tmp_path / hashlib.sha256(b"project").hexdigest()
        assert (runtime_root / "steps" / "conversation").is_dir()
        assert not (tmp_path / "steps").exists()
    finally:
        await steps.close()
        await runtime.close()


@pytest.mark.asyncio
async def test_sqlite_resources_uses_the_shared_sql_database(tmp_path: Path) -> None:
    primary = tmp_path / "runtime.db"
    storage = RuntimeStorage.sqlite(str(primary))
    async with open_sql_resources(storage, namespace="namespace") as resources:
        assert isinstance(resources.steps, RuntimeStepPersistence)
    with sqlite3.connect(primary) as connection:
        tables = {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
    assert {"step_runs", "step_snapshots"} <= tables


@pytest.mark.asyncio
async def test_persistence_namespace_and_tenant_are_orthogonal(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'isolation.db'}")
    plan = RuntimeStoragePlan({RuntimeDomain.CONVERSATION: RuntimeStorageRoute.durable()})
    await provision_runtime_database(engine, plan=plan)
    storage = RuntimeStorage.sql(engine, plan=plan)
    now = datetime.now(timezone.utc)
    async with open_runtime_persistence(storage, namespace="runtime-a", tenant_id="runtime-a") as first, open_runtime_persistence(storage, namespace="runtime-b", tenant_id="runtime-b") as second:
        await first.domain.conversation.sessions.create(SessionRecord("shared", "runtime-a", "principal-a", "binding", SessionStatus.OPEN, 0, 0, None, {}, now, now, None))
        await second.domain.conversation.sessions.create(SessionRecord("shared", "runtime-b", "principal-b", "binding", SessionStatus.OPEN, 0, 0, None, {}, now, now, None))

        assert (await first.domain.conversation.sessions.get("shared", tenant_id="runtime-a")).owner_principal_id == "principal-a"
        assert (await second.domain.conversation.sessions.get("shared", tenant_id="runtime-b")).owner_principal_id == "principal-b"
        assert first.domain.namespace != second.domain.namespace
        with pytest.raises(AIError) as sql_tenant_error:
            await first.domain.conversation.sessions.get("shared", tenant_id="runtime-b")
        assert sql_tenant_error.value.code is ErrorCode.STORAGE_NOT_FOUND
        memory_record = MemoryRecord("memory", "runtime-a", "a" * 64, "content", "digest", {}, 0, now, now)
        foreign_operation = OperationLedgerInput(
            "operation", "runtime-b", ResourceKind.MEMORY, "memory", None,
            OperationKind.MEMORY_WRITE, OperationStatus.SUCCEEDED, "request",
            None, None, None, True, now, now,
        )
        with pytest.raises(AIError) as sql_atomic_tenant_error:
            await first.domain.memory.records.put_with_operation(memory_record, expected_revision=0, operation=foreign_operation)
        assert sql_atomic_tenant_error.value.code is ErrorCode.REQUEST_FIELD_INVALID
    await engine.dispose()

    memory = build_in_memory_runtime(namespace="tenant-validation")
    await memory.initialize()
    try:
        with pytest.raises(AIError) as memory_tenant_error:
            await memory.persistence.conversation.sessions.get("shared", tenant_id="tenant-a ")
        assert memory_tenant_error.value.code is ErrorCode.REQUEST_FIELD_INVALID
        with pytest.raises(AIError) as memory_atomic_tenant_error:
            await memory.persistence.memory.records.put_with_operation(memory_record, expected_revision=0, operation=foreign_operation)
        assert memory_atomic_tenant_error.value.code is ErrorCode.REQUEST_FIELD_INVALID
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_storage_database_preparation_owns_sqlite_dialect_setup(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'runtime.db'}")
    plan = RuntimeStoragePlan({RuntimeDomain.CONVERSATION: RuntimeStorageRoute.durable()})
    metadata = build_runtime_sql_metadata(plan)
    try:
        await provision_sql(engine, metadata)
        await validate_sql(engine, metadata)
        async with engine.connect() as connection:
            from sqlalchemy import inspect

            tables = await connection.run_sync(lambda sync_connection: set(inspect(sync_connection).get_table_names()))
        assert {"runtime_sessions", "runtime_operation_counters", "runtime_operations"} <= tables
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_sql_schema_validation_allows_other_owner_tables(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'runtime.db'}")
    plan = RuntimeStoragePlan({RuntimeDomain.CONVERSATION: RuntimeStorageRoute.durable()})
    metadata = build_runtime_sql_metadata(plan)
    try:
        await provision_sql(engine, metadata)
        await validate_sql(engine, metadata)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_memory_claim_start_has_one_winner_and_reserves_segment() -> None:
    runtime = build_in_memory_runtime(namespace="claim")
    await runtime.initialize()
    now = datetime.now(timezone.utc)
    execution = ExecutionRecord(execution_id="execution", tenant_id="tenant", session_id=None, binding_digest="binding", parent_execution_id=None, root_execution_id="execution", source_execution_id=None, base_execution_id=None, lineage_kind=ExecutionLineageKind.RUN, status=ExecutionStatus.PENDING_START, revision=0, event_sequence=0, agent_run_sequence=0, error_code=None, safe_error_details={}, created_at=now, updated_at=now)
    key_hash = idempotency_key_hash("request")
    await runtime.persistence.execution.executions.create(execution)
    await runtime.persistence.execution.idempotency.reserve(IdempotencyRecord("tenant", RuntimeDomain.EXECUTION, "run", key_hash, "digest", ResourceKind.EXECUTION, "execution", IdempotencyStatus.RESERVED, None, None, now, now))
    claim = ExecutionStartClaim("execution", "tenant", 0, 0, "run", key_hash, "digest", now)
    outcomes = await asyncio.gather(runtime.persistence.execution.executions.claim_start(claim), runtime.persistence.execution.executions.claim_start(claim), return_exceptions=True)
    assert sum(isinstance(item, ExecutionRecord) for item in outcomes) == 1
    current = await runtime.persistence.execution.executions.get("execution", tenant_id="tenant")
    assert current is not None and current.agent_run_sequence == 1 and current.event_sequence == 1
    await runtime.close()


@pytest.mark.asyncio
async def test_sqlite_claim_start_updates_the_runtime_resources_atomically(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    storage = RuntimeStorage.sqlite(str(tmp_path / "claim.db"), plan=RuntimeStoragePlan.all())
    async with open_sql_resources(storage, namespace="claim") as resources:
        execution = ExecutionRecord(execution_id="execution", tenant_id="claim", session_id=None, binding_digest="binding", parent_execution_id=None, root_execution_id="execution", source_execution_id=None, base_execution_id=None, lineage_kind=ExecutionLineageKind.RUN, status=ExecutionStatus.PENDING_START, revision=0, event_sequence=0, agent_run_sequence=0, error_code=None, safe_error_details={}, created_at=now, updated_at=now)
        key_hash = idempotency_key_hash("request")
        await resources.domain.execution.executions.create(execution)
        await resources.domain.execution.idempotency.reserve(IdempotencyRecord("claim", RuntimeDomain.EXECUTION, "run", key_hash, "digest", ResourceKind.EXECUTION, "execution", IdempotencyStatus.RESERVED, None, None, now, now))
        claimed = await resources.domain.execution.executions.claim_start(ExecutionStartClaim("execution", "claim", 0, 0, "run", key_hash, "digest", now))
        assert claimed.status is ExecutionStatus.STARTED
        assert claimed.agent_run_sequence == 1


@pytest.mark.asyncio
async def test_memory_terminal_aggregate_commits_event_idempotency_without_session_head() -> None:
    runtime = build_in_memory_runtime(namespace="terminal")
    await runtime.initialize()
    now = datetime.now(timezone.utc)
    await runtime.persistence.conversation.sessions.create(SessionRecord(session_id="session", tenant_id="tenant", owner_principal_id="owner", binding_digest="binding", status=SessionStatus.OPEN, revision=0, resource_generation=0, cwd=None, metadata={}, created_at=now, updated_at=now, closed_at=None, continuation=None))
    execution = ExecutionRecord(execution_id="execution", tenant_id="tenant", session_id="session", binding_digest="binding", parent_execution_id=None, root_execution_id="execution", source_execution_id=None, base_execution_id=None, lineage_kind=ExecutionLineageKind.SESSION_RESUME, status=ExecutionStatus.STARTED, revision=1, event_sequence=0, agent_run_sequence=1, error_code=None, safe_error_details={}, created_at=now, updated_at=now)
    await runtime.persistence.execution.executions.create(execution)
    key_hash = idempotency_key_hash("terminal")
    await runtime.persistence.execution.idempotency.reserve(IdempotencyRecord("tenant", RuntimeDomain.EXECUTION, "execution.run", key_hash, "request", ResourceKind.EXECUTION, "execution", IdempotencyStatus.STARTED, None, None, now, now))
    terminal = ExecutionRecord(execution_id="execution", tenant_id="tenant", session_id="session", binding_digest="binding", parent_execution_id=None, root_execution_id="execution", source_execution_id=None, base_execution_id=None, lineage_kind=ExecutionLineageKind.SESSION_RESUME, status=ExecutionStatus.SUCCEEDED, revision=2, event_sequence=1, agent_run_sequence=1, error_code=None, safe_error_details={}, created_at=now, updated_at=now)
    result = ResultRecord("execution", "tenant", "text", 1, "digest", ObjectRef("memory", "result", "a" * 64, 0), StopReason.END_TURN, 0, 0, 0, now)
    await runtime.persistence.execution.commit_terminal(ExecutionTerminalCommit(1, 0, terminal, result, ExecutionEventType.EXECUTION_SUCCEEDED, {}, IdempotencyTerminalUpdate("execution.run", key_hash, IdempotencyStatus.STARTED, IdempotencyStatus.COMPLETED, "request", "digest", None), None))
    assert (await runtime.persistence.execution.idempotency.get("execution.run", key_hash, tenant_id="tenant")).status is IdempotencyStatus.COMPLETED
    events = await runtime.persistence.execution.events.list("execution", tenant_id="tenant", after_sequence=0, limit=10)
    assert tuple(item.event_type for item in events.items) == (ExecutionEventType.EXECUTION_SUCCEEDED,)
    await runtime.close()


@pytest.mark.asyncio
async def test_file_runtime_fault_does_not_publish_a_partial_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = build_filesystem_runtime(str(tmp_path), namespace="fault")
    await runtime.initialize()
    from linktools.ai.adapter import _persistence as runtime_persistence
    original = runtime_persistence.write_json_atomic
    failed = False

    def fail_once(path: object, value: object, *, fsync: bool = False) -> None:
        nonlocal failed
        if not failed and str(path).endswith("conversation/records.json"):
            failed = True
            raise OSError("injected write failure")
        original(path, value, fsync=fsync)

    monkeypatch.setattr(runtime_persistence, "write_json_atomic", fail_once)
    now = datetime.now(timezone.utc)
    with pytest.raises(AIError) as error:
        await runtime.persistence.conversation.sessions.create(SessionRecord("session", "fault", "owner", "binding", SessionStatus.OPEN, 0, 0, None, {}, now, now, None))
    assert error.value.code is ErrorCode.STORAGE_UNAVAILABLE
    monkeypatch.setattr(runtime_persistence, "write_json_atomic", original)
    assert await runtime.persistence.conversation.sessions.get("session", tenant_id="fault") is None
    await runtime.close()
