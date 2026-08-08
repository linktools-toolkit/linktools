#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Public RuntimeResources composition checks for in-memory, filesystem and SQLite."""

import sqlite3
import hashlib
from datetime import datetime, timezone
import asyncio
from pathlib import Path

import pytest
from pydantic_ai_harness.step_persistence import InMemoryStepStore, SqliteStepStore

from linktools.ai import RuntimePersistenceConfig, RuntimeResources, namespace_scoped_step_db_path, open_runtime_resources
from linktools.ai.adapter import DurableFilesystemStepStore, build_filesystem_runtime, build_in_memory_runtime
from linktools.ai.errors import ErrorCode, AIError
from linktools.ai.core import idempotency_key_hash
from linktools.ai.core import ExecutionEventType, ExecutionLineageKind, ExecutionStatus, IdempotencyStatus, SessionStatus, StopReason
from linktools.ai.runtime import ExecutionRecord, ExecutionStartClaim, ExecutionTerminalCommit, IdempotencyRecord, IdempotencyTerminalUpdate, ResultRecord, SessionHeadAdvance, SessionRecord
from tests.ai.persistence.helper import open_sql_resources


@pytest.mark.asyncio
async def test_in_memory_resources_owns_harness_store() -> None:
    async with open_runtime_resources(RuntimePersistenceConfig.in_memory(namespace="memory")) as resources:
        assert isinstance(resources, RuntimeResources)
        assert isinstance(resources.steps, InMemoryStepStore)
        assert resources.namespace == "memory"


@pytest.mark.asyncio
async def test_filesystem_resources_uses_shared_runtime_writer_and_durable_step_store(tmp_path: Path) -> None:
    async with open_runtime_resources(RuntimePersistenceConfig.filesystem(str(tmp_path), workspace_id="project")) as resources:
        assert isinstance(resources.steps, DurableFilesystemStepStore)
        runtime_root = tmp_path / ".linktools" / "runtime" / hashlib.sha256(b"project").hexdigest()
        assert (runtime_root / "steps").is_dir()
        assert not (tmp_path / "steps").exists()


@pytest.mark.asyncio
async def test_sqlite_resources_uses_sibling_harness_database(tmp_path: Path) -> None:
    primary = tmp_path / "runtime.db"
    config = RuntimePersistenceConfig.sqlite(str(primary), namespace="namespace", deployment_id="deployment")
    async with open_sql_resources(config) as resources:
        assert isinstance(resources.steps, SqliteStepStore)
    with sqlite3.connect(primary) as connection:
        tables = {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
    assert not any(name.startswith("ai_step_") for name in tables)
    sibling = tmp_path / f"runtime.db.steps.{hashlib.sha256(b'namespace').hexdigest()}.db"
    assert namespace_scoped_step_db_path(primary, "namespace") == sibling


@pytest.mark.asyncio
async def test_sql_resources_require_downstream_session_factory(tmp_path: Path) -> None:
    config = RuntimePersistenceConfig.sqlite(str(tmp_path / "runtime.db"), namespace="namespace", deployment_id="deployment")
    with pytest.raises(AIError) as error:
        async with open_runtime_resources(config):
            pass
    assert error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY


@pytest.mark.asyncio
async def test_memory_claim_start_has_one_winner_and_reserves_segment() -> None:
    runtime = build_in_memory_runtime(namespace="claim")
    await runtime.initialize()
    now = datetime.now(timezone.utc)
    execution = ExecutionRecord(execution_id="execution", tenant_id="tenant", session_id=None, binding_digest="binding", parent_execution_id=None, root_execution_id="execution", source_execution_id=None, base_execution_id=None, lineage_kind=ExecutionLineageKind.RUN, status=ExecutionStatus.PENDING_START, revision=0, event_sequence=0, agent_run_sequence=0, result_ref=None, result_digest=None, error_code=None, safe_error_details={}, created_at=now, updated_at=now)
    key_hash = idempotency_key_hash("request")
    await runtime.persistence.executions.create(execution)
    await runtime.persistence.idempotency.reserve(IdempotencyRecord("tenant", "run", key_hash, "digest", "execution", IdempotencyStatus.RESERVED, None, None, now, now))
    claim = ExecutionStartClaim("execution", "tenant", 0, 0, "run", key_hash, "digest", now)
    outcomes = await asyncio.gather(runtime.persistence.executions.claim_start(claim), runtime.persistence.executions.claim_start(claim), return_exceptions=True)
    assert sum(isinstance(item, ExecutionRecord) for item in outcomes) == 1
    current = await runtime.persistence.executions.get("execution", tenant_id="tenant")
    assert current is not None and current.agent_run_sequence == 1 and current.event_sequence == 1
    await runtime.close()


@pytest.mark.asyncio
async def test_sqlite_claim_start_updates_the_runtime_resources_atomically(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    config = RuntimePersistenceConfig.sqlite(str(tmp_path / "claim.db"), namespace="claim", deployment_id="test")
    async with open_sql_resources(config) as resources:
        execution = ExecutionRecord(execution_id="execution", tenant_id="tenant", session_id=None, binding_digest="binding", parent_execution_id=None, root_execution_id="execution", source_execution_id=None, base_execution_id=None, lineage_kind=ExecutionLineageKind.RUN, status=ExecutionStatus.PENDING_START, revision=0, event_sequence=0, agent_run_sequence=0, result_ref=None, result_digest=None, error_code=None, safe_error_details={}, created_at=now, updated_at=now)
        key_hash = idempotency_key_hash("request")
        await resources.domain.executions.create(execution)
        await resources.domain.idempotency.reserve(IdempotencyRecord("tenant", "run", key_hash, "digest", "execution", IdempotencyStatus.RESERVED, None, None, now, now))
        claimed = await resources.domain.executions.claim_start(ExecutionStartClaim("execution", "tenant", 0, 0, "run", key_hash, "digest", now))
        assert claimed.status is ExecutionStatus.STARTED
        assert claimed.agent_run_sequence == 1


@pytest.mark.asyncio
async def test_memory_terminal_aggregate_commits_event_idempotency_and_session_head() -> None:
    runtime = build_in_memory_runtime(namespace="terminal")
    await runtime.initialize()
    now = datetime.now(timezone.utc)
    await runtime.persistence.sessions.create(SessionRecord(session_id="session", tenant_id="tenant", owner_principal_id="owner", binding_digest="binding", status=SessionStatus.OPEN, revision=0, resource_generation=0, cwd=None, metadata={}, created_at=now, updated_at=now, closed_at=None, head_execution_id=None))
    execution = ExecutionRecord(execution_id="execution", tenant_id="tenant", session_id="session", binding_digest="binding", parent_execution_id=None, root_execution_id="execution", source_execution_id=None, base_execution_id=None, lineage_kind=ExecutionLineageKind.SESSION_RESUME, status=ExecutionStatus.STARTED, revision=1, event_sequence=0, agent_run_sequence=1, result_ref=None, result_digest=None, error_code=None, safe_error_details={}, created_at=now, updated_at=now)
    await runtime.persistence.executions.create(execution)
    key_hash = idempotency_key_hash("terminal")
    await runtime.persistence.idempotency.reserve(IdempotencyRecord("tenant", "execution.run", key_hash, "request", "execution", IdempotencyStatus.STARTED, None, None, now, now))
    terminal = ExecutionRecord(execution_id="execution", tenant_id="tenant", session_id="session", binding_digest="binding", parent_execution_id=None, root_execution_id="execution", source_execution_id=None, base_execution_id=None, lineage_kind=ExecutionLineageKind.SESSION_RESUME, status=ExecutionStatus.SUCCEEDED, revision=2, event_sequence=1, agent_run_sequence=1, result_ref="digest", result_digest="digest", error_code=None, safe_error_details={}, created_at=now, updated_at=now)
    result = ResultRecord("execution", "tenant", ExecutionStatus.SUCCEEDED, "none", 1, "none", "digest", "digest", StopReason.END_TURN, 0, 0, 0, now)
    await runtime.persistence.results.commit_terminal(ExecutionTerminalCommit(1, 0, terminal, result, ExecutionEventType.EXECUTION_SUCCEEDED, {}, IdempotencyTerminalUpdate("execution.run", key_hash, IdempotencyStatus.STARTED, IdempotencyStatus.COMPLETED, "request", "digest", None), None, SessionHeadAdvance("session", None, "execution")))
    assert (await runtime.persistence.idempotency.get("execution.run", key_hash, tenant_id="tenant")).status is IdempotencyStatus.COMPLETED
    assert (await runtime.persistence.sessions.get("session", tenant_id="tenant")).head_execution_id == "execution"
    events = await runtime.persistence.events.list("execution", tenant_id="tenant", after_sequence=0, limit=10)
    assert tuple(item.event_type for item in events.items) == (ExecutionEventType.EXECUTION_SUCCEEDED,)
    await runtime.close()


@pytest.mark.asyncio
async def test_file_runtime_fault_does_not_publish_a_partial_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = build_filesystem_runtime(str(tmp_path), workspace_id="fault")
    await runtime.initialize()
    from linktools.ai.adapter import _persistence as runtime_persistence
    original = runtime_persistence.write_json_atomic
    failed = False

    def fail_once(path: object, value: object, *, fsync: bool = False) -> None:
        nonlocal failed
        if not failed and str(path).endswith("state.json"):
            failed = True
            raise OSError("injected write failure")
        original(path, value, fsync=fsync)

    monkeypatch.setattr(runtime_persistence, "write_json_atomic", fail_once)
    now = datetime.now(timezone.utc)
    with pytest.raises(AIError) as error:
        await runtime.persistence.sessions.create(SessionRecord("session", "fault", "owner", "binding", SessionStatus.OPEN, 0, 0, None, {}, now, now, None))
    assert error.value.code is ErrorCode.STORAGE_UNAVAILABLE
    monkeypatch.setattr(runtime_persistence, "write_json_atomic", original)
    assert await runtime.persistence.sessions.get("session", tenant_id="fault") is None
    await runtime.close()
