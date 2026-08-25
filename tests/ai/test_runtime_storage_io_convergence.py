#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime filesystem and SQL physical-resource convergence evidence."""

import asyncio
import hashlib
import re
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.agent._capabilities import _RuntimeStepPersistence
from linktools.ai.agent._output import bind_output
from linktools.ai.core import (
    ExecutionLineageKind,
    ExecutionStatus,
    IdempotencyStatus,
    ResourceKind,
    SessionStatus,
    ToolOperationStatus,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import build_sql_schema_metadata, provision_database
from linktools.ai.runtime import RuntimeDomain, RuntimeState
from linktools.ai.runtime._tool import RuntimeToolOperationBridge
from linktools.ai.runtime.state import RuntimeStateCommands
from linktools.ai.runtime.state._contracts import (
    ExecutionRecord,
    ExecutionStartClaim,
    ExecutionStartReservation,
    IdempotencyRecord,
    RecoveryCheckpoint,
    RecoveryCheckpointState,
    RecoveryExecutionInput,
    RecoveryHandoffPhase,
    RecoveryIdempotencyInput,
    SessionRecord,
)
from linktools.ai.runtime.state._filesystem import FilesystemStateStore
from linktools.ai.runtime.state._materializer import materialize_runtime_state
from linktools.ai.runtime.state._plan import RuntimeStatePlan, RuntimeStateRoute
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import (
    FilesystemLeaseCoordinator,
    FilesystemJournal,
    FilesystemWriterLock,
    PayloadPolicy,
    SqlStorageContext,
    create_sql_storage_context,
)
from linktools.ai.storage import _database as database_module
from linktools.ai.storage import _files as files_module
from linktools.ai.storage import _object as object_module
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.step_persistence import RunRecord
from sqlalchemy import MetaData, event
from sqlalchemy.dialects import mysql
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.schema import CreateTable

pytestmark = pytest.mark.asyncio


class _SqlCursorCounter:
    def __init__(self, engine: object) -> None:
        self.entries: list[tuple[str, str, str, bool]] = []
        self.scope = ""
        self._engine = engine
        event.listen(engine.sync_engine, "before_cursor_execute", self._record)

    def close(self) -> None:
        event.remove(self._engine.sync_engine, "before_cursor_execute", self._record)

    def reset(self, scope: str) -> None:
        self.entries.clear()
        self.scope = scope

    def count(self, action: str, table: str) -> int:
        return sum(
            entry[1] == action and entry[2] == table
            for entry in self.entries
        )

    def executemany(self, action: str, table: str) -> bool:
        return any(
            entry[1] == action
            and entry[2] == table
            and entry[3]
            for entry in self.entries
        )

    def _record(
        self,
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        executemany: bool,
    ) -> None:
        normalized = re.sub(r"\s+", " ", statement.upper())
        action = normalized.split(" ", 1)[0]
        match = re.search(
            r"(?:FROM|INTO|UPDATE|JOIN) ([A-Z0-9_]+)",
            normalized,
        )
        table = "" if match is None else match.group(1)
        self.entries.append((self.scope, action, table, executemany))


def _binding_snapshot() -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", 1, "default"),
        agent_digest="d" * 64,
        output_schema_id=output.schema_id,
        output_schema_revision=output.schema_revision,
        output_schema_fingerprint=output.schema_fingerprint,
        local_runtime_capability_descriptors=(),
        binding_digest="a" * 64,
        global_runtime_capability_descriptors=(),
    )


def _tool_run(run_id: str) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        conversation_id="conversation",
        parent_run_id=None,
        agent_name="agent",
        metadata={},
        started_at=datetime.now(timezone.utc),
    )


async def test_sql_cursor_counter_captures_core_batch_gates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    await provision_database(engine)
    counter = _SqlCursorCounter(engine)
    state = RuntimeState.sql(engine)
    await state.initialize(namespace="cursor-gates", tenant_id="tenant")
    now = datetime.now(timezone.utc)
    session = SessionRecord(
        session_id="session",
        tenant_id="tenant",
        owner_principal_id="owner",
        agent_id="agent",
        status=SessionStatus.OPEN,
        revision=0,
        resource_generation=0,
        cwd=None,
        metadata={},
        created_at=now,
        updated_at=now,
        closed_at=None,
        active_execution_id=None,
    )
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
        status=ExecutionStatus.PENDING_START,
        revision=0,
        event_sequence=0,
        agent_run_sequence=0,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        planning=False,
        thinking=False,
        binding=_binding_snapshot(),
    )
    reserved_execution = replace(
        execution,
        execution_id="reserved",
        root_execution_id="reserved",
    )
    identity = IdempotencyRecord(
        "tenant",
        RuntimeDomain.EXECUTION,
        "scope",
        "key",
        "request",
        ResourceKind.EXECUTION,
        "reserved",
        IdempotencyStatus.RESERVED,
        None,
        None,
        now,
        now,
    )
    try:
        counter.reset("session.create")
        await state.conversation.sessions.create(session)
        assert counter.count("INSERT", "AI_STATE_RECORDS") == 1
        assert counter.count("INSERT", "AI_STATE_SEQUENCES") == 1

        counter.reset("execution.create_with_head")
        await state.execution.executions.create_with_history_head(execution)
        assert counter.count("SELECT", "AI_STATE_RECORDS") == 1
        assert counter.count("INSERT", "AI_STATE_RECORDS") == 1

        counter.reset("execution.reserve_start")
        await state.execution.executions.reserve_start(
            ExecutionStartReservation(reserved_execution, identity)
        )
        assert counter.count("SELECT", "AI_STATE_RECORDS") == 1
        assert counter.count("INSERT", "AI_STATE_RECORDS") == 1

        counter.reset("execution.claim_start")
        await state.execution.executions.claim_start(
            ExecutionStartClaim(
                "reserved",
                "tenant",
                0,
                0,
                "scope",
                "key",
                "request",
                now,
            )
        )
        assert counter.count("UPDATE", "AI_STATE_RECORDS") == 1
        assert counter.executemany("UPDATE", "AI_STATE_RECORDS")
        assert counter.count("INSERT", "AI_STATE_FACTS") == 1
    finally:
        counter.close()
        await state.close()
        await engine.dispose()


async def test_sql_state_group_maps_programming_failure_to_internal(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    await provision_database(engine)
    state = RuntimeState.sqlite(path)
    await state.initialize(namespace="sql-error-contract", tenant_id="tenant")
    store = state.execution.executions.state_store

    async def fail(_transaction: object) -> None:
        raise TypeError("boom")

    try:
        with pytest.raises(AIError) as raised:
            await store.storage_group.mutate((store,), fail)
        assert raised.value.code is ErrorCode.INTERNAL_ERROR
        assert raised.value.code is not ErrorCode.STORAGE_UNAVAILABLE
        assert raised.value.retryable is False
        assert raised.value.safe_details == {
            "phase": "runtime_state_sql_mutation"
        }
    finally:
        await state.close()
        await engine.dispose()


async def test_cli_sql_state_context_owns_one_engine_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linktools.commands.ai.run as run_module
    import sqlalchemy.ext.asyncio as sqlalchemy_asyncio

    created: list[object] = []
    provisioned: list[object] = []
    disposed: list[object] = []
    original_create = sqlalchemy_asyncio.create_async_engine
    original_dispose = sqlalchemy_asyncio.AsyncEngine.dispose
    original_provision = run_module.provision_runtime_database

    def create_engine(*args: object, **kwargs: object) -> object:
        engine = original_create(*args, **kwargs)
        created.append(engine)
        return engine

    async def provision(engine: object) -> None:
        provisioned.append(engine)
        await original_provision(engine)

    async def dispose(engine: object, *args: object, **kwargs: object) -> None:
        disposed.append(engine)
        await original_dispose(engine, *args, **kwargs)

    monkeypatch.setattr(sqlalchemy_asyncio, "create_async_engine", create_engine)
    monkeypatch.setattr(sqlalchemy_asyncio.AsyncEngine, "dispose", dispose)
    monkeypatch.setattr(run_module, "provision_runtime_database", provision)

    with pytest.raises(RuntimeError, match="request failed"):
        async with run_module._open_runtime_state(
            SimpleNamespace(storage_root=tmp_path),
            "sqlite",
        ) as state:
            assert state.plan.route(RuntimeDomain.EXECUTION).kind == "sqlite"
            assert disposed == created
            raise RuntimeError("request failed")

    assert len(created) == 1
    assert provisioned == created
    assert disposed == created


def _runtime_commands(
    state: RuntimeState,
    namespace: str,
    background_tasks: "set[asyncio.Task[object]] | None" = None,
) -> RuntimeStateCommands:
    return RuntimeStateCommands(
        state.execution.executions,
        namespace=namespace,
        events=state.execution.events,
        operations=state.execution.operations,
        conversation=state.conversation.sessions,
        recovery=state.recovery.checkpoints,
        conversation_history=state.conversation.histories,
        tools=state.recovery.tools,
        conversation_steps=state.steps.read_store(RuntimeDomain.CONVERSATION),
        execution_steps=state.steps.read_store(RuntimeDomain.EXECUTION),
        recovery_steps=state.steps.read_store(RuntimeDomain.RECOVERY),
        background_tasks=set() if background_tasks is None else background_tasks,
    )


async def test_sqlite_parallel_tool_lifecycle_uses_one_effect_writer(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.db"
    provisioning_engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    await provision_database(provisioning_engine)
    await provisioning_engine.dispose()

    state = RuntimeState.sqlite(path)
    await state.initialize(namespace="parallel-tools", tenant_id="tenant")
    try:
        run_id = "run"
        await state.steps.register_run(_tool_run(run_id))
        background_tasks: set[asyncio.Task[object]] = set()
        commands = _runtime_commands(
            state,
            "parallel-tools",
            background_tasks,
        )
        bridge = RuntimeToolOperationBridge(
            state.recovery.tools,
            state.object_store(RuntimeDomain.RECOVERY),
            namespace="parallel-tools",
            tenant_id="tenant",
            execution_id="execution",
            step_run_id=run_id,
            binding_fingerprint="a" * 64,
            owner="owner",
            background_tasks=background_tasks,
            payload_policy=PayloadPolicy(),
            terminal_commands=commands,
        )
        persistence = _RuntimeStepPersistence(
            tool_operations=bridge,
            store=state.steps,
            agent_name="agent",
            run_id=run_id,
        )
        call_ids = ("call-a", "call-b")
        entered = {call_id: asyncio.Event() for call_id in call_ids}
        release = asyncio.Event()
        handler_calls = {call_id: 0 for call_id in call_ids}

        async def execute(call_id: str) -> object:
            context = RunContext(
                deps=None,
                model=TestModel(),
                usage=RunUsage(),
                run_id=run_id,
            )
            call = ToolCallPart("tool", {}, tool_call_id=call_id)
            tool_def = ToolDefinition(
                name="tool",
                metadata={"linktools.ai.replay_safe": True},
            )
            await persistence.before_tool_execute(
                context,
                call=call,
                tool_def=tool_def,
                args={},
            )

            async def handler(_args: dict[str, object]) -> dict[str, str]:
                handler_calls[call_id] += 1
                entered[call_id].set()
                await release.wait()
                return {"call_id": call_id}

            result = await persistence.wrap_tool_execute(
                context,
                call=call,
                tool_def=tool_def,
                args={},
                handler=handler,
            )
            return await persistence.after_tool_execute(
                context,
                call=call,
                tool_def=tool_def,
                args={},
                result=result,
            )

        tasks = [asyncio.create_task(execute(call_id)) for call_id in call_ids]
        await asyncio.gather(*(entered[call_id].wait() for call_id in call_ids))
        release.set()
        assert await asyncio.gather(*tasks) == [
            {"call_id": "call-a"},
            {"call_id": "call-b"},
        ]

        recovery = state.steps.read_store(RuntimeDomain.RECOVERY)
        for call_id in call_ids:
            operation = await state.recovery.tools.get_by_call(
                run_id,
                call_id,
                tenant_id="tenant",
            )
            assert operation is not None
            assert operation.status is ToolOperationStatus.COMPLETED
            assert handler_calls[call_id] == 1
            effect = await recovery.get_tool_effect(
                run_id=run_id,
                tool_call_id=call_id,
            )
            assert effect is not None
            assert effect.status == "completed"
    finally:
        await state.close()


async def test_mysql_audit_columns_match_stg_contract() -> None:
    metadata = build_sql_schema_metadata()
    for table in metadata.tables.values():
        ddl = str(CreateTable(table).compile(dialect=mysql.dialect()))
        assert (
            "updated_at DATETIME NOT NULL COMMENT 'Update timestamp' "
            "DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
        ) in ddl
        assert "created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP" in ddl


async def test_filesystem_object_put_cancellation_transfers_publish_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = object_module.FilesystemObjectStore(tmp_path / "objects")
    data = b"object-payload"
    digest = hashlib.sha256(data).hexdigest()
    started = threading.Event()
    release = threading.Event()
    publish = object_module._publish_filesystem_object

    def delayed_publish(*args, **kwargs) -> bool:
        started.set()
        release.wait(timeout=5)
        return publish(*args, **kwargs)

    monkeypatch.setattr(object_module, "_publish_filesystem_object", delayed_publish)

    async def chunks():
        yield data

    operation = asyncio.create_task(
        store.put(
            "key",
            chunks(),
            expected_size=len(data),
            expected_digest=digest,
        )
    )
    await asyncio.to_thread(started.wait, 5)
    operation.cancel()
    with pytest.raises(AIError) as cancelled:
        await asyncio.wait_for(operation, 0.5)
    assert cancelled.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
    assert store._background_tasks

    release.set()
    for _ in range(1000):
        if not store._background_tasks:
            break
        await asyncio.sleep(0.001)
    assert not store._background_tasks
    stat = await store.stat("key")
    assert stat is not None
    assert stat.digest == digest
    assert stat.size == len(data)


async def test_filesystem_lease_acquire_cancellation_releases_late_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = FilesystemLeaseCoordinator(
        tmp_path / "leases",
        lease_seconds=5,
    )
    started = threading.Event()
    release = threading.Event()
    original_try_acquire = coordinator._try_acquire

    def delayed_try_acquire(key: str):
        started.set()
        release.wait(timeout=5)
        return original_try_acquire(key)

    monkeypatch.setattr(coordinator, "_try_acquire", delayed_try_acquire)
    operation = asyncio.create_task(
        coordinator.acquire("key", timeout=5),
    )
    await asyncio.to_thread(started.wait, 5)
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(operation, 0.5)

    release.set()
    lease = await asyncio.wait_for(
        coordinator.acquire("key", timeout=1),
        1,
    )
    await coordinator.release(lease)
    assert not tuple((tmp_path / "leases").glob("*.lease"))


async def test_filesystem_lease_release_cancellation_releases_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = FilesystemLeaseCoordinator(
        tmp_path / "leases",
        lease_seconds=5,
    )
    started = threading.Event()
    release = threading.Event()
    original_release = coordinator._release

    def delayed_release(value):
        started.set()
        release.wait(timeout=5)
        return original_release(value)

    monkeypatch.setattr(coordinator, "_release", delayed_release)
    async def release_lease() -> None:
        lease = await coordinator.acquire("key", timeout=1)
        await coordinator.release(lease)

    operation = asyncio.create_task(release_lease())
    await asyncio.to_thread(started.wait, 5)
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(operation, 0.5)

    release.set()
    replacement = await asyncio.wait_for(
        coordinator.acquire("key", timeout=1),
        1,
    )
    await coordinator.release(replacement)
    assert not tuple((tmp_path / "leases").glob("*.lease"))


async def test_filesystem_object_offline_exclusivity_is_task_owned(
    tmp_path: Path,
) -> None:
    store = object_module.FilesystemObjectStore(tmp_path / "objects")
    data = b"object-payload"
    digest = hashlib.sha256(data).hexdigest()

    async def chunks():
        yield data

    await store.put(
        "key",
        chunks(),
        expected_size=len(data),
        expected_digest=digest,
    )

    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold_offline() -> None:
        async with store.offline_exclusivity():
            entered.set()
            await release.wait()

    owner = asyncio.create_task(hold_offline())
    await entered.wait()
    deletion = asyncio.create_task(
        store.delete_object("key", expected_digest=digest)
    )
    await asyncio.sleep(0.01)
    assert not deletion.done()
    release.set()
    await owner
    assert await deletion is True


async def test_filesystem_state_store_is_single_writer_and_reopens(tmp_path: Path) -> None:
    first = FilesystemStateStore(tmp_path / "state", namespace="n", tenant_id="t", runtime_domain="conversation")
    second = FilesystemStateStore(tmp_path / "state", namespace="n", tenant_id="t", runtime_domain="conversation")

    await first.initialize()
    with pytest.raises(AIError) as raised:
        await second.initialize()
    assert raised.value.code is ErrorCode.STORAGE_CONFLICT

    await first.close()
    await second.initialize()
    await second.close()


async def test_filesystem_warm_path_does_not_reload_generation_or_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FilesystemStateStore(tmp_path / "state", namespace="n", tenant_id="t", runtime_domain="conversation")
    await store.initialize()

    def unexpected_generation() -> int:
        raise AssertionError("warm StateStore path read durable generation")

    def unexpected_index() -> object:
        raise AssertionError("warm StateStore path reloaded the index")

    monkeypatch.setattr(store, "_generation", unexpected_generation)
    monkeypatch.setattr(store, "_load_index", unexpected_index)

    key = b"k" * 32

    async def read(transaction) -> int:
        return await transaction.get_sequence(key)

    async def mutate(transaction) -> int:
        return await transaction.reserve_sequence(key, 2)

    assert await store.read(read) == 0
    assert await store.mutate(mutate) == 2
    assert await store.read(read) == 2
    await store.close()


async def test_filesystem_physical_initialization_and_commit_run_off_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FilesystemStateStore(tmp_path / "state", namespace="n", tenant_id="t", runtime_domain="conversation")
    loop_thread = threading.get_ident()
    threads: list[int] = []
    load_index = store._load_index
    commit = store._commit_sync

    def record_load() -> object:
        threads.append(threading.get_ident())
        return load_index()

    def record_commit(transaction, base: int, target: int) -> None:
        threads.append(threading.get_ident())
        commit(transaction, base, target)

    monkeypatch.setattr(store, "_load_index", record_load)
    monkeypatch.setattr(store, "_commit_sync", record_commit)
    await store.initialize()

    async def mutate(transaction) -> int:
        return await transaction.reserve_sequence(b"s" * 32, 1)

    assert await store.mutate(mutate) == 1
    assert threads and all(thread != loop_thread for thread in threads)
    await store.close()


async def test_filesystem_initialization_cancellation_retains_writer_until_owner_settles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "state"
    first = FilesystemStateStore(
        root,
        namespace="n",
        tenant_id="t",
        runtime_domain="conversation",
    )
    second = FilesystemStateStore(
        root,
        namespace="n",
        tenant_id="t",
        runtime_domain="conversation",
    )
    started = threading.Event()
    release = threading.Event()
    initialize = first._initialize_sync

    def delayed_initialize() -> object:
        started.set()
        release.wait(timeout=5)
        return initialize()

    monkeypatch.setattr(first, "_initialize_sync", delayed_initialize)
    task = asyncio.create_task(first.initialize())
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    with pytest.raises(AIError) as locked:
        await second.initialize()
    assert locked.value.code is ErrorCode.STORAGE_CONFLICT

    release.set()
    await first.initialize()
    await first.close()
    await second.initialize()
    await second.close()


async def test_filesystem_validation_cancellation_retains_mutation_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FilesystemStateStore(
        tmp_path / "state",
        namespace="n",
        tenant_id="t",
        runtime_domain="conversation",
    )
    await store.initialize()
    started = threading.Event()
    release = threading.Event()
    validate = store._validate_integrity_sync

    def delayed_validate() -> None:
        started.set()
        release.wait(timeout=5)
        validate()

    monkeypatch.setattr(store, "_validate_integrity_sync", delayed_validate)
    validation = asyncio.create_task(store.validate_integrity())
    await asyncio.to_thread(started.wait, 5)
    validation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await validation

    async def mutate(transaction) -> int:
        return await transaction.reserve_sequence(b"v" * 32, 1)

    mutation = asyncio.create_task(store.mutate(mutate))
    await asyncio.sleep(0.01)
    assert not mutation.done()
    release.set()
    assert await mutation == 1
    await store.close()


async def test_filesystem_commit_cancellation_reconciles_committed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FilesystemStateStore(tmp_path / "state", namespace="n", tenant_id="t", runtime_domain="conversation")
    await store.initialize()
    started = threading.Event()
    release = threading.Event()
    commit = store._commit_sync

    def delayed_commit(transaction, base: int, target: int) -> None:
        started.set()
        release.wait(timeout=5)
        commit(transaction, base, target)

    monkeypatch.setattr(store, "_commit_sync", delayed_commit)

    async def mutate(transaction) -> int:
        return await transaction.reserve_sequence(b"s" * 32, 1)

    operation = asyncio.create_task(store.mutate(mutate))
    await asyncio.to_thread(started.wait, 5)
    operation.cancel()
    with pytest.raises(AIError) as cancelled:
        await operation
    assert cancelled.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN

    async def read(transaction) -> int:
        return await transaction.get_sequence(b"s" * 32)

    with pytest.raises(AIError) as read_error:
        await store.read(read)
    assert read_error.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
    with pytest.raises(AIError) as close_error:
        await store.close()
    assert close_error.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN

    release.set()
    for _ in range(1000):
        if not store._pending_physical:
            break
        await asyncio.sleep(0.001)
    assert not store._pending_physical
    await store.close()

    reopened = FilesystemStateStore(
        tmp_path / "state",
        namespace="n",
        tenant_id="t",
        runtime_domain="conversation",
    )
    await reopened.initialize()
    assert await reopened.read(read) == 1
    await reopened.close()


async def test_filesystem_unknown_commit_poison_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FilesystemStateStore(tmp_path / "state", namespace="n", tenant_id="t", runtime_domain="conversation")
    await store.initialize()

    def failed_commit(*args, **kwargs) -> None:
        raise OSError("commit failed")

    async def unknown_outcome(*args, **kwargs) -> str:
        return "unknown"

    monkeypatch.setattr(store, "_commit_sync", failed_commit)
    monkeypatch.setattr(store, "_reconcile_commit", unknown_outcome)

    async def mutate(transaction) -> int:
        return await transaction.reserve_sequence(b"s" * 32, 1)

    with pytest.raises(AIError) as raised:
        await store.mutate(mutate)
    assert raised.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
    with pytest.raises(AIError) as read_error:
        await store.read(lambda transaction: transaction.get_sequence(b"s" * 32))
    assert read_error.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
    with pytest.raises(AIError) as validate_error:
        await store.validate_integrity()
    assert validate_error.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
    await store.close()


async def test_filesystem_close_cancellation_still_releases_writer(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    store = FilesystemStateStore(root, namespace="n", tenant_id="t", runtime_domain="conversation")
    await store.initialize()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def mutate(_transaction) -> None:
        entered.set()
        await release.wait()

    operation = asyncio.create_task(store.mutate(mutate))
    await entered.wait()
    closing = asyncio.create_task(store.close())
    await asyncio.sleep(0)
    closing.cancel()
    release.set()
    await operation
    with pytest.raises(asyncio.CancelledError):
        await closing
    await store.close()

    reopened = FilesystemStateStore(root, namespace="n", tenant_id="t", runtime_domain="conversation")
    await reopened.initialize()
    await reopened.close()


async def test_filesystem_writer_release_cancellation_clears_after_worker_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = FilesystemWriterLock(tmp_path / "state.lock")
    await lock.acquire()
    held = lock._lock
    assert held is not None
    started = threading.Event()
    release = threading.Event()
    original_release = held.release

    def delayed_release() -> None:
        started.set()
        release.wait(timeout=5)
        original_release()

    monkeypatch.setattr(held, "release", delayed_release)
    task = asyncio.create_task(lock.release())
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert lock.acquired

    release.set()
    for _ in range(1000):
        if not lock.acquired:
            break
        await asyncio.sleep(0.001)
    assert not lock.acquired

    replacement = FilesystemWriterLock(tmp_path / "state.lock")
    await replacement.acquire()
    await replacement.release()


async def test_journal_publish_syncs_each_affected_directory_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "state"
    journal = FilesystemJournal(root, error_code=ErrorCode.STORAGE_INTEGRITY_ERROR)
    writes = {f"records/group/{index}.json": str(index).encode("ascii") for index in range(10)}
    plan = journal.stage(writes, (), base_generation=0, target_generation=1)
    calls: list[Path] = []
    original_sync = files_module.sync_directory
    monkeypatch.setattr(files_module, "sync_directory", lambda path: (calls.append(path), original_sync(path))[1])

    journal.publish(plan)

    directory = root / "records" / "group"
    assert calls.count(directory) == 1
    assert calls.count(root) == 1
    journal.complete()


async def test_sql_context_validates_each_metadata_identity_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'state.db'}")
    context = create_sql_storage_context(engine)
    calls: list[MetaData] = []

    async def validate(_engine, metadata: MetaData) -> None:
        calls.append(metadata)

    monkeypatch.setattr(database_module, "_validate_sql_schema", validate)
    first = MetaData()
    second = MetaData()
    await context.initialize(metadata=first)
    await context.initialize(metadata=first)
    await context.initialize(metadata=second)

    assert calls == [first, second]
    assert context._validated_metadata is second
    await context.close()
    await engine.dispose()


async def test_builtin_sql_runtime_uses_one_engine_and_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "state.db"
    provisioning_engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    await provision_database(provisioning_engine)
    await provisioning_engine.dispose()

    import linktools.ai.runtime.state._materializer as materializer_module
    import sqlalchemy.ext.asyncio as sqlalchemy_asyncio

    context_count = 0
    engine_count = 0
    original_context = materializer_module.create_sql_storage_context
    original_engine = sqlalchemy_asyncio.create_async_engine

    def count_context(*args, **kwargs):
        nonlocal context_count
        context_count += 1
        return original_context(*args, **kwargs)

    def count_engine(*args, **kwargs):
        nonlocal engine_count
        engine_count += 1
        return original_engine(*args, **kwargs)

    monkeypatch.setattr(materializer_module, "create_sql_storage_context", count_context)
    monkeypatch.setattr(sqlalchemy_asyncio, "create_async_engine", count_engine)
    state = RuntimeState.sqlite(path)
    await state.initialize(namespace="n", tenant_id="t")
    try:
        assert context_count == 1
        assert engine_count == 1
    finally:
        await state.close()


async def test_sql_recovery_checkpoint_compare_and_swap_uses_split_records(
    tmp_path: Path,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'state.db'}")
    await provision_database(engine)
    state = RuntimeState.sql(engine)
    await state.initialize(namespace="n", tenant_id="t")
    now = datetime.now(timezone.utc)
    checkpoint = RecoveryCheckpoint(
        execution_id="execution",
        tenant_id="t",
        input=RecoveryExecutionInput(
            user_prompt="prompt",
            principal_id="principal",
            principal_kind="local_trusted",
            session_id=None,
            memory_scope=None,
            binding_digest="a" * 64,
            lineage_kind="RUN",
            parent_execution_id=None,
            root_execution_id="execution",
            source_execution_id=None,
            base_execution_id=None,
            conversation_step_run_id=None,
            idempotency=RecoveryIdempotencyInput(
                scope="execution.start",
                idempotency_key_digest="b" * 64,
                request_digest="c" * 64,
            ),
            planning=False,
            thinking=False,
            binding=_binding_snapshot(),
        ),
        step_run_id=None,
        agent_run_sequence=0,
        state=RecoveryCheckpointState.ADMITTED,
        handoff_phase=RecoveryHandoffPhase.NONE,
        terminal_handoff=None,
        handoff_contract_digest=None,
        pending_operation_id=None,
        revision=0,
        created_at=now,
        updated_at=now,
    )
    try:
        await state.recovery.checkpoints.create(checkpoint)
        assert await state.recovery.checkpoints.list(tenant_id="t") == (checkpoint,)
        updated = replace(
            checkpoint,
            step_run_id="run",
            agent_run_sequence=1,
            state=RecoveryCheckpointState.ACTIVE,
            revision=1,
            updated_at=datetime.now(timezone.utc),
        )
        assert await state.recovery.checkpoints.compare_and_swap(
            "execution",
            tenant_id="t",
            expected_revision=0,
            next_record=updated,
        ) == updated
        assert await state.recovery.checkpoints.get("execution", tenant_id="t") == updated
        assert await state.recovery.checkpoints.list(tenant_id="t") == (updated,)
    finally:
        await state.close()
        await engine.dispose()


async def test_external_sql_runtime_groups_by_engine_identity_and_borrows_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'first.db'}")
    second_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'second.db'}")
    await provision_database(first_engine)
    await provision_database(second_engine)

    import linktools.ai.runtime.state._materializer as materializer_module

    context_count = 0
    original_context = materializer_module.create_sql_storage_context

    def count_context(*args, **kwargs):
        nonlocal context_count
        context_count += 1
        return original_context(*args, **kwargs)

    monkeypatch.setattr(materializer_module, "create_sql_storage_context", count_context)
    plan = RuntimeStatePlan(
        conversation=RuntimeStateRoute.sql(first_engine),
        execution=RuntimeStateRoute.sql(first_engine),
        memory=RuntimeStateRoute.sql(second_engine),
        artifact=RuntimeStateRoute.sql(second_engine),
        task=RuntimeStateRoute.sql(second_engine),
        evaluation=RuntimeStateRoute.sql(second_engine),
        recovery=RuntimeStateRoute.sql(first_engine),
    )
    state = RuntimeState.from_plan(plan)
    await state.initialize(namespace="n", tenant_id="t")
    await state.close()
    assert context_count == 2

    async with first_engine.connect() as connection:
        assert (await connection.exec_driver_sql("SELECT 1")).scalar_one() == 1
    async with second_engine.connect() as connection:
        assert (await connection.exec_driver_sql("SELECT 1")).scalar_one() == 1
    await first_engine.dispose()
    await second_engine.dispose()


async def test_object_store_filesystem_operations_use_worker_threads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = object_module.FilesystemObjectStore(tmp_path)
    loop_thread = threading.get_ident()
    threads: list[int] = []
    publish = object_module._publish_filesystem_object
    read_metadata = object_module._read_filesystem_metadata

    def record_publish(*args, **kwargs):
        threads.append(threading.get_ident())
        return publish(*args, **kwargs)

    def record_metadata(*args, **kwargs):
        threads.append(threading.get_ident())
        return read_metadata(*args, **kwargs)

    monkeypatch.setattr(object_module, "_publish_filesystem_object", record_publish)
    monkeypatch.setattr(object_module, "_read_filesystem_metadata", record_metadata)
    data = b"object-data"
    digest = hashlib.sha256(data).hexdigest()

    async def chunks():
        yield data

    await store.put("key", chunks(), expected_size=len(data), expected_digest=digest)
    assert await store.stat("key") is not None
    assert b"".join([chunk async for chunk in store.open("key")]) == data
    await store.validate_integrity()
    assert threads and all(thread != loop_thread for thread in threads)


async def test_sql_object_store_large_payload_replays_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "state.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    await provision_database(engine)
    context: SqlStorageContext = create_sql_storage_context(engine)
    store = object_module.SqlObjectStore.from_context(context)
    monkeypatch.setattr(object_module, "_CHUNK_SIZE", 2)
    data = bytes(range(130))
    digest = hashlib.sha256(data).hexdigest()

    async def chunks():
        yield data

    await store.put("key", chunks(), expected_size=len(data), expected_digest=digest)
    assert b"".join([chunk async for chunk in store.open("key")]) == data
    await store.validate_integrity()
    await context.close()
    await engine.dispose()


async def test_materializer_keeps_domain_transactions_independent(tmp_path: Path) -> None:
    plan = RuntimeStatePlan(
        conversation=RuntimeStateRoute.filesystem(tmp_path / "conversation"),
        execution=RuntimeStateRoute.filesystem(tmp_path / "execution"),
        recovery=RuntimeStateRoute.filesystem(tmp_path / "recovery"),
    )
    materialized = await materialize_runtime_state(
        plan,
        namespace="n",
        tenant_id="t",
        object_store=None,
    )
    try:
        assert materialized.conversation is not materialized.execution
        assert materialized.execution is not materialized.recovery
    finally:
        for action in materialized.close_actions:
            await action()

async def test_filesystem_close_fails_closed_with_pending_maintenance_owner(
    tmp_path: Path,
) -> None:
    store = FilesystemStateStore(
        tmp_path / "state",
        namespace="close-owner",
        tenant_id="tenant",
        runtime_domain=RuntimeDomain.EXECUTION.value,
    )
    await store.initialize()
    blocker = asyncio.Event()
    owner = asyncio.create_task(blocker.wait())
    store._storage_group._maintenance_tasks.add(owner)
    try:
        with pytest.raises(AIError) as raised:
            await asyncio.wait_for(store.close(), timeout=0.2)
        assert raised.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
    finally:
        blocker.set()
        await owner
        store._storage_group._maintenance_tasks.discard(owner)
    await store.close()
