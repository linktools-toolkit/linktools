#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime filesystem and SQL physical-resource convergence evidence."""

import asyncio
import re
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from linktools.ai.agent import AgentBindingSnapshot
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
from linktools.ai.runtime._capabilities import _RuntimeStepPersistence
from linktools.ai.runtime._tool import RuntimeToolOperationBridge
from linktools.ai.runtime.state import RuntimeStateCommands
from linktools.ai.runtime.state._contracts import (
    ExecutionRecord,
    ExecutionStartClaim,
    ExecutionStartReservation,
    IdempotencyRecord,
    SessionRecord,
)
from linktools.ai.runtime.state._filesystem import FilesystemStateStore
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import PayloadPolicy, create_sql_storage_context
from linktools.ai.storage import _database as database_module
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
        return sum(entry[1] == action and entry[2] == table for entry in self.entries)

    def executemany(self, action: str, table: str) -> bool:
        return any(
            entry[1] == action and entry[2] == table and entry[3]
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
        match = re.search(r"(?:FROM|INTO|UPDATE|JOIN) ([A-Z0-9_]+)", normalized)
        table = "" if match is None else match.group(1)
        self.entries.append((self.scope, action, table, executemany))


def _binding_snapshot() -> AgentBindingSnapshot:
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", model="default"),
        model={"version": 1, "id": "default"},
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        binding_digest="a" * 64,
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


async def test_sql_cursor_counter_captures_core_batch_gates(tmp_path: Path) -> None:
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
        mode="run",
        planning=False,
        thinking=False,
        binding=_binding_snapshot(),
    )
    reserved_execution = replace(execution, execution_id="reserved", root_execution_id="reserved")
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
        await state.execution.executions.reserve_start(ExecutionStartReservation(reserved_execution, identity))
        assert counter.count("SELECT", "AI_STATE_RECORDS") == 1
        assert counter.count("INSERT", "AI_STATE_RECORDS") == 1

        counter.reset("execution.claim_start")
        await state.execution.executions.claim_start(
            ExecutionStartClaim("reserved", "tenant", 0, 0, "scope", "key", "request", now)
        )
        assert counter.count("UPDATE", "AI_STATE_RECORDS") == 1
        assert counter.executemany("UPDATE", "AI_STATE_RECORDS")
        assert counter.count("INSERT", "AI_STATE_FACTS") == 1
    finally:
        counter.close()
        await state.close()
        await engine.dispose()


async def test_sql_state_group_maps_programming_failure_to_internal(tmp_path: Path) -> None:
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
        assert raised.value.retryable is False
        assert raised.value.safe_details == {"phase": "runtime_state_sql_mutation"}
    finally:
        await state.close()
        await engine.dispose()


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


async def test_sqlite_parallel_tool_lifecycle_uses_one_effect_writer(tmp_path: Path) -> None:
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
        bridge = RuntimeToolOperationBridge(
            state.recovery.tools,
            state.object_store(RuntimeDomain.RECOVERY),
            namespace="parallel-tools",
            tenant_id="tenant",
            execution_id="execution",
            step_run_id=run_id,
            binding_digest="a" * 64,
            owner="owner",
            background_tasks=background_tasks,
            payload_policy=PayloadPolicy(),
            terminal_commands=_runtime_commands(state, "parallel-tools", background_tasks),
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

        async def execute(call_id: str) -> object:
            context = RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id=run_id)
            call = ToolCallPart("tool", {}, tool_call_id=call_id)
            tool_def = ToolDefinition(name="tool", metadata={"linktools.ai.replay_safe": True})
            await persistence.before_tool_execute(context, call=call, tool_def=tool_def, args={})

            async def handler(_args: dict[str, object]) -> dict[str, str]:
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
            operation = await state.recovery.tools.get_by_call(run_id, call_id, tenant_id="tenant")
            assert operation is not None
            assert operation.status is ToolOperationStatus.COMPLETED
            assert operation.binding_digest == "a" * 64
            effect = await recovery.get_tool_effect(run_id=run_id, tool_call_id=call_id)
            assert effect is not None
            assert effect.status == "completed"
    finally:
        await state.close()


async def test_mysql_audit_columns_match_stg_contract() -> None:
    metadata = build_sql_schema_metadata()
    for table in metadata.tables.values():
        ddl = str(CreateTable(table).compile(dialect=mysql.dialect()))
        assert "updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP" in ddl
        assert "created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP" in ddl


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

    async def read(transaction) -> int:
        return await transaction.get_sequence(b"k" * 32)

    async def mutate(transaction) -> int:
        return await transaction.reserve_sequence(b"k" * 32, 2)

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


async def test_filesystem_unknown_commit_poison_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FilesystemStateStore(tmp_path / "state", namespace="n", tenant_id="t", runtime_domain="conversation")
    await store.initialize()

    def failed_commit(*args, **kwargs) -> None:
        del args, kwargs
        raise OSError("commit failed")

    async def unknown_outcome(*args, **kwargs) -> str:
        del args, kwargs
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
    await store.close()


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
