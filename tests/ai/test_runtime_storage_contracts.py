#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime storage error, concurrency, and schema contracts."""

import asyncio
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.core import (
    OperationKind,
    OperationLedgerInput,
    OperationStatus,
    ResourceKind,
    ToolOperationStatus,
    canonical_sha256,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import build_sql_schema_metadata, provision_database
from linktools.ai.runtime import RuntimeDomain, RuntimeState
from linktools.ai.runtime._tool import RuntimeToolOperationBridge
from linktools.ai.runtime.state._commands import RuntimeStateCommands
from linktools.ai.runtime.state._filesystem import (
    FilesystemStateStorageGroup,
    FilesystemStateStore,
)
from linktools.ai.runtime.state._memory import MemoryStateStore
from linktools.ai.runtime.state._sql import SqlStateStore
from linktools.ai.runtime.state._store import (
    FactQuery,
    OperationQuery,
    RecordQuery,
    StateTransaction,
    StoredAlias,
    StoredFact,
    StoredOperation,
    StoredRecord,
)
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import FilesystemObjectStore, PayloadPolicy
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage
from linktools.ai.runtime.state._step_contracts import (
    RunRecord,
)
from sqlalchemy import event
from sqlalchemy.dialects import mysql
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.schema import CreateTable

pytestmark = pytest.mark.asyncio


def _binding_snapshot() -> AgentBindingSnapshot:
    return AgentBindingSnapshot(
        agent_spec=AgentSpec("agent", model_route="default"),
        model_contract={"version": 1, "id": "default"},
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={"type": "object", "properties": {"text": {"type": "string"}}},
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


async def test_memory_state_store_detaches_nested_record_data() -> None:
    store = MemoryStateStore()
    await store.initialize()
    source = {"nested": {"value": "original"}}
    record = StoredRecord(
        b"r" * 32,
        None,
        None,
        "test",
        "record",
        None,
        0,
        None,
        0,
        None,
        source,
    )
    try:
        await store.mutate(lambda transaction: transaction.insert_record(record))
        source["nested"]["value"] = "caller-mutated"

        async def fail_after_read(transaction: StateTransaction) -> None:
            stored = await transaction.get_record(record.key_digest)
            assert stored is not None
            nested = stored.data["nested"]
            assert isinstance(nested, dict)
            nested["value"] = "read-mutated"
            raise RuntimeError("rollback")

        with pytest.raises(RuntimeError):
            await store.mutate(fail_after_read)

        await store.validate_integrity()
        stored = await store.read(
            lambda transaction: transaction.get_record(record.key_digest)
        )
        assert stored is not None
        assert tuple(stored.data) == ("nested",)
        assert stored.data == {
            "nested": {"value": "original"},
        }
        assert stored.storage_version == 0
    finally:
        await store.close()


async def test_sql_state_group_maps_programming_failure_to_internal(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    await provision_database(engine)
    state = RuntimeState.sqlite(
        path,
        object_store=FilesystemObjectStore(tmp_path / "objects"),
    )
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


async def test_sql_latest_per_subject_uses_portable_aggregate_query(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    await provision_database(engine)
    store = SqlStateStore(engine)
    await store.initialize()
    statements: list[str] = []

    def capture_sql(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture_sql)
    owner = b"o" * 32
    stream = b"s" * 32
    subject_a = b"a" * 32
    subject_b = b"b" * 32
    record = StoredRecord(
        owner,
        None,
        None,
        "test",
        "owner",
        None,
        0,
        None,
        0,
        None,
        {"nested": {"value": "stored"}},
    )
    facts = (
        StoredFact(stream, 1, owner, "test", subject_a, None, {"value": 1}),
        StoredFact(stream, 2, owner, "test", subject_b, None, {"value": 2}),
        StoredFact(stream, 3, owner, "test", subject_a, None, {"value": 3}),
        StoredFact(stream, 4, owner, "test", None, None, {"value": 4}),
        StoredFact(stream, 5, owner, "test", None, None, {"value": 5}),
    )

    async def seed(transaction: StateTransaction) -> None:
        await transaction.insert_record(record)
        await transaction.insert_facts(facts)

    try:
        await store.mutate(seed)
        stored = await store.read(
            lambda transaction: transaction.get_record(owner)
        )
        assert stored is not None
        assert stored.data == {"nested": {"value": "stored"}}
        statements.clear()
        values = await store.read(
            lambda transaction: transaction.list_facts(
                FactQuery(stream, latest_per_subject=True)
            )
        )
        assert tuple(value.sequence for value in values) == (2, 3, 5)
        filtered = await store.read(
            lambda transaction: transaction.list_facts(
                FactQuery(
                    stream,
                    after_sequence=2,
                    limit=1,
                    latest_per_subject=True,
                )
            )
        )
        assert tuple(value.sequence for value in filtered) == (3,)
        sql = "\n".join(statements).upper()
        assert "ROW_NUMBER" not in sql
        assert " OVER " not in sql
        assert "GROUP BY" in sql
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture_sql)
        await store.close()
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


async def test_sqlite_parallel_tool_lifecycle_persists_each_terminal_effect(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.db"
    provisioning_engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    await provision_database(provisioning_engine)
    await provisioning_engine.dispose()
    state = RuntimeState.sqlite(
        path,
        object_store=FilesystemObjectStore(tmp_path / "objects"),
    )
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
            terminal_commands=_runtime_commands(
                state, "parallel-tools", background_tasks
            ),
        )
        call_ids = ("call-a", "call-b")
        entered = {call_id: asyncio.Event() for call_id in call_ids}
        release = asyncio.Event()

        async def execute(call_id: str) -> object:
            context = RunContext(
                deps=None, model=TestModel(), usage=RunUsage(), run_id=run_id
            )
            call = ToolCallPart("tool", {}, tool_call_id=call_id)
            tool_def = ToolDefinition(
                name="tool",
                metadata={"linktools.ai.replay_safe": True},
            )
            decision = await bridge.begin(
                context,
                call,
                tool_def,
                {},
                True,
            )

            async def handler(_args: dict[str, object]) -> dict[str, str]:
                entered[call_id].set()
                await release.wait()
                return {"call_id": call_id}

            result = await handler({})
            await bridge.complete(decision, result)
            return result

        tasks = [asyncio.create_task(execute(call_id)) for call_id in call_ids]
        await asyncio.gather(*(entered[call_id].wait() for call_id in call_ids))
        release.set()
        assert await asyncio.gather(*tasks) == [
            {"call_id": "call-a"},
            {"call_id": "call-b"},
        ]

        for call_id in call_ids:
            operation = await state.recovery.tools.get_by_call(
                run_id,
                call_id,
                tenant_id="tenant",
            )
            assert operation is not None
            assert operation.status is ToolOperationStatus.COMPLETED
            assert operation.binding_digest == "a" * 64
    finally:
        await state.close()


async def test_mysql_audit_columns_match_schema_contract() -> None:
    metadata = build_sql_schema_metadata()
    for table in metadata.tables.values():
        ddl = str(CreateTable(table).compile(dialect=mysql.dialect()))
        assert (
            "updated_at DATETIME NOT NULL COMMENT 'Update timestamp' "
            "DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
        ) in ddl
        assert (
            "created_at DATETIME NOT NULL COMMENT 'Creation timestamp' "
            "DEFAULT CURRENT_TIMESTAMP"
        ) in ddl


async def test_filesystem_state_store_is_single_writer_and_reopens(
    tmp_path: Path,
) -> None:
    first = FilesystemStateStore(
        tmp_path / "state",
        namespace="n",
        tenant_id="t",
        runtime_domain="conversation",
    )
    second = FilesystemStateStore(
        tmp_path / "state",
        namespace="n",
        tenant_id="t",
        runtime_domain="conversation",
    )
    await first.initialize()
    with pytest.raises(AIError) as raised:
        await second.initialize()
    assert raised.value.code is ErrorCode.STORAGE_CONFLICT

    await first.close()
    await second.initialize()
    await second.close()


async def test_filesystem_unknown_commit_poison_is_fail_closed(
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

    def failed_commit(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("commit failed")

    async def unknown_outcome(
        *args: object, **kwargs: object
    ) -> tuple[str, None]:
        del args, kwargs
        return "unknown", None

    monkeypatch.setattr(store, "_commit_sync", failed_commit)
    monkeypatch.setattr(store, "_reconcile_commit", unknown_outcome)

    async def mutate(transaction: StateTransaction) -> int:
        return await transaction.reserve_sequence(b"s" * 32, 1)

    try:
        with pytest.raises(AIError) as raised:
            await store.mutate(mutate)
        assert raised.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        with pytest.raises(AIError) as read_error:
            await store.read(lambda transaction: transaction.get_sequence(b"s" * 32))
        assert read_error.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
    finally:
        await store.close()


@pytest.mark.parametrize("grouped", (False, True))
async def test_filesystem_cancelled_commit_settles_before_propagating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    grouped: bool,
) -> None:
    if grouped:
        root = tmp_path / "group"
        group = FilesystemStateStorageGroup(
            root,
            namespace="n",
            tenant_id="t",
            scope_digest="scope",
        )
        store = FilesystemStateStore(
            root / "conversation",
            namespace="n",
            tenant_id="t",
            runtime_domain="conversation",
            group=group,
        )
        target = group
    else:
        store = FilesystemStateStore(
            tmp_path / "state",
            namespace="n",
            tenant_id="t",
            runtime_domain="conversation",
        )
        target = store
    await store.initialize()

    started = threading.Event()
    release = threading.Event()
    original_commit = target._commit_sync

    def slow_commit(*args: object, **kwargs: object) -> None:
        started.set()
        if not release.wait(5):
            raise RuntimeError("filesystem commit release timed out")
        original_commit(*args, **kwargs)

    monkeypatch.setattr(target, "_commit_sync", slow_commit)

    async def mutate(transaction: StateTransaction) -> int:
        return await transaction.reserve_sequence(b"s" * 32, 1)

    try:
        task = asyncio.create_task(store.mutate(mutate))
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert await store.read(
            lambda transaction: transaction.get_sequence(b"s" * 32)
        ) == 1
    finally:
        release.set()
        await store.close()


@pytest.mark.parametrize("grouped", (False, True))
async def test_filesystem_cancelled_reconcile_settles_known_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    grouped: bool,
) -> None:
    if grouped:
        root = tmp_path / "group"
        group = FilesystemStateStorageGroup(
            root,
            namespace="n",
            tenant_id="t",
            scope_digest="scope",
        )
        store = FilesystemStateStore(
            root / "conversation",
            namespace="n",
            tenant_id="t",
            runtime_domain="conversation",
            group=group,
        )
        target = group
        reconcile_name = "_reconcile_sync"
    else:
        store = FilesystemStateStore(
            tmp_path / "state",
            namespace="n",
            tenant_id="t",
            runtime_domain="conversation",
        )
        target = store
        reconcile_name = "_reconcile_commit_sync"
    await store.initialize()

    started = threading.Event()
    release = threading.Event()

    def failed_commit(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("forced commit failure")

    def delayed_reconcile(*args: object, **kwargs: object) -> str:
        del args, kwargs
        started.set()
        if not release.wait(5):
            raise RuntimeError("filesystem reconcile release timed out")
        return "not_committed"

    monkeypatch.setattr(target, "_commit_sync", failed_commit)
    monkeypatch.setattr(target, reconcile_name, delayed_reconcile)

    async def mutate(transaction: StateTransaction) -> int:
        return await transaction.reserve_sequence(b"s" * 32, 1)

    try:
        task = asyncio.create_task(store.mutate(mutate))
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert await store.read(
            lambda transaction: transaction.get_sequence(b"s" * 32)
        ) == 0
    finally:
        release.set()
        await store.close()


async def test_filesystem_kind_only_query_filters_other_record_kinds(
    tmp_path: Path,
) -> None:
    store = FilesystemStateStore(
        tmp_path / "kind-query",
        namespace="n",
        tenant_id="t",
        runtime_domain="conversation",
    )
    await store.initialize()
    first = StoredRecord(
        b"a" * 32,
        None,
        None,
        "first",
        "a",
        None,
        0,
        None,
        0,
        None,
        {},
    )
    second = StoredRecord(
        b"b" * 32,
        None,
        None,
        "second",
        "b",
        None,
        0,
        None,
        0,
        None,
        {},
    )
    try:
        await store.mutate(
            lambda transaction: transaction.insert_records((first, second))
        )
        values = await store.read(
            lambda transaction: transaction.list_records(
                RecordQuery(kind="second")
            )
        )
        assert values == (second,)
    finally:
        await store.close()


async def test_sql_state_store_scope_applies_to_point_and_collection_operations(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scoped-runtime.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    await provision_database(engine)
    first = SqlStateStore(engine, store_digest=b"a" * 32)
    second = SqlStateStore(engine, store_digest=b"b" * 32)
    await first.initialize()
    await second.initialize()

    record_key = b"r" * 32
    alias_key = b"a" * 32
    stream_key = b"s" * 32
    sequence_key = b"q" * 32
    operation_key = b"o" * 32
    operation_stream = b"p" * 32
    record = StoredRecord(
        record_key,
        None,
        None,
        "scope-test",
        "record",
        None,
        0,
        None,
        0,
        None,
        {},
    )
    fact = StoredFact(
        stream_key,
        1,
        record_key,
        "scope-test",
        None,
        None,
        {"value": 1},
    )
    operation = StoredOperation(
        operation_key,
        operation_stream,
        1,
        "RUNNING",
        False,
        {},
    )

    async def seed(transaction: StateTransaction) -> None:
        await transaction.insert_record(record)
        await transaction.insert_alias(StoredAlias(alias_key, record_key))
        await transaction.insert_fact(fact)
        assert await transaction.reserve_sequence(sequence_key, 1) == 1
        await transaction.insert_operation(operation)

    try:
        await first.mutate(seed)

        assert await second.read(
            lambda transaction: transaction.get_record(record_key)
        ) is None
        assert await second.read(
            lambda transaction: transaction.list_records(
                RecordQuery(kind="scope-test")
            )
        ) == ()
        assert await second.read(
            lambda transaction: transaction.resolve_alias(alias_key)
        ) is None
        assert await second.read(
            lambda transaction: transaction.get_sequence(sequence_key)
        ) == 0
        assert await second.read(
            lambda transaction: transaction.list_facts(FactQuery(stream_key))
        ) == ()
        assert await second.read(
            lambda transaction: transaction.get_operation(operation_key)
        ) is None

        assert await second.mutate(
            lambda transaction: transaction.guard_record(
                record_key,
                expected_storage_version=0,
            )
        ) is None
        with pytest.raises(AIError) as raised:
            await second.mutate(
                lambda transaction: transaction.advance_sequence(sequence_key, 1)
            )
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT

        await second.mutate(
            lambda transaction: transaction.delete_sequences((sequence_key,))
        )
        await second.mutate(
            lambda transaction: transaction.delete_fact_streams(record_key)
        )
        assert await second.mutate(
            lambda transaction: transaction.delete_operations(
                OperationQuery(stream_digest=operation_stream)
            )
        ) == ()

        assert await first.read(
            lambda transaction: transaction.get_sequence(sequence_key)
        ) == 1
        assert tuple(
            value.sequence
            for value in await first.read(
                lambda transaction: transaction.list_facts(
                    FactQuery(stream_key)
                )
            )
        ) == (1,)
        assert await first.read(
            lambda transaction: transaction.get_operation(operation_key)
        ) == operation
    finally:
        await first.close()
        await second.close()
        await engine.dispose()



async def test_operation_compaction_keeps_one_stream_anchor() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="operation-anchor", tenant_id="tenant")
    now = datetime.now(timezone.utc)

    def operation(index: int) -> OperationLedgerInput:
        return OperationLedgerInput(
            operation_id=canonical_sha256({"operation": index}),
            tenant_id="tenant",
            resource_kind=ResourceKind.SESSION,
            resource_id="session",
            execution_id=None,
            operation_kind=OperationKind.SESSION_UPDATE,
            status=OperationStatus.SUCCEEDED,
            request_digest=canonical_sha256({"request": index}),
            result_ref=None,
            result_digest=None,
            error_code=None,
            compactable=True,
            created_at=now,
            updated_at=now,
        )

    try:
        values = tuple(
            [
                await state.conversation.operations.append(operation(index))
                for index in range(1, 4)
            ]
        )
        assert tuple(value.sequence for value in values) == (1, 2, 3)

        await state.conversation.operations.compact_terminal(
            ResourceKind.SESSION,
            "session",
            tenant_id="tenant",
            through_sequence=3,
        )

        assert await state.conversation.operations.get(
            values[0].operation_id,
            tenant_id="tenant",
        ) is None
        assert await state.conversation.operations.get(
            values[1].operation_id,
            tenant_id="tenant",
        ) is None
        assert await state.conversation.operations.get(
            values[2].operation_id,
            tenant_id="tenant",
        ) == values[2]

        next_value = await state.conversation.operations.append(operation(4))
        assert next_value.sequence == 4
    finally:
        await state.close()
