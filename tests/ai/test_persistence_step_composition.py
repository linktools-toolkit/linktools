#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public native composition preserves SQL ownership and single-file backups."""

import asyncio
import sqlite3
import shutil
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai_harness.step_persistence import RunRecord, StepPersistence
from sqlalchemy import event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import Persistence, PersistencePlan, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import StorageRoute

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_atomic_step_sql import _records
from .test_state_group_record_batch import _record


def _backup(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as reader, sqlite3.connect(target) as writer:
        reader.backup(writer)


@pytest.mark.asyncio
@pytest.mark.parametrize("exclusive", (False, True))
async def test_local_native_factory_joins_control_commit_rollback_and_backup(tmp_path: Path, exclusive: bool) -> None:
    root = tmp_path / "source"
    storage = PersistenceStorage.from_root(root, exclusive_writer=exclusive)
    run, events, snapshot = _records()
    control = _record("control")
    async with Persistence.open("joined", storage=storage, tenant_id="one") as persistence:
        native = await persistence.create_step_store()
        try:
            state = persistence.execution.executions.state_store
            prepared = await native.prepare_step(run=run, events=events, snapshot=snapshot)

            async def write(transaction):
                await transaction.insert_record(control)
                return await native.write_prepared_step(transaction, prepared)

            async def abort(transaction):
                await write(transaction)
                raise RuntimeError("control tail failed")

            with pytest.raises(AIError) as failure:
                await state.mutate(abort)
            assert failure.value.code is ErrorCode.INTERNAL_ERROR
            assert isinstance(failure.value.__cause__, RuntimeError)
            assert await state.read(lambda tx: tx.get_record(control.key)) is None
            assert await native.get_run(run_id=run.run_id) is None
            assert await native.latest_snapshot(run_id=run.run_id) is None
            receipt = await state.mutate(write)
            native.confirm_prepared_step(receipt)
            await asyncio.to_thread(_backup, root / "runtime.db", tmp_path / "backup" / "runtime.db")
        finally:
            await native.close()
        # Closing a borrowed adapter must not revoke the live resource owner.
        second = await persistence.create_step_store()
        try:
            await second.register_run(RunRecord(run_id="after-adapter-close"))
        finally:
            await second.close()
    with pytest.raises(AIError) as closed:
        await native.get_run(run_id=run.run_id)
    assert closed.value.code is ErrorCode.STORAGE_CLOSED

    async with Persistence.open("joined", storage=PersistenceStorage.from_root(tmp_path / "backup", exclusive_writer=exclusive), tenant_id="one") as restored:
        native = await restored.create_step_store()
        try:
            state = restored.execution.executions.state_store
            assert await state.read(lambda tx: tx.get_record(control.key)) == control
            assert await native.get_run(run_id=run.run_id) == run
            assert await native.list_events(run_id=run.run_id) == events
            assert await native.latest_snapshot(run_id=run.run_id) == snapshot
            assert await native.get_run(run_id="after-adapter-close") is None
        finally:
            await native.close()
    assert not (root / "steps.db").exists()


@pytest.mark.asyncio
async def test_local_native_factory_runs_checkpointed_agent_and_reopens(tmp_path: Path) -> None:
    async def model(_messages, _info):
        return ModelResponse(parts=[TextPart("retained")])

    storage = PersistenceStorage.from_root(tmp_path, exclusive_writer=True)
    async with Persistence.open("native-agent", storage=storage) as persistence:
        native = await persistence.create_step_store()
        try:
            features = RuntimeFeatures("native-agent")
            features.capability(ExecutionCapability(retain_history=True))
            features.capability(StepPersistence(store=native, capture_frontier=True))
            features.agent("default", model="default", allow_tools=())
            async with Runtime.attach(persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
                session = await runtime.agents.get().create_session("session")
                execution = await session.start("question", idempotency_key="input")
                assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
                with pytest.raises(AIError):
                    await persistence.close()
            runs = await native.list_runs()
            assert runs
            snapshots = [await native.latest_snapshot(run_id=run.run_id) for run in runs]
            assert all(snapshot is not None for snapshot in snapshots)
        finally:
            await native.close()
    async with Persistence.open("native-agent", storage=storage) as reopened:
        native = await reopened.create_step_store()
        try:
            assert await native.list_runs() == runs
            assert [await native.latest_snapshot(run_id=run.run_id) for run in runs] == snapshots
        finally:
            await native.close()


@pytest.mark.asyncio
async def test_local_native_schema_failure_does_not_publish_partial_database(tmp_path: Path) -> None:
    storage = PersistenceStorage.from_root(tmp_path, exclusive_writer=True)

    def fail(connection, cursor, statement, parameters, context, many):
        if "CREATE TABLE ai_step_snapshots" in statement:
            raise RuntimeError("native schema failed")

    event.listen(Engine, "before_cursor_execute", fail)
    try:
        with pytest.raises(RuntimeError, match="native schema failed"):
            async with Persistence.open("partial", storage=storage):
                pytest.fail("partial schema was published")
    finally:
        event.remove(Engine, "before_cursor_execute", fail)
    assert not (tmp_path / "runtime.db").exists()
    assert not tuple(tmp_path.glob(".runtime.db.init-*"))
    async with Persistence.open("partial", storage=storage) as persistence:
        native = await persistence.create_step_store()
        try:
            await native.register_run(RunRecord(run_id="retry"))
        finally:
            await native.close()


@pytest.mark.asyncio
async def test_existing_local_native_schema_is_not_repaired_implicitly(tmp_path: Path) -> None:
    storage = PersistenceStorage.from_root(tmp_path, exclusive_writer=True)
    async with Persistence.open("existing", storage=storage):
        pass
    with sqlite3.connect(tmp_path / "runtime.db") as connection:
        connection.execute("DROP TABLE ai_step_runs")
        before = connection.execute("SELECT name, sql FROM sqlite_master ORDER BY name").fetchall()
    async with Persistence.open("existing", storage=storage) as persistence:
        native = await persistence.create_step_store()
        try:
            with pytest.raises(OperationalError, match="no such table: ai_step_runs"):
                await native.get_run(run_id="missing")
        finally:
            await native.close()
    with sqlite3.connect(tmp_path / "runtime.db") as connection:
        assert connection.execute("SELECT name, sql FROM sqlite_master ORDER BY name").fetchall() == before


@pytest.mark.asyncio
async def test_shared_native_factory_does_not_provision_or_dispose_callers_engine(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'shared.db'}")
    statements = []

    def capture(connection, cursor, statement, parameters, context, many):
        statements.append(statement)

    try:
        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        async with Persistence.open("shared", storage=PersistenceStorage.sql(engine)) as persistence:
            with pytest.raises(OperationalError, match="no such table: ai_state_records"):
                await persistence.create_step_store()
        assert not any(statement.lstrip().upper().startswith(("CREATE", "ALTER", "DROP")) for statement in statements)
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await provision_database(engine)
        async with Persistence.open("shared", storage=PersistenceStorage.sql(engine)) as persistence:
            native = await persistence.create_step_store()
            try:
                await native.register_run(RunRecord(run_id="borrowed"))
            finally:
                await native.close()
        # Engine disposal changes the pool even when SQLAlchemy can reopen it.
        pool = engine.pool
        async with Persistence.open("shared", storage=PersistenceStorage.sql(engine)) as persistence:
            native = await persistence.create_step_store()
            await native.close()
        assert engine.pool is pool
        async with engine.connect() as connection:
            assert await connection.scalar(text("SELECT 1")) == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_native_factory_keeps_namespace_and_tenant_partitions(tmp_path: Path) -> None:
    storage = PersistenceStorage.sqlite(tmp_path / "partitions.db")
    for namespace, tenant_id in (("first", "one"), ("first", "two"), ("second", "one")):
        async with Persistence.open(namespace, storage=storage, tenant_id=tenant_id) as persistence:
            native = await persistence.create_step_store()
            try:
                assert await native.get_run(run_id="same") is None
                await native.register_run(RunRecord(run_id="same", conversation_id=f"{namespace}:{tenant_id}"))
            finally:
                await native.close()
    async with Persistence.open("first", storage=storage, tenant_id="one") as persistence:
        native = await persistence.create_step_store()
        try:
            assert (await native.get_run(run_id="same")).conversation_id == "first:one"
        finally:
            await native.close()


@pytest.mark.asyncio
async def test_native_factory_rejects_inactive_read_only_and_memory_scopes(tmp_path: Path) -> None:
    storage = PersistenceStorage.from_root(tmp_path)
    with pytest.raises(AIError) as inactive:
        await Persistence(storage).create_step_store()
    assert inactive.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
    async with Persistence.open("scope", storage=storage):
        pass
    async with Persistence.open("scope", storage=storage, read_only=True) as persistence:
        with pytest.raises(AIError) as readonly:
            await persistence.create_step_store()
        assert readonly.value.code is ErrorCode.STORAGE_READ_ONLY
    async with Persistence.open("scope", storage=PersistenceStorage.in_memory()) as persistence:
        with pytest.raises(AIError) as memory:
            await persistence.create_step_store()
        assert memory.value.code is ErrorCode.STORAGE_CAPABILITY_MISSING


@pytest.mark.asyncio
async def test_native_factory_uses_execution_route_in_split_plan(tmp_path: Path) -> None:
    execution = tmp_path / "execution.db"
    conversation = tmp_path / "conversation.db"
    storage = PersistenceStorage.from_plan(PersistencePlan(
        execution=StorageRoute.sqlite(execution), conversation=StorageRoute.sqlite(conversation),
    ))
    async with Persistence.open("routed", storage=storage) as persistence:
        native = await persistence.create_step_store()
        try:
            await native.register_run(RunRecord(run_id="routed"))
        finally:
            await native.close()
    with sqlite3.connect(execution) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert {"ai_step_runs", "ai_step_events", "ai_step_snapshots", "ai_step_messages", "ai_step_sequences"} <= tables
        assert connection.execute("SELECT run_id FROM ai_step_runs").fetchall() == [("routed",)]
    with sqlite3.connect(conversation) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert {name for name in tables if name.startswith("ai_step_")} == {
            "ai_step_runs", "ai_step_messages", "ai_step_sequences",
        }
        assert connection.execute("SELECT * FROM ai_step_runs").fetchall() == []
        assert connection.execute("SELECT * FROM ai_step_messages").fetchall() == []
        assert connection.execute("SELECT * FROM ai_step_sequences").fetchall() == []


@pytest.mark.asyncio
async def test_local_native_handoff_backup_recovers_same_execution(tmp_path: Path) -> None:
    entered = asyncio.Event()

    async def blocked(_messages, _info):
        entered.set()
        await asyncio.Event().wait()
        return ModelResponse(parts=[TextPart("unreachable")])

    async def answer(_messages, _info):
        return ModelResponse(parts=[TextPart("recovered")])

    def features(native, model):
        group = RuntimeFeatures("native-handoff")
        group.capability(ExecutionCapability(retain_history=True))
        group.capability(StepPersistence(store=native, capture_frontier=True))
        group.agent("default", model="default", allow_tools=())
        return model_features(_Models(_UsageFunctionModel(model)), group)

    source, target = tmp_path / "source", tmp_path / "restored"
    async with Persistence.open("native-handoff", storage=PersistenceStorage.from_root(source, exclusive_writer=True)) as persistence:
        native = await persistence.create_step_store()
        try:
            async with Runtime.attach(persistence=persistence, features=features(native, blocked)) as owner:
                execution = await (await owner.agents.get().create_session("session")).start("Remember blue", idempotency_key="first")
                await asyncio.wait_for(entered.wait(), 5)
                await asyncio.wait_for(owner.handoff(), 5)
            await asyncio.to_thread(_backup, source / "runtime.db", target / "runtime.db")
            if (source / "objects").exists():
                await asyncio.to_thread(shutil.copytree, source / "objects", target / "objects")
        finally:
            await native.close()

    async with Persistence.open("native-handoff", storage=PersistenceStorage.from_root(target, exclusive_writer=True)) as persistence:
        native = await persistence.create_step_store()
        try:
            async with Runtime.attach(persistence=persistence, features=features(native, answer), auto_recover=False) as successor:
                principal = successor.default_principal
                before = await successor.executions.inspect(execution.execution_id, principal=principal)
                assert before.status is ExecutionStatus.STARTED
                recovered = await successor.executions.recover(execution.execution_id, principal=principal, idempotency_key="recover")
                assert recovered.execution_id == execution.execution_id
                await successor.executions.wait(recovered.execution_id, principal=principal, timeout_seconds=5)
                assert (await successor.executions.inspect(recovered.execution_id, principal=principal)).status is ExecutionStatus.SUCCEEDED
                session = successor.agents.get().session("session")
                replay = await session.start("Remember blue", idempotency_key="first")
                assert replay.execution_id == execution.execution_id
        finally:
            await native.close()
