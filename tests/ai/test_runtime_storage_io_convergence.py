"""Runtime filesystem and SQL physical-resource convergence evidence."""

import asyncio
import hashlib
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import build_sql_schema_metadata, provision_database
from linktools.ai.runtime.state import RuntimeState
from linktools.ai.runtime.state._contracts import (
    RecoveryCheckpoint,
    RecoveryCheckpointState,
    RecoveryExecutionInput,
    RecoveryHandoffPhase,
    RecoveryIdempotencyInput,
)
from linktools.ai.runtime.state._filesystem import FilesystemStateStore
from linktools.ai.runtime.state._materializer import materialize_runtime_state
from linktools.ai.runtime.state._plan import RuntimeStatePlan, RuntimeStateRoute
from linktools.ai.storage import (
    FilesystemJournal,
    FilesystemWriterLock,
    SqlStorageContext,
    create_sql_storage_context,
)
from linktools.ai.storage import _database as database_module
from linktools.ai.storage import _files as files_module
from linktools.ai.storage import _object as object_module
from sqlalchemy import MetaData
from sqlalchemy.dialects import mysql
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.schema import CreateTable

pytestmark = pytest.mark.asyncio


async def test_mysql_audit_columns_match_stg_contract() -> None:
    metadata = build_sql_schema_metadata()
    for table in metadata.tables.values():
        ddl = str(CreateTable(table).compile(dialect=mysql.dialect()))
        assert (
            "updated_at DATETIME NOT NULL COMMENT 'Update timestamp' "
            "DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
        ) in ddl
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
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await operation

    async def read(transaction) -> int:
        return await transaction.get_sequence(b"s" * 32)

    assert await store.read(read) == 1
    await store.close()


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
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
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
            agent_id="agent",
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
