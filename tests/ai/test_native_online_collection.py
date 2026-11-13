#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit offline reclamation follows committed owners, not cache proofs."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from linktools.ai.steps import native_store_locator
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import NativeMessageRoot, NativeMessageSlice, SqlStepStore
from linktools.ai.persistence._archive import ContentArchive
from linktools.ai.persistence._sql import SqlStateStore
from linktools.ai.persistence._step_schema import build_step_sql_metadata
from linktools.ai.persistence.codec import encode_domain
from linktools.ai.persistence.contracts import ArchivedContextBlock
from linktools.ai.persistence.state import OperationQuery, StoredFact, StoredOperation, StoredRecord, operation_key
from linktools.ai.storage import SqlObjectStore

from .test_step_content_cleanup import _snapshot, _Offline


@asynccontextmanager
async def _shared(path: Path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    await provision_database(engine)
    state = SqlStateStore(engine)
    await state.initialize()
    group, archive_store = await state.storage_group.open_archive(create=True)
    objects = SqlObjectStore.from_context(state.storage_group.context)
    archive = ContentArchive(archive_store, group, namespace="steps", tenant_id="one", object_store=objects)
    native = SqlStepStore(engine, namespace="steps", tenant_id="one", max_snapshots_per_run=1)
    native.bind_message_archive(archive)
    try:
        await native.save_snapshot(_snapshot("run", 1, "old"))
        table = build_step_sql_metadata().tables["ai_step_snapshots"]
        async with engine.connect() as connection:
            first = await connection.scalar(select(table.c.sequence_root))
        root = NativeMessageRoot(group, native.store_key, (NativeMessageSlice(first, 1, 0, 1),))
        yield engine, state, native, root
    finally:
        await native.close()
        await state.close()
        await engine.dispose()


def _record(data, *, key: bytes = b"r" * 32, state: str = "DONE") -> StoredRecord:
    return StoredRecord(key, None, None, "owner", b'owner', state, 0, None, 0, None, data)


async def _count(engine, table: str) -> int:
    async with engine.connect() as connection:
        return await connection.scalar(text(f"SELECT count(*) FROM {table}"))


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ("record", "fact", "operation"))
@pytest.mark.parametrize("status", ("UNKNOWN", "DONE"))
async def test_offline_collection_preserves_all_committed_root_owners_including_unknown(
    tmp_path: Path, owner: str, status: str,
) -> None:
    async with _shared(tmp_path / "owners.db") as (engine, state, native, root):
        payload = encode_domain(ArchivedContextBlock(root, "c" * 64, 2))
        record = _record(payload if owner == "record" else {}, state=status if owner == "record" else "DONE")

        async def publish(tx):
            await tx.insert_record(record)
            if owner == "fact":
                await tx.insert_fact(StoredFact(b"s" * 32, 1, record.key, "chunk", None, status, payload))
            elif owner == "operation":
                await tx.insert_operation(StoredOperation(operation_key("steps", "one", "execution", "界" * 42 + "xx"), b"s" * 32, 1, status, False, payload))

        await state.mutate(publish)
        current = _snapshot("run", 2, "current")
        await native.save_snapshot(current)
        assert await _count(engine, "ai_step_messages") == 2
        if status == "UNKNOWN":
            with pytest.raises(AIError) as unknown:
                await native.purge_unreferenced_content(exclusive=_Offline())
            assert unknown.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
            assert await _count(engine, "ai_step_messages") == 2
        else:
            retained = await native.purge_unreferenced_content(exclusive=_Offline())
            assert retained.retained_messages == 2
        assert (await native.read_message_roots((root,)))[0]
        if owner == "operation":
            async def transition(tx):
                previous = await tx.get_operation(operation_key("steps", "one", "execution", "界" * 42 + "xx"))
                assert await tx.replace_operation(replace(previous, state="DONE"), expected_state=status)
            await state.mutate(transition)
            assert (await native.read_message_roots((root,)))[0]
            await state.mutate(lambda tx: tx.delete_operations(OperationQuery(stream_key=b"s" * 32)))
        elif owner == "record":
            assert await state.mutate(lambda tx: tx.replace_record(
                replace(record, data={}, state="DONE", storage_revision=1), expected_storage_revision=0,
            ))
        else:
            assert await state.mutate(lambda tx: tx.delete_record(record.key))
        await native.purge_unreferenced_content(exclusive=_Offline())
        assert await _count(engine, "ai_step_messages") == 1
        assert await _count(engine, "ai_step_sequences") == 1
        assert await native.latest_snapshot(run_id="run") == current


@pytest.mark.asyncio
async def test_native_binding_enables_retention_after_state_only_initialization(tmp_path: Path) -> None:
    from linktools.ai.persistence._schema import build_runtime_sql_metadata
    from linktools.ai.persistence import PersistenceDomain
    from linktools.ai.storage import provision_sql

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'late-native.db'}")
    await provision_sql(engine, build_runtime_sql_metadata(frozenset({PersistenceDomain.CONVERSATION})))
    state = SqlStateStore(engine)
    await state.initialize()
    native = SqlStepStore(engine, namespace="steps", tenant_id="one", max_snapshots_per_run=1)
    try:
        record = _record({})
        await state.mutate(lambda tx: tx.insert_record(record))
        await provision_database(engine)
        await native.initialize()
        group, archive_store = await state.storage_group.open_archive(create=True)
        archive = ContentArchive(archive_store, group, namespace="steps", tenant_id="one",
                                 object_store=SqlObjectStore.from_context(state.storage_group.context))
        native.bind_message_archive(archive)
        from linktools.ai.persistence._archive import PreparedArchive
        prepared = PreparedArchive()
        root = await native.prepare_archived_messages(_snapshot("retained", 1, "retained").messages, prepared)
        retained = replace(record, storage_revision=1, data=encode_domain(ArchivedContextBlock(root, "c" * 64, 2)))
        async def publish(tx):
            assert await tx.replace_record(retained, expected_storage_revision=0)
            await native.publish_archived_messages(tx, prepared)
        await state.mutate(publish)
        assert (await native.read_message_roots((root,)))[0]
        assert await state.mutate(lambda tx: tx.delete_record(record.key))
        await native.purge_unreferenced_content(exclusive=_Offline())
        assert await _count(engine, "ai_step_messages") == 0
    finally:
        await native.close()
        await state.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("release", ("replace", "delete"))
async def test_cold_state_owner_retains_and_releases_native_content(tmp_path: Path, release: str) -> None:
    async with _shared(tmp_path / "cold-owner.db") as (engine, _state, native, root):
        other_engine = create_async_engine(str(engine.url))
        owner = SqlStateStore(other_engine, native_content_index=True)
        await owner.initialize()
        try:
            record = _record(encode_domain(ArchivedContextBlock(root, "c" * 64, 2)))
            await owner.mutate(lambda tx: tx.insert_record(record))
            await native.save_snapshot(_snapshot("run", 2, "current"))
            assert (await native.read_message_roots((root,)))[0]
            assert await _count(engine, "ai_step_messages") == 2
            await owner.close()
            owner = SqlStateStore(other_engine, native_content_index=True)
            await owner.initialize()
            assert await owner.read(lambda tx: tx.get_record(record.key)) == record
            if release == "replace":
                async def release_owner(tx):
                    previous = await tx.get_record(record.key)
                    assert previous == record
                    return await tx.replace_record(
                        replace(previous, storage_revision=1, data={}), expected_storage_revision=0,
                    )
                assert await owner.mutate(release_owner)
            else:
                assert await owner.mutate(lambda tx: tx.delete_record(record.key))
            await native.purge_unreferenced_content(exclusive=_Offline())
            assert await _count(engine, "ai_step_messages") == 1
        finally:
            await owner.close()
            await other_engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("record", "fact", "operation"))
async def test_undeclared_native_owner_fails_closed_without_rejecting_business_json(
    tmp_path: Path, kind: str,
) -> None:
    async with _shared(tmp_path / "undeclared.db") as (engine, _state, _native, root):
        owner = SqlStateStore(engine, store_key=b"state-partition")
        await owner.initialize()
        try:
            payload = encode_domain(ArchivedContextBlock(root, "c" * 64, 2))
            record = _record(payload if kind == "record" else {})
            async def publish(tx):
                await tx.insert_record(record)
                try:
                    if kind == "fact":
                        await tx.insert_fact(StoredFact(b"s" * 32, 1, record.key, "chunk", None, "UNKNOWN", payload))
                    elif kind == "operation":
                        await tx.insert_operation(StoredOperation(operation_key("steps", "one", "execution", "界" * 42 + "xx"), b"s" * 32, 1, "UNKNOWN", False, payload))
                except AIError:
                    # Catching the direct failure must not allow a partial commit.
                    pass
            with pytest.raises(AIError) as raised:
                await owner.mutate(publish)
            assert raised.value.code is ErrorCode.STORAGE_CAPABILITY_MISSING
            assert await owner.read(lambda tx: tx.get_record(record.key)) is None
            business = _record(encode_domain({"root": payload}))
            await owner.mutate(lambda tx: tx.insert_record(business))
            assert await owner.read(lambda tx: tx.get_record(business.key)) == business
        finally:
            await owner.close()


@pytest.mark.asyncio
async def test_reopened_writer_rebuilds_reclaimed_root_without_resurrecting_replay(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'cache.db'}"
    engines = [create_async_engine(url), create_async_engine(url)]
    stores = [SqlStepStore(engine, namespace="steps", tenant_id="one", max_snapshots_per_run=1) for engine in engines]
    try:
        await stores[0].initialize(local=True)
        old = _snapshot("run", 1, "old")
        await stores[0].save_snapshot(old)
        await stores[1].save_snapshot(_snapshot("run", 2, "current"))
        # Offline collection requires every other adapter to discard its caches.
        await stores[0].close()
        await stores[1].purge_unreferenced_content(exclusive=_Offline())
        assert await _count(engines[0], "ai_step_messages") == 1
        stores[0] = SqlStepStore(engines[0], namespace="steps", tenant_id="one", max_snapshots_per_run=1)
        await stores[0].save_snapshot(old)
        assert await _count(engines[0], "ai_step_messages") == 1
        assert await stores[0].latest_snapshot(run_id="run") == _snapshot("run", 2, "current")
        revived = replace(old, run_id="revived")
        await stores[0].save_snapshot(revived)
        assert await stores[1].latest_snapshot(run_id="revived") == revived
        assert await _count(engines[0], "ai_step_messages") == 2
    finally:
        for store in stores:
            await store.close()
        for engine in engines:
            await engine.dispose()


@pytest.mark.asyncio
async def test_failed_owner_cas_cannot_release_live_content(tmp_path: Path) -> None:
    async with _shared(tmp_path / "cas.db") as (engine, state, native, root):
        record = _record(encode_domain(ArchivedContextBlock(root, "c" * 64, 2)))
        await state.mutate(lambda tx: tx.insert_record(record))
        await native.save_snapshot(_snapshot("run", 2, "current"))
        assert not await state.mutate(lambda tx: tx.replace_record(
            replace(record, data={}, storage_revision=10), expected_storage_revision=9,
        ))
        await native.purge_unreferenced_content(exclusive=_Offline())
        assert await _count(engine, "ai_step_messages") == 2
        assert (await native.read_message_roots((root,)))[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("ordinary", "cancelled"))
async def test_collection_failure_leaves_already_committed_owner_intact(tmp_path: Path, monkeypatch, failure: str) -> None:
    async with _shared(tmp_path / "failure.db") as (engine, state, native, root):
        record = _record(encode_domain(ArchivedContextBlock(root, "c" * 64, 2)))
        await state.mutate(lambda tx: tx.insert_record(record))
        current = _snapshot("run", 2, "current")
        await native.save_snapshot(current)
        execute = AsyncSession.execute
        async def fail_delete(session, statement, *args, **kwargs):
            if str(statement).startswith("DELETE FROM ai_step_messages"):
                if failure == "cancelled":
                    raise asyncio.CancelledError()
                raise RuntimeError("collector failure")
            return await execute(session, statement, *args, **kwargs)
        # Retire only an unrelated body; the committed UNKNOWN owner remains authoritative.
        await native.save_snapshot(_snapshot("unrelated", 1, "orphan"))
        await native.save_snapshot(_snapshot("unrelated", 2, "live"))
        before = await _count(engine, "ai_step_messages")
        with monkeypatch.context() as patch:
            patch.setattr(AsyncSession, "execute", fail_delete)
            with pytest.raises(asyncio.CancelledError if failure == "cancelled" else RuntimeError):
                await native.purge_unreferenced_content(exclusive=_Offline())
        assert await state.read(lambda tx: tx.get_record(record.key)) == record
        assert await _count(engine, "ai_step_messages") == before
        assert await native.latest_snapshot(run_id="run") == current
        assert (await native.read_message_roots((root,)))[0]


@pytest.mark.asyncio
async def test_unknown_business_commit_never_cold_retries_or_collects_as_rollback(tmp_path: Path, monkeypatch) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'unknown.db'}")
    store = SqlStepStore(engine, namespace="steps", tenant_id="one", max_snapshots_per_run=1)
    await store.initialize(local=True)
    original = type(store.storage_context).run_mutation
    writes = 0
    try:
        await store.save_snapshot(_snapshot("run", 1, "old"))
        async def lose_ack(context, callback, **kwargs):
            nonlocal writes
            result = await original(context, callback, **kwargs)
            if kwargs.get("domain") == "steps.snapshot":
                writes += 1
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return result
        monkeypatch.setattr(type(store.storage_context), "run_mutation", lose_ack)
        current = _snapshot("run", 2, "current")
        with pytest.raises(AIError) as raised:
            await store.save_snapshot(current)
        assert raised.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        assert writes == 1
        assert await store.latest_snapshot(run_id="run") == current
        assert await _count(engine, "ai_step_snapshots") == 2
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_offline_sweep_is_idempotent_and_preserves_replay_tombstones(tmp_path: Path) -> None:
    async with _shared(tmp_path / "sweep.db") as (engine, state, native, root):
        await native.save_snapshot(_snapshot("run", 2))
        result = await native.purge_unreferenced_content(exclusive=_Offline())
        assert result.deleted_messages == result.deleted_sequences == 1
        again = await native.purge_unreferenced_content(exclusive=_Offline())
        assert again.deleted_messages == again.deleted_sequences == 0
        assert await _count(engine, "ai_step_messages") == await _count(engine, "ai_step_sequences") == 0
        assert await _count(engine, "ai_step_snapshots") == 2


@pytest.mark.asyncio
async def test_owner_committed_before_offline_exclusion_is_kept(tmp_path: Path) -> None:
    async with _shared(tmp_path / "guard.db") as (engine, state, native, root):
        await native.save_snapshot(_snapshot("run", 2, "current"))
        lock = asyncio.Lock()
        entered = asyncio.Event()
        release = asyncio.Event()
        record = _record(encode_domain(ArchivedContextBlock(root, "c" * 64, 2)))
        class Guard:
            @asynccontextmanager
            async def offline_exclusivity(self):
                async with lock:
                    yield
        async def writer():
            async with lock:
                entered.set()
                await release.wait()
                await state.mutate(lambda tx: tx.insert_record(record))
        writing = asyncio.create_task(writer())
        await entered.wait()
        collection = asyncio.create_task(native.purge_unreferenced_content(exclusive=Guard()))
        await asyncio.sleep(0)
        assert not collection.done(), "offline collection must wait for admitted writers"
        release.set()
        await writing
        await collection
        assert (await native.read_message_roots((root,)))[0]
        assert await _count(engine, "ai_step_messages") == 2


@pytest.mark.asyncio
async def test_owner_and_native_content_publication_roll_back_together(tmp_path: Path) -> None:
    from linktools.ai.persistence._archive import PreparedArchive

    async with _shared(tmp_path / "rollback.db") as (engine, state, native, _root):
        prepared = PreparedArchive()
        root = await native.prepare_archived_messages(_snapshot("new", 1, "new").messages, prepared)
        record = _record(encode_domain(ArchivedContextBlock(root, "c" * 64, 2)))
        before = await _count(engine, "ai_step_sequences")

        async def publish(tx):
            await tx.insert_record(record)
            await native.publish_archived_messages(tx, prepared)
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        with pytest.raises(AIError):
            await state.mutate(publish)
        assert await state.read(lambda tx: tx.get_record(record.key)) is None
        assert await _count(engine, "ai_step_sequences") == before
        assert await _count(engine, "ai_step_messages") == 1


@pytest.mark.asyncio
async def test_scoped_offline_collection_leaves_other_namespace_content(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'scope.db'}")
    stores = [SqlStepStore(engine, namespace="steps", tenant_id=tenant, max_snapshots_per_run=1)
              for tenant in ("one", "two")]
    await stores[0].initialize(local=True)
    try:
        for store in stores:
            await store.save_snapshot(_snapshot("run", 1, "old"))
            await store.save_snapshot(_snapshot("run", 2, "current"))
        await stores[0].purge_unreferenced_content(exclusive=_Offline())
        table = build_step_sql_metadata().tables["ai_step_messages"]
        async with engine.connect() as connection:
            for store, expected in zip(stores, (1, 2)):
                assert len((await connection.execute(select(table.c.id).where(table.c.store_key == store.store_key))).all()) == expected
    finally:
        for store in stores:
            await store.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_offline_invalidation_requires_fresh_candidates_before_atomic_publication(tmp_path: Path) -> None:
    from linktools.ai.persistence._archive import PreparedArchive
    from linktools.ai.persistence.contracts import ContextProjection

    async with _shared(tmp_path / "multiple-stale.db") as (engine, state, native, _root):
        originals = (_snapshot("a", 1, "A"), _snapshot("b", 1, "B"))
        for snapshot in originals:
            await native.save_snapshot(snapshot)
        prepared = PreparedArchive()
        roots = tuple([await native.prepare_archived_messages(snapshot.messages, prepared) for snapshot in originals])
        for name in ("a", "b"):
            await native.save_snapshot(_snapshot(name, 2, "new-" + name))
        await native.purge_unreferenced_content(exclusive=_Offline())
        before = await _count(engine, "ai_step_messages")
        record = _record(encode_domain(ContextProjection(tuple(ArchivedContextBlock(root, "c" * 64, 2) for root in roots))))
        attempts = 0

        async def publish(tx):
            nonlocal attempts
            attempts += 1
            await tx.insert_record(record)
            await native.publish_archived_messages(tx, prepared)

        with pytest.raises(AIError) as stale:
            await state.mutate(publish)
        assert stale.value.code is ErrorCode.INTERNAL_ERROR
        assert await state.read(lambda tx: tx.get_record(record.key)) is None
        # Offline exclusion expires this adapter's prepared proofs. Re-prepare
        # from the original bytes; never republish an invalidated proof.
        prepared = PreparedArchive()
        refreshed = tuple([await native.prepare_archived_messages(snapshot.messages, prepared) for snapshot in originals])
        assert refreshed == roots
        await state.mutate(publish)
        assert attempts == 2, "one rejected stale publication and one fresh atomic publication"
        assert await state.read(lambda tx: tx.get_record(record.key)) == record
        assert await _count(engine, "ai_step_messages") == before + 2
        assert all(await native.read_message_roots(roots))


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("same", "savepoint_commit", "savepoint_rollback", "outer_rollback", "other_root"))
async def test_state_owner_consumes_only_live_matching_publication_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    from sqlalchemy import event
    from linktools.ai.persistence._archive import PreparedArchive
    from linktools.ai.persistence._step_message_content import NativeMessageContent

    async with _shared(tmp_path / "owner-handoff.db") as (engine, state, native, old_root):
        prepared = PreparedArchive()
        root = await native.prepare_archived_messages(_snapshot("new", 1, "new").messages, prepared)
        retained = old_root if boundary == "other_root" else root
        record = _record(encode_domain(ArchivedContextBlock(retained, "c" * 64, 2)))
        publish = NativeMessageContent.publish_retained
        first = True
        queries = []

        class RollbackSavepoint(Exception):
            pass

        async def publish_at_boundary(content, transaction, candidate, **kwargs):
            nonlocal first
            if not first or not boundary.startswith("savepoint_"):
                return await publish(content, transaction, candidate, **kwargs)
            first = False

            async def nested(session):
                try:
                    async with session.begin_nested():
                        receipt = await publish(content, transaction, candidate, **kwargs)
                        if boundary == "savepoint_rollback":
                            raise RollbackSavepoint()
                except RollbackSavepoint:
                    pass
                return receipt

            return await transaction.run_sql(native.storage_context, nested)

        def capture(_connection, _cursor, statement, parameters, _context, _many):
            if statement.startswith("SELECT ai_step_sequences.sha256, ai_step_sequences.id"):
                queries.append((statement, parameters))

        monkeypatch.setattr(NativeMessageContent, "publish_retained", publish_at_boundary)
        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        try:
            async def write(transaction, *, abort=False):
                await transaction.insert_record(record)
                await native.publish_archived_messages(transaction, prepared)
                if abort:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)

            if boundary == "outer_rollback":
                with pytest.raises(AIError) as failed:
                    await state.mutate(lambda transaction: write(transaction, abort=True))
                assert failed.value.code is ErrorCode.STORAGE_CONFLICT
                assert await state.read(lambda transaction: transaction.get_record(record.key)) is None
            await state.mutate(write)
            assert await state.read(lambda transaction: transaction.get_record(record.key)) == record
            # Node indexing supplies publication roots; only an expired receipt
            # or a different retained root needs this standalone owner lookup.
            assert len(queries) == (1 if boundary in {"savepoint_commit", "savepoint_rollback", "other_root"} else 0)
            assert all(parameters[:2] == (native_store_locator(native.store_key), native.store_key) and len(parameters) == 3 for _, parameters in queries)
            assert (await native.read_message_roots((retained,)))[0]
            if boundary == "other_root":
                await native.purge_unreferenced_content(exclusive=_Offline())
                assert await _count(engine, "ai_step_messages") == 1
            await native.save_snapshot(_snapshot("run", 2, "current"))
            await native.purge_unreferenced_content(exclusive=_Offline())
            assert (await native.read_message_roots((retained,)))[0]
            assert await _count(engine, "ai_step_messages") == 2
            assert await state.mutate(lambda transaction: transaction.delete_record(record.key))
            await native.purge_unreferenced_content(exclusive=_Offline())
            assert await _count(engine, "ai_step_messages") == 1
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
