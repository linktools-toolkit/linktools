#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native-only online reclamation follows committed owners, not cache proofs."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import NativeMessageRoot, NativeMessageSlice, SqlStepStore
from linktools.ai.persistence import _native_content_index as index
from linktools.ai.persistence._archive import ContentArchive
from linktools.ai.persistence._sql import SqlStateStore
from linktools.ai.persistence._step_schema import build_step_sql_metadata
from linktools.ai.persistence.codec import encode_domain
from linktools.ai.persistence.contracts import ArchivedContextBlock
from linktools.ai.persistence.state import OperationQuery, StoredFact, StoredOperation, StoredRecord, operation_key
from linktools.ai.storage import SqlObjectStore

from .test_step_content_cleanup import _snapshot


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
        root = NativeMessageRoot(group, native.store_digest, (NativeMessageSlice(first, 1, 0, 1),))
        yield engine, state, native, root
    finally:
        await native.close()
        await state.close()
        await engine.dispose()


def _record(data, *, key: bytes = b"r" * 32, state: str = "UNKNOWN") -> StoredRecord:
    return StoredRecord(key, None, None, "owner", b'owner', state, 0, None, 0, None, data)


async def _count(engine, table: str) -> int:
    async with engine.connect() as connection:
        return await connection.scalar(text(f"SELECT count(*) FROM {table}"))


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ("record", "fact", "operation"))
async def test_online_collection_preserves_all_committed_root_owners_including_unknown(
    tmp_path: Path, owner: str,
) -> None:
    async with _shared(tmp_path / "owners.db") as (engine, state, native, root):
        payload = encode_domain(ArchivedContextBlock(root, "c" * 64, 2))
        record = _record(payload if owner == "record" else {})

        async def publish(tx):
            await tx.insert_record(record)
            if owner == "fact":
                await tx.insert_fact(StoredFact(b"s" * 32, 1, record.key_digest, "chunk", None, "UNKNOWN", payload))
            elif owner == "operation":
                await tx.insert_operation(StoredOperation(operation_key("steps", "one", "execution", "界" * 85 + "x"), b"s" * 32, 1, "UNKNOWN", False, payload))

        await state.mutate(publish)
        current = _snapshot("run", 2, "current")
        await native.save_snapshot(current)
        assert await _count(engine, "ai_step_messages") == 2
        assert (await native.read_message_roots((root,)))[0]
        if owner == "operation":
            async def transition(tx):
                previous = await tx.get_operation(operation_key("steps", "one", "execution", "界" * 85 + "x"))
                assert await tx.replace_operation(replace(previous, state="DONE"), expected_state="UNKNOWN")
            await state.mutate(transition)
            assert (await native.read_message_roots((root,)))[0]
            await state.mutate(lambda tx: tx.delete_operations(OperationQuery(stream_digest=b"s" * 32)))
        elif owner == "record":
            assert await state.mutate(lambda tx: tx.replace_record(
                replace(record, data={}, storage_revision=1), expected_storage_revision=0,
            ))
        else:
            assert await state.mutate(lambda tx: tx.delete_record(record.key_digest))
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
        assert await state.mutate(lambda tx: tx.delete_record(record.key_digest))
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
            assert await owner.read(lambda tx: tx.get_record(record.key_digest)) == record
            if release == "replace":
                async def release_owner(tx):
                    previous = await tx.get_record(record.key_digest)
                    assert previous == record
                    return await tx.replace_record(
                        replace(previous, storage_revision=1, data={}), expected_storage_revision=0,
                    )
                assert await owner.mutate(release_owner)
            else:
                assert await owner.mutate(lambda tx: tx.delete_record(record.key_digest))
            await index.maintain_native_content(native.storage_context, store_digests=(native.store_digest,))
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
        owner = SqlStateStore(engine, store_digest=b"u" * 32)
        await owner.initialize()
        try:
            payload = encode_domain(ArchivedContextBlock(root, "c" * 64, 2))
            record = _record(payload if kind == "record" else {})
            async def publish(tx):
                await tx.insert_record(record)
                try:
                    if kind == "fact":
                        await tx.insert_fact(StoredFact(b"s" * 32, 1, record.key_digest, "chunk", None, "UNKNOWN", payload))
                    elif kind == "operation":
                        await tx.insert_operation(StoredOperation(operation_key("steps", "one", "execution", "界" * 85 + "x"), b"s" * 32, 1, "UNKNOWN", False, payload))
                except AIError:
                    # Catching the direct failure must not allow a partial commit.
                    pass
            with pytest.raises(AIError) as raised:
                await owner.mutate(publish)
            assert raised.value.code is ErrorCode.STORAGE_CAPABILITY_MISSING
            assert await owner.read(lambda tx: tx.get_record(record.key_digest)) is None
            business = _record(encode_domain({"root": payload}))
            await owner.mutate(lambda tx: tx.insert_record(business))
            assert await owner.read(lambda tx: tx.get_record(business.key_digest)) == business
        finally:
            await owner.close()


@pytest.mark.asyncio
async def test_reclaimed_cross_process_cache_rebuilds_same_root_without_resurrecting_replay(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'cache.db'}"
    engines = [create_async_engine(url), create_async_engine(url)]
    stores = [SqlStepStore(engine, namespace="steps", tenant_id="one", max_snapshots_per_run=1) for engine in engines]
    try:
        await stores[0].initialize(local=True)
        old = _snapshot("run", 1, "old")
        await stores[0].save_snapshot(old)
        await stores[1].save_snapshot(_snapshot("run", 2, "current"))
        assert await _count(engines[0], "ai_step_messages") == 1
        await stores[0].save_snapshot(old)
        assert await _count(engines[0], "ai_step_messages") == 1
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
async def test_failed_owner_cas_cannot_release_live_edges(tmp_path: Path) -> None:
    async with _shared(tmp_path / "cas.db") as (engine, state, native, root):
        record = _record(encode_domain(ArchivedContextBlock(root, "c" * 64, 2)))
        await state.mutate(lambda tx: tx.insert_record(record))
        await native.save_snapshot(_snapshot("run", 2, "current"))
        assert not await state.mutate(lambda tx: tx.replace_record(
            replace(record, data={}, storage_revision=10), expected_storage_revision=9,
        ))
        assert await _count(engine, "ai_step_messages") == 2
        assert (await native.read_message_roots((root,)))[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("ordinary", "cancelled"))
async def test_collection_failure_does_not_retry_committed_owner(tmp_path: Path, monkeypatch, failure: str) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'maintenance.db'}")
    store = SqlStepStore(engine, namespace="steps", tenant_id="one", max_snapshots_per_run=1)
    await store.initialize(local=True)
    try:
        await store.save_snapshot(_snapshot("run", 1, "old"))
        async def fail(*args, **kwargs):
            if failure == "cancelled":
                raise asyncio.CancelledError()
            raise RuntimeError("collector failure")
        monkeypatch.setattr(index, "collect_native_content", fail)
        current = _snapshot("run", 2, "current")
        if failure == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await store.save_snapshot(current)
        else:
            await store.save_snapshot(current)
        assert await store.latest_snapshot(run_id="run") == current
        assert await _count(engine, "ai_step_snapshots") == 2
        assert await _count(engine, "ai_step_messages") == 2
    finally:
        await store.close()
        await engine.dispose()


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
async def test_collection_budget_bounds_deleted_rows_and_preserves_pending_work(tmp_path: Path, monkeypatch) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'budget.db'}")
    store = SqlStepStore(engine, namespace="steps", tenant_id="one", max_snapshots_per_run=1)
    await store.initialize(local=True)
    async def defer(_context, **kwargs):
        pass
    monkeypatch.setattr(index, "maintain_native_content", defer)
    try:
        await store.save_snapshot(_snapshot("run", 1, *(str(i) for i in range(12))))
        await store.save_snapshot(_snapshot("run", 2))
        before = await _count(engine, "ai_step_messages") + await _count(engine, "ai_step_sequences")
        await store.storage_context.run_mutation(lambda session: index.collect_native_content(
            store.storage_context, session, store_digests=(store.store_digest,), budget=3,
        ))
        after = await _count(engine, "ai_step_messages") + await _count(engine, "ai_step_sequences")
        assert 0 < before - after <= 3
        assert await _count(engine, "ai_native_content_edges") > 0
        for _ in range(12):
            await store.storage_context.run_mutation(lambda session: index.collect_native_content(
                store.storage_context, session, store_digests=(store.store_digest,), budget=3,
            ))
        assert await _count(engine, "ai_step_messages") == await _count(engine, "ai_step_sequences") == 0
        assert await _count(engine, "ai_step_snapshots") == 2
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_pin_winning_against_collector_keeps_target_and_unknown_owner(tmp_path: Path, monkeypatch) -> None:
    async with _shared(tmp_path / "pin-race.db") as (engine, state, native, root):
        async def defer(_context, **kwargs):
            pass
        monkeypatch.setattr(index, "maintain_native_content", defer)
        await native.save_snapshot(_snapshot("run", 2, "current"))
        # WAL permits the independent writer to commit while the collector
        # still holds its earlier read snapshot; rollback journal mode cannot.
        async with engine.connect() as connection:
            await connection.exec_driver_sql("PRAGMA journal_mode=WAL")
        other_engine = create_async_engine(str(engine.url))
        other = SqlStateStore(other_engine, native_content_index=True)
        await other.initialize()
        await other.storage_group.open_archive()
        execute = AsyncSession.execute
        inserted = False
        record = _record(encode_domain(ArchivedContextBlock(root, "c" * 64, 2)))

        async def publish_before_delete(session, statement, *args, **kwargs):
            nonlocal inserted
            if not inserted and str(statement).startswith("DELETE FROM ai_step_sequences"):
                inserted = True
                # An independent writer wins after the collector's old metadata
                # read. Its target guard must invalidate that deletion attempt.
                await other.mutate(lambda tx: tx.insert_record(record))
            return await execute(session, statement, *args, **kwargs)

        try:
            with monkeypatch.context() as patch:
                patch.setattr(AsyncSession, "execute", publish_before_delete)
                await native.storage_context.run_mutation(lambda session: index.collect_native_content(
                    native.storage_context, session, store_digests=(native.store_digest,),
                ))
            assert inserted
            assert await _count(engine, "ai_step_messages") == 2
            assert (await native.read_message_roots((root,)))[0]
        finally:
            await other.close()
            await other_engine.dispose()


@pytest.mark.asyncio
async def test_candidate_reenqueue_cannot_be_consumed_by_an_older_collector(tmp_path: Path, monkeypatch) -> None:
    async with _shared(tmp_path / "candidate.db") as (_engine, _state, native, root):
        sequence = build_step_sql_metadata().tables["ai_step_sequences"]
        edges = build_step_sql_metadata().tables["ai_native_content_edges"]
        async with native.storage_context.engine.connect() as connection:
            target = await connection.scalar(select(sequence.c.id).where(sequence.c.sha256 == root.slices[0].first))
        await native.storage_context.run_mutation(lambda session: index.enqueue(native.storage_context, session, "sequence", target))
        execute = AsyncSession.execute
        reenqueued = False

        async def renew_before_ack(session, statement, *args, **kwargs):
            nonlocal reenqueued
            if not reenqueued and str(statement).startswith("DELETE FROM ai_native_content_edges"):
                reenqueued = True
                await index.enqueue(native.storage_context, session, "sequence", target)
            return await execute(session, statement, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(AsyncSession, "execute", renew_before_ack)
            await native.storage_context.run_mutation(lambda session: index.collect_native_content(
                native.storage_context, session, store_digests=(native.store_digest,), budget=1,
            ))
        assert reenqueued
        async with native.storage_context.engine.connect() as connection:
            assert await connection.scalar(select(edges.c.id).where(
                edges.c.edge_kind == "candidate_sequence", edges.c.target_id == target,
            )) is not None
        assert (await native.read_message_roots((root,)))[0]


@pytest.mark.asyncio
async def test_owner_and_native_edge_publication_roll_back_together(tmp_path: Path) -> None:
    from linktools.ai.persistence._archive import PreparedArchive

    async with _shared(tmp_path / "rollback.db") as (engine, state, native, _root):
        prepared = PreparedArchive()
        root = await native.prepare_archived_messages(_snapshot("new", 1, "new").messages, prepared)
        record = _record(encode_domain(ArchivedContextBlock(root, "c" * 64, 2)))
        before = await _count(engine, "ai_native_content_edges")

        async def publish(tx):
            await tx.insert_record(record)
            await native.publish_archived_messages(tx, prepared)
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        with pytest.raises(AIError):
            await state.mutate(publish)
        assert await state.read(lambda tx: tx.get_record(record.key_digest)) is None
        assert await _count(engine, "ai_native_content_edges") == before
        assert await _count(engine, "ai_step_messages") == 1


@pytest.mark.asyncio
async def test_scoped_collection_leaves_other_namespace_content_and_clears_dead_candidates(tmp_path: Path, monkeypatch) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'scope.db'}")
    stores = [SqlStepStore(engine, namespace="steps", tenant_id=tenant, max_snapshots_per_run=1)
              for tenant in ("one", "two")]
    await stores[0].initialize(local=True)
    async def defer(_context, **kwargs):
        pass
    monkeypatch.setattr(index, "maintain_native_content", defer)
    try:
        for store in stores:
            await store.save_snapshot(_snapshot("run", 1, "old"))
            await store.save_snapshot(_snapshot("run", 2, "current"))
        await stores[0].storage_context.run_mutation(lambda session: index.enqueue(
            stores[0].storage_context, session, "sequence", 999999,
        ))
        await stores[0].storage_context.run_mutation(lambda session: index.collect_native_content(
            stores[0].storage_context, session, store_digests=(stores[0].store_digest,),
        ))
        table = build_step_sql_metadata().tables["ai_step_messages"]
        edges = build_step_sql_metadata().tables["ai_native_content_edges"]
        async with engine.connect() as connection:
            for store, expected in zip(stores, (1, 2)):
                assert len((await connection.execute(select(table.c.id).where(
                    table.c.store_digest == store.store_digest,
                ))).all()) == expected
            assert await connection.scalar(select(edges.c.id).where(edges.c.target_id == 999999)) is None
    finally:
        for store in stores:
            await store.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_multiple_stale_candidates_each_rebuild_once_before_atomic_publication(tmp_path: Path) -> None:
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
        before = await _count(engine, "ai_step_messages")
        record = _record(encode_domain(ContextProjection(tuple(ArchivedContextBlock(root, "c" * 64, 2) for root in roots))))
        attempts = 0

        async def publish(tx):
            nonlocal attempts
            attempts += 1
            await tx.insert_record(record)
            await native.publish_archived_messages(tx, prepared)

        await state.mutate(publish)
        assert attempts == 3, "each newly encountered stale candidate permits only one cold retry"
        assert await state.read(lambda tx: tx.get_record(record.key_digest)) == record
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
                assert await state.read(lambda transaction: transaction.get_record(record.key_digest)) is None
            await state.mutate(write)
            assert await state.read(lambda transaction: transaction.get_record(record.key_digest)) == record
            assert len(queries) == (1 if boundary == "same" else 3 if boundary == "savepoint_rollback" else 2)
            assert all(parameters[0] == native.store_digest and len(parameters) == 2 for _, parameters in queries)
            assert (await native.read_message_roots((retained,)))[0]
            if boundary == "other_root":
                assert await _count(engine, "ai_step_messages") == 1
            await native.save_snapshot(_snapshot("run", 2, "current"))
            assert (await native.read_message_roots((retained,)))[0]
            assert await _count(engine, "ai_step_messages") == 2
            assert await state.mutate(lambda transaction: transaction.delete_record(record.key_digest))
            assert await _count(engine, "ai_step_messages") == 1
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
