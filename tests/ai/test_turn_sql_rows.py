#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL coordination source rows preserve exact identity and release evidence."""

from datetime import datetime, timedelta, timezone
from hashlib import sha256

import pytest
from sqlalchemy import MetaData, create_engine, insert, literal, update
from sqlalchemy.exc import IntegrityError

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence._turn_sql import SqlTurnProof, SqlTurnRows
from linktools.ai.persistence._turn_sql_schema import (
    TURN_ADMISSION_TABLE, RUNTIME_TURN_SCOPE, TURN_SLOT_TABLE, build_turn_sql_metadata,
)
from linktools.ai.storage import TurnScopeIdentity
from linktools.ai.persistence.state import state_store_key
from linktools.ai.storage._dialects import SQLiteDialect


def test_sql_turn_request_route_is_exact_and_reusable_after_retirement() -> None:
    metadata = MetaData()
    build_turn_sql_metadata(metadata)
    engine = create_engine("sqlite://")
    try:
        metadata.create_all(engine)
        slots, admissions = metadata.tables[TURN_SLOT_TABLE], metadata.tables[TURN_ADMISSION_TABLE]
        scope = TurnScopeIdentity("namespace", "tenant", RUNTIME_TURN_SCOPE)
        store_key = state_store_key(scope.namespace, scope.tenant_id, "execution")
        rows = SqlTurnRows(metadata, store_key, scope)
        partition = {"store_key": store_key, "session_kind": 1}
        now = datetime(2030, 1, 1, tzinfo=timezone.utc)

        def slot(session_key: bytes, execution_id: bytes, fence: int):
            return dict(**partition, session_key=session_key, active_execution_id=execution_id,
                        lease_token=b"a" * 32, lease_fence=fence, lease_expires_at=now + timedelta(seconds=5))

        def admission(session_key: bytes, execution_id: bytes, request_key: bytes, fence: int):
            return dict(**partition, session_key=session_key, execution_id=execution_id, retired_request_key=None,
                        route_kind=1, route_key=request_key, request_digest="d" * 64, owner_id=b"e" * 32, lease_fence=fence,
                        cancel_requested=False, receipt_id=b"a" * 32, receipt_kind="claim")

        with engine.begin() as conn:
            conn.execute(insert(slots), [slot(b"s", b"e1", 1), slot(b"S", b"e2", 1)])
            conn.execute(insert(admissions), [admission(b"s", b"e1", b"a" * 64, 1), admission(b"S", b"e2", b"b" * 64, 1)])

            def live(route, identity, stamp=now):
                result = conn.execute(rows.select_live(route=route, identity=identity, clock=literal(stamp))).mappings().one_or_none()
                return None if result is None else rows.live_proof(result, decode_time=SQLiteDialect().decode_database_time)

            assert live("request", "a" * 64).execution_id == "e1"
            assert live("request", "b" * 64).execution_id == "e2"
            assert live("request", "a" * 64, now + timedelta(seconds=5)) is None
            assert live("execution", "missing") is None
            conn.execute(update(slots).where(slots.c.session_key == b"s").values(active_execution_id=b"other"))
            for route, key in (("session", "session:s"), ("request", "a" * 64), ("execution", "e1")):
                with pytest.raises(AIError) as damaged:
                    live(route, key)
                assert damaged.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            conn.execute(update(slots).where(slots.c.session_key == b"s").values(active_execution_id=b"e1"))
            conn.execute(update(admissions).where(admissions.c.execution_id == b"e1").values(route_key=b"wrong"))
            with pytest.raises(AIError) as projection:
                live("execution", "e1")
            assert projection.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            conn.execute(update(admissions).where(admissions.c.execution_id == b"e1").values(route_key=b"a" * 64))
            with pytest.raises(IntegrityError), conn.begin_nested():
                conn.execute(insert(admissions), admission(b"S", b"e3", b"a" * 64, 2))

            assert conn.execute(rows.serialize_slot(session_key="session:s", execution_id="e1", token="b" * 32, fence=1)).rowcount == 0
            assert conn.execute(rows.serialize_slot(session_key="session:s", execution_id="e1", token="a" * 32, fence=1)).rowcount == 1
            conn.execute(update(admissions).where(admissions.c.execution_id == b"e1").values(
                route_kind=2, route_key=b"e1", retired_request_key=b"a" * 64, receipt_id=b"c" * 32, receipt_kind="release",
            ))
            conn.execute(update(admissions).where(admissions.c.execution_id == b"e2").values(
                route_kind=2, route_key=b"e2", retired_request_key=b"b" * 64, receipt_id=b"d" * 32, receipt_kind="expire",
            ))
            conn.execute(update(slots).where(slots.c.session_key == b"s").values(
                active_execution_id=None, lease_token=None, lease_expires_at=None,
            ))
            conn.execute(update(slots).where(slots.c.session_key == b"S").values(active_execution_id=b"e3", lease_fence=2))
            conn.execute(insert(admissions), admission(b"S", b"e3", b"a" * 64, 2))
            assert live("request", "a" * 64).execution_id == "e3"
            assert live("execution", "e1") is None
            assert live("session", "session:s") is None
            retired = SqlTurnProof.from_row(conn.execute(rows.select_proof("e1")).mappings().one())
            assert not retired.active and (retired.receipt_kind, retired.receipt_id) == ("release", "c" * 32)
    finally:
        engine.dispose()


def test_mysql57_turn_identity_indexes_fit_exact_binary_keys() -> None:
    from sqlalchemy.dialects.mysql import dialect
    from sqlalchemy.schema import CreateIndex, CreateTable

    metadata = MetaData()
    build_turn_sql_metadata(metadata)
    mysql = dialect()
    mysql.server_version_info = (5, 7, 44)
    for table in metadata.tables.values():
        assert "ROW_FORMAT=DYNAMIC" in str(CreateTable(table).compile(dialect=mysql))
        for index in table.indexes:
            if not index.unique:
                continue
            assert "UNIQUE INDEX" in str(CreateIndex(index).compile(dialect=mysql))
            widths = [getattr(column.type.dialect_impl(mysql), "length", None) or 2 for column in index.columns]
            assert max(widths) <= 128 and sum(widths) <= 162
            assert all(not column.nullable for column in index.columns)
            assert all(not hasattr(column.type.dialect_impl(mysql), "collation") for column in index.columns)


@pytest.mark.asyncio
async def test_state_sql_release_serializes_and_preserves_proof_without_select(
    tmp_path,
) -> None:
    from sqlalchemy import event, select
    from linktools.ai.persistence._sql import SqlStateStore
    from .test_state_batch_delete_backends import _open_store

    scope = TurnScopeIdentity("namespace", "tenant", RUNTIME_TURN_SCOPE)
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    async with _open_store("sql", tmp_path) as store:
        store = await store.storage_group._open_partition(state_store_key(scope.namespace, scope.tenant_id, "execution"))
        store.bind_runtime_turn_scope(scope)
        assert isinstance(store, SqlStateStore)
        group = store.storage_group
        engine = group.context.engine
        slots, admissions = group.metadata.tables[TURN_SLOT_TABLE], group.metadata.tables[TURN_ADMISSION_TABLE]
        partition = {"store_key": store.store_key, "session_kind": 1}
        async with engine.begin() as conn:
            await conn.execute(insert(slots), [
                dict(**partition, session_key=b"s", active_execution_id=b"e", lease_token=b"a" * 32,
                     lease_fence=1, lease_expires_at=future),
                dict(**partition, session_key=b"expired", active_execution_id=b"e-expired", lease_token=b"a" * 32,
                     lease_fence=1, lease_expires_at=future - timedelta(hours=2)),
            ])
            await conn.execute(insert(admissions), dict(
                **partition, execution_id=b"e", session_key=b"s", retired_request_key=None, route_kind=1, route_key=b"a" * 64,
                request_digest="d" * 64, owner_id=b"e" * 32, lease_fence=1,
                cancel_requested=False, receipt_id=b"a" * 32, receipt_kind="claim",
            ))

        statements = []

        def count(_conn, _cursor, statement, *_args):
            statements.append(statement.split()[0])

        event.listen(engine.sync_engine, "before_cursor_execute", count)
        try:
            live = await store.observe_turn_admission(scope, route="request", identity="a" * 64)
            assert live is not None and live.execution_id == "e"
            assert statements.count("SELECT") == 1
            statements.clear()
            assert await store.retire_turn_admission(
                scope, session_key="session:s", request_key="a" * 64, execution_id="e", token="a" * 32, fence=1, receipt="f" * 32,
            )
            assert statements.count("SELECT") == 0 and statements.count("UPDATE") == 3
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", count)

        retired = await store.get_turn_operation_proof(scope, "e")
        assert not retired.active and (retired.receipt_kind, retired.receipt_id) == ("release", "f" * 32)
        assert await store.observe_turn_admission(scope, route="session", identity="session:s") is None
        with pytest.raises(AIError) as stale:
            await store.retire_turn_admission(
                scope, session_key="session:expired", request_key="b" * 64, execution_id="e-expired", token="a" * 32,
                fence=1, receipt="f" * 32,
            )
        assert stale.value.code is ErrorCode.STORAGE_CONFLICT
        async with engine.begin() as conn:
            assert (await conn.execute(select(slots.c.revision).where(slots.c.session_key == b"expired"))).scalar_one() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("takeover", [False, True])
async def test_sqlite_release_guard_waits_before_observing_authority(tmp_path, monkeypatch, takeover) -> None:
    import asyncio
    from sqlalchemy import event, select
    from linktools.ai.persistence._sql import SqlStateStore
    from .test_state_batch_delete_backends import _open_store

    scope = TurnScopeIdentity("namespace", "tenant", RUNTIME_TURN_SCOPE)
    now = datetime(2030, 1, 1, tzinfo=timezone.utc)
    expiry = now + timedelta(seconds=1)
    sampled = now
    clock_samples = []

    def source_time(_self):
        clock_samples.append(sampled)
        return literal(sampled)

    monkeypatch.setattr(SQLiteDialect, "read_snapshot_time_expression", source_time)
    async with _open_store("sql", tmp_path) as store:
        store = await store.storage_group._open_partition(state_store_key(scope.namespace, scope.tenant_id, "execution"))
        store.bind_runtime_turn_scope(scope)
        assert isinstance(store, SqlStateStore)
        group = store.storage_group
        engine = group.context.engine
        slots, admissions = group.metadata.tables[TURN_SLOT_TABLE], group.metadata.tables[TURN_ADMISSION_TABLE]
        partition = {"store_key": store.store_key, "session_kind": 1}
        async with engine.begin() as conn:
            await conn.execute(insert(slots), dict(
                **partition, session_key=b"s", active_execution_id=b"e", lease_token=b"a" * 32,
                lease_fence=1, lease_expires_at=expiry,
            ))
            await conn.execute(insert(admissions), dict(
                **partition, execution_id=b"e", session_key=b"s", retired_request_key=None, route_kind=1, route_key=b"a" * 64,
                request_digest="d" * 64, owner_id=b"e" * 32, lease_fence=1,
                cancel_requested=False, receipt_id=b"a" * 32, receipt_kind="claim",
            ))

        blocked_guard = asyncio.Event()

        def before_guard(_conn, _cursor, statement, *_args):
            if statement.startswith("UPDATE ai_turn_slots SET revision="):
                blocked_guard.set()

        async with engine.begin() as other:
            values = ({"lease_token": b"new", "lease_fence": 2} if takeover else {"lease_fence": slots.c.lease_fence})
            await other.execute(update(slots).values(**values))
            event.listen(engine.sync_engine, "before_cursor_execute", before_guard)
            waiter = asyncio.create_task(store.retire_turn_admission(
                scope, session_key="session:s", request_key="a" * 64, execution_id="e", token="a" * 32, fence=1, receipt="f" * 32,
            ))
            try:
                await asyncio.wait_for(blocked_guard.wait(), 5)
                assert not waiter.done()
                assert not clock_samples
                sampled = expiry
            finally:
                event.remove(engine.sync_engine, "before_cursor_execute", before_guard)
        if takeover:
            assert not await asyncio.wait_for(waiter, 5)
            assert not clock_samples
        else:
            with pytest.raises(AIError) as expired:
                await asyncio.wait_for(waiter, 5)
            assert expired.value.code is ErrorCode.STORAGE_CONFLICT
            assert clock_samples == [expiry]
        async with engine.begin() as conn:
            assert (await conn.execute(select(slots.c.revision))).scalar_one() == 0
            assert (await conn.execute(select(admissions.c.receipt_id))).scalar_one() == b"a" * 32


@pytest.mark.asyncio
async def test_sql_turn_lifecycle_and_cross_session_reclamation(tmp_path, monkeypatch) -> None:
    from sqlalchemy import select
    from linktools.ai.persistence._sql import SqlStateStore, _SqlTransaction
    from .test_state_batch_delete_backends import _open_store
    from .test_state_lease_availability import _clock

    scope = TurnScopeIdentity("namespace", "tenant", RUNTIME_TURN_SCOPE)
    now = datetime(2030, 1, 1, tzinfo=timezone.utc)
    sampled = now

    async with _open_store("sql", tmp_path) as store:
        store = await store.storage_group._open_partition(state_store_key(scope.namespace, scope.tenant_id, "execution"))
        store.bind_runtime_turn_scope(scope)
        assert isinstance(store, SqlStateStore)
        _clock(monkeypatch, store, "sql", lambda: sampled)
        # The read and write clocks use the same deterministic source instant.
        monkeypatch.setattr(SQLiteDialect, "read_snapshot_time_expression", lambda _self: literal(sampled))

        async def claim(session, request, execution, *, owner="o", digest="d"):
            return await store.claim_turn_admission(
                scope, session_key="session:" + session, request_key=request, request_digest=sha256(digest.encode()).hexdigest(),
                execution_id=execution, owner_id=sha256(owner.encode()).hexdigest()[:32], duration=1,
                token=sha256(("claim-" + execution + owner).encode()).hexdigest()[:32],
            )

        first = await claim("s", "a" * 64, "e")
        assert first is not None and first.lease is not None
        admissions = store.storage_group.metadata.tables[TURN_ADMISSION_TABLE]
        async with store.storage_group.context.engine.begin() as conn:
            physical = (await conn.execute(select(admissions).where(admissions.c.execution_id == b"e"))).mappings().one()
            assert (physical["route_kind"], physical["route_key"], physical["retired_request_key"]) == (1, b"a" * 64, None)
        assert (await store.confirm_turn_admission(scope, execution_id="e", receipt=first.lease.token, kind="claim")) == first
        duplicate = await claim("s", "a" * 64, "ignored")
        assert duplicate.execution_id == "e" and duplicate.lease is None
        assert await claim("s", "c" * 64, "never-written") is None
        with pytest.raises(AIError) as foreign:
            await claim("different-session", "a" * 64, "foreign")
        assert foreign.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        assert await store.get_turn_operation_proof(scope, "never-written") is None

        assert (await store.change_turn_admission(scope, execution_id="e", receipt="a" * 32, kind="cancel")).cancel_requested
        assert await store.confirm_turn_admission(scope, execution_id="e", receipt="a" * 32, kind="cancel") is True
        renewed = await store.change_turn_admission(
            scope, execution_id="e", receipt="b" * 32, kind="renew",
            lease=("session:s", "a" * 64, first.lease.token, first.lease.fence), duration=1,
        )
        assert renewed == (await store.confirm_turn_admission(scope, execution_id="e", receipt="b" * 32, kind="renew"))
        assert renewed.lease == first.lease and renewed.cancel_requested
        assert await store.retire_turn_admission(
            scope, session_key="session:s", request_key="a" * 64, execution_id="e",
            token=first.lease.token, fence=first.lease.fence, receipt="c" * 32,
        )
        admissions = store.storage_group.metadata.tables[TURN_ADMISSION_TABLE]
        async with store.storage_group.context.engine.begin() as conn:
            physical = (await conn.execute(select(admissions).where(admissions.c.execution_id == b"e"))).mappings().one()
            assert (physical["route_kind"], physical["route_key"], physical["retired_request_key"]) == (2, b"e", b"a" * 64)
        assert await store.confirm_turn_admission(scope, execution_id="e", receipt="c" * 32, kind="release") is True
        old1 = await claim("s", "a" * 64, "rec")
        old2 = await claim("x", "b" * 64, "old")
        assert old1.lease.fence == 2 and old2.lease.fence == 1
        assert await store.confirm_turn_admission(scope, execution_id="e", receipt="c" * 32, kind="release") is True

        sampled += timedelta(seconds=2)
        execute = _SqlTransaction._execute

        async def fail_publication(transaction, statement, parameters=None):
            if statement.is_insert and statement.table.name == TURN_ADMISSION_TABLE:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return await execute(transaction, statement, parameters)

        with monkeypatch.context() as failure:
            failure.setattr(_SqlTransaction, "_execute", fail_publication)
            with pytest.raises(AIError) as rolled_back:
                await claim("s", "b" * 64, "new", owner="new-o", digest="new-d")
            assert rolled_back.value.code is ErrorCode.STORAGE_CONFLICT
        for stale_execution in ("rec", "old"):
            assert (await store.get_turn_operation_proof(scope, stale_execution)).active
        assert await store.get_turn_operation_proof(scope, "new") is None
        reclaimed = await claim("s", "b" * 64, "new", owner="new-o", digest="new-d")
        assert reclaimed.lease.fence == 3
        assert (await store.observe_turn_admission(scope, route="request", identity="b" * 64)).execution_id == "new"
        assert await store.observe_turn_admission(scope, route="session", identity="session:x") is None
        assert await store.observe_turn_admission(scope, route="execution", identity="rec") is None
        assert await store.observe_turn_admission(scope, route="request", identity="a" * 64) is None
        for execution in ("old", "rec"):
            proof = await store.get_turn_operation_proof(scope, execution)
            assert not proof.active and proof.receipt_kind == "expire"
        assert not await store.retire_turn_admission(
            scope, session_key="session:x", request_key="b" * 64, execution_id="old",
            token=old2.lease.token, fence=old2.lease.fence, receipt="d" * 32,
        )
        recovered = await claim("s", "b" * 64, "new", owner="next-o", digest="new-d")
        assert recovered.lease is None
        sampled += timedelta(seconds=2)
        recovered = await claim("s", "b" * 64, "new", owner="next-o", digest="new-d")
        assert recovered.lease.fence == 4
        assert await store.confirm_turn_admission(
            scope, execution_id="new", receipt=reclaimed.lease.token, kind="claim",
        ) is None
        slots = store.storage_group.metadata.tables[TURN_SLOT_TABLE]
        async with store.storage_group.context.engine.begin() as conn:
            assert (await conn.execute(select(slots.c.lease_fence).where(slots.c.session_key == b"x"))).scalar_one() == 1


@pytest.mark.asyncio
async def test_sql_turn_two_contenders_return_one_owner(tmp_path) -> None:
    import asyncio
    from linktools.ai.persistence._sql import SqlStateStore
    from .test_state_batch_delete_backends import _open_store

    scope = TurnScopeIdentity("namespace", "tenant", RUNTIME_TURN_SCOPE)
    async with _open_store("sql", tmp_path) as store:
        store = await store.storage_group._open_partition(state_store_key(scope.namespace, scope.tenant_id, "execution"))
        store.bind_runtime_turn_scope(scope)
        assert isinstance(store, SqlStateStore)

        async def claim(label):
            return await store.claim_turn_admission(
                scope, session_key="session:session", request_key="a" * 64, request_digest="b" * 64,
                execution_id=label, owner_id=sha256(label.encode()).hexdigest()[:32], duration=30,
                token=sha256(label.encode()).hexdigest()[:32],
            )

        first, second = await asyncio.gather(claim("first"), claim("second"))
        assert {first.execution_id, second.execution_id} in ({"first"}, {"second"})
        assert (first.lease is None) != (second.lease is None)
        loser = "first" if first.lease is None else "second"
        assert await store.get_turn_operation_proof(scope, loser) is None


@pytest.mark.asyncio
async def test_live_replay_and_busy_do_not_write_or_consume_the_new_request(tmp_path) -> None:
    from sqlalchemy import event
    from linktools.ai.persistence.turns import StateTurnCoordination
    from .test_state_batch_delete_backends import _open_store

    scope = TurnScopeIdentity("namespace", "tenant", RUNTIME_TURN_SCOPE)
    async with _open_store("sql", tmp_path) as store:
        store = await store.storage_group._open_partition(state_store_key(scope.namespace, scope.tenant_id, "execution"))
        store.bind_runtime_turn_scope(scope)
        authority = StateTurnCoordination(store, namespace=scope.namespace, tenant_id=scope.tenant_id, _runtime_sql=True)

        async def claim(session, request, execution):
            return await authority.claim_turn(
                scope, session_key="session:" + session, request_key=request, request_digest="b" * 64,
                execution_id=execution, owner_id=sha256(execution.encode()).hexdigest()[:32], lease_seconds=60,
            )

        first = await claim("s", "a" * 64, "original")
        assert first.lease is not None
        seen = []

        def capture(_conn, _cursor, stmt, *_args):
            seen.append(stmt.lstrip().split(None, 1)[0].upper())

        engine = store.context.engine.sync_engine
        event.listen(engine, "before_cursor_execute", capture)
        try:
            replay = await claim("s", "a" * 64, "proposed")
            assert replay.lease is None and replay.execution_id == first.execution_id
            assert await claim("s", "c" * 64, "busy") is None
            with pytest.raises(AIError) as foreign:
                await claim("foreign", "a" * 64, "other")
            assert foreign.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
            assert seen.count("SELECT") == 3
            assert all(verb not in {"INSERT", "UPDATE", "DELETE"} for verb in seen)
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        assert await authority.read_turn_by_request(scope, "c" * 64) is None
        assert await authority.release_turn(first.lease)
        assert (await claim("s", "c" * 64, "next")).lease is not None


def test_compact_addresses_preserve_raw_width_and_separate_slot_families() -> None:
    from linktools.ai.persistence._turn_sql import turn_session_address, turn_session_identity

    maximum = "案" * 42 + ":x"
    raw = maximum.encode("utf-8")
    assert len(raw) == 128
    assert turn_session_address("session:" + maximum) == (1, raw)
    assert turn_session_identity(1, raw) == "session:" + maximum
    with pytest.raises(ValueError):
        turn_session_address("session:" + maximum + "x")
    identical = "a" * 64
    assert turn_session_address("session:" + identical) == (1, identical.encode())
    assert turn_session_address("request:" + identical) == (2, identical.encode())
    with pytest.raises(ValueError):
        turn_session_address("request:" + identical.upper())


@pytest.mark.asyncio
async def test_compact_sql_route_requires_exact_bound_execution_partition(tmp_path) -> None:
    from dataclasses import replace
    from sqlalchemy import event
    from .test_state_batch_delete_backends import _open_store

    scope = TurnScopeIdentity("namespace", "tenant", RUNTIME_TURN_SCOPE)
    async with _open_store("sql", tmp_path) as generic:
        with pytest.raises(AIError) as unbound:
            generic.bind_runtime_turn_scope(scope)
        assert unbound.value.code is ErrorCode.STORAGE_OWNER_MISMATCH
        store = await generic.storage_group._open_partition(state_store_key(scope.namespace, scope.tenant_id, "execution"))
        store.bind_runtime_turn_scope(scope)
        engine = store.context.engine.sync_engine
        seen = []

        def observe(*args):
            seen.append(args[2])

        event.listen(engine, "before_cursor_execute", observe)
        try:
            for foreign in (replace(scope, namespace="other"), replace(scope, tenant_id="other")):
                with pytest.raises(AIError) as invalid:
                    store.bind_runtime_turn_scope(foreign)
                assert invalid.value.code is ErrorCode.STORAGE_OWNER_MISMATCH
                with pytest.raises(AIError) as false_route:
                    await store.observe_turn_admission(foreign, route="execution", identity="e")
                assert false_route.value.code is ErrorCode.STORAGE_OWNER_MISMATCH
            assert not seen
        finally:
            event.remove(engine, "before_cursor_execute", observe)
