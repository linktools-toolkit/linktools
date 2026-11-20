#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL coordination source rows preserve exact identity and release evidence."""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import MetaData, create_engine, insert, literal, update
from sqlalchemy.exc import IntegrityError

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence._record_key import encode_key_components
from linktools.ai.persistence._turn_sql import SqlTurnProof, SqlTurnRows
from linktools.ai.persistence._turn_sql_schema import (
    TURN_ADMISSION_TABLE, TURN_SCOPE_MAX_BYTES, TURN_SLOT_TABLE, build_turn_sql_metadata,
)
from linktools.ai.storage import TurnScopeIdentity
from linktools.ai.storage._dialects import SQLiteDialect


def test_sql_turn_request_route_is_exact_and_reusable_after_retirement() -> None:
    metadata = MetaData()
    build_turn_sql_metadata(metadata)
    engine = create_engine("sqlite://")
    try:
        metadata.create_all(engine)
        slots, admissions = metadata.tables[TURN_SLOT_TABLE], metadata.tables[TURN_ADMISSION_TABLE]
        scope = TurnScopeIdentity("namespace", "tenant", "scope")
        store_key = encode_key_components(("turns",), 128)
        rows = SqlTurnRows(metadata, store_key, scope)
        partition = {"store_key": store_key, "scope_key": encode_key_components(scope.as_tuple(), TURN_SCOPE_MAX_BYTES)}
        now = datetime(2030, 1, 1, tzinfo=timezone.utc)

        def slot(session_key: bytes, execution_id: bytes, fence: int):
            return dict(**partition, session_key=session_key, active_execution_id=execution_id,
                        lease_token=b"token", lease_fence=fence, lease_expires_at=now + timedelta(seconds=5))

        def admission(session_key: bytes, execution_id: bytes, request_key: bytes, fence: int):
            return dict(**partition, session_key=session_key, execution_id=execution_id, request_key=request_key,
                        live_request_key=request_key, request_digest="digest", owner_id="owner", lease_fence=fence,
                        cancel_requested=False, receipt_id="token", receipt_kind="claim")

        with engine.begin() as conn:
            conn.execute(insert(slots), [slot(b"s", b"e1", 1), slot(b"S", b"e2", 1)])
            conn.execute(insert(admissions), [admission(b"s", b"e1", b"r", 1), admission(b"S", b"e2", b"r ", 1)])

            def live(route, identity, stamp=now):
                result = conn.execute(rows.select_live(route=route, identity=identity, clock=literal(stamp))).mappings().one_or_none()
                return None if result is None else rows.live_proof(result, decode_time=SQLiteDialect().decode_database_time)

            assert live("request", "r").execution_id == "e1"
            assert live("request", "r ").execution_id == "e2"
            assert live("request", "r", now + timedelta(seconds=5)) is None
            assert live("execution", "missing") is None
            conn.execute(update(slots).where(slots.c.session_key == b"s").values(active_execution_id=b"other"))
            for route, key in (("session", "s"), ("request", "r"), ("execution", "e1")):
                with pytest.raises(AIError) as damaged:
                    live(route, key)
                assert damaged.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            conn.execute(update(slots).where(slots.c.session_key == b"s").values(active_execution_id=b"e1"))
            conn.execute(update(admissions).where(admissions.c.execution_id == b"e1").values(live_request_key=None))
            with pytest.raises(AIError) as projection:
                live("execution", "e1")
            assert projection.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            conn.execute(update(admissions).where(admissions.c.execution_id == b"e1").values(live_request_key=b"r"))
            with pytest.raises(IntegrityError), conn.begin_nested():
                conn.execute(insert(admissions), admission(b"S", b"e3", b"r", 2))

            assert conn.execute(rows.serialize_slot(session_key="s", execution_id="e1", token="stale", fence=1)).rowcount == 0
            assert conn.execute(rows.serialize_slot(session_key="s", execution_id="e1", token="token", fence=1)).rowcount == 1
            conn.execute(update(admissions).where(admissions.c.execution_id == b"e1").values(
                live_request_key=None, receipt_id="release-proof", receipt_kind="release",
            ))
            conn.execute(update(admissions).where(admissions.c.execution_id == b"e2").values(
                live_request_key=None, receipt_id="expire-proof", receipt_kind="expire",
            ))
            conn.execute(update(slots).where(slots.c.session_key == b"s").values(
                active_execution_id=None, lease_token=None, lease_expires_at=None,
            ))
            conn.execute(update(slots).where(slots.c.session_key == b"S").values(active_execution_id=b"e3", lease_fence=2))
            conn.execute(insert(admissions), admission(b"S", b"e3", b"r", 2))
            assert live("request", "r").execution_id == "e3"
            assert live("execution", "e1") is None
            assert live("session", "s") is None
            retired = SqlTurnProof.from_row(conn.execute(rows.select_proof("e1")).mappings().one())
            assert not retired.active and (retired.receipt_kind, retired.receipt_id) == ("release", "release-proof")
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
            widths = [column.type.dialect_impl(mysql).length for column in index.columns]
            assert sum(widths) <= 783
            assert all(not hasattr(column.type.dialect_impl(mysql), "collation") for column in index.columns)


@pytest.mark.asyncio
async def test_state_sql_release_serializes_and_preserves_proof_without_select(
    tmp_path,
) -> None:
    from sqlalchemy import event, select
    from linktools.ai.persistence._sql import SqlStateStore
    from .test_state_batch_delete_backends import _open_store

    scope = TurnScopeIdentity("namespace", "tenant", "scope")
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    async with _open_store("sql", tmp_path) as store:
        assert isinstance(store, SqlStateStore)
        group = store.storage_group
        engine = group.context.engine
        slots, admissions = group.metadata.tables[TURN_SLOT_TABLE], group.metadata.tables[TURN_ADMISSION_TABLE]
        partition = {"store_key": store.store_key, "scope_key": encode_key_components(scope.as_tuple(), TURN_SCOPE_MAX_BYTES)}
        async with engine.begin() as conn:
            await conn.execute(insert(slots), [
                dict(**partition, session_key=b"s", active_execution_id=b"e", lease_token=b"token",
                     lease_fence=1, lease_expires_at=future),
                dict(**partition, session_key=b"expired", active_execution_id=b"e-expired", lease_token=b"token",
                     lease_fence=1, lease_expires_at=future - timedelta(hours=2)),
            ])
            await conn.execute(insert(admissions), dict(
                **partition, execution_id=b"e", session_key=b"s", request_key=b"r", live_request_key=b"r",
                request_digest="digest", owner_id="owner", lease_fence=1,
                cancel_requested=False, receipt_id="token", receipt_kind="claim",
            ))

        statements = []

        def count(_conn, _cursor, statement, *_args):
            statements.append(statement.split()[0])

        event.listen(engine.sync_engine, "before_cursor_execute", count)
        try:
            live = await store.observe_turn_admission(scope, route="request", identity="r")
            assert live is not None and live.execution_id == "e"
            assert statements.count("SELECT") == 1
            statements.clear()
            assert await store.retire_turn_admission(
                scope, session_key="s", request_key="r", execution_id="e", token="token", fence=1, receipt="release",
            )
            assert statements.count("SELECT") == 0 and statements.count("UPDATE") == 3
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", count)

        retired = await store.get_turn_operation_proof(scope, "e")
        assert not retired.active and (retired.receipt_kind, retired.receipt_id) == ("release", "release")
        assert await store.observe_turn_admission(scope, route="session", identity="s") is None
        with pytest.raises(AIError) as stale:
            await store.retire_turn_admission(
                scope, session_key="expired", request_key="missing", execution_id="e-expired", token="token",
                fence=1, receipt="must not commit",
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

    scope = TurnScopeIdentity("namespace", "tenant", "scope")
    now = datetime(2030, 1, 1, tzinfo=timezone.utc)
    expiry = now + timedelta(seconds=1)
    sampled = now
    clock_samples = []

    def source_time(_self):
        clock_samples.append(sampled)
        return literal(sampled)

    monkeypatch.setattr(SQLiteDialect, "read_snapshot_time_expression", source_time)
    async with _open_store("sql", tmp_path) as store:
        assert isinstance(store, SqlStateStore)
        group = store.storage_group
        engine = group.context.engine
        slots, admissions = group.metadata.tables[TURN_SLOT_TABLE], group.metadata.tables[TURN_ADMISSION_TABLE]
        partition = {"store_key": store.store_key, "scope_key": encode_key_components(scope.as_tuple(), TURN_SCOPE_MAX_BYTES)}
        async with engine.begin() as conn:
            await conn.execute(insert(slots), dict(
                **partition, session_key=b"s", active_execution_id=b"e", lease_token=b"token",
                lease_fence=1, lease_expires_at=expiry,
            ))
            await conn.execute(insert(admissions), dict(
                **partition, execution_id=b"e", session_key=b"s", request_key=b"r", live_request_key=b"r",
                request_digest="digest", owner_id="owner", lease_fence=1,
                cancel_requested=False, receipt_id="token", receipt_kind="claim",
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
                scope, session_key="s", request_key="r", execution_id="e", token="token", fence=1, receipt="release",
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
            assert (await conn.execute(select(admissions.c.receipt_id))).scalar_one() == "token"


@pytest.mark.asyncio
async def test_sql_turn_lifecycle_and_cross_session_reclamation(tmp_path, monkeypatch) -> None:
    from sqlalchemy import select
    from linktools.ai.persistence._sql import SqlStateStore, _SqlTransaction
    from .test_state_batch_delete_backends import _open_store
    from .test_state_lease_availability import _clock

    scope = TurnScopeIdentity("namespace", "tenant", "scope")
    now = datetime(2030, 1, 1, tzinfo=timezone.utc)
    sampled = now

    async with _open_store("sql", tmp_path) as store:
        assert isinstance(store, SqlStateStore)
        _clock(monkeypatch, store, "sql", lambda: sampled)
        # The read and write clocks use the same deterministic source instant.
        monkeypatch.setattr(SQLiteDialect, "read_snapshot_time_expression", lambda _self: literal(sampled))

        async def claim(session, request, execution, *, owner="o", digest="d"):
            return await store.claim_turn_admission(
                scope, session_key=session, request_key=request, request_digest=digest,
                execution_id=execution, owner_id=owner, duration=1, token="claim-" + execution + owner,
            )

        first = await claim("s", "r", "e")
        assert first is not None and first.lease is not None
        assert (await store.confirm_turn_admission(scope, execution_id="e", receipt="claim-eo", kind="claim")) == first
        duplicate = await claim("s", "r", "ignored")
        assert duplicate.execution_id == "e" and duplicate.lease is None
        assert await claim("s", "unused", "never-written") is None
        with pytest.raises(AIError) as foreign:
            await claim("different-session", "r", "foreign")
        assert foreign.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        assert await store.get_turn_operation_proof(scope, "never-written") is None

        assert (await store.change_turn_admission(scope, execution_id="e", receipt="cancel", kind="cancel")).cancel_requested
        assert await store.confirm_turn_admission(scope, execution_id="e", receipt="cancel", kind="cancel") is True
        renewed = await store.change_turn_admission(
            scope, execution_id="e", receipt="renew", kind="renew",
            lease=("s", "r", first.lease.token, first.lease.fence), duration=1,
        )
        assert renewed == (await store.confirm_turn_admission(scope, execution_id="e", receipt="renew", kind="renew"))
        assert renewed.lease == first.lease and renewed.cancel_requested
        assert await store.retire_turn_admission(
            scope, session_key="s", request_key="r", execution_id="e",
            token=first.lease.token, fence=first.lease.fence, receipt="release-first",
        )
        assert await store.confirm_turn_admission(scope, execution_id="e", receipt="release-first", kind="release") is True
        old1 = await claim("s", "r", "rec")
        old2 = await claim("x", "y", "old")
        assert old1.lease.fence == 2 and old2.lease.fence == 1
        assert await store.confirm_turn_admission(scope, execution_id="e", receipt="release-first", kind="release") is True

        sampled += timedelta(seconds=2)
        execute = _SqlTransaction._execute

        async def fail_publication(transaction, statement, parameters=None):
            if statement.is_insert and statement.table.name == TURN_ADMISSION_TABLE:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return await execute(transaction, statement, parameters)

        with monkeypatch.context() as failure:
            failure.setattr(_SqlTransaction, "_execute", fail_publication)
            with pytest.raises(AIError) as rolled_back:
                await claim("s", "y", "new", owner="new-o", digest="new-d")
            assert rolled_back.value.code is ErrorCode.STORAGE_CONFLICT
        for stale_execution in ("rec", "old"):
            assert (await store.get_turn_operation_proof(scope, stale_execution)).active
        assert await store.get_turn_operation_proof(scope, "new") is None
        reclaimed = await claim("s", "y", "new", owner="new-o", digest="new-d")
        assert reclaimed.lease.fence == 3
        assert (await store.observe_turn_admission(scope, route="request", identity="y")).execution_id == "new"
        assert await store.observe_turn_admission(scope, route="session", identity="x") is None
        assert await store.observe_turn_admission(scope, route="execution", identity="rec") is None
        assert await store.observe_turn_admission(scope, route="request", identity="r") is None
        for execution in ("old", "rec"):
            proof = await store.get_turn_operation_proof(scope, execution)
            assert not proof.active and proof.receipt_kind == "expire"
        assert not await store.retire_turn_admission(
            scope, session_key="x", request_key="y", execution_id="old",
            token=old2.lease.token, fence=old2.lease.fence, receipt="stale-release",
        )
        recovered = await claim("s", "y", "new", owner="next-o", digest="new-d")
        assert recovered.lease is None
        sampled += timedelta(seconds=2)
        recovered = await claim("s", "y", "new", owner="next-o", digest="new-d")
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

    scope = TurnScopeIdentity("namespace", "tenant", "scope")
    async with _open_store("sql", tmp_path) as store:
        assert isinstance(store, SqlStateStore)

        async def claim(label):
            return await store.claim_turn_admission(
                scope, session_key="session", request_key="request", request_digest="digest",
                execution_id=label, owner_id=label, duration=30, token=label,
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

    scope = TurnScopeIdentity("namespace", "tenant", "scope")
    async with _open_store("sql", tmp_path) as store:
        authority = StateTurnCoordination(store, namespace=scope.namespace, tenant_id=scope.tenant_id)

        async def claim(session, request, execution):
            return await authority.claim_turn(
                scope, session_key=session, request_key=request, request_digest="digest",
                execution_id=execution, owner_id=execution, lease_seconds=60,
            )

        first = await claim("s", "r", "original")
        assert first.lease is not None
        seen = []

        def capture(_conn, _cursor, stmt, *_args):
            seen.append(stmt.lstrip().split(None, 1)[0].upper())

        engine = store.context.engine.sync_engine
        event.listen(engine, "before_cursor_execute", capture)
        try:
            replay = await claim("s", "r", "proposed")
            assert replay.lease is None and replay.execution_id == first.execution_id
            assert await claim("s", "unused", "busy") is None
            with pytest.raises(AIError) as foreign:
                await claim("foreign", "r", "other")
            assert foreign.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
            assert seen.count("SELECT") == 3
            assert all(verb not in {"INSERT", "UPDATE", "DELETE"} for verb in seen)
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        assert await authority.read_turn_by_request(scope, "unused") is None
        assert await authority.release_turn(first.lease)
        assert (await claim("s", "unused", "next")).lease is not None
