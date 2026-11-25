#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native Execution receipts preserve exact SQL routes and release evidence."""

from datetime import datetime, timedelta, timezone
from hashlib import sha256

import pytest
from sqlalchemy import event, literal, update

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence._execution_authority import ExecutionAuthority
from linktools.ai.persistence._schema import build_runtime_sql_metadata
from linktools.ai.persistence._split_turn_anchor import SplitSessionTurnTransactions
from linktools.ai.persistence._sql import _SqlTransaction
from linktools.ai.persistence._turn_sql_schema import RUNTIME_TURN_SCOPE
from linktools.ai.persistence.state import state_store_key
from linktools.ai.persistence.turns import StateTurnCoordination
from linktools.ai.storage import TurnScopeIdentity
from linktools.ai.storage._dialects import SQLiteDialect

from .test_state_batch_delete_backends import _open_store
from .test_state_lease_availability import _clock


_SCOPE = TurnScopeIdentity("namespace", "tenant", RUNTIME_TURN_SCOPE)


async def _bind(store):
    store = await store.storage_group._open_partition(state_store_key(_SCOPE.namespace, _SCOPE.tenant_id, "execution"))
    return store, StateTurnCoordination(store, namespace=_SCOPE.namespace, tenant_id=_SCOPE.tenant_id)


async def _claim(authority, slot, request, execution, *, owner=None, seconds=60):
    return await authority.claim_turn(
        _SCOPE, session_key="session:" + slot, request_key=request, request_digest="b" * 64,
        execution_id=execution, owner_id=sha256((owner or execution).encode()).hexdigest()[:32], lease_seconds=seconds,
    )


async def _receipt(store, execution):
    return await store.read(lambda tx: SplitSessionTurnTransactions(tx, _SCOPE).read(route="execution", identity=execution))


def test_mysql57_turn_identity_indexes_fit_exact_binary_keys() -> None:
    from sqlalchemy.dialects.mysql import dialect
    from sqlalchemy.schema import CreateIndex, CreateTable

    table = build_runtime_sql_metadata(frozenset({PersistenceDomain.EXECUTION})).tables["ai_execution_records"]
    mysql = dialect()
    mysql.server_version_info = (5, 7, 44)
    assert "ROW_FORMAT=DYNAMIC" in str(CreateTable(table).compile(dialect=mysql))
    for route, length in (("turn_request_route", 64), ("turn_occupied_route", 128)):
        column = table.c[route]
        index, = [i for i in table.indexes if i.unique and tuple(i.columns.keys()) == ("store_key", route)]
        assert "UNIQUE INDEX" in str(CreateIndex(index).compile(dialect=mysql))
        assert column.type.dialect_impl(mysql).length == length
        assert not hasattr(column.type.dialect_impl(mysql), "collation") and column.nullable


@pytest.mark.asyncio
async def test_sql_turn_request_route_is_exact_and_reusable_after_retirement(tmp_path) -> None:
    async with _open_store("sql", tmp_path) as generic:
        store, authority = await _bind(generic)
        first = await _claim(authority, "s", "a" * 64, "e1")
        second = await _claim(authority, "S", "b" * 64, "e2")
        assert first.lease.fence == second.lease.fence == 1
        assert (await authority.read_turn_by_session(_SCOPE, "session:s")).execution_id == "e1"
        assert (await authority.read_turn_by_session(_SCOPE, "session:S")).execution_id == "e2"
        assert (await authority.read_turn_by_request(_SCOPE, "a" * 64)).execution_id == "e1"
        with pytest.raises(AIError) as conflict:
            await _claim(authority, "different", "a" * 64, "ignored")
        assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        assert await _receipt(store, "ignored") is None
        table = store.storage_group.metadata.tables["ai_execution_records"]
        key = ExecutionAuthority(_SCOPE, "e1", "session:s", first.lease.token, 1).key
        async with store.context.engine.begin() as conn:
            await conn.execute(update(table).where(table.c.store_key == store.store_key, table.c.key == key)
                              .values(turn_slot_key=b"corrupt"))
        for reader, route in ((authority.read_turn, "e1"), (authority.read_turn_by_session, "session:s"),
                              (authority.read_turn_by_request, "a" * 64)):
            with pytest.raises(AIError) as damaged:
                await reader(_SCOPE, route)
            assert damaged.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        async with store.context.engine.begin() as conn:
            await conn.execute(update(table).where(table.c.store_key == store.store_key, table.c.key == key)
                              .values(turn_slot_key=b"s"))
        assert await authority.release_turn(first.lease)
        successor = await _claim(authority, "s", "a" * 64, "e3")
        assert successor.lease.fence == 2
        assert await authority.read_turn(_SCOPE, "e1") is None
        parked = await _receipt(store, "e1")
        assert not parked[1].active and parked[1].receipt_kind == "release"
        assert parked[0]["turn_occupied_route"] is None and parked[0]["turn_request_route"] is None
        assert (await authority.read_turn_by_request(_SCOPE, "a" * 64)).execution_id == "e3"


@pytest.mark.asyncio
async def test_native_release_preserves_exact_receipt_at_existing_statement_budget(tmp_path) -> None:
    async with _open_store("sql", tmp_path) as generic:
        store, authority = await _bind(generic)
        first = await _claim(authority, "s", "a" * 64, "e")
        verbs = []
        def count(_conn, _cursor, statement, *_):
            verbs.append(statement.split()[0])
        engine = store.context.engine.sync_engine
        event.listen(engine, "before_cursor_execute", count)
        try:
            assert (await authority.read_turn_by_request(_SCOPE, "a" * 64)).execution_id == "e"
            assert verbs.count("SELECT") == 1
            verbs.clear()
            assert await authority.release_turn(first.lease)
            assert verbs.count("SELECT") == 1 and verbs.count("UPDATE") == 2
        finally:
            event.remove(engine, "before_cursor_execute", count)
        row, proof, observation = await _receipt(store, "e")
        assert not proof.active and observation is None and proof.receipt_kind == "release"
        assert row["turn_occupied_route"] == b"s" and row["turn_request_route"] is None
        assert await authority.read_turn_by_session(_SCOPE, "session:s") is None
        assert not await authority.release_turn(first.lease)


@pytest.mark.asyncio
async def test_sqlite_release_rechecks_deadline_after_exact_guard(tmp_path, monkeypatch) -> None:
    now = datetime(2030, 1, 1, tzinfo=timezone.utc)
    sampled = now
    async with _open_store("sql", tmp_path) as generic:
        store, authority = await _bind(generic)
        _clock(monkeypatch, store, "sql", lambda: sampled)
        monkeypatch.setattr(SQLiteDialect, "read_snapshot_time_expression", lambda _self: literal(sampled))
        claim = await _claim(authority, "s", "a" * 64, "e", seconds=1)
        before, _, _ = await _receipt(store, "e")
        execute = _SqlTransaction._execute

        async def wait_at_guard(tx, statement, parameters=None):
            nonlocal sampled
            result = await execute(tx, statement, parameters)
            if statement.is_update and statement.table.name == "ai_execution_records":
                sampled = now + timedelta(seconds=1)
            return result

        with monkeypatch.context() as waiting:
            waiting.setattr(_SqlTransaction, "_execute", wait_at_guard)
            assert not await authority.release_turn(claim.lease)
        after, proof, _ = await _receipt(store, "e")
        for name in ("turn_token", "turn_fence", "turn_revision", "turn_receipt", "turn_occupied_route"):
            assert before[name] == after[name]
        assert proof.active


@pytest.mark.asyncio
async def test_sql_turn_failed_successor_rolls_back_predecessor_retirement(tmp_path, monkeypatch) -> None:
    now = datetime(2030, 1, 1, tzinfo=timezone.utc)
    sampled = now
    async with _open_store("sql", tmp_path) as generic:
        store, authority = await _bind(generic)
        _clock(monkeypatch, store, "sql", lambda: sampled)
        monkeypatch.setattr(SQLiteDialect, "read_snapshot_time_expression", lambda _self: literal(sampled))
        first = await _claim(authority, "s", "a" * 64, "e", seconds=1)
        sampled += timedelta(seconds=2)
        execute = _SqlTransaction._execute

        async def fail_publication(tx, statement, parameters=None):
            if statement.is_insert and statement.table.name == "ai_execution_records":
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return await execute(tx, statement, parameters)

        with monkeypatch.context() as failure:
            failure.setattr(_SqlTransaction, "_execute", fail_publication)
            with pytest.raises(AIError) as rolled_back:
                await _claim(authority, "s", "c" * 64, "new")
            assert rolled_back.value.code is ErrorCode.STORAGE_CONFLICT
        assert (await _receipt(store, "e"))[1].active
        assert await _receipt(store, "new") is None
        successor = await _claim(authority, "s", "c" * 64, "new")
        assert successor.lease.fence == 2
        old = (await _receipt(store, "e"))[1]
        assert not old.active and old.receipt_kind == "expire"
        assert not await authority.release_turn(first.lease)


@pytest.mark.asyncio
async def test_sql_turn_two_contenders_return_one_owner(tmp_path) -> None:
    import asyncio
    async with _open_store("sql", tmp_path) as generic:
        store, authority = await _bind(generic)
        first, second = await asyncio.gather(
            _claim(authority, "s", "a" * 64, "first"), _claim(authority, "s", "a" * 64, "second"),
        )
        assert first.execution_id == second.execution_id
        assert (first.lease is None) != (second.lease is None)
        loser = "first" if first.lease is None else "second"
        assert await _receipt(store, loser) is None


@pytest.mark.asyncio
async def test_live_replay_and_busy_do_not_write_or_consume_the_new_request(tmp_path) -> None:
    scope = TurnScopeIdentity("namespace", "tenant", RUNTIME_TURN_SCOPE)
    async with _open_store("sql", tmp_path) as store:
        store = await store.storage_group._open_partition(state_store_key(scope.namespace, scope.tenant_id, "execution"))
        store.bind_runtime_turn_scope(scope)
        authority = StateTurnCoordination(store, namespace=scope.namespace, tenant_id=scope.tenant_id)

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

    async with _open_store("sql", tmp_path) as generic:
        with pytest.raises(AIError) as unbound:
            generic.bind_runtime_turn_scope(_SCOPE)
        assert unbound.value.code is ErrorCode.STORAGE_OWNER_MISMATCH
        store, authority = await _bind(generic)
        engine = store.context.engine.sync_engine
        seen = []
        def observe(*args):
            seen.append(args[2])
        event.listen(engine, "before_cursor_execute", observe)
        try:
            for foreign in (replace(_SCOPE, namespace="other"), replace(_SCOPE, tenant_id="other")):
                with pytest.raises(AIError) as invalid:
                    store.bind_runtime_turn_scope(foreign)
                assert invalid.value.code is ErrorCode.STORAGE_OWNER_MISMATCH
                with pytest.raises(AIError) as false_route:
                    await authority.read_turn(foreign, "e")
                assert false_route.value.code is ErrorCode.STORAGE_OWNER_MISMATCH
            assert not seen
        finally:
            event.remove(engine, "before_cursor_execute", observe)
