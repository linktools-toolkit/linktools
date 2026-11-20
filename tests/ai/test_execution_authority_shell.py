#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unpublished SQL authority must remain invisible to ordinary Execution reads."""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._execution_authority import ExecutionAuthority
from linktools.ai.storage import TurnScopeIdentity

from .test_history_projection_conformance import _record


@pytest.mark.asyncio
async def test_control_shell_hidden_and_promoted_in_place(tmp_path) -> None:
    namespace, tenant = "execution-control", "tenant"
    async with Persistence.open(namespace, tenant_id=tenant, storage=PersistenceStorage.sqlite(tmp_path / "control.db")) as state:
        repository = state.execution.executions
        store = repository.state_store
        authority = ExecutionAuthority(
            TurnScopeIdentity(namespace, tenant, "runtime-pending-turn"), "execution", "session:r", "b" * 32, 1,
        )
        expiry = datetime.now(timezone.utc) + timedelta(hours=1)
        candidate = _record(ExecutionStatus.STARTED, 1)
        assert candidate.execution_id == authority.execution_id
        await repository.prepare_content(candidate)
        record = await repository._stored_content("execution", (authority.execution_id,), candidate, state=candidate.status.value)
        assert record.key == authority.key

        await store.mutate(lambda tx: tx.insert_execution_shell(authority, expiry=expiry))
        assert await repository.get(authority.execution_id, tenant_id=tenant) is None
        assert (await repository.get_many((authority.execution_id,), tenant_id=tenant)) == {}
        assert not (await repository.list_recoverable_page(tenant_id=tenant, cursor=None, limit=5)).items
        assert await store.read(lambda tx: tx.get_record(authority.key)) is None
        assert not any(item.key == authority.key for item in await store.read(lambda tx: tx.scan_records()))
        assert await store.mutate(lambda tx: tx.guard_record_revision(authority.key, expected_storage_revision=0)) is None
        assert not await store.mutate(lambda tx: tx.delete_record(authority.key))
        table = store.storage_group.metadata.tables["ai_execution_records"]
        engine = store.storage_group.context.engine

        async def physical():
            async with engine.begin() as conn:
                return (await conn.execute(select(table).where(table.c.key == authority.key))).mappings().one()

        hidden = await physical()
        assert hidden["kind"] == "execution_shell" and hidden["turn_token"] == b"b" * 32
        from dataclasses import replace

        with pytest.raises(AIError) as lost:
            await store.mutate(lambda tx: tx.publish_execution_shell(replace(authority, token="c" * 32), record))
        assert lost.value.code is ErrorCode.STORAGE_OWNER_MISMATCH
        await store.mutate(lambda tx: tx.publish_execution_shell(authority, record))
        public = await physical()
        assert public["id"] == hidden["id"] and public["kind"] == "execution"
        assert public["turn_token"] == hidden["turn_token"] and public["turn_expiry"] == hidden["turn_expiry"]
        assert await repository.get(authority.execution_id, tenant_id=tenant) == candidate


@pytest.mark.asyncio
async def test_control_slot_claim_renew_release_uses_one_execution_authority(tmp_path) -> None:
    namespace, tenant = "execution-control", "tenant"
    async with Persistence.open(namespace, tenant_id=tenant, storage=PersistenceStorage.sqlite(tmp_path / "lifecycle.db")) as state:
        store = state.execution.executions.state_store
        scope = TurnScopeIdentity(namespace, tenant, "runtime-pending-turn")
        store.bind_runtime_turn_scope(scope)
        fields = dict(session_key="session:session", request_key="a" * 64, request_digest="b" * 64, execution_id="execution")
        claim = await store.claim_turn_admission(scope, **fields, owner_id="a" * 32, duration=60, token="b" * 32, coowned=True)
        assert claim.lease.fence == 1
        authority = ExecutionAuthority(scope, "execution", "session:session", "b" * 32, 1)
        table = store.storage_group.metadata.tables["ai_execution_records"]

        async def live_shell():
            async with store.storage_group.context.engine.begin() as connection:
                return (await connection.execute(select(table.c.id).where(
                    table.c.store_key == store.store_key, table.c.key == authority.key,
                ))).scalar_one_or_none()

        first_shell = await live_shell()
        assert first_shell is not None
        with pytest.raises(AIError) as wrong:
            await store.retire_turn_admission(
                scope, session_key="session:session", request_key="a" * 64, execution_id="execution",
                token="e" * 32, fence=1, receipt="f" * 32, coowned=True,
            )
        assert wrong.value.code is ErrorCode.STORAGE_CONFLICT
        assert await live_shell() == first_shell
        for route, identity in (("session", "session:session"), ("request", "a" * 64), ("execution", "execution")):
            proof = await store.observe_turn_admission(scope, route=route, identity=identity, coowned=True)
            assert proof.execution_id == "execution"
        assert await state.execution.executions.get("execution", tenant_id=tenant) is None
        renewed = await store.change_turn_admission(
            scope, execution_id="execution", receipt="c" * 32, kind="renew",
            lease=("session:session", "a" * 64, "b" * 32, 1), duration=60, coowned=True,
        )
        assert renewed.lease == claim.lease
        assert await store.confirm_turn_admission(
            scope, execution_id="execution", receipt="c" * 32, kind="renew", coowned=True,
        ) == renewed
        assert await store.retire_turn_admission(
            scope, session_key="session:session", request_key="a" * 64, execution_id="execution",
            token="b" * 32, fence=1, receipt="d" * 32, coowned=True,
        )
        assert await store.confirm_turn_admission(
            scope, execution_id="execution", receipt="d" * 32, kind="release", coowned=True,
        ) is True
        assert await live_shell() is None
        assert await store.observe_turn_admission(scope, route="session", identity="session:session", coowned=True) is None
        after = await store.claim_turn_admission(
            scope, **fields, owner_id="e" * 32, duration=60, token="f" * 32, coowned=True,
        )
        assert after.lease.fence == 2
        assert await state.execution.executions.get("execution", tenant_id=tenant) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("expiry", ["2000-01-01 00:00:00.000000", "2030-01-01 00:00:00.000001", "malformed"])
async def test_expired_sql_release_rolls_back_tentative_revocation_and_slot_clear(tmp_path, monkeypatch, expiry) -> None:
    from sqlalchemy import literal, text
    from linktools.ai.storage import SQLiteDialect

    namespace, tenant = "execution-control", "tenant"
    async with Persistence.open(namespace, tenant_id=tenant,
                                storage=PersistenceStorage.sqlite(tmp_path / "expiry.db")) as state:
        store = state.execution.executions.state_store
        scope = TurnScopeIdentity(namespace, tenant, "runtime-pending-turn")
        store.bind_runtime_turn_scope(scope)
        fields = dict(session_key="session:session", request_key="a" * 64, request_digest="b" * 64, execution_id="execution")
        claim = await store.claim_turn_admission(scope, **fields, owner_id="a" * 32, duration=60, token="b" * 32, coowned=True)
        engine = store.storage_group.context.engine
        table = store.storage_group.metadata.tables["ai_execution_records"]
        slots = store.storage_group.metadata.tables["ai_turn_slots"]
        exact = ExecutionAuthority(scope, "execution", "session:session", "b" * 32, 1)

        async with engine.begin() as connection:
            await connection.execute(text(
                'UPDATE ai_execution_records SET turn_expiry = :expiry WHERE "key" = :key AND store_key = :store'
            ), {"expiry": expiry, "key": exact.key, "store": store.store_key})
        monkeypatch.setattr(SQLiteDialect, "read_snapshot_time_expression", lambda self: literal("2030-01-01T00:00:00.000"))
        with pytest.raises(AIError) as expired:
            await store.retire_turn_admission(
                scope, session_key="session:session", request_key="a" * 64, execution_id="execution",
                token=claim.lease.token, fence=claim.lease.fence, receipt="c" * 32, coowned=True,
            )
        assert expired.value.code is ErrorCode.STORAGE_CONFLICT
        async with engine.begin() as connection:
            token = (await connection.execute(select(table.c.turn_token).where(
                table.c.store_key == store.store_key, table.c.key == exact.key,
            ))).scalar_one()
            pointer = (await connection.execute(select(slots.c.active_execution_id).where(
                slots.c.store_key == store.store_key,
            ))).scalar_one()
        assert token == b"b" * 32 and pointer == b"execution"


def test_server_deadline_uses_native_database_clock_when_compiled() -> None:
    from sqlalchemy import select
    from sqlalchemy.dialects import mysql, postgresql, sqlite

    from linktools.ai.persistence._sql import _initial_turn_deadline
    from linktools.ai.storage import MySQLDialect, PostgreSQLDialect, SQLiteDialect

    for adapter, dialect, expected in (
        (MySQLDialect(), mysql.dialect(), ("timestampadd", "microsecond", "utc_timestamp", "1501000")),
        (PostgreSQLDialect(), postgresql.dialect(), ("clock_timestamp", "interval")),
        (SQLiteDialect(), sqlite.dialect(), ("strftime", "+1.501 seconds", "000")),
    ):
        statement = select(_initial_turn_deadline(adapter, timedelta(milliseconds=1501)))
        compiled = str(statement.compile(dialect=dialect, compile_kwargs={"literal_binds": True})).lower()
        assert all(part in compiled for part in expected), compiled


@pytest.mark.asyncio
async def test_execution_authority_point_queries_use_sqlite_unique_locator(tmp_path) -> None:
    from hashlib import sha256

    from sqlalchemy import literal

    from linktools.ai.persistence._turn_sql import SqlTurnRows

    namespace, tenant = "execution-control", "tenant"
    async with Persistence.open(
        namespace, tenant_id=tenant, storage=PersistenceStorage.sqlite(tmp_path / "locator.db"),
    ) as state:
        store = state.execution.executions.state_store
        scope = TurnScopeIdentity(namespace, tenant, "runtime-pending-turn")
        source = SqlTurnRows(store.storage_group.metadata, store.store_key, scope, coowned=True)
        execution_id, session_key, token = "execution", "session:session", "b" * 32
        clock = literal("2026-11-20T12:00:00.000")
        key = source.execution_key(execution_id)
        statements = (
            ("proposed_control", source.select_claim(
                session_key=session_key, request_key="a" * 64, execution_id=execution_id, clock=clock,
            )),
            ("ai_execution_records", source.control_guard(
                session_key=session_key, execution_id=execution_id, token=token, fence=1,
            )),
            ("ai_execution_records", source.control_set_expiry(
                execution_id=execution_id, token=token, fence=1, expiry=None,
            )),
            ("ai_execution_records", source.revoked_control_deadline(
                execution_id=execution_id, fence=1, clock=clock, sqlite=True,
            )),
            ("ai_execution_records", source.reclaim_retired_shell(execution_id)),
        )
        engine = store.storage_group.context.engine
        async with engine.connect() as connection:
            for alias, statement in statements:
                compiled = statement.compile(dialect=engine.dialect, compile_kwargs={"render_postcompile": True})
                parameters = tuple(compiled.params[name] for name in compiled.positiontup or ())
                assert key in parameters and sha256(key).hexdigest() in parameters
                plan = (await connection.exec_driver_sql("EXPLAIN QUERY PLAN " + str(compiled), parameters)).all()
                assert any(
                    detail.startswith(f"SEARCH {alias} USING INDEX ")
                    and "store_key=? AND key_locator=?" in detail
                    for _, _, _, detail in plan
                ), plan
