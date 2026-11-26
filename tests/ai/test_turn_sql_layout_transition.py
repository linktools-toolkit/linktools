#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split SQL execution receipts preserve exact ownership and fencing."""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from hashlib import sha256

import pytest
from sqlalchemy import select, update

from linktools.ai.core import PersistenceDomain
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistencePlan, PersistenceStorage
from linktools.ai.persistence._split_turn_anchor import SplitSessionTurnTransactions
from linktools.ai.persistence.contracts import SessionRecord, SessionStatus
from linktools.ai.persistence.state import record_key
from linktools.ai.persistence._turn_sql_schema import unoccupied_turn_route
from linktools.ai.storage import FilesystemObjectStore, StorageRoute, TurnScopeIdentity


_SCOPE = TurnScopeIdentity("turn-layout", "tenant", "runtime-pending-turn")
_REQUEST = "a" * 64
_FIELDS = dict(session_key="session:s", request_key=_REQUEST, request_digest="b" * 64, duration=3600)


@asynccontextmanager
async def _stores(path):
    execution = StorageRoute.sqlite(path / "execution.db")
    conversation = StorageRoute.sqlite(path / "conversation.db")
    plan = PersistencePlan(conversation=conversation, execution=execution)
    storage = PersistenceStorage(plan, object_store=FilesystemObjectStore(path / "objects"))
    async with Persistence.open(_SCOPE.namespace, tenant_id=_SCOPE.tenant_id, storage=storage) as state:
        stores = state.snapshot_resources().stores
        yield state, stores[PersistenceDomain.EXECUTION], stores[PersistenceDomain.CONVERSATION]


async def _declare_session(state):
    now = datetime.now(timezone.utc)
    await state.conversation.sessions.create(
        SessionRecord("s", "owner", SessionStatus.OPEN, 0, None, {}, now, now, None, None, "agent")
    )


async def _claim(execution, *, execution_id, owner_id, token, **fields):
    values = {**_FIELDS, **fields, "execution_id": execution_id, "owner_id": owner_id, "token": token}
    return await execution.mutate(lambda tx: SplitSessionTurnTransactions(tx, _SCOPE).claim(**values))


async def _read(execution, *, route, identity):
    return await execution.read(lambda _: execution._turn_transaction()._read_native_execution_receipt(
        _SCOPE, route=route, identity=identity, split=True,
    ))


async def _confirm(execution, *, execution_id, receipt, kind="claim"):
    return await execution.read(lambda _: execution._turn_transaction()._confirm_native_execution_receipt(
        _SCOPE, execution_id=execution_id, receipt=receipt, kind=kind, split=True,
    ))


async def _update_control(execution, identity, **values):
    table = execution.storage_group.metadata.tables["ai_execution_records"]
    key = record_key(_SCOPE.namespace, _SCOPE.tenant_id, "execution", "execution", (identity,))
    async with execution.context.engine.begin() as conn:
        changed = await conn.execute(update(table).where(table.c.store_key == execution.store_key, table.c.key == key).values(**values))
        assert changed.rowcount == 1


@pytest.mark.asyncio
async def test_native_split_layout_replay_and_fence_expired_takeover(tmp_path):
    path = tmp_path / "turns"
    async with _stores(path) as (state, first, _first_conversation):
        await _declare_session(state)
        async with _stores(path) as (_other_state, other, _other_conversation):
            first_claim = await _claim(first, execution_id="first",
                                       owner_id="1" * 32, token="2" * 32)
            assert first_claim.lease.fence == 1
            assert await _confirm(other, execution_id="first", receipt="2" * 32) == first_claim
            for route, identity in (("execution", "first"), ("request", _REQUEST), ("session", "session:s")):
                _, proof, observed = await _read(other, route=route, identity=identity)
                assert proof.execution_id == "first" and not observed.expired
            replay = await _claim(other, execution_id="replay",
                                  owner_id="3" * 32, token="4" * 32)
            assert replay.execution_id == "first" and replay.lease is None
            assert await _claim(other, execution_id="competing",
                                owner_id="3" * 32, token="4" * 32, request_key="c" * 64) is None

            cancelled = await first.mutate(lambda tx: tx._change_native_execution_receipt(
                _SCOPE, execution_id="first", receipt="5" * 32, kind="cancel", lease=None, split=True,
            ))
            assert cancelled is True
            assert await _confirm(other, execution_id="first", receipt="5" * 32, kind="cancel") is True

            await _update_control(first, "first", turn_expiry=datetime.now(timezone.utc) - timedelta(minutes=1))
            _, proof, observed = await _read(other, route="execution", identity="first")
            assert proof.cancel_requested and observed.expired
            successor = await _claim(other, execution_id="second",
                                     owner_id="3" * 32, token="4" * 32)
            assert successor.lease.fence == 2
            _, owner, observed = await _read(first, route="request", identity=_REQUEST)
            assert owner.execution_id == "second" and not observed.expired and not owner.cancel_requested
            old, retired, observed = await _read(first, route="execution", identity="first")
            assert retired.receipt_kind == "expire" and observed is None
            assert old["turn_request_route"] == unoccupied_turn_route(old["key_locator"])
            assert await _confirm(first, execution_id="first", receipt="2" * 32) is None


@pytest.mark.asyncio
async def test_native_split_broken_current_owner_is_not_read_as_idle(tmp_path):
    async with _stores(tmp_path / "broken") as (state, execution, _conversation):
        await _declare_session(state)
        await _claim(execution, execution_id="current", owner_id="1" * 32, token="2" * 32)
        key = record_key(_SCOPE.namespace, _SCOPE.tenant_id, "execution", "execution", ("current",))
        await _update_control(execution, "current", turn_occupied_route=unoccupied_turn_route(sha256(key).hexdigest()))
        with pytest.raises(AIError) as caught:
            await _read(execution, route="execution", identity="current")
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_exact_published_identity_requires_recovery_instead_of_rewriting_its_body(tmp_path):
    async with _stores(tmp_path / "published") as (state, execution, _conversation):
        await _declare_session(state)
        await _claim(execution, execution_id="current", owner_id="1" * 32, token="2" * 32)
        original = {"fixture": "unpublished output must not be replaced by an admission"}
        await _update_control(execution, "current", kind="execution", payload_json=original,
                              turn_expiry=datetime.now(timezone.utc) - timedelta(minutes=1))
        with pytest.raises(AIError) as caught:
            await _claim(execution, execution_id="current", owner_id="3" * 32, token="4" * 32)
        assert caught.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        control = execution.storage_group.metadata.tables["ai_execution_records"]
        async with execution.context.engine.begin() as connection:
            result = await connection.execute(select(control.c.payload_json).where(control.c.store_key == execution.store_key))
            assert result.scalar_one() == original
