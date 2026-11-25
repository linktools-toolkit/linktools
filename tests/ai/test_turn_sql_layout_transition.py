#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Current strict SQL layouts agree on exact receipts and fencing."""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select, update

from linktools.ai.core import PersistenceDomain
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistencePlan, PersistenceStorage
from linktools.ai.persistence._split_turn_anchor import SplitSessionTurnTransactions
from linktools.ai.persistence.contracts import SessionRecord, SessionStatus
from linktools.ai.persistence.state import record_key
from linktools.ai.storage import FilesystemObjectStore, StorageRoute, TurnScopeIdentity


_SCOPE = TurnScopeIdentity("turn-layout", "tenant", "runtime-pending-turn")
_REQUEST = "a" * 64
_FIELDS = dict(session_key="session:s", request_key=_REQUEST, request_digest="b" * 64, duration=3600)


@asynccontextmanager
async def _stores(path, *, split):
    if split:
        execution = StorageRoute.sqlite(path / "execution.db")
        conversation = StorageRoute.sqlite(path / "conversation.db")
        plan = PersistencePlan(conversation=conversation, execution=execution)
        storage = PersistenceStorage(plan, object_store=FilesystemObjectStore(path / "objects"))
    else:
        storage = PersistenceStorage.from_root(path)
    async with Persistence.open(_SCOPE.namespace, tenant_id=_SCOPE.tenant_id, storage=storage) as state:
        stores = state.snapshot_resources().stores
        yield state, stores[PersistenceDomain.EXECUTION], stores[PersistenceDomain.CONVERSATION]


async def _declare_session(state, conversation, *, split):
    now = datetime.now(timezone.utc)
    await state.conversation.sessions.create(
        SessionRecord("s", "owner", SessionStatus.OPEN, 0, None, {}, now, now, None, None, "agent", tenant_id=_SCOPE.tenant_id)
    )
    if not split:
        table = conversation.storage_group.metadata.tables["ai_state_records"]
        key = state.conversation.sessions.session_record_key("s")
        async with conversation.context.engine.begin() as conn:
            changed = await conn.execute(update(table).where(
                table.c.store_key == conversation.store_key, table.c.key == key,
            ).values(strict_turn_fence=0))
            assert changed.rowcount == 1


async def _claim(execution, conversation, *, split, execution_id, owner_id, token, **fields):
    values = {**_FIELDS, **fields, "execution_id": execution_id, "owner_id": owner_id, "token": token}
    if split:
        return await execution.mutate(lambda tx: SplitSessionTurnTransactions(tx, _SCOPE).claim(**values))
    return await execution.storage_group.mutate(
        (conversation, execution),
        lambda group: group.transaction(execution)._claim_strict_session(_SCOPE, session_store=conversation, **values),
    )


async def _read(execution, *, split, route, identity):
    return await execution.read(lambda tx: tx._read_native_execution_receipt(
        _SCOPE, route=route, identity=identity, split=split,
    ))


async def _confirm(execution, *, split, execution_id, receipt, kind="claim"):
    return await execution.read(lambda tx: tx._confirm_native_execution_receipt(
        _SCOPE, execution_id=execution_id, receipt=receipt, kind=kind, split=split,
    ))


async def _update_control(execution, identity, **values):
    table = execution.storage_group.metadata.tables["ai_execution_records"]
    key = record_key(_SCOPE.namespace, _SCOPE.tenant_id, "execution", "execution", (identity,))
    async with execution.context.engine.begin() as conn:
        changed = await conn.execute(update(table).where(table.c.store_key == execution.store_key, table.c.key == key).values(**values))
        assert changed.rowcount == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("split", (False, True))
async def test_native_strict_current_layouts_replay_and_fence_expired_takeover(tmp_path, split):
    path = tmp_path / "turns"
    async with _stores(path, split=split) as (state, first, first_conversation):
        await _declare_session(state, first_conversation, split=split)
        async with _stores(path, split=split) as (_other_state, other, other_conversation):
            first_claim = await _claim(first, first_conversation, split=split, execution_id="first",
                                       owner_id="1" * 32, token="2" * 32)
            assert first_claim.lease.fence == 1
            assert await _confirm(other, split=split, execution_id="first", receipt="2" * 32) == first_claim
            for route, identity in (("execution", "first"), ("request", _REQUEST), ("session", "session:s")):
                _, proof, observed = await _read(other, split=split, route=route, identity=identity)
                assert proof.execution_id == "first" and not observed.expired
            replay = await _claim(other, other_conversation, split=split, execution_id="replay",
                                  owner_id="3" * 32, token="4" * 32)
            assert replay.execution_id == "first" and replay.lease is None
            assert await _claim(other, other_conversation, split=split, execution_id="competing",
                                owner_id="3" * 32, token="4" * 32, request_key="c" * 64) is None

            cancelled = await first.mutate(lambda tx: tx._change_native_execution_receipt(
                _SCOPE, execution_id="first", receipt="5" * 32, kind="cancel", lease=None, split=split,
            ))
            assert cancelled is True
            assert await _confirm(other, split=split, execution_id="first", receipt="5" * 32, kind="cancel") is True

            await _update_control(first, "first", turn_expiry=datetime.now(timezone.utc) - timedelta(minutes=1))
            _, proof, observed = await _read(other, split=split, route="execution", identity="first")
            assert proof.cancel_requested and observed.expired
            successor = await _claim(other, other_conversation, split=split, execution_id="second",
                                     owner_id="3" * 32, token="4" * 32)
            assert successor.lease.fence == 2
            _, owner, observed = await _read(first, split=split, route="request", identity=_REQUEST)
            assert owner.execution_id == "second" and not observed.expired and not owner.cancel_requested
            old, retired, observed = await _read(first, split=split, route="execution", identity="first")
            assert retired.receipt_kind == "expire" and old["turn_request_route"] is None and observed is None
            assert await _confirm(first, split=split, execution_id="first", receipt="2" * 32) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("split", (False, True))
async def test_native_strict_broken_current_owner_is_not_read_as_idle(tmp_path, split):
    async with _stores(tmp_path / "broken", split=split) as (state, execution, conversation):
        await _declare_session(state, conversation, split=split)
        await _claim(execution, conversation, split=split, execution_id="current", owner_id="1" * 32, token="2" * 32)
        await _update_control(execution, "current", **(
            {"turn_occupied_route": None} if split else {"turn_session_owner_id": 100000}
        ))
        with pytest.raises(AIError) as caught:
            await _read(execution, split=split, route="execution", identity="current")
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("split", (False, True))
async def test_exact_published_identity_requires_recovery_instead_of_rewriting_its_body(tmp_path, split):
    async with _stores(tmp_path / "published", split=split) as (state, execution, conversation):
        await _declare_session(state, conversation, split=split)
        await _claim(execution, conversation, split=split, execution_id="current", owner_id="1" * 32, token="2" * 32)
        original = {"fixture": "unpublished output must not be replaced by an admission"}
        await _update_control(execution, "current", kind="execution", payload_json=original,
                              turn_expiry=datetime.now(timezone.utc) - timedelta(minutes=1))
        with pytest.raises(AIError) as caught:
            await _claim(execution, conversation, split=split, execution_id="current", owner_id="3" * 32, token="4" * 32)
        assert caught.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        control = execution.storage_group.metadata.tables["ai_execution_records"]
        async with execution.context.engine.begin() as connection:
            result = await connection.execute(select(control.c.payload_json).where(control.c.store_key == execution.store_key))
            assert result.scalar_one() == original
