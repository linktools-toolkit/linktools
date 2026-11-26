#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable SQL snapshots restore fencing evidence, not active producers."""

from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from linktools.ai.core import PersistenceDomain
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistencePlan, PersistenceStorage
from linktools.ai.persistence._execution_authority import ExecutionAuthority
from linktools.ai.persistence._turn_snapshot import (
    decode_turn_snapshot, execution_receipt_snapshot_record,
)
from linktools.ai.persistence._turn_sql import SqlTurnProof
from linktools.ai.persistence._turn_sql_schema import unoccupied_turn_route
from linktools.ai.persistence.contracts import SessionRecord, SessionStatus
from linktools.ai.persistence.snapshots import restore_snapshot
from linktools.ai.snapshot import SnapshotLimits
from linktools.ai.storage import FilesystemObjectStore, InMemoryObjectStore, StorageRoute, TurnScopeIdentity


_SCOPE = TurnScopeIdentity("turn-snapshot", "tenant", "runtime-pending-turn")
_LIMITS = SnapshotLimits(max_entries=1000, max_bytes=1_000_000)
_IDENTITY = {"namespace": _SCOPE.namespace, "tenant_id": _SCOPE.tenant_id}
_DEADLINE = datetime(2035, 1, 1, tzinfo=timezone.utc)
_EXPIRED = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _storage(root, *, split):
    if not split:
        return PersistenceStorage.from_root(root)
    execution = StorageRoute.sqlite(root / "execution.db")
    conversation = StorageRoute.sqlite(root / "conversation.db")
    return PersistenceStorage(replace(PersistencePlan.from_default(execution), conversation=conversation),
                              object_store=FilesystemObjectStore(root / "objects"))


def _proof(execution, slot, request, fence, *, active):
    receipt = str(fence) * 32
    return SqlTurnProof(execution, slot, request, "d" * 64, "e" * 32, fence, False,
                        receipt, "claim" if active else "release", active)


def _receipt(proof, *, session):
    result = {name: getattr(proof, name) for name in (
        "owner_id", "request_digest", "receipt_id", "receipt_kind", "cancel_requested",
    )}
    return {**result, "request_key": proof.request_key} if session else result


async def _session(state, identity):
    now = datetime.now(timezone.utc)
    await state.conversation.sessions.create(
        SessionRecord(identity, "owner", SessionStatus.OPEN, 0, None, {}, now, now, None, None, "agent")
    )


async def _add_receipt(state, proof, *, parked=False, split=False):
    store = state.snapshot_resources().stores[PersistenceDomain.EXECUTION]
    control = store.storage_group.metadata.tables["ai_execution_records"]
    authority = ExecutionAuthority(_SCOPE, proof.execution_id, proof.session_key, proof.receipt_id, proof.fence)
    values = authority.shell_values(store_key=store.store_key, expiry=_DEADLINE)
    session = proof.session_key.startswith("session:")
    request = proof.request_key.encode() if proof.active or (parked and not session) else values["turn_request_route"]
    async with store.context.engine.begin() as conn:
        await conn.execute(control.insert().values({
            **values, "turn_token": values["turn_token"] if proof.active else None,
            "turn_revision": proof.fence + 1, "turn_receipt": _receipt(proof, session=session),
            "turn_request_route": request,
            "turn_occupied_route": (values["turn_slot_key"] if split and (proof.active or parked)
                                    else values["turn_occupied_route"]),
        }))


@pytest.mark.asyncio
@pytest.mark.parametrize("split", (False, True))
async def test_native_turn_snapshot_restores_expired_authority_and_preserves_fencing(tmp_path, split):
    archive = InMemoryObjectStore()
    source = _storage(tmp_path / "source", split=split)
    async with Persistence.open(_SCOPE.namespace, storage=source, tenant_id=_SCOPE.tenant_id) as state:
        if split:
            await _session(state, "active")
            await _session(state, "idle")
            first = _proof("older-session", "session:active", "a" * 64, 1, active=False)
            live = _proof("session-live", "session:active", "a" * 64, 3, active=True)
            parked = _proof("session-parked", "session:idle", "b" * 64, 2, active=False)
            for proof in (first, live, parked):
                await _add_receipt(state, proof, parked=proof is parked, split=True)
        for proof, parked_value in (
            (_proof("standalone-old", "request:" + "c" * 64, "c" * 64, 1, active=False), False),
            (_proof("standalone-parked", "request:" + "c" * 64, "c" * 64, 2, active=False), True),
            (_proof("standalone-live", "request:" + "d" * 64, "d" * 64, 1, active=True), False),
        ):
            await _add_receipt(state, proof, parked=parked_value)
    async with Persistence.open(_SCOPE.namespace, storage=source, tenant_id=_SCOPE.tenant_id, read_only=True) as state:
        reference = await state.export_snapshot(object_store=archive, limits=_LIMITS)

    @asynccontextmanager
    async def open_target(path, namespace, tenant_id):
        async with Persistence.open(namespace, storage=_storage(path, split=split), tenant_id=tenant_id) as state:
            yield state.snapshot_resources()

    target = tmp_path / "restored"
    await restore_snapshot(reference, object_store=archive, root=target, limits=_LIMITS, open_target=open_target)
    async with Persistence.open(_SCOPE.namespace, storage=_storage(target, split=split), tenant_id=_SCOPE.tenant_id) as state:
        resources = state.snapshot_resources()
        store = resources.stores[PersistenceDomain.EXECUTION]
        control = store.storage_group.metadata.tables["ai_execution_records"]
        async with store.context.engine.begin() as conn:
            physical = (await conn.execute(select(control).where(control.c.store_key == store.store_key))).mappings()
            rows = {}
            for row in physical:
                for name in ("older-session", "session-live", "session-parked", "standalone-old", "standalone-parked", "standalone-live"):
                    if row["key"] == ExecutionAuthority(_SCOPE, name, "session:active", "1" * 32, 1).key:
                        rows[name] = row
            assert len(rows) == (6 if split else 3)
            for name, row in rows.items():
                if name.endswith("-live"):
                    assert row["turn_token"] and row["turn_request_route"]
                    assert store.context.dialect.decode_database_time(row["turn_expiry"]) == _EXPIRED
                else:
                    assert row["turn_token"] is None
                    assert row["turn_receipt"]["receipt_kind"] == "release"
                    assert store.context.dialect.decode_database_time(row["turn_expiry"]) == _DEADLINE
            unoccupied = {name: unoccupied_turn_route(row["key_locator"]) for name, row in rows.items()}
            assert len(set(unoccupied.values())) == len(rows)
            assert rows["standalone-old"]["turn_request_route"] == unoccupied["standalone-old"]
            assert rows["standalone-parked"]["turn_request_route"] == b"c" * 64
            if split:
                assert rows["session-live"]["turn_fence"] == 3 and rows["session-live"]["turn_revision"] == 4
                assert rows["session-parked"]["turn_fence"] == 2
                assert rows["session-parked"]["turn_request_route"] == unoccupied["session-parked"]
                assert rows["older-session"]["turn_occupied_route"] == unoccupied["older-session"]
                assert rows["session-parked"]["turn_occupied_route"] == b"idle"
                assert rows["session-live"]["turn_occupied_route"] == b"active"


def test_native_snapshot_rejects_invalid_split_receipt_and_route():
    proof = _proof("native", "session:example", "a" * 64, 3, active=True)
    active = execution_receipt_snapshot_record({
        "turn_request_route": b"a" * 64, "turn_occupied_route": b"example",
        "turn_token": proof.receipt_id.encode(), "turn_revision": 1,
        "turn_expiry": _DEADLINE, "kind": "execution_shell", "turn_receipt": _receipt(proof, session=True),
    }, proof, split_session=True, **_IDENTITY)
    [receipt] = decode_turn_snapshot((active,), **_IDENTITY)
    assert receipt["turn_fence"] == 3 and receipt["turn_occupied_route"] == b"example"
    for record in (
        replace(active, data={key: value for key, value in active.data.items() if key != "split_session"}),
        replace(active, data={**active.data, "receipt_id": "f" * 32}),
        replace(active, key=b"wrong"),
    ):
        with pytest.raises(AIError) as caught:
            decode_turn_snapshot((record,), **_IDENTITY)
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
