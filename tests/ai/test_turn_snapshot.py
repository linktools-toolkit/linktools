#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable snapshots preserve source coordination proofs, not active producers."""

from dataclasses import replace
from hashlib import sha256

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._turn_snapshot import decode_turn_snapshot
from linktools.ai.snapshot import SnapshotLimits
from linktools.ai.storage import InMemoryObjectStore, TurnScopeIdentity


_SCOPE = TurnScopeIdentity("turn-snapshot", "tenant", "runtime-pending-turn")
_LIMITS = SnapshotLimits(max_entries=1000, max_bytes=1_000_000)


async def _claim(store, *, session, execution, request, token, coowned):
    outcome = await store.claim_turn_admission(
        _SCOPE, session_key=session, execution_id=execution, request_key=request,
        request_digest="d" * 64, owner_id=sha256(execution.encode()).hexdigest()[:32], duration=120, token=token, coowned=coowned,
    )
    assert outcome is not None and outcome.lease is not None
    return outcome.lease


@pytest.mark.asyncio
@pytest.mark.parametrize("coowned", (False, True))
async def test_sql_turn_snapshot_restores_live_admission_and_retired_release(tmp_path, coowned):
    archive = InMemoryObjectStore()
    source = PersistenceStorage.from_root(tmp_path / "source")
    async with Persistence.open(_SCOPE.namespace, storage=source, tenant_id=_SCOPE.tenant_id) as state:
        store = state.execution.executions.state_store
        store.bind_runtime_turn_scope(_SCOPE)
        first = await _claim(store, session="session:session", execution="first", request="a" * 64, token="1" * 32, coowned=coowned)
        assert await store.retire_turn_admission(
            _SCOPE, session_key="session:session", execution_id="first", request_key="a" * 64,
            token=first.token, fence=first.fence, receipt="a" * 32, coowned=coowned,
        )
        idle = await _claim(store, session="session:idle", execution="idle-execution", request="b" * 64, token="2" * 32, coowned=coowned)
        assert await store.retire_turn_admission(
            _SCOPE, session_key="session:idle", execution_id="idle-execution", request_key="b" * 64,
            token=idle.token, fence=idle.fence, receipt="b" * 32, coowned=coowned,
        )
        current = await _claim(store, session="session:session", execution="next", request="a" * 64, token="3" * 32, coowned=coowned)
    async with Persistence.open(_SCOPE.namespace, storage=source, tenant_id=_SCOPE.tenant_id, read_only=True) as state:
        reference = await state.export_snapshot(object_store=archive, limits=_LIMITS)

    destination = tmp_path / "restored"
    await Persistence.restore_snapshot(reference, object_store=archive, root=destination, limits=_LIMITS)
    async with Persistence.open(_SCOPE.namespace, storage=PersistenceStorage.from_root(destination),
                                tenant_id=_SCOPE.tenant_id) as state:
        store = state.execution.executions.state_store
        store.bind_runtime_turn_scope(_SCOPE)
        assert await store.confirm_turn_admission(_SCOPE, execution_id="first", receipt="a" * 32, kind="release", coowned=coowned) is True
        assert await store.confirm_turn_admission(_SCOPE, execution_id="first", receipt="d" * 32, kind="release", coowned=coowned) is None
        proof = await store.observe_turn_admission(_SCOPE, route="request", identity="a" * 64, coowned=coowned)
        assert (proof.execution_id, proof.fence) == ("next", current.fence)
        assert proof.coordination(_SCOPE).lease is None
        following = await _claim(store, session="session:idle", execution="following", request="c" * 64, token="4" * 32, coowned=coowned)
        assert following.fence == idle.fence + 1
        assert not await store.retire_turn_admission(
            _SCOPE, session_key="session:session", execution_id="next", request_key="a" * 64,
            token=first.token, fence=first.fence, receipt="d" * 32, coowned=coowned,
        )
        assert await store.retire_turn_admission(
            _SCOPE, session_key="session:session", execution_id="next", request_key="a" * 64,
            token=current.token, fence=current.fence, receipt="c" * 32, coowned=coowned,
        )


def test_sql_turn_snapshot_rejects_missing_active_owner():
    from datetime import datetime, timezone
    from linktools.ai.persistence._turn_snapshot import turn_snapshot_record
    from linktools.ai.persistence._turn_sql_schema import TURN_SLOT_TABLE

    row = {
        "session_kind": 1,
        "session_key": b"session", "active_execution_id": b"execution", "lease_token": b"a" * 32,
        "lease_fence": 1, "revision": 1, "lease_expires_at": datetime.now(timezone.utc),
    }
    record = turn_snapshot_record(TURN_SLOT_TABLE, row, namespace=_SCOPE.namespace, tenant_id=_SCOPE.tenant_id)
    for value in (record, replace(record, key=b"wrong")):
        with pytest.raises(AIError) as exc:
            decode_turn_snapshot((value,), namespace=_SCOPE.namespace, tenant_id=_SCOPE.tenant_id, store_key=b"state")
        assert exc.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
