#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable snapshots preserve source coordination proofs, not active producers."""

from dataclasses import replace

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._turn_snapshot import decode_turn_snapshot
from linktools.ai.snapshot import SnapshotLimits
from linktools.ai.storage import InMemoryObjectStore, TurnScopeIdentity


_SCOPE = TurnScopeIdentity("turn-snapshot", "tenant", "purpose")
_LIMITS = SnapshotLimits(max_entries=1000, max_bytes=1_000_000)


async def _claim(store, *, session, execution, request, token):
    outcome = await store.claim_turn_admission(
        _SCOPE, session_key=session, execution_id=execution, request_key=request,
        request_digest="input", owner_id="owner-" + execution, duration=120, token=token,
    )
    assert outcome is not None and outcome.lease is not None
    return outcome.lease


@pytest.mark.asyncio
async def test_sql_turn_snapshot_restores_live_admission_and_retired_release(tmp_path):
    archive = InMemoryObjectStore()
    source = PersistenceStorage.from_root(tmp_path / "source")
    async with Persistence.open(_SCOPE.namespace, storage=source, tenant_id=_SCOPE.tenant_id) as state:
        store = state.execution.executions.state_store
        first = await _claim(store, session="session", execution="first", request="request", token="first")
        assert await store.retire_turn_admission(
            _SCOPE, session_key="session", execution_id="first", request_key="request",
            token=first.token, fence=first.fence, receipt="exact-release",
        )
        idle = await _claim(store, session="idle", execution="idle-execution", request="idle", token="idle")
        assert await store.retire_turn_admission(
            _SCOPE, session_key="idle", execution_id="idle-execution", request_key="idle",
            token=idle.token, fence=idle.fence, receipt="idle-release",
        )
        current = await _claim(store, session="session", execution="next", request="request", token="next")
    async with Persistence.open(_SCOPE.namespace, storage=source, tenant_id=_SCOPE.tenant_id, read_only=True) as state:
        reference = await state.export_snapshot(object_store=archive, limits=_LIMITS)

    destination = tmp_path / "restored"
    await Persistence.restore_snapshot(reference, object_store=archive, root=destination, limits=_LIMITS)
    async with Persistence.open(_SCOPE.namespace, storage=PersistenceStorage.from_root(destination),
                                tenant_id=_SCOPE.tenant_id) as state:
        store = state.execution.executions.state_store
        assert await store.confirm_turn_admission(_SCOPE, execution_id="first", receipt="exact-release", kind="release") is True
        assert await store.confirm_turn_admission(_SCOPE, execution_id="first", receipt="wrong", kind="release") is None
        proof = await store.observe_turn_admission(_SCOPE, route="request", identity="request")
        assert (proof.execution_id, proof.fence) == ("next", current.fence)
        assert proof.coordination(_SCOPE).lease is None
        following = await _claim(store, session="idle", execution="following", request="other", token="following")
        assert following.fence == idle.fence + 1
        assert not await store.retire_turn_admission(
            _SCOPE, session_key="session", execution_id="next", request_key="request",
            token=first.token, fence=first.fence, receipt="wrong",
        )
        assert await store.retire_turn_admission(
            _SCOPE, session_key="session", execution_id="next", request_key="request",
            token=current.token, fence=current.fence, receipt="next-release",
        )


def test_sql_turn_snapshot_rejects_missing_active_owner():
    from datetime import datetime, timezone
    from linktools.ai.persistence._record_key import encode_key_components
    from linktools.ai.persistence._turn_snapshot import turn_snapshot_record
    from linktools.ai.persistence._turn_sql_schema import TURN_SLOT_TABLE, TURN_SCOPE_MAX_BYTES

    row = {
        "scope_key": encode_key_components(_SCOPE.as_tuple(), TURN_SCOPE_MAX_BYTES),
        "session_key": b"session", "active_execution_id": b"execution", "lease_token": b"token",
        "lease_fence": 1, "revision": 1, "lease_expires_at": datetime.now(timezone.utc),
    }
    record = turn_snapshot_record(TURN_SLOT_TABLE, row, namespace=_SCOPE.namespace, tenant_id=_SCOPE.tenant_id)
    for value in (record, replace(record, key=b"wrong")):
        with pytest.raises(AIError) as exc:
            decode_turn_snapshot((value,), namespace=_SCOPE.namespace, tenant_id=_SCOPE.tenant_id, store_key=b"state")
        assert exc.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
