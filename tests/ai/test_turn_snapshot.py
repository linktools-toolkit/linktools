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


async def _claim(store, *, session, execution, request, token, coowned, owner=None):
    outcome = await store.claim_turn_admission(
        _SCOPE, session_key=session, execution_id=execution, request_key=request,
        request_digest="d" * 64, owner_id=owner or sha256(execution.encode()).hexdigest()[:32], duration=120, token=token, coowned=coowned,
    )
    assert outcome is not None and outcome.lease is not None
    return outcome.lease


@pytest.mark.asyncio
@pytest.mark.parametrize("coowned", (False, True))
async def test_sql_turn_snapshot_restores_proofs_but_not_live_authority(tmp_path, coowned):
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
        old_lease = ("session:session", "a" * 64, current.token, current.fence)
        renewed = await store.change_turn_admission(
            _SCOPE, execution_id="next", receipt="e" * 32, kind="renew", lease=old_lease, duration=120, coowned=coowned,
        )
        assert renewed.lease == current
        original = await store.get_turn_operation_proof(_SCOPE, "next", coowned=coowned)
        assert (original.receipt_kind, original.receipt_id) == ("renew", "e" * 32)
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
        captured = await store.capture_turn_admission(_SCOPE, route="request", identity="a" * 64, coowned=coowned)
        assert captured.expired and captured.proof == original
        for route, identity in (("request", "a" * 64), ("execution", "next"), ("session", "session:session")):
            assert await store.observe_turn_admission(_SCOPE, route=route, identity=identity, coowned=coowned) is None
        assert await store.confirm_turn_admission(_SCOPE, execution_id="next", receipt="e" * 32, kind="renew", coowned=coowned) is None
        with pytest.raises(AIError) as denied:
            await store.change_turn_admission(
                _SCOPE, execution_id="next", receipt="d" * 32, kind="renew", lease=old_lease, duration=120, coowned=coowned,
            )
        assert denied.value.code is ErrorCode.STORAGE_CONFLICT
        with pytest.raises(AIError) as denied:
            await store.retire_turn_admission(
                _SCOPE, session_key="session:session", execution_id="next", request_key="a" * 64,
                token=current.token, fence=current.fence, receipt="c" * 32, coowned=coowned,
            )
        assert denied.value.code is ErrorCode.STORAGE_CONFLICT
        assert await store.confirm_turn_admission(_SCOPE, execution_id="next", receipt="c" * 32, kind="release", coowned=coowned) is None
        assert await store.get_turn_operation_proof(_SCOPE, "next", coowned=coowned) == original
        takeover = await _claim(store, session="session:session", execution="next", request="a" * 64,
                                token="5" * 32, owner="6" * 32, coowned=coowned)
        assert takeover.fence == current.fence + 1
        assert await store.confirm_turn_admission(_SCOPE, execution_id="next", receipt="e" * 32, kind="renew", coowned=coowned) is None
        assert await store.confirm_turn_admission(_SCOPE, execution_id="first", receipt="a" * 32, kind="release", coowned=coowned) is True
        following = await _claim(store, session="session:idle", execution="following", request="c" * 64, token="4" * 32, coowned=coowned)
        assert following.fence == idle.fence + 1



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


@pytest.fixture
def standalone_snapshot_row():
    from datetime import datetime, timezone

    from linktools.ai.persistence._turn_sql import ACTIVE_REQUEST, STANDALONE
    from linktools.ai.persistence.state import record_key

    key = b"a" * 64
    token = b"b" * 32
    return {
        "session_kind": STANDALONE, "session_key": key, "execution_id": b"standalone",
        "route_kind": ACTIVE_REQUEST, "route_key": key, "retired_request_key": None,
        "standalone_idle": False, "control_id": 12, "request_digest": "c" * 64,
        "owner_id": b"d" * 32, "lease_fence": 3, "cancel_requested": False,
        "receipt_id": token, "receipt_kind": "claim", "lease_token": token,
        "lease_expires_at": datetime(2026, 11, 25, tzinfo=timezone.utc), "authority_revision": 4,
        "execution_key": record_key(_SCOPE.namespace, _SCOPE.tenant_id, "execution", "execution", ("standalone",)),
    }


@pytest.mark.parametrize("parked", (False, True))
def test_standalone_snapshot_anchors_preserve_only_authoritative_fields(standalone_snapshot_row, parked):
    from linktools.ai.persistence._turn_snapshot import turn_snapshot_record
    from linktools.ai.persistence._turn_sql_schema import TURN_ADMISSION_TABLE

    row = standalone_snapshot_row
    if parked:
        row.update(standalone_idle=True, control_id=None, receipt_kind="release", receipt_id=b"e" * 32,
                   lease_token=None, lease_expires_at=None, authority_revision=None, execution_key=None)
    identity = {"namespace": _SCOPE.namespace, "tenant_id": _SCOPE.tenant_id}
    record = turn_snapshot_record(TURN_ADMISSION_TABLE, row, **identity)
    assert record.data["standalone_idle"] is parked
    assert "control_id" not in record.data
    assert record.state == ("PARKED" if parked else "ACTIVE")
    slots, (admission,) = decode_turn_snapshot((record,), **identity)
    assert not slots
    for field in ("standalone_idle", "session_key", "execution_id", "route_kind", "route_key",
                  "lease_fence", "receipt_kind", "receipt_id"):
        assert admission[field] == row[field]
    if parked:
        assert record.lease_owner is None and record.lease_expires_at is None
        assert "_standalone_authority" not in admission
    else:
        assert admission["_standalone_authority"] == {
            "token": row["lease_token"], "expiry": row["lease_expires_at"], "revision": row["authority_revision"],
        }


def test_standalone_snapshot_rejects_wrong_control_identity_fence_and_receipt(standalone_snapshot_row):
    from linktools.ai.persistence._turn_snapshot import turn_snapshot_record
    from linktools.ai.persistence._turn_sql_schema import TURN_ADMISSION_TABLE

    identity = {"namespace": _SCOPE.namespace, "tenant_id": _SCOPE.tenant_id}
    row = standalone_snapshot_row
    record = turn_snapshot_record(TURN_ADMISSION_TABLE, row, **identity)
    for corrupt in (replace(record, lease_fence=record.lease_fence + 1),
                    replace(record, data={**record.data, "receipt_id": "e" * 32})):
        with pytest.raises(AIError) as caught:
            decode_turn_snapshot((corrupt,), **identity)
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    with pytest.raises(AIError) as caught:
        turn_snapshot_record(TURN_ADMISSION_TABLE, {**row, "execution_key": b"wrong"}, **identity)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_standalone_snapshot_legacy_admission_still_requires_its_slot(standalone_snapshot_row):
    from linktools.ai.persistence._turn_snapshot import turn_snapshot_record
    from linktools.ai.persistence._turn_sql_schema import TURN_ADMISSION_TABLE, TURN_SLOT_TABLE

    identity = {"namespace": _SCOPE.namespace, "tenant_id": _SCOPE.tenant_id}
    row = {**standalone_snapshot_row, "standalone_idle": None}
    admission = turn_snapshot_record(TURN_ADMISSION_TABLE, row, **identity)
    assert admission.data["coowned"] is True and "standalone_idle" not in admission.data
    assert admission.lease_owner is None
    slot = turn_snapshot_record(TURN_SLOT_TABLE, {
        **row, "active_execution_id": row["execution_id"], "revision": row["authority_revision"],
    }, **identity)
    (slot_row,), (admission_row,) = decode_turn_snapshot((slot, admission), **identity)
    assert slot_row["active_execution_id"] == row["execution_id"]
    assert admission_row["standalone_idle"] is None and admission_row["_coowned"] is True
    with pytest.raises(AIError) as caught:
        decode_turn_snapshot((admission,), **identity)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
