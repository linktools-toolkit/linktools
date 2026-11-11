#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable run owners are stored once and references require exact registrations."""

from dataclasses import replace

import pytest

from linktools.ai.errors import AIError
from linktools.ai.persistence._official_step_reader import OfficialStepReader
from linktools.ai.persistence._run_registration import RunRegistrationReference
from linktools.ai.persistence.snapshots import (
    _decode_checkpoint_registration_refs, _decode_run_registrations,
    _encode_checkpoint_registration_refs, _expand_run_reference, _run_json,
)
from linktools.ai.persistence.state import StoredRecord, record_key

from .test_native_backup_projection import _inventory


def test_native_backup_reuses_single_canonical_registration_body() -> None:
    run, _, backup = _inventory()
    owners = {run.run_id: run}
    wire = _encode_checkpoint_registration_refs(backup, owners)
    assert "runs" not in wire
    assert wire["run_refs"] == [{"run_id": run.run_id, "managed_start_key": "c3RhcnQ="}]
    assert _decode_checkpoint_registration_refs(wire, owners) == backup
    with pytest.raises(AIError):
        _decode_checkpoint_registration_refs(wire, {})
    with pytest.raises(AIError):
        _decode_checkpoint_registration_refs(dict(wire, run_refs=wire["run_refs"] * 2), owners)
    with pytest.raises(AIError):
        _encode_checkpoint_registration_refs(backup, {run.run_id: replace(run, registration_id="other")})


def test_run_inventory_rejects_duplicate_identity_and_preserves_metadata() -> None:
    run, _, _ = _inventory()
    wire = _run_json(run)
    assert _decode_run_registrations([wire]) == {run.run_id: run}
    with pytest.raises(AIError):
        _decode_run_registrations([wire, wire])


def test_domain_deltas_share_registration_and_fail_closed_without_owner() -> None:
    run, _, _ = _inventory()
    projection = replace(OfficialStepReader.project_run(run), metadata={"local": "delta"})
    reference = RunRegistrationReference.capture(run, projection).to_data()
    record = StoredRecord(record_key("portable", "tenant", "trace", "agent_run", (run.run_id,)),
                          None, None, "agent_run", b"run", None, 1, None, 0, None, reference)
    from linktools.ai.persistence.codec import _decode_step_envelope
    assert _decode_step_envelope(_expand_run_reference(record, {run.run_id: run}).data) == projection
    with pytest.raises(AIError):
        _expand_run_reference(record, {})
    with pytest.raises(AIError):
        _expand_run_reference(record, {run.run_id: replace(run, registration_id="other")})


@pytest.mark.asyncio
@pytest.mark.parametrize("matching", (True, False))
async def test_native_restore_reuses_only_matching_published_registration(tmp_path, matching: bool) -> None:
    from sqlalchemy.ext.asyncio import create_async_engine
    from linktools.ai.persistence import SqlStepStore
    from .test_native_step_backup import _Reader

    run, _, backup = _inventory()
    # No history bodies are required to exercise the registration ownership gate.
    backup = replace(backup, events=(), runs=(dict(backup.runs[0], managed_start_key=None),))
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'registrations.db'}")
    store = SqlStepStore(engine, namespace="portable", tenant_id="tenant")
    await store.initialize(local=True)
    try:
        await store.register_run(run if matching else replace(run, registration_id="other"))
        if matching:
            async with store.storage_context.sessions() as session, session.begin():
                await store.restore_backup(session, backup)
                restored = await store.capture_backup(_Reader(session))
            assert restored == backup
        else:
            with pytest.raises(ValueError, match="registration differs"):
                async with store.storage_context.sessions() as session, session.begin():
                    await store.restore_backup(session, backup)
        assert (await store.get_run(run_id=run.run_id)).registration_id == (run.registration_id if matching else "other")
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.parametrize("key", (None, b"", b"start"))
def test_portable_run_ref_preserves_start_identity_and_checks_its_event(key) -> None:
    from linktools.ai.persistence import SqlStepStore
    from linktools.ai.persistence._step_backup import NativeStepBackup
    from linktools.ai.steps import encode_native_replay_key

    run, event, backup = _inventory()
    if key is None:
        backup = replace(backup, runs=(dict(backup.runs[0], managed_start_key=None),), events=())
    elif key == b"":
        event = replace(event, idempotency_key="")
        import hashlib
        raw = dict(backup.events[0], idempotency_key=encode_native_replay_key(""),
                   body_sha256=hashlib.sha256(SqlStepStore._started_body(event).encode()).hexdigest(),
                   fields_sha256=hashlib.sha256(SqlStepStore._started_fields_body(event).encode()).hexdigest())
        backup = replace(backup, runs=(dict(backup.runs[0], managed_start_key=b""),), events=(raw,))
    owners = {run.run_id: run}
    restored = _decode_checkpoint_registration_refs(_encode_checkpoint_registration_refs(backup, owners), owners)
    assert restored == backup
    assert NativeStepBackup.from_json(backup.to_json()) == backup
    SqlStepStore.validate_backup(restored)
    if key is not None:
        with pytest.raises(ValueError, match="missing.*managed start"):
            SqlStepStore.validate_backup(replace(restored, events=()))
