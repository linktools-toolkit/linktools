#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable Trace locators must prove their exact native-backed projection."""

import hashlib
from dataclasses import replace

import pytest
from pydantic_ai_harness.step_persistence import RunRecord, StepEvent

from linktools.ai.core import canonical_sha256
from linktools.ai.errors import AIError
from linktools.ai.persistence import SqlStepStore
from linktools.ai.persistence._native_backup_projection import NativeBackupProjections
from linktools.ai.persistence._native_event_reference import NativeEventReferences
from linktools.ai.persistence._official_step_reader import OfficialStepReader
from linktools.ai.persistence._step_backup import NativeStepBackup
from linktools.ai.persistence._step_store import _replay_digest
from linktools.ai.persistence.codec import _encode_step_envelope, encode_envelope
from linktools.ai.persistence.state import StoredRecord, StoredFact, record_key, stream_digest


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _inventory():
    store = SqlStepStore.scope_digest(namespace="portable", tenant_id="tenant")
    run = RunRecord(run_id="run", registration_id="registration", metadata={"native": "body"})
    row = dict(SqlStepStore._record_values(run), store_digest=store)
    event = StepEvent(run_id="run", kind="run_started", step_index=1, idempotency_key="start")
    event_row = dict(SqlStepStore._event_values(event), store_digest=store,
                     key=_replay_digest(store, run.run_id, "start"), outcome=None, run_fields=0,
                     body_sha256=_digest(SqlStepStore._started_body(event)),
                     fields_sha256=_digest(SqlStepStore._started_fields_body(event)))
    return run, event, NativeStepBackup(store, (row,), (event_row,), ())


def test_portable_run_projection_restores_reference_and_rejects_registration_substitution() -> None:
    run, _, backup = _inventory()
    data = _encode_step_envelope(OfficialStepReader.project_run(run))
    reference = encode_envelope({"type": "native_agent_run_reference", "payload": {
        "run": run.run_id, "registration": run.registration_id,
        "projection_sha256": canonical_sha256(data), "metadata": {}, "removed": [], "started_at": None,
    }})
    projected = StoredRecord(record_key("portable", "tenant", "trace", "agent_run", (run.run_id,)),
        None, None, "agent_run", b'run', None, 1, None, 0, None, data)
    validator = NativeBackupProjections(backup, namespace="portable", tenant_id="tenant")
    physical = validator.validate(projected, reference)
    assert physical.data == reference
    assert replace(physical, data=projected.data) == projected
    substituted = replace(backup, runs=(dict(backup.runs[0], registration_id="other"),))
    with pytest.raises(AIError):
        NativeBackupProjections(substituted, namespace="portable", tenant_id="tenant").validate(projected, reference)


def test_portable_event_projection_requires_exact_source_and_trace_position() -> None:
    _, event, backup = _inventory()
    projection = NativeEventReferences.project(event, event_index=2)
    reference = encode_envelope({"type": "native_event_reference", "payload": {
        "key": backup.events[0]["key_digest"], "index": 2,
    }})
    projected = StoredFact(stream_digest("portable", "tenant", "trace", "event", "run"), 7,
        record_key("portable", "tenant", "trace", "agent_run", ("run",)),
        "step_event", None, "AGENT_RUN_STARTED", _encode_step_envelope(projection))
    validator = NativeBackupProjections(backup, namespace="portable", tenant_id="tenant")
    assert validator.validate(projected, reference).data == reference
    with pytest.raises(AIError):
        validator.validate(replace(projected, owner_key=b"x" * 32), reference)
    with pytest.raises(AIError):
        validator.validate(replace(projected, data=_encode_step_envelope(replace(projection, step_index=99))), reference)
