#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end evidence for Runtime persistence evolution invariants."""

from __future__ import annotations

import copy
from dataclasses import is_dataclass, replace
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.core import SessionStatus
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime.state import _codec as codec
from linktools.ai.runtime.state._contracts import ConversationCursor, SessionRecord
from scripts.build.persistence import validate_append_only


def _session(tenant_id: str = "tenant") -> SessionRecord:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return SessionRecord(
        session_id="session",
        tenant_id=tenant_id,
        owner_principal_id="owner",
        agent_digest="a" * 64,
        status=SessionStatus.OPEN,
        revision=0,
        resource_generation=0,
        cwd=None,
        metadata={},
        created_at=now,
        updated_at=now,
        closed_at=None,
        active_execution_id=None,
        continuation=None,
        history_quality="complete",
        history_id="history",
    )


async def _insert_legacy_session(state: RuntimeState, session: SessionRecord):
    repository = state.conversation.sessions
    stored = repository._stored(
        "session", "session", session, state=session.status.value
    )
    legacy_data = copy.deepcopy(stored.data)
    legacy_data["value"]["payload"].pop("schema")
    legacy = replace(stored, data=legacy_data)
    await repository.state_store.mutate(lambda tx: tx.insert_record(legacy))
    raw = await repository.state_store.read(
        lambda tx: tx.get_record(legacy.key_digest)
    )
    assert raw is not None
    return legacy.key_digest, raw


async def _assert_legacy_read_is_pure(
    state: RuntimeState,
    session: SessionRecord,
    key: bytes,
    raw_before: object,
) -> None:
    loaded = await state.conversation.sessions.get(
        session.session_id,
        tenant_id=session.tenant_id,
    )
    assert loaded == session
    raw_after = await state.conversation.sessions.state_store.read(
        lambda tx: tx.get_record(key)
    )
    assert raw_after is not None
    assert raw_after.storage_version == raw_before.storage_version
    assert raw_after.data == raw_before.data
    assert "schema" not in raw_after.data["value"]["payload"]


def test_full_schema_upgrade_chain_decodes_old_revision_and_rewrites_latest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_fingerprint = codec._V1_SCHEMA_FINGERPRINTS["conversation_cursor"]

    def upgrade_v1_to_v2(raw: object, _codec: object) -> dict[str, object]:
        assert isinstance(raw, dict)
        assert set(raw) == {"step_run_id", "legacy_history_id", "message_count"}
        return {
            "step_run_id": raw["step_run_id"],
            "history_id": raw["legacy_history_id"],
            "message_count": raw["message_count"],
        }

    contracts = dict(codec._V1_DATACLASS_PERSISTENCE)
    contracts["conversation_cursor"] = codec._DataclassPersistenceContract(
        fingerprints={1: "0" * 64, 2: current_fingerprint},
        upgrades={1: upgrade_v1_to_v2},
        legacy_revision=1,
    )
    monkeypatch.setattr(codec, "_DATACLASS_PERSISTENCE_BY_VERSION", {1: contracts})

    legacy = {
        "v": 1,
        "value": {
            "type": "conversation_cursor",
            "payload": {
                "$dataclass": "conversation_cursor",
                "schema": 1,
                "fields": {
                    "step_run_id": "run",
                    "legacy_history_id": "history",
                    "message_count": 7,
                },
            },
        },
    }
    decoded = codec._decode_enveloped_domain(legacy, ConversationCursor)
    assert decoded == ConversationCursor("run", "history", 7)

    rewritten = codec.encode_envelope(
        {
            "type": codec.wire_type_id(decoded),
            "payload": codec._encode_persisted_domain(decoded),
        }
    )
    payload = rewritten["value"]["payload"]
    assert payload["schema"] == 2
    assert payload["fields"] == {
        "step_run_id": "run",
        "history_id": "history",
        "message_count": 7,
    }


@pytest.mark.asyncio
async def test_filesystem_legacy_read_is_pure_and_survives_reopen(tmp_path) -> None:
    root = tmp_path / "runtime"
    namespace = "persistence-no-read-repair-fs"
    session = _session()

    state = RuntimeState.filesystem(root)
    await state.initialize(namespace=namespace, tenant_id=session.tenant_id)
    key, raw_before = await _insert_legacy_session(state, session)
    await state.close()

    reopened = RuntimeState.filesystem(root)
    await reopened.initialize(namespace=namespace, tenant_id=session.tenant_id)
    try:
        await _assert_legacy_read_is_pure(
            reopened, session, key, raw_before
        )
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_sqlite_legacy_read_is_pure_and_survives_reopen(tmp_path) -> None:
    path = tmp_path / "runtime.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    await provision_runtime_database(engine)
    await engine.dispose()

    namespace = "persistence-no-read-repair-sqlite"
    session = _session()
    state = RuntimeState.sqlite(path)
    await state.initialize(namespace=namespace, tenant_id=session.tenant_id)
    key, raw_before = await _insert_legacy_session(state, session)
    await state.close()

    reopened = RuntimeState.sqlite(path)
    await reopened.initialize(namespace=namespace, tenant_id=session.tenant_id)
    try:
        await _assert_legacy_read_is_pure(
            reopened, session, key, raw_before
        )
    finally:
        await reopened.close()


def test_generic_dataclass_source_shape_matches_current_schema_revision() -> None:
    custom = set(codec._V1_DATACLASS_ENCODERS)
    for wire_id, target in codec._V1_WIRE_TYPES:
        if wire_id in custom or not is_dataclass(target):
            continue
        contract = codec._V1_DATACLASS_PERSISTENCE[wire_id]
        assert codec._dataclass_schema_fingerprint(target, codec._V1_CODEC) == (
            contract.fingerprints[contract.current_revision]
        )


def test_append_only_gate_rejects_historical_contract_mutation() -> None:
    baseline = codec._runtime_persistence_manifest()

    changed_revision = copy.deepcopy(baseline)
    changed_revision["dataclasses"]["conversation_cursor"]["revisions"]["1"] = (
        "f" * 64
    )
    assert "historical revision changed: conversation_cursor@1" in validate_append_only(
        baseline, changed_revision
    )

    removed_enum = copy.deepcopy(baseline)
    removed_enum["enums"]["session_status"].remove("OPEN")
    assert "historical enum values removed: session_status" in validate_append_only(
        baseline, removed_enum
    )

    changed_external = copy.deepcopy(baseline)
    changed_external["external"]["agent_binding_snapshot"]["version"] = 2
    assert "historical external contract changed: agent_binding_snapshot" in (
        validate_append_only(baseline, changed_external)
    )

    invented_legacy = copy.deepcopy(baseline)
    invented_legacy["dataclasses"]["future_record"] = {
        "legacy_revision": 1,
        "revisions": {"1": "e" * 64},
    }
    assert "new dataclass cannot claim unversioned legacy: future_record" in (
        validate_append_only(baseline, invented_legacy)
    )
