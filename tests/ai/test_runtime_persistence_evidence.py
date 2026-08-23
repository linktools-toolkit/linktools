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
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime.state import _codec as codec
from linktools.ai.runtime.state._contracts import ConversationCursor, SessionRecord
from scripts.build.persistence import (
    validate_append_only,
    validate_fixture_append_only,
    validate_upgrade_fixtures,
)


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
        expected = {"step_run_id", "legacy_history_id", "message_count"}
        if not isinstance(raw, dict) or set(raw) != expected:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
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

    legacy_fields = {
        "step_run_id": "run",
        "legacy_history_id": "history",
        "message_count": 7,
    }
    current_fields = {
        "step_run_id": "run",
        "history_id": "history",
        "message_count": 7,
    }
    legacy = {
        "v": 1,
        "value": {
            "type": "conversation_cursor",
            "payload": {
                "$dataclass": "conversation_cursor",
                "schema": 1,
                "fields": legacy_fields,
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
    assert payload["fields"] == current_fields

    manifest = codec._runtime_persistence_manifest()
    assert validate_upgrade_fixtures(manifest, {}) == (
        "missing upgrade fixture: conversation_cursor@1",
    )
    fixtures = {
        "conversation_cursor@1": {
            "fields": legacy_fields,
            "current_fields": current_fields,
        }
    }
    assert validate_upgrade_fixtures(manifest, fixtures) == ()
    changed = copy.deepcopy(fixtures)
    changed["conversation_cursor@1"]["current_fields"]["message_count"] = 8
    assert validate_fixture_append_only(fixtures, changed) == (
        "historical upgrade fixture changed: conversation_cursor@1",
    )


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

    advanced_external = copy.deepcopy(baseline)
    advanced_external["external"]["agent_binding_snapshot"]["version"] = 2
    assert validate_append_only(baseline, advanced_external) == ()

    changed_owner = copy.deepcopy(baseline)
    changed_owner["external"]["agent_binding_snapshot"]["owner"] = "runtime"
    assert "agent_binding_snapshot owner changed" in validate_append_only(
        baseline, changed_owner
    )

    newer_baseline = copy.deepcopy(baseline)
    newer_baseline["external"]["agent_binding_snapshot"]["version"] = 2
    regressed_external = copy.deepcopy(baseline)
    assert "agent_binding_snapshot version regressed" in validate_append_only(
        newer_baseline, regressed_external
    )

    invented_legacy = copy.deepcopy(baseline)
    invented_legacy["dataclasses"]["future_record"] = {
        "legacy_revision": 1,
        "revisions": {"1": "e" * 64},
    }
    assert "new dataclass cannot claim unversioned legacy: future_record" in (
        validate_append_only(baseline, invented_legacy)
    )


def test_append_only_fixture_allows_new_revision_but_not_rewrite() -> None:
    baseline = {"task_node@1": {"schema": 1}}
    extended = {
        "task_node@1": {"schema": 1},
        "task_node@2": {"schema": 2},
    }
    assert validate_fixture_append_only(
        baseline,
        extended,
        label="custom wire fixture",
    ) == ()

    rewritten = copy.deepcopy(extended)
    rewritten["task_node@1"] = {"schema": 2}
    assert validate_fixture_append_only(
        baseline,
        rewritten,
        label="custom wire fixture",
    ) == ("historical custom wire fixture changed: task_node@1",)
