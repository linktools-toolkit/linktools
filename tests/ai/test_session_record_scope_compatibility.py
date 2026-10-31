#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Committed SessionRecord v1 scope-memory compatibility stays narrowly bounded."""

import copy
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from linktools.ai.core import SessionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence
from linktools.ai.persistence.codec import (
    _decode_enveloped_domain,
    _encode_persisted_domain,
    decode_domain,
)
from linktools.ai.session import SessionRecord

from .test_incremental_projection_boundaries import _storage


def _fixture() -> dict[str, object]:
    path = Path(__file__).with_name("fixtures") / "persistence" / "session_record_without_scope_memory_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _session() -> SessionRecord:
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    return SessionRecord(
        session_id="synthetic-session", owner_principal_id="synthetic-owner",
        status=SessionStatus.OPEN, revision=1, cwd="src",
        metadata={"purpose": "compatibility-fixture"}, created_at=now, updated_at=now,
        closed_at=None, active_execution_id=None, agent_id="synthetic-agent",
        history_id="synthetic-history",
    )


def test_session_without_scope_memory_reads_and_reencodes_current_shape() -> None:
    wire = _fixture()
    before = copy.deepcopy(wire)
    record = _decode_enveloped_domain(wire, SessionRecord)
    assert record == _session()
    assert record.repository_instruction_scopes == ()
    current = _encode_persisted_domain(record)
    assert current["fields"] == {
        **wire["value"]["payload"]["fields"],
        "repository_instruction_scopes": {"$tuple": []},
    }
    assert _decode_enveloped_domain({"v": 1, "value": {
        "type": "session_record", "payload": current,
    }}, SessionRecord) == record
    assert wire == before


def test_current_session_preserves_explicit_nonempty_scope_memory() -> None:
    record = replace(_session(), repository_instruction_scopes=("src", "src/component"))
    wire = {"v": 1, "value": {"type": "session_record", "payload": _encode_persisted_domain(record)}}
    assert _decode_enveloped_domain(wire, SessionRecord) == record


def test_scope_compatibility_does_not_fill_other_missing_session_fields() -> None:
    wire = _fixture()
    del wire["value"]["payload"]["fields"]["agent_id"]
    with pytest.raises(AIError) as caught:
        _decode_enveloped_domain(wire, SessionRecord)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert caught.value.safe_details == {"wire_type": "session_record", "missing_fields": ["agent_id"]}


def test_canonical_session_decoding_uses_the_same_declared_default() -> None:
    payload = _fixture()["value"]["payload"]
    del payload["schema"]
    assert decode_domain(payload, SessionRecord) == _session()


@pytest.mark.parametrize("scopes", (
    None,
    [],
    {"$tuple": [1]},
    {"$tuple": ["../outside"]},
    {"$tuple": ["src", "src"]},
    {"$tuple": ["z", "a"]},
))
def test_scope_compatibility_rejects_malformed_explicit_scope_memory(scopes: object) -> None:
    wire = _fixture()
    wire["value"]["payload"]["fields"]["repository_instruction_scopes"] = scopes
    with pytest.raises(AIError) as caught:
        _decode_enveloped_domain(wire, SessionRecord)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_session_repository_reads_scope_compatible_record_without_rewriting(
    tmp_path: Path, backend: str,
) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="session-scope-compatibility", tenant_id="tenant")
    try:
        repository = state.conversation.sessions
        await repository.create(_session())
        key = repository.session_record_key("synthetic-session")

        async def seed(transaction):
            current = await transaction.get_record(key)
            assert current is not None
            assert await transaction.replace_record(
                replace(current, data=_fixture(), storage_revision=current.storage_revision + 1),
                expected_storage_revision=current.storage_revision,
            )

        await repository.state_store.mutate(seed)
        before = await repository.state_store.read(lambda transaction: transaction.get_record(key))
        record = await repository.get("synthetic-session", tenant_id="tenant")
        assert record == _session()
        after = await repository.state_store.read(lambda transaction: transaction.get_record(key))
        assert after == before
        updated = replace(record, revision=record.revision + 1, repository_instruction_scopes=("src",))
        assert await repository.compare_and_swap(
            record.session_id, tenant_id="tenant", expected_revision=record.revision, next_record=updated,
        ) == updated
        assert await repository.get(record.session_id, tenant_id="tenant") == updated
    finally:
        await state.close()
