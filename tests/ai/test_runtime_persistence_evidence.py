#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end evidence for tolerant Runtime persistence reads."""

import copy
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from linktools.ai.core import (
    HmacCursorSigner,
    Principal,
    SessionStatus,
    TenantAuthorizationPolicy,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.runtime import DefaultSessionService, RuntimeState
from linktools.ai.runtime._session import _session_agent_id
from linktools.ai.runtime.state._codec import (
    _decode_enveloped_domain,
    _encode_persisted_domain,
    encode_envelope,
)
from linktools.ai.runtime.state._contracts import ContextProjection, SessionRecord
from sqlalchemy.ext.asyncio import create_async_engine


def _session() -> SessionRecord:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return SessionRecord(
        session_id="session",
        tenant_id="tenant",
        owner_principal_id="owner",
        agent_id="agent",
        status=SessionStatus.OPEN,
        revision=0,
        resource_generation=0,
        cwd=None,
        metadata={},
        created_at=now,
        updated_at=now,
        closed_at=None,
        active_execution_id=None,
        history_id="history",
    )


async def _insert_compatible_old_session(
    state: RuntimeState,
    session: SessionRecord,
) -> tuple[bytes, object]:
    repository = state.conversation.sessions
    stored = repository._stored(
        "session",
        "session",
        session,
        state=session.status.value,
    )
    legacy_data = copy.deepcopy(stored.data)
    fields = legacy_data["value"]["payload"]["fields"]
    fields.pop("history_id")
    fields["removed_historical_field"] = {"not": "decoded"}
    legacy = replace(stored, data=legacy_data)
    await repository.state_store.mutate(lambda tx: tx.insert_record(legacy))
    raw = await repository.state_store.read(
        lambda tx: tx.get_record(legacy.key_digest)
    )
    assert raw is not None
    return legacy.key_digest, raw


async def _assert_read_preserves_raw_record(
    state: RuntimeState,
    session: SessionRecord,
    key: bytes,
    raw_before: object,
) -> None:
    loaded = await state.conversation.sessions.get(
        session.session_id,
        tenant_id=session.tenant_id,
    )
    assert loaded == replace(session, history_id=None)
    raw_after = await state.conversation.sessions.state_store.read(
        lambda tx: tx.get_record(key)
    )
    assert raw_after is not None
    assert raw_after.storage_version == raw_before.storage_version
    assert raw_after.data == raw_before.data


@pytest.mark.asyncio
async def test_filesystem_compatible_read_is_pure_and_survives_reopen(
    tmp_path,
) -> None:
    root = tmp_path / "runtime"
    namespace = "persistence-tolerant-fs"
    session = _session()

    state = RuntimeState.filesystem(root)
    await state.initialize(namespace=namespace, tenant_id=session.tenant_id)
    key, raw_before = await _insert_compatible_old_session(state, session)
    await state.close()

    reopened = RuntimeState.filesystem(root)
    await reopened.initialize(namespace=namespace, tenant_id=session.tenant_id)
    try:
        await _assert_read_preserves_raw_record(
            reopened,
            session,
            key,
            raw_before,
        )
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_sqlite_compatible_read_is_pure_and_survives_reopen(
    tmp_path,
) -> None:
    path = tmp_path / "runtime.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    await provision_runtime_database(engine)
    await engine.dispose()

    namespace = "persistence-tolerant-sqlite"
    session = _session()
    state = RuntimeState.sqlite(path)
    await state.initialize(namespace=namespace, tenant_id=session.tenant_id)
    key, raw_before = await _insert_compatible_old_session(state, session)
    await state.close()

    reopened = RuntimeState.sqlite(path)
    await reopened.initialize(namespace=namespace, tenant_id=session.tenant_id)
    try:
        await _assert_read_preserves_raw_record(
            reopened,
            session,
            key,
            raw_before,
        )
    finally:
        await reopened.close()


async def _insert_historical_identity_session(
    state: RuntimeState,
    session: SessionRecord,
    historical_field: str,
) -> tuple[bytes, object]:
    repository = state.conversation.sessions
    legacy = replace(
        session,
        agent_id=None,
        metadata={"linktools.ai.agent_id": "agent"},
    )
    stored = repository._stored(
        "session",
        "session",
        legacy,
        state=legacy.status.value,
    )
    data = copy.deepcopy(stored.data)
    fields = data["value"]["payload"]["fields"]
    fields.pop("agent_id")
    fields[historical_field] = "b" * 64
    historical = replace(stored, data=data)
    await repository.state_store.mutate(lambda tx: tx.insert_record(historical))
    raw = await repository.state_store.read(
        lambda tx: tx.get_record(historical.key_digest)
    )
    assert raw is not None
    return historical.key_digest, raw


async def _assert_historical_identity_read(
    state: RuntimeState,
    key: bytes,
    raw_before: object,
) -> None:
    principal = Principal("owner", "tenant")
    service = DefaultSessionService(
        state.conversation,
        state.execution.executions,
        TenantAuthorizationPolicy(),
        object(),
        HmacCursorSigner("session", b"session-key"),
        history_reader=object(),
    )
    view = await service.get("session", principal=principal)
    assert view.agent_id == "agent"
    record = await state.conversation.sessions.get("session", tenant_id="tenant")
    assert record is not None
    assert record.agent_id is None
    raw_after = await state.conversation.sessions.state_store.read(
        lambda tx: tx.get_record(key)
    )
    assert raw_after is not None
    assert raw_after.data == raw_before.data
    assert raw_after.storage_version == raw_before.storage_version


@pytest.mark.asyncio
@pytest.mark.parametrize("historical_field", ["binding_digest", "agent_digest"])
@pytest.mark.parametrize("backend", ["filesystem", "sqlite"])
async def test_historical_session_identity_fallback_is_read_only(
    tmp_path,
    historical_field: str,
    backend: str,
) -> None:
    namespace = f"historical-session-{backend}-{historical_field}"
    if backend == "filesystem":
        state = RuntimeState.filesystem(tmp_path / "runtime")
    else:
        path = tmp_path / "runtime.db"
        engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
        await provision_runtime_database(engine)
        await engine.dispose()
        state = RuntimeState.sqlite(path)
    session = _session()
    await state.initialize(namespace=namespace, tenant_id=session.tenant_id)
    try:
        key, raw_before = await _insert_historical_identity_session(
            state,
            session,
            historical_field,
        )
        await _assert_historical_identity_read(state, key, raw_before)
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["filesystem", "sqlite"])
async def test_historical_session_business_write_persists_resolved_agent_id(
    tmp_path,
    backend: str,
) -> None:
    namespace = f"historical-session-write-{backend}"
    if backend == "filesystem":
        state = RuntimeState.filesystem(tmp_path / "runtime")
    else:
        path = tmp_path / "runtime.db"
        engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
        await provision_runtime_database(engine)
        await engine.dispose()
        state = RuntimeState.sqlite(path)
    session = _session()
    await state.initialize(namespace=namespace, tenant_id=session.tenant_id)
    try:
        key, _raw_before = await _insert_historical_identity_session(
            state,
            session,
            "binding_digest",
        )
        await state.conversation.sessions.admit_execution(
            session.session_id,
            tenant_id=session.tenant_id,
            execution_id="execution",
            expected=None,
        )
        raw_after = await state.conversation.sessions.state_store.read(
            lambda tx: tx.get_record(key)
        )
        assert raw_after is not None
        fields = raw_after.data["value"]["payload"]["fields"]
        assert fields["agent_id"] == "agent"
    finally:
        await state.close()


def test_historical_session_identity_failures_are_scoped() -> None:
    session = replace(_session(), agent_id=None, metadata={})
    with pytest.raises(AIError) as raised:
        _session_agent_id(session)
    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED

    conflicting = replace(
        _session(),
        metadata={"linktools.ai.agent_id": "other"},
    )
    with pytest.raises(AIError) as raised:
        _session_agent_id(conflicting)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("historical_field", ["agent_digest", "binding_digest"])
def test_historical_projection_identity_is_ignored_and_digest_is_opaque(
    historical_field: str,
) -> None:
    projection = ContextProjection((), "legacy-digest")
    payload = copy.deepcopy(_encode_persisted_domain(projection))
    payload["fields"][historical_field] = "a" * 64
    decoded = _decode_enveloped_domain(
        encode_envelope({"type": "context_projection", "payload": payload}),
        ContextProjection,
    )
    assert decoded == projection
