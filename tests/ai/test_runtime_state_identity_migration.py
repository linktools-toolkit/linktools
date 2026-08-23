#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility tests for one-time Agent identity state normalization."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from linktools.ai.agent import AgentCatalog, AgentCompiler
from linktools.ai.core import ExecutionLineageKind, ExecutionStatus, SessionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime.state import (
    ContextProjection,
    RuntimeDomain,
    SessionRecord,
    migrate_v1_agent_identity_state,
)
from linktools.ai.runtime.state._codec import encode_domain, encode_envelope
from linktools.ai.runtime.state._contracts import ExecutionRecord
from linktools.ai.runtime.state._repositories import _domain_data
from linktools.ai.spec import AgentSpec


def _compiler_catalog() -> tuple[AgentCompiler, AgentCatalog]:
    compiler = AgentCompiler(
        model_resolver=ModelRegistry.openai(model="gpt-test").snapshot(),
        runtime_fingerprint="a" * 64,
    )
    root = compiler.compile(AgentSpec("default"))
    return compiler, AgentCatalog({"default": root})


def _legacy_snapshot(binding: object, legacy_digest: str) -> dict[str, object]:
    payload = dict(binding.snapshot.to_payload())
    payload.pop("agent_digest")
    spec = dict(payload["agent_spec"])
    spec["metadata"] = {}
    payload["agent_spec"] = spec
    payload["binding_digest"] = legacy_digest
    return payload


def _legacy_session_data(record: SessionRecord, legacy_digest: str) -> dict[str, object]:
    payload = encode_domain(record)
    fields = dict(payload["fields"])
    fields["binding_digest"] = encode_domain(legacy_digest)
    fields.pop("agent_digest")
    return encode_envelope(
        {"type": "session_record", "payload": {"$dataclass": "session_record", "fields": fields}}
    )


def _legacy_execution_data(record: ExecutionRecord, binding: object, legacy_digest: str) -> dict[str, object]:
    data = _domain_data(record)
    envelope = dict(data)
    value = dict(envelope["value"])
    payload = dict(value["payload"])
    fields = dict(payload["fields"])
    fields["binding_digest"] = encode_domain(legacy_digest)
    fields["binding"] = _legacy_snapshot(binding, legacy_digest)
    payload["fields"] = fields
    value["payload"] = payload
    envelope["value"] = value
    return envelope


@pytest.mark.asyncio
async def test_migration_rewrites_legacy_session_and_execution_exactly() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="legacy-agent-identity", tenant_id="tenant")
    compiler, catalog = _compiler_catalog()
    definition = catalog.root_definition("default")
    binding = compiler.bind(definition)
    legacy_digest = "f" * 64
    now = datetime.now(timezone.utc)
    session = SessionRecord(
        "session",
        "tenant",
        "owner",
        definition.digest,
        SessionStatus.OPEN,
        0,
        0,
        None,
        {"linktools.ai.agent_id": "default"},
        now,
        now,
        None,
        None,
        None,
        "complete",
        "history",
    )
    execution = ExecutionRecord(
        execution_id="execution",
        tenant_id="tenant",
        session_id="session",
        binding_digest=binding.digest,
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=ExecutionStatus.SUCCEEDED,
        revision=0,
        event_sequence=0,
        agent_run_sequence=0,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        planning=False,
        thinking=False,
        binding=binding.snapshot,
    )
    sessions = state.conversation.sessions
    executions = state.execution.executions
    session_stored = sessions._stored("session", "session", session, state=session.status.value)
    session_stored = replace(session_stored, data=_legacy_session_data(session, legacy_digest))
    execution_stored = executions._stored(
        "execution", "execution", execution, state=execution.status.value
    )
    execution_stored = replace(
        execution_stored,
        data=_legacy_execution_data(execution, binding, legacy_digest),
    )
    await sessions.state_store.mutate(lambda tx: tx.insert_record(session_stored))
    await executions.state_store.mutate(lambda tx: tx.insert_record(execution_stored))
    try:
        with pytest.raises(AIError) as before:
            await sessions.get("session", tenant_id="tenant")
        assert before.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

        changed = await migrate_v1_agent_identity_state(
            state, catalog, compiler, tenant_id="tenant"
        )
        assert changed == 2
        migrated_session = await sessions.get("session", tenant_id="tenant")
        migrated_execution = await executions.get("execution", tenant_id="tenant")
        assert migrated_session is not None
        assert migrated_execution is not None
        assert migrated_session.agent_digest == definition.digest
        assert migrated_execution.binding_digest == binding.digest
        assert migrated_execution.binding == binding.snapshot
        assert await migrate_v1_agent_identity_state(
            state, catalog, compiler, tenant_id="tenant"
        ) == 0
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_migration_allows_empty_legacy_session_from_root_identity() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="legacy-empty-session", tenant_id="tenant")
    compiler, catalog = _compiler_catalog()
    definition = catalog.root_definition("default")
    now = datetime.now(timezone.utc)
    session = SessionRecord(
        "session",
        "tenant",
        "owner",
        definition.digest,
        SessionStatus.OPEN,
        0,
        0,
        None,
        {"linktools.ai.agent_id": "default"},
        now,
        now,
        None,
        None,
    )
    sessions = state.conversation.sessions
    stored = sessions._stored("session", "session", session, state=session.status.value)
    stored = replace(stored, data=_legacy_session_data(session, "e" * 64))
    await sessions.state_store.mutate(lambda tx: tx.insert_record(stored))
    try:
        assert await migrate_v1_agent_identity_state(
            state, catalog, compiler, tenant_id="tenant"
        ) == 1
        migrated = await sessions.get("session", tenant_id="tenant")
        assert migrated is not None
        assert migrated.agent_digest == definition.digest
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_migration_rejects_legacy_execution_without_exact_snapshot() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="legacy-missing-binding", tenant_id="tenant")
    compiler, catalog = _compiler_catalog()
    definition = catalog.root_definition("default")
    binding = compiler.bind(definition)
    now = datetime.now(timezone.utc)
    execution = ExecutionRecord(
        execution_id="execution",
        tenant_id="tenant",
        session_id=None,
        binding_digest=binding.digest,
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=ExecutionStatus.SUCCEEDED,
        revision=0,
        event_sequence=0,
        agent_run_sequence=0,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        planning=False,
        thinking=False,
        binding=binding.snapshot,
    )
    executions = state.execution.executions
    stored = executions._stored("execution", "execution", execution, state=execution.status.value)
    data = _domain_data(execution)
    value = dict(data["value"])
    payload = dict(value["payload"])
    fields = dict(payload["fields"])
    fields["binding"] = None
    fields["binding_digest"] = encode_domain("d" * 64)
    payload["fields"] = fields
    value["payload"] = payload
    data = {**data, "value": value}
    stored = replace(stored, data=data)
    await executions.state_store.mutate(lambda tx: tx.insert_record(stored))
    try:
        with pytest.raises(AIError) as error:
            await migrate_v1_agent_identity_state(
                state, catalog, compiler, tenant_id="tenant"
            )
        assert error.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED
    finally:
        await state.close()
