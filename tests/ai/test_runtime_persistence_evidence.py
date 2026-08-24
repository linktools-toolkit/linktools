#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end evidence for tolerant Runtime persistence reads."""

import copy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.agent._output import bind_output
from linktools.ai.core import (
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    HmacCursorSigner,
    Principal,
    SessionStatus,
    StopReason,
    TenantAuthorizationPolicy,
    UsageMetrics,
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
from linktools.ai.runtime.state._contracts import (
    ContextProjection,
    ExecutionRecord,
    ExecutionTerminalCommit,
    ResultRecord,
    SessionRecord,
)
from linktools.ai.spec import AgentSpec, AgentSpecCodec
from linktools.ai.storage import ObjectRef, StoredPayload
from linktools.ai.workspace import Workspace, open_workspace_runtime
from pydantic import BaseModel
from pydantic_ai.models.test import TestModel
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


def _binding_snapshot() -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", 1, "default"),
        agent_digest="b" * 64,
        output_type_module=output.value_type.__module__,
        output_type_qualname=output.value_type.__qualname__,
        output_schema_id=output.schema_id,
        output_schema_revision=output.schema_revision,
        output_schema_fingerprint=output.schema_fingerprint,
        local_runtime_capability_descriptors=(),
        binding_digest="a" * 64,
    )


def _execution() -> ExecutionRecord:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return ExecutionRecord(
        execution_id="execution",
        tenant_id="tenant",
        session_id=None,
        binding_digest="a" * 64,
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=ExecutionStatus.STARTED,
        revision=1,
        event_sequence=0,
        agent_run_sequence=1,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        planning=False,
        thinking=False,
        binding=_binding_snapshot(),
    )


def _result(execution_id: str, payload_kind: str, now: datetime) -> ResultRecord:
    output = (
        StoredPayload.inline_json({"text": "result"})
        if payload_kind == "inline"
        else StoredPayload.object(
            ObjectRef("runtime", "result", "c" * 64, 7)
        )
    )
    return ResultRecord(
        execution_id=execution_id,
        tenant_id="tenant",
        output_schema_id="schema",
        output_schema_revision=1,
        output_schema_fingerprint="fingerprint",
        output=output,
        stop_reason=StopReason.END_TURN,
        usage=UsageMetrics(),
        created_at=now,
    )


def test_execution_record_writer_accepts_nested_json_result() -> None:
    execution = _execution()
    result = ResultRecord(
        execution_id=execution.execution_id,
        tenant_id=execution.tenant_id,
        output_schema_id="schema",
        output_schema_revision=1,
        output_schema_fingerprint="fingerprint",
        output=StoredPayload.inline_json(
            {
                "findings": [
                    {
                        "trace_id": "trace",
                        "labels": [{"name": "priority", "value": "high"}],
                    }
                ]
            }
        ),
        stop_reason=StopReason.END_TURN,
        usage=UsageMetrics(),
        created_at=execution.updated_at,
    )
    value = replace(execution, result=result)

    payload = _encode_persisted_domain(value)

    assert _decode_enveloped_domain(
        encode_envelope({"type": "execution_record", "payload": payload}),
        ExecutionRecord,
    ) == value


class _PersistenceNestedLabel(BaseModel):
    name: str
    value: str


class _PersistenceNestedFinding(BaseModel):
    trace_id: str
    labels: list[_PersistenceNestedLabel]


class _PersistenceNestedOutput(BaseModel):
    findings: list[_PersistenceNestedFinding]


class _PersistenceTestModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:test"
    connection_identity = "test"
    fingerprint = "a" * 64

    def materialize(self) -> TestModel:
        return TestModel(
            custom_output_args={
                "findings": [
                    {
                        "trace_id": "trace",
                        "labels": [
                            {"name": "priority", "value": "high"},
                        ],
                    }
                ]
            }
        )


class _PersistenceTestModels:
    def snapshot(self) -> "_PersistenceTestModels":
        return self

    def resolve(self, route_id: str) -> _PersistenceTestModelBinding:
        if route_id != "default":
            raise AssertionError(f"unexpected model route: {route_id}")
        return _PersistenceTestModelBinding()


@pytest.mark.asyncio
async def test_session_runtime_persists_and_reads_terminal_result(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    agent_path = workspace_root / ".linktools" / "agents" / "default"
    agent_path.parent.mkdir(parents=True)
    agent_path.write_bytes(
        AgentSpecCodec().encode(
            AgentSpec("default", model="default", allow_tools=())
        )
    )
    database = tmp_path / "runtime.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{database}")
    await provision_runtime_database(engine)
    await engine.dispose()
    state = RuntimeState.sqlite(database)

    try:
        async with open_workspace_runtime(
            Workspace.load(workspace_root),
            models=_PersistenceTestModels(),
            state=state,
        ) as runtime:
            created = await runtime.agent("default").create_session("session")
            loaded = await runtime.session.get(
                created.session_id,
                principal=runtime.default_principal,
            )
            assert loaded.session_id == created.session_id

            result = await runtime.agent("default").run(
                "hello",
                output=_PersistenceNestedOutput,
                session_id=created.session_id,
                timeout_seconds=10,
            )
            assert result.status is ExecutionStatus.SUCCEEDED

            session_record = await state.conversation.sessions.get(
                created.session_id,
                tenant_id=runtime.tenant_id,
            )
            execution = await state.execution.executions.get(
                result.execution_id,
                tenant_id=runtime.tenant_id,
            )
            persisted_result = await state.execution.executions.get_result(
                result.execution_id,
                tenant_id=runtime.tenant_id,
            )
            assert session_record is not None
            assert session_record.active_execution_id is None
            assert execution is not None
            assert execution.status is ExecutionStatus.SUCCEEDED
            assert execution.result is not None
            assert persisted_result == execution.result
            assert persisted_result.output is not None
            assert persisted_result.output.value == {
                "findings": [
                    {
                        "trace_id": "trace",
                        "labels": [
                            {"name": "priority", "value": "high"},
                        ],
                    }
                ]
            }

            terminal_execution = replace(execution, result=None)
            next_execution = replace(
                terminal_execution,
                result=persisted_result,
            )
            _encode_persisted_domain(persisted_result)
            _encode_persisted_domain(next_execution)

            inspected = await runtime.execution.inspect(
                result.execution_id,
                principal=runtime.default_principal,
            )
            waited = await runtime.execution.wait(
                result.execution_id,
                principal=runtime.default_principal,
            )
            assert inspected.status is ExecutionStatus.SUCCEEDED
            assert waited.status is ExecutionStatus.SUCCEEDED
            assert waited.output == persisted_result.output.value
    finally:
        await state.close()


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
@pytest.mark.parametrize("backend", ["filesystem", "sqlite"])
@pytest.mark.parametrize("payload_kind", ["inline", "object"])
async def test_terminal_result_readback_survives_reopen(
    tmp_path,
    backend: str,
    payload_kind: str,
) -> None:
    path = tmp_path / f"terminal-{backend}-{payload_kind}"
    if backend == "filesystem":
        state = RuntimeState.filesystem(path)
    else:
        database = path.with_suffix(".db")
        engine = create_async_engine(f"sqlite+aiosqlite:///{database}")
        await provision_runtime_database(engine)
        await engine.dispose()
        state = RuntimeState.sqlite(database)

    execution = _execution()
    now = execution.updated_at
    result = _result(execution.execution_id, payload_kind, now)
    terminal = replace(
        execution,
        status=ExecutionStatus.SUCCEEDED,
        revision=2,
        event_sequence=1,
        result=result,
    )
    commit = ExecutionTerminalCommit(
        expected_revision=1,
        expected_event_sequence=0,
        execution=terminal,
        result=result,
        terminal_event_type=ExecutionEventType.EXECUTION_SUCCEEDED,
        terminal_event_payload={},
    )
    await state.initialize(namespace=f"terminal-{backend}", tenant_id="tenant")
    await state.execution.executions.create(execution)
    try:
        await state.execution.executions.commit_terminal(commit)
        current = await state.execution.executions.get(
            execution.execution_id,
            tenant_id="tenant",
        )
        current_result = await state.execution.executions.get_result(
            execution.execution_id,
            tenant_id="tenant",
        )
        assert current is not None
        assert current.status is ExecutionStatus.SUCCEEDED
        assert current.result == result
        assert current_result == result
    finally:
        await state.close()

    reopened = (
        RuntimeState.filesystem(path)
        if backend == "filesystem"
        else RuntimeState.sqlite(path.with_suffix(".db"))
    )
    await reopened.initialize(namespace=f"terminal-{backend}", tenant_id="tenant")
    try:
        current = await reopened.execution.executions.get(
            execution.execution_id,
            tenant_id="tenant",
        )
        current_result = await reopened.execution.executions.get_result(
            execution.execution_id,
            tenant_id="tenant",
        )
        assert current is not None
        assert current.status is ExecutionStatus.SUCCEEDED
        assert current.result == result
        assert current_result == result
    finally:
        await reopened.close()


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
