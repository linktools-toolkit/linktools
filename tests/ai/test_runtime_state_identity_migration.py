#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility tests for one-time Agent identity state normalization."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from linktools.ai.agent import AgentCatalog, AgentCompiler
from linktools.ai.core import (
    EvaluationStatus,
    ExecutionLineageKind,
    ExecutionStatus,
    SessionStatus,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime.state import RuntimeDomain, SessionRecord
from linktools.ai.runtime.state._migration import migrate_v1_agent_identity_state
from linktools.ai.runtime.state._codec import (
    _decode_enveloped_domain,
    encode_domain,
    encode_envelope,
)
from linktools.ai.runtime.state._contracts import (
    ContextProjection,
    EvaluationRecord,
    ExecutionRecord,
    RecoveryAdmissionRecord,
    RecoveryExecutionInput,
    RecoveryIdempotencyInput,
)
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


def _legacy_session_data(
    record: SessionRecord,
    legacy_digest: str,
) -> dict[str, object]:
    payload = encode_domain(record)
    fields = dict(payload["fields"])
    fields["binding_digest"] = encode_domain(legacy_digest)
    fields.pop("agent_digest")
    return encode_envelope(
        {
            "type": "session_record",
            "payload": {"$dataclass": "session_record", "fields": fields},
        }
    )


def _legacy_execution_data(
    record: ExecutionRecord,
    binding: object,
    legacy_digest: str,
) -> dict[str, object]:
    data = _domain_data(record)
    value = dict(data["value"])
    payload = dict(value["payload"])
    fields = dict(payload["fields"])
    fields["binding_digest"] = encode_domain(legacy_digest)
    fields["binding"] = _legacy_snapshot(binding, legacy_digest)
    payload["fields"] = fields
    value["payload"] = payload
    data["value"] = value
    return data


def _legacy_recovery_data(
    admission: RecoveryAdmissionRecord,
    binding: object,
    legacy_digest: str,
) -> dict[str, object]:
    data = _domain_data(admission)
    value = dict(data["value"])
    payload = dict(value["payload"])
    fields = dict(payload["fields"])
    raw_input = dict(fields["input"])
    input_fields = dict(raw_input["fields"])
    input_fields["agent_id"] = encode_domain(binding.definition.spec.id)
    input_fields["binding_digest"] = encode_domain(legacy_digest)
    input_fields["binding"] = _legacy_snapshot(binding, legacy_digest)
    raw_input["fields"] = input_fields
    fields["input"] = raw_input
    payload["fields"] = fields
    value["payload"] = payload
    data["value"] = value
    return data


def _execution(binding: object, *, session_id: str | None = None) -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    return ExecutionRecord(
        execution_id="execution",
        tenant_id="tenant",
        session_id=session_id,
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


def _session(agent_digest: str) -> SessionRecord:
    now = datetime.now(timezone.utc)
    return SessionRecord(
        "session",
        "tenant",
        "owner",
        agent_digest,
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


@pytest.mark.asyncio
async def test_migration_rewrites_legacy_session_and_execution_exactly() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="legacy-agent-identity", tenant_id="tenant")
    compiler, catalog = _compiler_catalog()
    definition = catalog.root_definition("default")
    binding = compiler.bind(definition)
    legacy_digest = "f" * 64
    session = _session(definition.digest)
    execution = _execution(binding, session_id="session")
    sessions = state.conversation.sessions
    executions = state.execution.executions
    session_stored = sessions._stored(
        "session", "session", session, state=session.status.value
    )
    session_stored = replace(
        session_stored,
        data=_legacy_session_data(session, legacy_digest),
    )
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

        # Completion markers make subsequent startup checks O(1), not a rescan.
        assert await migrate_v1_agent_identity_state(
            state, catalog, compiler, tenant_id="tenant"
        ) == 0
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_migration_rejects_legacy_session_without_binding_evidence() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="legacy-unprovable-session", tenant_id="tenant")
    compiler, catalog = _compiler_catalog()
    definition = catalog.root_definition("default")
    session = _session(definition.digest)
    sessions = state.conversation.sessions
    stored = sessions._stored("session", "session", session, state=session.status.value)
    stored = replace(stored, data=_legacy_session_data(session, "e" * 64))
    await sessions.state_store.mutate(lambda tx: tx.insert_record(stored))
    try:
        with pytest.raises(AIError) as error:
            await migrate_v1_agent_identity_state(
                state, catalog, compiler, tenant_id="tenant"
            )
        assert error.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_migration_rejects_legacy_execution_without_exact_snapshot() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="legacy-missing-binding", tenant_id="tenant")
    compiler, catalog = _compiler_catalog()
    binding = compiler.bind(catalog.root_definition("default"))
    execution = _execution(binding)
    executions = state.execution.executions
    stored = executions._stored(
        "execution", "execution", execution, state=execution.status.value
    )
    data = _domain_data(execution)
    value = dict(data["value"])
    payload = dict(value["payload"])
    fields = dict(payload["fields"])
    fields["binding"] = None
    fields["binding_digest"] = encode_domain("d" * 64)
    payload["fields"] = fields
    value["payload"] = payload
    data["value"] = value
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


@pytest.mark.asyncio
async def test_migration_rewrites_full_legacy_recovery_shape() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="legacy-recovery", tenant_id="tenant")
    compiler, catalog = _compiler_catalog()
    definition = catalog.root_definition("default")
    binding = compiler.bind(definition)
    legacy_digest = "c" * 64
    now = datetime.now(timezone.utc)
    recovery_input = RecoveryExecutionInput(
        user_prompt="prompt",
        principal_id="owner",
        principal_kind="service",
        session_id=None,
        memory_scope=None,
        binding_digest=binding.digest,
        lineage_kind=ExecutionLineageKind.RUN.value,
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        conversation_step_run_id=None,
        idempotency=RecoveryIdempotencyInput("scope", "key", "request"),
        planning=False,
        thinking=False,
        binding=binding.snapshot,
    )
    admission = RecoveryAdmissionRecord("execution", "tenant", recovery_input, now)
    recovery = state.recovery.checkpoints
    stored = recovery._stored("recovery_admission", "execution", admission)
    stored = replace(
        stored,
        data=_legacy_recovery_data(admission, binding, legacy_digest),
    )
    await recovery.state_store.mutate(lambda tx: tx.insert_record(stored))
    try:
        assert await migrate_v1_agent_identity_state(
            state, catalog, compiler, tenant_id="tenant"
        ) == 1
        raw = await recovery.state_store.read(
            lambda tx: tx.get_record(recovery._key("recovery_admission", "execution"))
        )
        assert raw is not None
        migrated = _decode_enveloped_domain(raw.data, RecoveryAdmissionRecord)
        assert migrated.input.binding_digest == binding.digest
        assert migrated.input.binding == binding.snapshot
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_migration_preserves_historical_projection_digest() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="legacy-projection", tenant_id="tenant")
    compiler, catalog = _compiler_catalog()
    definition = catalog.root_definition("default")
    binding = compiler.bind(definition)
    legacy_digest = "b" * 64
    execution = _execution(binding)
    executions = state.execution.executions
    execution_stored = executions._stored(
        "execution", "execution", execution, state=execution.status.value
    )
    execution_stored = replace(
        execution_stored,
        data=_legacy_execution_data(execution, binding, legacy_digest),
    )
    await executions.state_store.mutate(lambda tx: tx.insert_record(execution_stored))

    history = state.steps.read_store(RuntimeDomain.EXECUTION).transcript_repository
    projection = ContextProjection(definition.digest, (), "historic-projection-digest")
    await history._store.mutate(
        lambda tx: history.store_projection(tx, "run", projection)
    )
    record = await history._store.read(
        lambda tx: tx.get_record(history._projection_key("run"))
    )
    assert record is not None
    data = dict(record.data)
    value = dict(data["value"])
    payload = dict(value["payload"])
    fields = dict(payload["fields"])
    fields["binding_digest"] = encode_domain(legacy_digest)
    fields.pop("agent_digest")
    payload["fields"] = fields
    value["payload"] = payload
    data["value"] = value
    replaced = await history._store.mutate(
        lambda tx: tx.replace_record(
            replace(record, data=data, storage_version=record.storage_version + 1),
            expected_storage_version=record.storage_version,
        )
    )
    assert replaced
    try:
        assert await migrate_v1_agent_identity_state(
            state, catalog, compiler, tenant_id="tenant"
        ) == 2
        migrated = await history.load_projection("run")
        assert migrated is not None
        assert migrated.agent_digest == definition.digest
        assert migrated.digest == "historic-projection-digest"
    finally:
        await state.close()

@pytest.mark.asyncio
async def test_migration_rewrites_evaluation_from_linked_legacy_execution() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="legacy-evaluation", tenant_id="tenant")
    compiler, catalog = _compiler_catalog()
    binding = compiler.bind(catalog.root_definition("default"))
    legacy_digest = "a" * 64
    execution = _execution(binding)
    executions = state.execution.executions
    stored = executions._stored(
        "execution", "execution", execution, state=execution.status.value
    )
    stored = replace(
        stored,
        data=_legacy_execution_data(execution, binding, legacy_digest),
    )
    await executions.state_store.mutate(lambda tx: tx.insert_record(stored))
    now = datetime.now(timezone.utc)
    evaluation = EvaluationRecord(
        evaluation_id="evaluation",
        tenant_id="tenant",
        execution_id="execution",
        dataset_id="dataset",
        dataset_revision=1,
        evaluator_id="default",
        evaluator_revision=1,
        binding_digest=legacy_digest,
        output_schema_fingerprint=binding.snapshot.output_schema_fingerprint,
        artifact_digest=None,
        status=EvaluationStatus.SUCCEEDED,
        revision=0,
        metrics={},
        created_at=now,
        updated_at=now,
    )
    await state.evaluation.records.create(evaluation)
    try:
        assert await migrate_v1_agent_identity_state(
            state, catalog, compiler, tenant_id="tenant"
        ) == 2
        migrated = await state.evaluation.records.get(
            "evaluation", tenant_id="tenant"
        )
        assert migrated is not None
        assert migrated.binding_digest == binding.digest
        assert migrated.output_schema_fingerprint == binding.snapshot.output_schema_fingerprint
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_migration_converges_evaluation_after_execution_was_already_migrated() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="partial-evaluation", tenant_id="tenant")
    compiler, catalog = _compiler_catalog()
    binding = compiler.bind(catalog.root_definition("default"))
    execution = _execution(binding)
    executions = state.execution.executions
    await executions.state_store.mutate(
        lambda tx: tx.insert_record(
            executions._stored(
                "execution",
                execution.execution_id,
                execution,
                state=execution.status.value,
            )
        )
    )
    now = datetime.now(timezone.utc)
    evaluation = EvaluationRecord(
        evaluation_id="evaluation",
        tenant_id="tenant",
        execution_id=execution.execution_id,
        dataset_id="dataset",
        dataset_revision=1,
        evaluator_id="default",
        evaluator_revision=1,
        binding_digest="f" * 64,
        output_schema_fingerprint=binding.snapshot.output_schema_fingerprint,
        artifact_digest=None,
        status=EvaluationStatus.SUCCEEDED,
        revision=0,
        metrics={},
        created_at=now,
        updated_at=now,
    )
    await state.evaluation.records.create(evaluation)
    try:
        assert await migrate_v1_agent_identity_state(
            state, catalog, compiler, tenant_id="tenant"
        ) == 1
        migrated = await state.evaluation.records.get(
            "evaluation", tenant_id="tenant"
        )
        assert migrated is not None
        assert migrated.binding_digest == binding.digest
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_migration_rejects_evaluation_output_contract_drift() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="evaluation-contract-drift", tenant_id="tenant")
    compiler, catalog = _compiler_catalog()
    binding = compiler.bind(catalog.root_definition("default"))
    execution = _execution(binding)
    executions = state.execution.executions
    await executions.state_store.mutate(
        lambda tx: tx.insert_record(
            executions._stored(
                "execution",
                execution.execution_id,
                execution,
                state=execution.status.value,
            )
        )
    )
    now = datetime.now(timezone.utc)
    await state.evaluation.records.create(
        EvaluationRecord(
            evaluation_id="evaluation",
            tenant_id="tenant",
            execution_id=execution.execution_id,
            dataset_id="dataset",
            dataset_revision=1,
            evaluator_id="default",
            evaluator_revision=1,
            binding_digest="e" * 64,
            output_schema_fingerprint="d" * 64,
            artifact_digest=None,
            status=EvaluationStatus.SUCCEEDED,
            revision=0,
            metrics={},
            created_at=now,
            updated_at=now,
        )
    )
    try:
        with pytest.raises(AIError) as raised:
            await migrate_v1_agent_identity_state(
                state, catalog, compiler, tenant_id="tenant"
            )
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()

