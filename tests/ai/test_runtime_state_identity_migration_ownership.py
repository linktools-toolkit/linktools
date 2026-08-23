#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ownership coverage for Runtime composition and identity migration."""

from datetime import datetime, timezone

import pytest

from linktools.ai.agent import AgentBinding, AgentCatalog, AgentCompiler
from linktools.ai.core import EvaluationStatus, ExecutionLineageKind, ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime._factory import _require_state_identity
from linktools.ai.runtime.state._contracts import (
    EvaluationRecord,
    ExecutionRecord,
    RecoveryAdmissionRecord,
    RecoveryExecutionInput,
    RecoveryIdempotencyInput,
)
from linktools.ai.runtime.state._migration import migrate_v1_agent_identity_state
from linktools.ai.spec import AgentSpec


def _compiler_catalog() -> tuple[AgentCompiler, AgentCatalog]:
    compiler = AgentCompiler(
        model_resolver=ModelRegistry.openai(model="gpt-test").snapshot(),
        runtime_fingerprint="a" * 64,
    )
    definition = compiler.compile(AgentSpec("default"))
    return compiler, AgentCatalog({"default": definition})


def _execution(binding: AgentBinding, *, tenant_id: str) -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    return ExecutionRecord(
        execution_id="execution",
        tenant_id=tenant_id,
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


def _recovery(binding: AgentBinding, *, tenant_id: str) -> RecoveryAdmissionRecord:
    now = datetime.now(timezone.utc)
    recovery_input = RecoveryExecutionInput(
        user_prompt="prompt",
        principal_id="principal",
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
    return RecoveryAdmissionRecord("execution", tenant_id, recovery_input, now)


def _evaluation(binding: AgentBinding, *, tenant_id: str) -> EvaluationRecord:
    now = datetime.now(timezone.utc)
    return EvaluationRecord(
        evaluation_id="evaluation",
        tenant_id=tenant_id,
        execution_id="execution",
        dataset_id="dataset",
        dataset_revision=1,
        evaluator_id="default",
        evaluator_revision=1,
        binding_digest=binding.digest,
        output_schema_fingerprint=binding.snapshot.output_schema_fingerprint,
        artifact_digest=None,
        status=EvaluationStatus.SUCCEEDED,
        revision=0,
        metrics={},
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_runtime_factory_rejects_state_identity_mismatch() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="runtime-owner", tenant_id="tenant")
    try:
        assert state.namespace == "runtime-owner"
        assert state.tenant_id == "tenant"
        _require_state_identity(
            state,
            namespace="runtime-owner",
            tenant_id="tenant",
        )
        for namespace, tenant_id in (
            ("other-runtime", "tenant"),
            ("runtime-owner", "other-tenant"),
        ):
            with pytest.raises(AIError) as raised:
                _require_state_identity(
                    state,
                    namespace=namespace,
                    tenant_id=tenant_id,
                )
            assert raised.value.code is ErrorCode.STORAGE_OWNER_MISMATCH
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_migration_rejects_runtime_tenant_mismatch() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="migration-runtime-owner", tenant_id="tenant")
    compiler, catalog = _compiler_catalog()
    try:
        with pytest.raises(AIError) as raised:
            await migrate_v1_agent_identity_state(
                state,
                catalog,
                compiler,
                tenant_id="other-tenant",
            )
        assert raised.value.code is ErrorCode.STORAGE_OWNER_MISMATCH
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("record_kind", ["execution", "recovery", "evaluation"])
async def test_migration_rejects_embedded_tenant_mismatch(record_kind: str) -> None:
    state = RuntimeState.in_memory()
    await state.initialize(
        namespace=f"migration-record-owner-{record_kind}",
        tenant_id="tenant",
    )
    compiler, catalog = _compiler_catalog()
    binding = compiler.bind(catalog.root_definition("default"))

    if record_kind == "execution":
        repository = state.execution.executions
        value = _execution(binding, tenant_id="other-tenant")
        stored = repository._stored(
            "execution",
            value.execution_id,
            value,
            state=value.status.value,
        )
    elif record_kind == "recovery":
        repository = state.recovery.checkpoints
        value = _recovery(binding, tenant_id="other-tenant")
        stored = repository._stored("recovery_admission", value.execution_id, value)
    else:
        repository = state.evaluation.records
        value = _evaluation(binding, tenant_id="other-tenant")
        stored = repository._stored(
            "evaluation",
            value.evaluation_id,
            value,
            state=value.status.value,
        )

    await repository.state_store.mutate(lambda tx: tx.insert_record(stored))
    try:
        with pytest.raises(AIError) as raised:
            await migrate_v1_agent_identity_state(
                state,
                catalog,
                compiler,
                tenant_id="tenant",
            )
        assert raised.value.code is ErrorCode.STORAGE_OWNER_MISMATCH
    finally:
        await state.close()
