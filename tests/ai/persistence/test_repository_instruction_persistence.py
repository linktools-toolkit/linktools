#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V1 persistence regressions for repository instruction and approval pins."""

from copy import deepcopy
from datetime import datetime, timezone

from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.agent._output import bind_output
from linktools.ai.core import ExecutionLineageKind, ExecutionStatus
from linktools.ai.runtime.state import (
    ExecutionRecord,
    PendingApprovalContinuation,
    RecoveryCheckpointState,
    RecoveryHandoffPhase,
    RecoveryStateRecord,
    RuntimeDomain,
    RuntimePayloadRef,
    iter_runtime_object_refs,
)
from linktools.ai.runtime.state._codec import decode_domain, encode_domain
from linktools.ai.runtime.state._contracts import (
    RecoveryExecutionInput,
    RecoveryIdempotencyInput,
)
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import ObjectRef, StoredPayload


def _binding() -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", model="model"),
        model={"route_id": "model", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
        binding_digest="a" * 64,
    )


def _instruction_ref(*, object_backed: bool = False) -> RuntimePayloadRef:
    if object_backed:
        payload = StoredPayload.object(
            ObjectRef("execution", "repository/instructions", "b" * 64, 17)
        )
    else:
        payload = StoredPayload.inline_json({"version": 1, "documents": []})
    return RuntimePayloadRef(payload, RuntimeDomain.EXECUTION)


def _execution(repository_instructions: RuntimePayloadRef | None) -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    binding = _binding()
    return ExecutionRecord(
        execution_id="execution",
        tenant_id="tenant",
        session_id=None,
        binding_digest=binding.binding_digest,
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=ExecutionStatus.PENDING_START,
        revision=0,
        event_sequence=0,
        agent_run_sequence=0,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        mode="run",
        planning=False,
        thinking=False,
        binding=binding,
        repository_instructions=repository_instructions,
    )


def _recovery(repository_instructions: RuntimePayloadRef | None) -> RecoveryExecutionInput:
    binding = _binding()
    return RecoveryExecutionInput(
        user_prompt="prompt",
        user_prompt_codec="text",
        principal_id="principal",
        principal_kind="service",
        session_id=None,
        memory_scope=None,
        binding_digest=binding.binding_digest,
        lineage_kind=ExecutionLineageKind.RUN.value,
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        conversation_step_run_id=None,
        idempotency=RecoveryIdempotencyInput("scope", "key", "request"),
        mode="run",
        planning=False,
        thinking=False,
        binding=binding,
        repository_instructions=repository_instructions,
    )


def test_older_v1_execution_and_recovery_payloads_default_instruction_pin_to_none() -> None:
    execution_wire = deepcopy(encode_domain(_execution(None)))
    recovery_wire = deepcopy(encode_domain(_recovery(None)))
    assert isinstance(execution_wire, dict)
    assert isinstance(recovery_wire, dict)
    execution_wire["fields"].pop("repository_instructions", None)
    recovery_wire["fields"].pop("repository_instructions", None)

    execution = decode_domain(execution_wire, ExecutionRecord)
    recovery = decode_domain(recovery_wire, RecoveryExecutionInput)
    assert execution.repository_instructions is None
    assert recovery.repository_instructions is None


def test_instruction_aware_execution_and_recovery_remain_v1_and_round_trip_exact_pin() -> None:
    reference = _instruction_ref()
    execution = _execution(reference)
    recovery = _recovery(reference)
    execution_wire = encode_domain(execution)
    recovery_wire = encode_domain(recovery)

    assert isinstance(execution_wire, dict)
    assert isinstance(recovery_wire, dict)
    assert execution_wire["$dataclass"] == "execution_record"
    assert recovery_wire["$dataclass"] == "recovery_execution_input"
    decoded_execution = decode_domain(execution_wire, ExecutionRecord)
    decoded_recovery = decode_domain(recovery_wire, RecoveryExecutionInput)
    assert decoded_execution.repository_instructions == reference
    assert decoded_recovery.repository_instructions == reference
    assert decoded_execution.repository_instructions == decoded_recovery.repository_instructions


def test_older_v1_recovery_state_defaults_pending_approval_to_none() -> None:
    now = datetime.now(timezone.utc)
    current = RecoveryStateRecord(
        execution_id="execution",
        tenant_id="tenant",
        step_run_id="run-1",
        agent_run_sequence=1,
        state=RecoveryCheckpointState.ACTIVE,
        handoff_phase=RecoveryHandoffPhase.NONE,
        terminal_handoff=None,
        handoff_contract_digest=None,
        pending_operation_id=None,
        revision=1,
        updated_at=now,
    )
    wire = deepcopy(encode_domain(current))
    assert isinstance(wire, dict)
    wire["fields"].pop("pending_approval", None)
    assert decode_domain(wire, RecoveryStateRecord).pending_approval is None


def test_pending_approval_recovery_state_round_trips_without_schema_version_bump() -> None:
    now = datetime.now(timezone.utc)
    continuation = PendingApprovalContinuation("c" * 64, "run-1")
    current = RecoveryStateRecord(
        execution_id="execution",
        tenant_id="tenant",
        step_run_id="run-1",
        agent_run_sequence=1,
        state=RecoveryCheckpointState.WAITING,
        handoff_phase=RecoveryHandoffPhase.NONE,
        terminal_handoff=None,
        handoff_contract_digest=None,
        pending_operation_id=None,
        revision=2,
        updated_at=now,
        pending_approval=continuation,
    )
    wire = encode_domain(current)
    assert isinstance(wire, dict)
    assert wire["$dataclass"] == "recovery_state"
    assert decode_domain(wire, RecoveryStateRecord) == current


def test_object_ref_traversal_finds_repository_instruction_object() -> None:
    reference = _instruction_ref(object_backed=True)
    execution = _execution(reference)
    refs = tuple(
        iter_runtime_object_refs(
            encode_domain(execution),
            default_domain=RuntimeDomain.EXECUTION,
        )
    )
    assert len(refs) == 1
    assert refs[0] == (RuntimeDomain.EXECUTION, reference.payload.ref)
