#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistence regressions for instruction and deferred-work pins."""

from datetime import datetime, timezone

from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.agent._output import bind_output
from linktools.ai.core import ExecutionLineageKind, ExecutionStatus, canonical_sha256
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._codec import (
    _encode_persisted_domain,
    decode_domain,
    encode_domain,
    iter_runtime_object_refs,
)
from linktools.ai.runtime.state._contracts import (
    ExecutionRecord,
    PendingDeferredCall,
    PendingToolContinuation,
    RecoveryCheckpoint,
    RecoveryCheckpointState,
    RecoveryHandoffPhase,
    RuntimePayloadRef,
    StoredUserInput,
)
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import ObjectRef, StoredPayload


def _binding() -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", model="model"),
        base_model={"route_id": "model", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
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
        principal_id="principal",
        principal_kind="service",
        stored_user_input=StoredUserInput(
            "text",
            StoredPayload.inline_text("prompt"),
        ),
        repository_instructions=repository_instructions,
    )


def _continuation() -> PendingToolContinuation:
    arguments = StoredPayload.inline_json({"path": "file.txt"})
    call = PendingDeferredCall(
        "call-1",
        "read_file",
        arguments,
        arguments.digest,
        {"kind": "workspace_approval"},
    )
    return PendingToolContinuation(
        "step-1",
        canonical_sha256({"call_id": call.tool_call_id}),
        approvals=(call,),
    )


def _checkpoint(pending_tools: PendingToolContinuation | None) -> RecoveryCheckpoint:
    now = datetime.now(timezone.utc)
    waiting = pending_tools is not None
    return RecoveryCheckpoint(
        execution_id="execution",
        tenant_id="tenant",
        step_run_id="step-1" if waiting else None,
        agent_run_sequence=1 if waiting else 0,
        state=(
            RecoveryCheckpointState.WAITING
            if waiting
            else RecoveryCheckpointState.ADMITTED
        ),
        revision=1,
        created_at=now,
        updated_at=now,
        pending_tools=pending_tools,
        handoff_phase=RecoveryHandoffPhase.NONE,
    )


def test_instruction_aware_execution_round_trips_exact_pin() -> None:
    reference = _instruction_ref()
    wire = encode_domain(_execution(reference))

    assert isinstance(wire, dict)
    assert wire["$dataclass"] == "execution_record"
    decoded = decode_domain(wire, ExecutionRecord)
    assert decoded.repository_instructions == reference


def test_deferred_frontier_round_trips_current_contract() -> None:
    current = _checkpoint(_continuation())
    wire = encode_domain(current)

    assert isinstance(wire, dict)
    assert wire["$dataclass"] == "recovery_checkpoint"
    assert decode_domain(wire, RecoveryCheckpoint) == current


def test_object_ref_traversal_finds_repository_instruction_object() -> None:
    reference = _instruction_ref(object_backed=True)
    execution = _execution(reference)
    refs = tuple(
        iter_runtime_object_refs(
            _encode_persisted_domain(execution),
            default_domain=RuntimeDomain.EXECUTION,
        )
    )
    assert refs == ((RuntimeDomain.EXECUTION, reference.payload.ref),)
