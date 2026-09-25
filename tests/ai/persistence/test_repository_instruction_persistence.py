#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistence regressions for instruction and deferred-work pins."""

from dataclasses import replace
from datetime import datetime, timezone

from linktools.ai.agent import AgentBindingSnapshot, CapabilityPin
from linktools.ai.agent._output import bind_output
from linktools.ai.asset import AssetKey, AssetVersionRef
from linktools.ai.capability import SkillDefinition, SkillResourceVersion, SkillSourceRef
from linktools.ai.core import ExecutionLineageKind, ExecutionStatus
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
from linktools.ai.spec import AgentSpec, SkillSpec
from linktools.ai.storage import ObjectRef, StorageEntryRevision, StoredPayload


def _binding() -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        agent_spec=AgentSpec("agent", model_route="model"),
        base_model={"route_id": "model", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
    )


def _instruction_ref(*, object_backed: bool = False) -> RuntimePayloadRef:
    if object_backed:
        payload = StoredPayload.object(
            ObjectRef("runtime", "repository/instructions", "b" * 64, 17)
        )
    else:
        payload = StoredPayload.inline_json({"version": 1, "documents": []})
    return RuntimePayloadRef(payload, RuntimeDomain.EXECUTION)


def _execution(repository_instructions: RuntimePayloadRef | None) -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    binding = _binding()
    return ExecutionRecord(
        execution_id="execution",
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
        {"kind": "workspace_approval"},
    )
    return PendingToolContinuation(
        "step-1",
        approvals=(call,),
    )


def _checkpoint(pending_tools: PendingToolContinuation | None) -> RecoveryCheckpoint:
    now = datetime.now(timezone.utc)
    waiting = pending_tools is not None
    return RecoveryCheckpoint(
        execution_id="execution",
        step_run_id="step-1" if waiting else None,
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


def test_object_ref_traversal_allows_additive_skill_asset_fields() -> None:
    reference = _instruction_ref(object_backed=True)
    output = bind_output()
    asset = AssetVersionRef(
        AssetKey("skill", "review/guide.md"),
        "application",
        StorageEntryRevision(1),
        "a" * 64,
        23,
    )
    contract = SkillDefinition(
        SkillSpec("review", "review instructions"),
        SkillSourceRef("application", "review").with_asset_versions(
            (SkillResourceVersion("guide.md", asset),),
            "d" * 64,
        ),
    ).contract
    source = contract["source"]
    assert isinstance(source, dict)
    source["future_metadata"] = {"version": 2}
    binding = AgentBindingSnapshot(
        agent_spec=AgentSpec("agent", model_route="model"),
        base_model={"route_id": "model", "model_identity": "test:model"},
        selected=(CapabilityPin("skill", "review", contract),),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
    )
    execution = replace(_execution(reference), binding=binding)

    refs = tuple(
        iter_runtime_object_refs(
            _encode_persisted_domain(execution),
            default_domain=RuntimeDomain.EXECUTION,
        )
    )

    assert refs == ((RuntimeDomain.EXECUTION, reference.payload.ref),)


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
