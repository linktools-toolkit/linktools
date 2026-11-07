#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistence regressions for instruction and deferred-work pins."""

from dataclasses import replace
from datetime import datetime, timezone

from linktools.ai.agent import AgentBindingContract
from linktools.ai.agent._output import bind_output
from linktools.ai.asset import AssetKey, AssetVersionRef
from linktools.ai.capability import SkillDefinition, SkillResource, SkillSourceRef
from linktools.ai.core import ExecutionLineageKind, ExecutionStatus
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence.codec import (
    _encode_persisted_domain,
    decode_domain,
    encode_domain,
    iter_runtime_object_refs,
)
from linktools.ai.persistence.contracts import (
    ExecutionRecord,
    PendingDeferredCall,
    PendingToolContinuation,
    ExecutionRecoveryDetail,
    ExecutionRecoveryState,
    RecoveryHandoffPhase,
    RuntimePayloadRef,
    StoredUserInput,
)
from linktools.ai.spec import AgentSpec, SkillSpec
from linktools.ai.storage import ObjectRef, StorageEntryRevision, StoredPayload


def _binding() -> AgentBindingContract:
    output = bind_output()
    return AgentBindingContract(
        agent_spec=AgentSpec("agent", model="model", tool_retries=3, output_retries=3),
        model_contract={"route_id": "model", "model_identity": "test:model"},
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
    return RuntimePayloadRef(payload, PersistenceDomain.EXECUTION)


def _execution(repository_instructions: RuntimePayloadRef | None) -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    binding = _binding()
    return ExecutionRecord(
        execution_id="execution",
        session_id=None,
        parent_execution_id=None,
        root_execution_id="execution",
        previous_execution_id=None,
        fork_base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=ExecutionStatus.PENDING_START,
        revision=0,
        event_seq=0,
        agent_run_seq=0,
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


def _checkpoint(pending_tools: PendingToolContinuation | None) -> ExecutionRecord:
    return replace(_execution(None),
                   status=ExecutionStatus.WAITING_DEFERRED if pending_tools is not None else ExecutionStatus.STARTED,
                   recovery=ExecutionRecoveryDetail(agent_run_id="step-1" if pending_tools is not None else None,
                                                    pending_tools=pending_tools))


def test_instruction_aware_execution_round_trips_exact_pin() -> None:
    reference = _instruction_ref()
    wire = encode_domain(_execution(reference))

    assert isinstance(wire, dict)
    assert wire["$dataclass"] == "execution_record"
    decoded = decode_domain(wire, ExecutionRecord)
    assert decoded.repository_instructions == reference


def test_skill_asset_fields_remain_separate_from_runtime_object_refs() -> None:
    reference = _instruction_ref(object_backed=True)
    asset = AssetVersionRef(
        AssetKey("skill", "review/guide.md"),
        "application",
        StorageEntryRevision(1),
        "a" * 64,
        23,
    )
    contract = SkillDefinition(
        SkillSpec("review", "review instructions"),
        SkillSourceRef(
            "application",
            "review",
            (SkillResource("guide.md", asset),),
        ),
    ).contract
    source = contract["source"]
    assert isinstance(source, dict)
    source["future_metadata"] = {"version": 2}
    restored = SkillDefinition.from_contract(contract)
    assert restored.source_ref.resources[0].asset == asset
    execution = _execution(reference)

    refs = tuple(
        iter_runtime_object_refs(
            _encode_persisted_domain(execution),
            default_domain=PersistenceDomain.EXECUTION,
        )
    )

    assert refs == ((PersistenceDomain.EXECUTION, reference.payload.ref),)


def test_deferred_frontier_round_trips_current_contract() -> None:
    current = _checkpoint(_continuation())
    wire = encode_domain(current)

    assert isinstance(wire, dict)
    assert wire["$dataclass"] == "execution_record"
    assert decode_domain(wire, ExecutionRecord) == current


def test_object_ref_traversal_finds_repository_instruction_object() -> None:
    reference = _instruction_ref(object_backed=True)
    execution = _execution(reference)
    refs = tuple(
        iter_runtime_object_refs(
            _encode_persisted_domain(execution),
            default_domain=PersistenceDomain.EXECUTION,
        )
    )
    assert refs == ((PersistenceDomain.EXECUTION, reference.payload.ref),)
