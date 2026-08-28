#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core JSON, Agent binding, Task, and Recovery boundary contracts."""

from collections.abc import Callable
from dataclasses import fields
from datetime import date, datetime, timezone
from enum import Enum, IntEnum

import pytest
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.core import (
    JsonValue,
    OperationLedgerRecord,
    Principal,
    PrincipalKind,
    ResourceKind,
    ResourceRef,
    normalize_json_value,
    validate_external_payload,
    validate_observation_payload,
    validate_tool_arguments,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import ExecutionRequest
from linktools.ai.capability import SubagentCapability
from linktools.ai.runtime._subagent_adapter import _PydanticSubagentCapability
from linktools.ai.runtime._tool import ToolOperationRecord
from linktools.ai.runtime.state import RuntimeStatePlan
from linktools.ai.runtime.state._codec import decode_domain, encode_domain
from linktools.ai.runtime.state._contracts import (
    RecoveryCheckpoint,
    RecoveryCheckpointState,
    RecoveryExecutionInput,
    RecoveryHandoffPhase,
    RecoveryIdempotencyInput,
)
from linktools.ai.spec import AgentSpec, SubagentRef
from linktools.ai.storage import StoredPayload
from linktools.ai.task import TaskGraph, TaskLease, TaskNode
from linktools.ai.workspace import trusted_workspace_principal


class _JsonEnum(Enum):
    VALUE = "value"


class _JsonStrEnum(str, Enum):
    __str__ = str.__str__
    __format__ = str.__format__
    VALUE = "value"


class _JsonIntEnum(IntEnum):
    VALUE = 1


def _binding_snapshot(
    *,
    agent_id: str = "default",
    digest: str = "a" * 64,
) -> AgentBindingSnapshot:
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec(agent_id, model="route"),
        model={"version": 1, "id": "route"},
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        binding_digest=digest,
    )


def test_normalize_json_value_detaches_nested_values() -> None:
    source: dict[str, object] = {"items": [{"value": 1}], "text": "ok"}
    normalized = normalize_json_value(source)
    source["items"][0]["value"] = 2  # type: ignore[index]
    assert normalized == {"items": [{"value": 1}], "text": "ok"}
    assert normalized is not source


@pytest.mark.parametrize(
    "value",
    [
        ("tuple",),
        {"set"},
        frozenset({"frozen"}),
        b"bytes",
        bytearray(b"bytearray"),
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        date(2026, 1, 1),
        _JsonEnum.VALUE,
        _JsonStrEnum.VALUE,
        _JsonIntEnum.VALUE,
        {1: "non-string key"},
        float("nan"),
        float("inf"),
        float("-inf"),
        object(),
    ],
)
def test_normalize_json_value_rejects_non_json_runtime_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalize_json_value(value)


def test_normalize_json_value_rejects_cycles() -> None:
    value: list[object] = []
    value.append(value)
    with pytest.raises(ValueError):
        normalize_json_value(value)


@pytest.mark.parametrize(
    "validator",
    [validate_external_payload, validate_tool_arguments, validate_observation_payload],
)
def test_json_validators_return_detached_normalized_values(
    validator: Callable[[JsonValue], JsonValue],
) -> None:
    source: dict[str, JsonValue] = {"items": [{"value": 1}]}
    normalized = validator(source)
    source["items"][0]["value"] = 2  # type: ignore[index]
    assert normalized == {"items": [{"value": 1}]}


def test_stored_payload_inline_json_detaches_source() -> None:
    source: dict[str, JsonValue] = {"items": [{"value": 1}]}
    payload = StoredPayload.inline_json(source)
    source["items"][0]["value"] = 2  # type: ignore[index]
    assert payload == StoredPayload.inline_json({"items": [{"value": 1}]})


def test_task_graph_rejects_cycles_and_agent_spec_is_stable() -> None:
    with pytest.raises(ValueError):
        TaskGraph("cycle", (TaskNode("a", ("b",)), TaskNode("b", ("a",))))
    spec = AgentSpec(
        "agent",
        model="route",
        system_prompt="system",
        instructions=("answer",),
        allow_tools=("bash",),
    )
    assert spec == AgentSpec(
        "agent",
        model="route",
        system_prompt="system",
        instructions=("answer",),
        allow_tools=("bash",),
    )


def test_model_registry_snapshot_is_instance_owned() -> None:
    registry = ModelRegistry()
    registry.register_openai("route", model="model")
    snapshot = registry.snapshot()
    assert snapshot.resolve("route").model_identity == "openai:model"
    assert snapshot.resolve("route").route_id == "route"


def test_runtime_state_plan_rejects_an_invalid_domain() -> None:
    with pytest.raises(ValueError):
        RuntimeStatePlan(conversation="invalid")  # type: ignore[arg-type]


def test_recovery_checkpoint_enforces_v1_execution_identity() -> None:
    now = datetime.now(timezone.utc)
    snapshot = _binding_snapshot(digest="c" * 64)
    recovery_input = RecoveryExecutionInput(
        user_prompt="prompt",
        user_prompt_codec="text",
        principal_id="principal",
        principal_kind="user",
        session_id=None,
        memory_scope=None,
        binding_digest=snapshot.binding_digest,
        lineage_kind="RUN",
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        conversation_step_run_id=None,
        idempotency=RecoveryIdempotencyInput("scope", "key", "request"),
        mode="run",
        planning=False,
        thinking=False,
        binding=snapshot,
    )

    def checkpoint(
        state: RecoveryCheckpointState,
        sequence: int,
        step_run_id: str | None,
    ) -> RecoveryCheckpoint:
        return RecoveryCheckpoint(
            execution_id="execution",
            tenant_id="tenant",
            input=recovery_input,
            step_run_id=step_run_id,
            agent_run_sequence=sequence,
            state=state,
            handoff_phase=RecoveryHandoffPhase.NONE,
            terminal_handoff=None,
            handoff_contract_digest=None,
            pending_operation_id=None,
            revision=0,
            created_at=now,
            updated_at=now,
        )

    with pytest.raises(ValueError):
        checkpoint(RecoveryCheckpointState.ADMITTED, 1, None)
    with pytest.raises(ValueError):
        checkpoint(RecoveryCheckpointState.ACTIVE, 0, None)
    assert checkpoint(RecoveryCheckpointState.COMPLETED, 0, None).agent_run_sequence == 0


def test_recovery_input_requires_exact_binding_digest_and_mode_contract() -> None:
    snapshot = _binding_snapshot(digest="d" * 64)
    with pytest.raises(ValueError):
        RecoveryExecutionInput(
            user_prompt="prompt",
            user_prompt_codec="text",
            principal_id="principal",
            principal_kind="user",
            session_id=None,
            memory_scope=None,
            binding_digest="e" * 64,
            lineage_kind="RUN",
            parent_execution_id=None,
            root_execution_id="execution",
            source_execution_id=None,
            base_execution_id=None,
            conversation_step_run_id=None,
            idempotency=RecoveryIdempotencyInput("scope", "key", "request"),
            mode="run",
            planning=False,
            thinking=False,
            binding=snapshot,
        )
    with pytest.raises(ValueError):
        RecoveryExecutionInput(
            user_prompt="prompt",
            user_prompt_codec="text",
            principal_id="principal",
            principal_kind="user",
            session_id=None,
            memory_scope=None,
            binding_digest=snapshot.binding_digest,
            lineage_kind="RUN",
            parent_execution_id=None,
            root_execution_id="execution",
            source_execution_id=None,
            base_execution_id=None,
            conversation_step_run_id=None,
            idempotency=RecoveryIdempotencyInput("scope", "key", "request"),
            mode="plan",
            planning=False,
            thinking=False,
            binding=snapshot,
        )


def test_domain_codec_preserves_mapping_payloads_in_nullable_json_values() -> None:
    payload = StoredPayload.inline_json({"text": "你好！"})
    assert decode_domain(encode_domain(payload), StoredPayload) == payload


def test_subagent_tool_schema_accepts_json_payload() -> None:
    async def delegate(
        _ref: SubagentRef,
        _task: str,
        *,
        invocation_id: str,
    ) -> dict[str, JsonValue]:
        assert invocation_id
        return {
            "execution_id": "child",
            "status": "SUCCEEDED",
            "output": {"value": True},
        }

    capability = _PydanticSubagentCapability(
        SubagentCapability((SubagentRef("agent", "child"),), delegate)
    )
    assert capability.get_toolset() is not None


def test_authorization_kinds_are_canonical() -> None:
    assert Principal("principal", "tenant", "service").kind == PrincipalKind.SERVICE.value
    assert Principal("principal", "tenant", "custom").kind == "custom"
    assert ResourceRef(ResourceKind.EXECUTION.value, "execution", "tenant").kind is ResourceKind.EXECUTION
    with pytest.raises(ValueError, match="principal identity is invalid"):
        Principal("principal", "tenant", " custom")


def test_contextual_classification_fields_stay_concise() -> None:
    operation_fields = {field.name for field in fields(OperationLedgerRecord)}
    task_lease_fields = {field.name for field in fields(TaskLease)}
    tool_operation_fields = {field.name for field in fields(ToolOperationRecord)}
    assert "operation_kind" in operation_fields and "kind" not in operation_fields
    assert "owner" in task_lease_fields and "lease_owner" not in task_lease_fields
    assert "owner" in tool_operation_fields and "lease_owner" not in tool_operation_fields
    assert "binding_digest" in tool_operation_fields
    assert "binding_fingerprint" not in tool_operation_fields


@pytest.mark.parametrize("value", [" memory", "memory ", "memory\nvalue", "界" * 129])
def test_memory_scope_rejects_noncanonical_values(value: str) -> None:
    with pytest.raises(AIError) as error:
        ExecutionRequest(
            "prompt",
            "text",
            trusted_workspace_principal("workspace"),
            "request",
            value,
            "run",
            False,
            False,
        )
    assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID
