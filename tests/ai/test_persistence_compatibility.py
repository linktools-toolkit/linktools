#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from linktools.ai.agent import AgentBindingSnapshot, SemanticPin, bind_output, restore_output
from linktools.ai.capability import workspace_tool_contributions
from linktools.ai.core import IdempotencyStatus, JsonValue, OperationStatus, canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._message import decode_model_messages, encode_model_messages
from linktools.ai.runtime.state import _codec as runtime_codec
from linktools.ai.runtime.state._contracts import (
    ContextProjection,
    IdempotencyTerminalUpdate,
    OperationTerminalUpdate,
)
from linktools.ai.spec import AgentSpec
from linktools.ai.task import TaskNode
from linktools.ai.workspace import Workspace
from pydantic_ai.messages import ModelRequest, UserPromptPart

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "persistence"


def _load_json(name: str) -> object:
    return json.loads((_FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _binding_fixture_value() -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("runtime-persistence-v1"),
        model={"route_id": "default", "model_identity": "fixture:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
        binding_digest="c" * 64,
    )


def test_agent_binding_v1_fixture_and_additive_fields() -> None:
    value = _load_json("runtime_agent_binding_snapshot_v1.json")
    expected = _binding_fixture_value()
    assert value == expected.to_payload()
    decoded = AgentBindingSnapshot.from_payload(value)
    assert decoded == expected
    assert decoded.to_payload() == value

    additive = dict(cast(Mapping[str, object], value))
    additive["future_metadata"] = {"$future_v2": ["must", "not", "decode"]}
    assert AgentBindingSnapshot.from_payload(additive) == decoded


def test_output_binding_round_trips_from_durable_semantics() -> None:
    binding = bind_output()
    restored = restore_output(binding.mode, binding.schema_definition)
    assert restored == binding
    assert restored.fingerprint == binding.fingerprint


def _model_message_values() -> tuple[ModelRequest, ...]:
    fixed = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return (
        ModelRequest(
            parts=[
                UserPromptPart(
                    content="runtime-persistence-v1",
                    timestamp=fixed,
                )
            ]
        ),
    )


def test_model_message_v1_fixture() -> None:
    value = _load_json("runtime_model_messages_v1.json")
    expected = json.loads(encode_model_messages(_model_message_values()).decode("utf-8"))
    assert value == expected
    decoded = decode_model_messages(canonical_json_bytes(cast(JsonValue, value)))
    assert decoded == _model_message_values()


def _custom_wire_values() -> dict[str, JsonValue]:
    task_node = TaskNode(
        "node",
        ("dependency",),
        input={"key": "value"},
        budget_cost=2,
    )
    task_wire = cast(dict[str, JsonValue], runtime_codec.encode_domain(task_node))
    task_wire = dict(task_wire)
    task_wire["schema"] = runtime_codec.CURRENT_DATA_VERSION
    idempotency = IdempotencyTerminalUpdate(
        scope="scope",
        idempotency_key_digest="a" * 64,
        expected_status=IdempotencyStatus.STARTED,
        next_status=IdempotencyStatus.COMPLETED,
        request_digest="b" * 64,
        result_digest="c" * 64,
        error_code="terminal-error",
    )
    operation = OperationTerminalUpdate(
        operation_id="operation",
        expected_status=OperationStatus.RUNNING,
        next_status=OperationStatus.SUCCEEDED,
        result_ref="result",
        result_digest="d" * 64,
        error_code="terminal-error",
    )
    version = runtime_codec.CURRENT_DATA_VERSION
    return {
        f"task_node@{version}": task_wire,
        f"execution_terminal_commit@{version}:idempotency_terminal_update": runtime_codec.encode_domain(idempotency),
        f"execution_terminal_commit@{version}:operation_terminal_update": runtime_codec.encode_domain(operation),
    }


def _decode_custom_wire_values(
    value: Mapping[str, object],
) -> tuple[TaskNode, IdempotencyTerminalUpdate, OperationTerminalUpdate]:
    version = runtime_codec.CURRENT_DATA_VERSION
    task_wire = value[f"task_node@{version}"]
    if not isinstance(task_wire, Mapping) or task_wire.get("schema") != version:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    task_payload = dict(task_wire)
    task_payload.pop("schema")
    task = runtime_codec.decode_domain(cast(JsonValue, task_payload), TaskNode)
    idempotency = runtime_codec.decode_domain(
        cast(JsonValue, value[f"execution_terminal_commit@{version}:idempotency_terminal_update"]),
        IdempotencyTerminalUpdate,
    )
    operation = runtime_codec.decode_domain(
        cast(JsonValue, value[f"execution_terminal_commit@{version}:operation_terminal_update"]),
        OperationTerminalUpdate,
    )
    return task, idempotency, operation


def test_custom_wire_v1_fixture_and_additive_fields() -> None:
    value = _load_json("runtime_custom_wire_v1.json")
    assert isinstance(value, Mapping)
    expected = _custom_wire_values()
    assert value == expected
    assert _decode_custom_wire_values(value) == _decode_custom_wire_values(expected)

    additive = dict(expected)
    version = runtime_codec.CURRENT_DATA_VERSION
    task_key = f"task_node@{version}"
    task_wire = dict(cast(Mapping[str, object], additive[task_key]))
    task_fields = dict(cast(Mapping[str, object], task_wire["fields"]))
    task_fields["future_field"] = {"$future_v2": ["must", "not", "decode"]}
    task_wire["fields"] = task_fields
    additive[task_key] = cast(JsonValue, task_wire)
    additive_task, _, _ = _decode_custom_wire_values(additive)
    expected_task, _, _ = _decode_custom_wire_values(expected)
    assert additive_task == expected_task


def test_generic_v1_envelope_tolerates_additive_dataclass_fields() -> None:
    value = ContextProjection((), "d" * 64)
    payload = runtime_codec._encode_persisted_domain(value)
    canonical_json_bytes(payload)
    assert isinstance(payload, Mapping)
    fields_value = payload.get("fields")
    assert isinstance(fields_value, Mapping)

    additive = dict(payload)
    additive_fields = dict(fields_value)
    additive_fields["future_field"] = {"$future_v2": ["must", "not", "decode"]}
    additive["fields"] = additive_fields
    decoded = runtime_codec._decode_enveloped_domain(
        runtime_codec.encode_envelope(
            {
                "type": runtime_codec.wire_type_id(value),
                "payload": cast(JsonValue, additive),
            }
        ),
        ContextProjection,
    )
    assert decoded == value


def test_workspace_tool_semantics_match_v1_fixture(tmp_path: Path) -> None:
    expected = _load_json("workspace_tool_semantics_v1.json")
    actual = {
        contribution.id: contribution.semantic_contract
        for contribution in workspace_tool_contributions(Workspace.load(tmp_path))
    }
    assert actual == expected


def test_workspace_tool_pin_excludes_runtime_capability_provenance(tmp_path: Path) -> None:
    contribution = workspace_tool_contributions(Workspace.load(tmp_path))[0]
    pin = SemanticPin(
        "tool",
        contribution.id,
        cast(int, contribution.semantic_contract["version"]),
        contribution.semantic_contract,
    )
    payload = pin.to_payload()
    assert set(payload) == {"kind", "id", "contract_version", "contract"}
    assert "capability_id" not in cast(Mapping[str, object], payload["contract"])
