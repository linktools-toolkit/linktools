#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest
from linktools.ai.agent import AgentBindingSnapshot, AgentCompiler, SemanticPin, bind_output, restore_output
from linktools.ai.capability import CapabilityGroup, workspace_capabilities
from linktools.ai.core import IdempotencyStatus, JsonValue, OperationStatus, canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime._message import decode_model_messages, encode_model_messages
from linktools.ai.runtime.state import _codec as runtime_codec
from linktools.ai.runtime.state._contracts import (
    ContextProjection,
    IdempotencyTerminalUpdate,
    OperationTerminalUpdate,
)
from linktools.ai.spec import AgentSpec
from linktools.ai.task import TaskNode
from linktools.ai.workspace import DisabledSandbox, Workspace
from pydantic_ai.messages import ModelRequest, UserPromptPart


def _workspace_tool_contributions(workspace: Workspace):
    return tuple(
        CapabilityGroup("workspace", workspace=workspace, discover_workspace_assets=False)._contributions
    )


_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "persistence"


def _load_json(name: str) -> object:
    return json.loads((_FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _binding_fixture_value() -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        agent_spec=AgentSpec("runtime-persistence-v1", tool_retries=10000),
        base_model={"route_id": "default", "model_identity": "fixture:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
    )


def test_agent_binding_fixture_matches_current_contract() -> None:
    value = _load_json("runtime_agent_binding_snapshot_v1.json")
    expected = _binding_fixture_value()
    assert value == expected.to_payload()
    decoded = AgentBindingSnapshot.from_payload(value)
    assert decoded == expected
    assert decoded.binding_digest == expected.binding_digest


def test_agent_binding_ignores_unknown_fields() -> None:
    value = cast(dict[str, object], _load_json("runtime_agent_binding_snapshot_v1.json"))
    value["future_metadata"] = {"future": True}

    decoded = AgentBindingSnapshot.from_payload(value)

    assert decoded == _binding_fixture_value()
    assert "future_metadata" not in decoded.to_payload()


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


def test_custom_wire_v1_fixture() -> None:
    value = _load_json("runtime_custom_wire_v1.json")
    assert isinstance(value, Mapping)
    expected = _custom_wire_values()
    assert value == expected
    assert _decode_custom_wire_values(value) == _decode_custom_wire_values(expected)


def test_generic_v1_envelope_round_trips_current_shape() -> None:
    value = ContextProjection(())
    payload = runtime_codec._encode_persisted_domain(value)
    canonical_json_bytes(payload)
    decoded = runtime_codec._decode_enveloped_domain(
        runtime_codec.encode_envelope(
            {
                "type": runtime_codec.wire_type_id(value),
                "payload": payload,
            }
        ),
        ContextProjection,
    )
    assert decoded == value


def test_workspace_tool_pin_contains_one_version_source(tmp_path: Path) -> None:
    contribution = _workspace_tool_contributions(Workspace.load(tmp_path, workspace_id="workspace"))[0]
    pin = SemanticPin(
        "tool",
        contribution.id,
        contribution.semantic_contract,
    )
    payload = pin.to_payload()
    assert set(payload) == {"kind", "id", "contract"}
    assert cast(Mapping[str, object], payload["contract"])["version"] == 1
    assert "capability_id" not in cast(Mapping[str, object], payload["contract"])


@pytest.mark.asyncio
async def test_workspace_tool_binding_restores_before_disabled_sandbox_materialization(
    tmp_path: Path,
) -> None:
    workspace = Workspace.load(tmp_path, workspace_id="workspace", sandbox=DisabledSandbox())
    candidates = _workspace_tool_contributions(workspace)
    spec = AgentSpec(
        "workspace-persistence-v1",
        model="default",
        allow_tools=("read_file",),
        allow_skills=(),
        allow_subagents=(),
    )
    models = ModelRegistry.openai(model="gpt-test").snapshot()
    compiler = AgentCompiler(
        model_resolver=models,
        candidates=candidates,
        agents={spec.id: spec},
    )
    binding = compiler.bind(compiler.compile(spec))
    baseline = {
        contribution.id: contribution.semantic_contract
        for contribution in candidates
    }
    assert len(binding.snapshot.selected) == 1
    pin = binding.snapshot.selected[0]
    assert pin.kind == "tool"
    assert pin.id == "read_file"
    assert dict(pin.contract) == baseline["read_file"]

    restored = AgentCompiler(
        model_resolver=models,
        candidates=candidates,
        agents={spec.id: spec},
    ).restore(binding.snapshot)
    assert restored.snapshot == binding.snapshot
    selected = tuple(candidate.id for candidate in restored.definition.selected_tools)
    with pytest.raises(AIError) as missing_session:
        workspace_capabilities(workspace, selected)
    assert missing_session.value.code is ErrorCode.SANDBOX_SESSION_CLOSED
    with pytest.raises(AIError) as raised:
        await workspace.sandbox.open(root=workspace.root)  # type: ignore[union-attr]
    assert raised.value.code is ErrorCode.SANDBOX_UNAVAILABLE


def _environment_compiler(
    workspace_ref: "Mapping[str, JsonValue] | None",
) -> tuple[AgentCompiler, AgentSpec]:
    spec = AgentSpec(
        "environment",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
        allow_capabilities=(),
    )
    return (
        AgentCompiler(
            model_resolver=ModelRegistry.openai(model="gpt-test").snapshot(),
            candidates=(),
            agents={spec.id: spec},
            namespace="runtime",
            workspace_ref=workspace_ref,
        ),
        spec,
    )


def test_workspace_ref_distinguishes_legacy_workspace_and_workspace_less() -> None:
    legacy_compiler, spec = _environment_compiler(None)
    legacy = legacy_compiler.bind(legacy_compiler.compile(spec))
    assert legacy.snapshot.workspace_ref is None
    assert "workspace_ref" not in legacy.snapshot.to_payload()
    assert legacy_compiler.restore(legacy.snapshot).digest == legacy.digest

    no_workspace_compiler, spec = _environment_compiler({"id": None})
    no_workspace = no_workspace_compiler.bind(no_workspace_compiler.compile(spec))
    assert no_workspace.snapshot.to_payload()["workspace_ref"] == {"id": None}
    assert no_workspace_compiler.restore(no_workspace.snapshot).digest == no_workspace.digest

    with pytest.raises(AIError) as missing_workspace:
        no_workspace_compiler.restore(legacy.snapshot)
    assert missing_workspace.value.code is ErrorCode.AGENT_DEFINITION_UNAVAILABLE
    assert missing_workspace.value.safe_details == {"reason": "workspace_mismatch"}

    with pytest.raises(AIError) as extra_workspace:
        legacy_compiler.restore(no_workspace.snapshot)
    assert extra_workspace.value.code is ErrorCode.AGENT_DEFINITION_UNAVAILABLE
    assert extra_workspace.value.safe_details == {"reason": "workspace_mismatch"}


def test_workspace_ref_requires_exact_stable_workspace_id() -> None:
    project_compiler, spec = _environment_compiler({"id": "project-a"})
    binding = project_compiler.bind(project_compiler.compile(spec))
    assert binding.snapshot.to_payload()["workspace_ref"] == {"id": "project-a"}
    assert project_compiler.restore(binding.snapshot).digest == binding.digest

    other_compiler, _ = _environment_compiler({"id": "project-b"})
    with pytest.raises(AIError) as mismatch:
        other_compiler.restore(binding.snapshot)
    assert mismatch.value.code is ErrorCode.AGENT_DEFINITION_UNAVAILABLE
    assert mismatch.value.safe_details == {"reason": "workspace_mismatch"}


@pytest.mark.parametrize(
    "workspace_ref",
    ({}, {"id": 1}),
)
def test_workspace_ref_rejects_invalid_durable_shape(
    workspace_ref: Mapping[str, object],
) -> None:
    payload = _binding_fixture_value().to_payload()
    payload["workspace_ref"] = cast(JsonValue, dict(workspace_ref))

    with pytest.raises(AIError) as raised:
        AgentBindingSnapshot.from_payload(payload)

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
