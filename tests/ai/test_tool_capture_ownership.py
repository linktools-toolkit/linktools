#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Captured tool declarations own execution semantics, not authoring SDK objects."""
from ._runtime_test_helpers import model_features

import pytest
from pydantic_ai import Agent, DeferredToolRequests, RunContext, Tool
from pydantic_ai.messages import (
    ModelMessage, ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition

from linktools.ai.agent import AgentCompiler
from linktools.ai.capability import CapabilityContribution, tool_metadata
from linktools.ai.core import ExecutionStatus, JsonValue
from linktools.ai.runtime import Runtime, RuntimeFeatures, RuntimeStorage
from linktools.ai.spec import AgentSpec


class _ModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:tool-capture"
    contract: dict[str, JsonValue] = {"provider": "test", "model": "tool-capture"}

    def __init__(self) -> None:
        self.model = TestModel(call_tools=["lookup"], custom_output_text="done")

    def materialize(self) -> TestModel:
        return self.model


@pytest.mark.asyncio
async def test_runtime_uses_registered_tool_semantics_after_authoring_mutation() -> None:
    features = RuntimeFeatures("capture")
    calls: list[str] = []

    def lookup(ctx: object, key: str) -> str:
        """Original model-visible description."""
        calls.append(key)
        return key

    source = features.tool(lookup, effect_policy="none")
    source.description = "Changed before capture"
    spec = features.agent(
        "default", allow_tools=("lookup",), allow_skills=(),
        allow_subagents=(), allow_capabilities=(),
    )
    model = _ModelBinding()
    features.model("default", model)
    capture = await features.capture()
    candidate = next(item for item in capture.contributions if item.kind == "tool")
    original = candidate.contract
    source.function_schema.json_schema["properties"]["key"]["description"] = "Changed schema"
    candidate.value.description = "Changed through captured value"
    candidate.value.timeout = 0.000001
    compiler = AgentCompiler(
        models=capture.models, candidates=capture.contributions,
        agents={spec.id: spec},
    )
    binding = compiler.bind(compiler.compile(spec))
    restored = compiler.restore(binding.binding_contract)
    assert restored.binding_contract == binding.binding_contract

    async with Runtime.open(
        "capture", storage=RuntimeStorage.in_memory(), features=capture,
    ) as runtime:
        result = (await runtime.agents.get().run("Call lookup", timeout_seconds=5)).result
        assert result.status is ExecutionStatus.SUCCEEDED

    definition = next(
        tool for tool in model.model.last_model_request_parameters.function_tools
        if tool.name == "lookup"
    )
    assert definition.description == original["description"]
    assert definition.parameters_json_schema == original["parameters"]
    assert definition.timeout == original.get("timeout")
    assert calls


@pytest.mark.parametrize("include_return_schema", (None, False, True))
def test_materialized_tool_contract_and_mutable_views_are_isolated(
    include_return_schema: bool | None,
) -> None:
    def lookup(value: int, /) -> str:
        return str(value)

    source = Tool(
        lookup,
        description="Captured",
        max_retries=2,
        strict=True,
        sequential=True,
        requires_approval=True,
        timeout=2.0,
        defer_loading=True,
        include_return_schema=include_return_schema,
        metadata=tool_metadata(
            effect_policy="replay_safe", tool_class="business",
            base={"upstream": {"labels": ["original"]}},
        ),
    )
    candidate = CapabilityContribution.from_opaque("tool", "lookup", source, revision=3)
    original = candidate.contract
    source.name = "changed"
    source.description = "changed"
    source.max_retries = 0
    source.strict = False
    source.sequential = False
    source.requires_approval = False
    source.timeout = None
    source.defer_loading = False
    source.include_return_schema = not include_return_schema
    source.metadata["upstream"]["labels"].append("changed")
    source.function_schema.json_schema["properties"]["value"]["type"] = "string"
    source.function_schema.return_schema["type"] = "object"
    source.function_schema.positional_fields.clear()

    materialized = candidate.materialize_tool()
    assert CapabilityContribution.from_opaque(
        "tool", "lookup", materialized, revision=3,
    ).contract == original
    materialized.metadata["upstream"]["labels"].append("execution annotation")
    materialized.function_schema.json_schema["properties"]["value"]["type"] = "boolean"
    materialized.function_schema.return_schema["type"] = "array"
    materialized.function_schema.positional_fields.clear()
    fresh = candidate.materialize_tool()
    assert CapabilityContribution.from_opaque(
        "tool", "lookup", fresh, revision=3,
    ).contract == original
    assert fresh.function_schema.positional_fields == ["value"]


@pytest.mark.asyncio
async def test_restored_tool_preserves_validator_callable_and_prepare_hook() -> None:
    executed: list[int] = []
    validated: list[int] = []

    async def checked(value: int, /) -> str:
        executed.append(value)
        return f"accepted {value}"

    async def validate(_ctx: RunContext[None], value: int) -> None:
        validated.append(value)

    async def prepare(
        _ctx: RunContext[None], definition: ToolDefinition,
    ) -> ToolDefinition:
        assert definition.description == "Captured"
        assert definition.return_schema == {"type": "string"}
        assert definition.include_return_schema is True
        definition.description = "Prepared"
        definition.parameters_json_schema["properties"]["value"]["description"] = "Prepared argument"
        return definition

    source = Tool(
        checked, description="Captured", args_validator=validate, prepare=prepare,
        max_retries=1, include_return_schema=True,
        metadata=tool_metadata(effect_policy="none", tool_class="business"),
    )
    candidate = CapabilityContribution.from_opaque("tool", "checked", source)
    spec = AgentSpec("default", allow_tools=("checked",))
    models = {"default": _ModelBinding()}
    compiler = AgentCompiler(
        models=models, candidates=(candidate,), agents={spec.id: spec},
    )
    original_binding = compiler.bind(compiler.compile(spec))

    def changed(value: str) -> str:
        pytest.fail("mutated implementation must not execute")

    source.function = changed
    source.function_schema.function = changed
    source.function_schema.validator = Tool(changed).function_schema.validator
    source.function_schema.positional_fields.clear()
    source.prepare = None
    source.args_validator = None
    source.max_retries = 0
    source.function_schema.json_schema["properties"]["value"]["type"] = "string"
    restored = compiler.restore(original_binding.binding_contract)
    restored_tool = restored.compiled_agent.selected_tools[0]

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        definition = info.function_tools[0]
        assert definition.description.startswith("Prepared")
        assert definition.parameters_json_schema["properties"]["value"]["type"] == "integer"
        assert definition.parameters_json_schema["properties"]["value"]["description"] == "Prepared argument"
        assert definition.include_return_schema is True
        last = messages[-1].parts[0]
        if isinstance(last, RetryPromptPart):
            assert last.content[0]["type"] == "int_parsing"
            return ModelResponse(parts=[ToolCallPart("checked", {"value": "7"})])
        if isinstance(last, ToolReturnPart):
            assert last.content == "accepted 7"
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("checked", {"value": "not an integer"})])

    for _ in range(2):
        tool = restored_tool.materialize_tool()
        assert tool.tool_def.description == candidate.contract["description"] == "Captured"
        assert tool.tool_def.parameters_json_schema == candidate.contract["parameters"]
        agent = Agent(FunctionModel(respond), tools=[tool])
        assert (await agent.run("Use checked")).output == "done"
    assert validated == [7, 7]
    assert executed == [7, 7]
    assert restored_tool.contract == candidate.contract
    assert "description" not in candidate.contract["parameters"]["properties"]["value"]


@pytest.mark.asyncio
async def test_captured_tool_approval_is_not_removed_by_authoring_mutation() -> None:
    async def checked(value: int) -> str:
        pytest.fail("approval must precede execution")

    source = Tool(
        checked, requires_approval=True,
        metadata=tool_metadata(effect_policy="non_replay_safe", tool_class="business"),
    )
    candidate = CapabilityContribution.from_opaque("tool", "checked", source)
    source.requires_approval = False
    agent = Agent(
        TestModel(call_tools=["checked"]),
        tools=[candidate.materialize_tool()],
        output_type=[str, DeferredToolRequests],
    )
    result = await agent.run("Use checked")
    assert isinstance(result.output, DeferredToolRequests)
    assert [call.tool_name for call in result.output.approvals] == ["checked"]
