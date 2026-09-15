#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for the remaining runtime audit boundaries."""

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.capabilities import (
    AbstractCapability,
    ValidatedToolArgs,
    WrapToolExecuteHandler,
)
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RunUsage

from linktools.ai.agent._output import bind_output, canonicalize_output_schema_v1
from linktools.ai.capability import CapabilityGroup
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._agent_executor import _thinking_capability
from linktools.ai.runtime._tool import ToolOperationDecision
from linktools.ai.runtime._tool_boundary import (
    ManagedToolDescriptor,
    RuntimeToolBoundaryToolset,
)
from linktools.ai.spec import AgentSpec, AgentSpecCodec
from ._runtime_test_helpers import semantic_tool


class _GeneratedOutputAlpha(BaseModel):
    title: str


class _GeneratedOutputBeta(BaseModel):
    title: str


class _ExplicitOutputAlpha(BaseModel):
    model_config = ConfigDict(title="alpha-contract")
    value: str


class _ExplicitOutputBeta(BaseModel):
    model_config = ConfigDict(title="beta-contract")
    value: str


def test_output_fingerprint_ignores_generated_type_titles() -> None:
    alpha = bind_output(_GeneratedOutputAlpha)
    beta = bind_output(_GeneratedOutputBeta)

    assert alpha.fingerprint == beta.fingerprint
    assert alpha.schema_definition == beta.schema_definition
    assert alpha.schema_definition["properties"]["title"]["type"] == "string"


def test_output_fingerprint_preserves_explicit_schema_titles() -> None:
    assert bind_output(_ExplicitOutputAlpha).fingerprint != bind_output(
        _ExplicitOutputBeta
    ).fingerprint


def test_output_schema_literal_ref_is_not_treated_as_schema_ref() -> None:
    schema = {
        "type": "object",
        "properties": {
            "payload": {
                "const": {"$ref": "literal-value"},
            }
        },
    }

    normalized = canonicalize_output_schema_v1(schema)

    assert normalized["properties"]["payload"]["const"] == {
        "$ref": "literal-value"
    }


def test_agent_tool_retry_default_is_finite() -> None:
    assert AgentSpec("agent").tool_retries == 10000
    assert AgentSpecCodec().decode(b'{"version":1,"id":"agent"}').tool_retries == 10000
    assert CapabilityGroup("group").agent("agent").tool_retries == 10000


def _request_context(model: TestModel) -> ModelRequestContext:
    return ModelRequestContext(
        model=model,
        messages=[],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )


def _run_context(model: TestModel) -> RunContext[None]:
    return RunContext(
        deps=None,
        model=model,
        usage=RunUsage(),
        run_id="run",
    )


@pytest.mark.asyncio
async def test_thinking_validates_the_effective_request_model() -> None:
    capability = _thinking_capability(True)
    assert capability.get_ordering().position == "innermost"

    bootstrap = TestModel(profile={"supports_thinking": False})
    selected = TestModel(
        model_name="selected",
        profile={"supports_thinking": True},
    )
    seen: list[TestModel] = []

    async def handler(request_context: ModelRequestContext) -> ModelResponse:
        seen.append(request_context.model)  # type: ignore[arg-type]
        return ModelResponse(parts=[], model_name="selected")

    await capability.wrap_model_request(
        _run_context(bootstrap),
        request_context=_request_context(selected),
        handler=handler,
    )
    assert seen == [selected]

    rejected = TestModel(
        model_name="selected-without-thinking",
        profile={"supports_thinking": False},
    )
    with pytest.raises(AIError) as error:
        await capability.wrap_model_request(
            _run_context(TestModel(profile={"supports_thinking": True})),
            request_context=_request_context(rejected),
            handler=handler,
        )
    assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID
    assert error.value.safe_details["reason"] == "model_not_supported"


class _MutateToolArgs(AbstractCapability[None]):
    async def wrap_tool_execute(
        self,
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> object:
        del ctx, call, tool_def
        mutated = dict(args)
        mutated["value"] = "mutated"
        return await handler(mutated)


class _RecordingToolOperations:
    def __init__(self) -> None:
        self.arguments: list[dict[str, Any]] = []

    async def begin(
        self,
        ctx: RunContext[None],
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        replay_safe: bool,
    ) -> ToolOperationDecision:
        del ctx, call, tool_def
        self.arguments.append(dict(args))
        return ToolOperationDecision("operation", "owner", 1, replay_safe)

    async def complete(self, decision: ToolOperationDecision, result: object) -> bool:
        del decision, result
        return False


@pytest.mark.asyncio
async def test_tool_operation_records_the_args_that_reach_the_handler() -> None:
    executed: list[str] = []

    async def business(value: str) -> str:
        executed.append(value)
        return value

    operations = _RecordingToolOperations()
    descriptor = ManagedToolDescriptor(
        effect_owner="tool_operation",
        effect="replay_safe",
        tool_class="business",
    )
    boundary = RuntimeToolBoundaryToolset(
        (FunctionToolset([semantic_tool(business, descriptor)]),),
        {"business": descriptor},
        id="business",
        tool_operations=operations,  # type: ignore[arg-type]
    )
    agent = PydanticAgent(
        TestModel(call_tools=["business"], custom_output_text="done"),
        toolsets=(boundary,),
    )

    result = await agent.run("run", capabilities=(_MutateToolArgs(),))

    assert result.output == "done"
    assert operations.arguments == [{"value": "mutated"}]
    assert executed == ["mutated"]
