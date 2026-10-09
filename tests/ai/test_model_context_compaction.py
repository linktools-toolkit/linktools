#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model windows guide compaction without turning estimates into request limits."""

from copy import deepcopy
from dataclasses import replace

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    BinaryContent,
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import OutputObjectDefinition
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RequestUsage, RunUsage

from linktools.ai.core import PromptLimits
from linktools.ai.runtime._compaction import CompactionCapability
from linktools.ai.runtime._journal import ModelRequestJournal


def _tool_history(result_chars: int = 400) -> list[ModelMessage]:
    messages: list[ModelMessage] = []
    for index in range(5):
        messages.extend((
            ModelResponse(parts=[ToolCallPart("inspect", {}, f"call-{index}")]),
            ModelRequest(parts=[ToolReturnPart("inspect", "x" * result_chars, f"call-{index}")]),
        ))
    messages.append(ModelRequest(parts=[UserPromptPart("continue")]))
    return messages


async def _project(
    model: TestModel,
    *,
    messages: list[ModelMessage] | None = None,
    settings: ModelSettings | None = None,
    parameters: ModelRequestParameters | None = None,
    target: int | None = None,
) -> list[ModelMessage]:
    source = _tool_history() if messages is None else messages
    original = deepcopy(source)
    captured: list[ModelRequestContext] = []
    request = ModelRequestContext(
        model=model,
        messages=source,
        model_settings=settings,
        model_request_parameters=parameters or ModelRequestParameters(),
    )
    ctx = RunContext(deps=None, model=model, usage=RunUsage(), messages=list(source), run_id="run")

    async def handler(current: ModelRequestContext) -> ModelResponse:
        captured.append(current)
        return ModelResponse(parts=[TextPart("done")])

    await CompactionCapability(target, limits=PromptLimits()).wrap_model_request(
        ctx, request_context=request, handler=handler,
    )
    assert source == original
    assert ctx.messages == original
    return captured[0].messages


@pytest.mark.asyncio
@pytest.mark.parametrize(("window", "target", "cleared"), [
    (None, None, False),
    (None, 300, True),
    (300, None, True),
    (300, 10_000, True),
    (10_000, 300, True),
    (750, None, False),
])
async def test_known_windows_and_explicit_targets_select_compaction(
    window: int | None, target: int | None, cleared: bool,
) -> None:
    messages = await _project(TestModel(profile={"context_window": window}), target=target)
    assert ("[tool result cleared]" in str(messages)) is cleared


@pytest.mark.asyncio
@pytest.mark.parametrize(("defaults", "overrides", "cleared"), [
    ({"max_tokens": 300}, None, True),
    ({"max_tokens": 300}, {"max_tokens": 10}, False),
    ({"max_tokens": 10}, {"max_tokens": 300}, True),
])
async def test_output_reserve_uses_effective_model_settings(
    defaults: ModelSettings, overrides: ModelSettings | None, cleared: bool,
) -> None:
    messages = await _project(
        TestModel(profile={"context_window": 750}, settings=defaults), settings=overrides,
    )
    assert ("[tool result cleared]" in str(messages)) is cleared


@pytest.mark.asyncio
@pytest.mark.parametrize("request_content", ["instructions", "function", "output_tool", "output_schema"])
async def test_request_configuration_participates_in_first_request_estimate(request_content: str) -> None:
    parameters = ModelRequestParameters()
    if request_content == "instructions":
        parameters.instruction_parts = [InstructionPart(content="i" * 1600, dynamic=True)]
    elif request_content == "function":
        parameters.function_tools = [ToolDefinition(name="inspect", description="d" * 1600)]
    elif request_content == "output_tool":
        parameters.output_tools = [ToolDefinition(name="answer", description="d" * 1600, kind="output")]
        parameters.output_mode = "tool"
    else:
        parameters.output_object = OutputObjectDefinition(
            json_schema={"type": "object", "description": "d" * 1600},
        )
        parameters.output_mode = "native"
    messages = await _project(TestModel(profile={"context_window": 750}), parameters=parameters)
    assert "[tool result cleared]" in str(messages)


@pytest.mark.asyncio
async def test_usage_anchor_is_not_charged_for_the_same_schemas_twice() -> None:
    messages = _tool_history(40)
    response = messages[-3]
    assert isinstance(response, ModelResponse)
    response.usage = RequestUsage(input_tokens=700, output_tokens=10)
    parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name="inspect", description="d" * 2000)],
    )
    projected = await _project(
        TestModel(profile={"context_window": 1000}), messages=messages, parameters=parameters,
    )
    assert projected == messages


@pytest.mark.asyncio
async def test_auto_output_schema_alternatives_are_not_charged_twice() -> None:
    schema = {"type": "object", "description": "d" * 2000}
    parameters = ModelRequestParameters(
        output_mode="auto",
        output_tools=[ToolDefinition(name="answer", parameters_json_schema=schema, kind="output")],
        output_object=OutputObjectDefinition(name="answer", json_schema=schema),
    )
    messages = _tool_history(40)
    projected = await _project(
        TestModel(profile={"context_window": 750}), messages=messages, parameters=parameters,
    )
    assert projected == messages


@pytest.mark.asyncio
async def test_unrevealed_tool_schemas_do_not_trigger_compaction() -> None:
    parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name="hidden", description="d" * 4000, defer_loading=True)],
    )
    projected = await _project(TestModel(profile={"context_window": 750}), parameters=parameters)
    assert "[tool result cleared]" not in str(projected)
    parameters = replace(parameters, revealed_tool_names={"hidden"})
    projected = await _project(TestModel(profile={"context_window": 750}), parameters=parameters)
    assert "[tool result cleared]" in str(projected)


@pytest.mark.asyncio
async def test_irreducible_estimate_and_binary_content_are_not_hard_token_limits() -> None:
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart([
        "x" * 10_000,
        BinaryContent(b"image" * 10_000, media_type="image/png"),
    ])])]
    projected = await _project(TestModel(profile={"context_window": 100}), messages=messages)
    assert projected == messages


class _PreparedTools(AbstractCapability[None]):
    async def prepare_tools(
        self, ctx: RunContext[None], tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        return [replace(tool, description="d" * 1600) for tool in tool_defs]


@pytest.mark.asyncio
@pytest.mark.parametrize("dynamic_content", ["instructions", "tool_schema"])
async def test_sdk_dynamic_request_configuration_is_resolved_before_compaction(dynamic_content: str) -> None:
    projected: list[tuple[ModelMessage, ...] | None] = []
    capability = CompactionCapability(
        None, limits=PromptLimits(), projection_sink=lambda _source, result: projected.append(result),
    )
    agent = Agent(
        TestModel(profile={"context_window": 750}, custom_output_text="done", call_tools=[]),
        capabilities=[capability, _PreparedTools()],
    )
    if dynamic_content == "instructions":
        @agent.instructions
        def instructions() -> str:
            return "i" * 1600
    else:
        @agent.tool_plain
        def inspect() -> str:
            return "value"
    result = await agent.run("continue", message_history=_tool_history()[:-1])
    assert result.output == "done"
    assert projected and projected[0] is not None
    assert "[tool result cleared]" in str(projected[0])


@pytest.mark.asyncio
async def test_repeated_automatic_summaries_preserve_raw_history_and_request_identity() -> None:
    model = TestModel(profile={"context_window": 1200}, settings={"max_tokens": 200}, custom_output_text="summary")
    journal = ModelRequestJournal(
        source_namespace="workspace", tenant_id="tenant", execution_id="execution", agent_run_id="run",
    )
    observed: list[tuple[int, str]] = []

    async def observer(ctx, fact, phase, selected_model, response, error) -> None:
        assert selected_model is model
        assert selected_model.context_window == 1200
        assert selected_model.settings == {"max_tokens": 200}
        assert error is None
        observed.append((fact.model_request_seq, phase))

    capability = CompactionCapability(None, limits=PromptLimits(), journal=journal, observer=observer)
    messages: list[ModelMessage] = []
    for _cycle in range(2):
        for index in range(20):
            messages.extend((
                ModelRequest(parts=[UserPromptPart(f"user {index} " + "x" * 240)]),
                ModelResponse(parts=[TextPart(f"answer {index} " + "y" * 240)]),
            ))
        messages.append(ModelRequest(parts=[UserPromptPart("continue")]))
        original = deepcopy(messages)
        captured: list[ModelRequestContext] = []

        async def handler(request: ModelRequestContext) -> ModelResponse:
            captured.append(request)
            return ModelResponse(parts=[TextPart("done")])

        await capability.wrap_model_request(
            RunContext(deps=None, model=model, usage=RunUsage(), messages=list(messages), run_id="run"),
            request_context=ModelRequestContext(
                model=model, messages=messages, model_settings=None, model_request_parameters=ModelRequestParameters(),
            ),
            handler=handler,
        )
        assert messages == original
        assert len(captured[0].messages) < len(messages)
        messages = captured[0].messages
    assert observed == [(1, "started"), (1, "completed"), (2, "started"), (2, "completed")]
