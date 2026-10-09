#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LinkTools tool signal and final Pydantic boundary contracts."""

import pytest
from linktools.ai.capability import ToolCallFailed, ToolCallRetry
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._pydantic_tool_control import (
    PydanticToolControlCapability,
    build_model_retry,
    build_tool_failed,
)
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.exceptions import ModelRetry, ToolFailed, UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition

from ._runtime_test_helpers import _UsageFunctionModel


@pytest.mark.parametrize("signal_type", (ToolCallRetry, ToolCallFailed))
def test_tool_signal_has_only_a_valid_message(signal_type: type[Exception]) -> None:
    signal = signal_type("valid message")

    assert isinstance(signal, Exception)
    assert not isinstance(signal, AIError)
    assert signal.message == "valid message"
    assert vars(signal) == {"message": "valid message"}
    assert not hasattr(signal, "code")
    assert not hasattr(signal, "retryable")
    assert not hasattr(signal, "safe_details")


@pytest.mark.parametrize("message", ("", "   ", "x" * 2049, 1, None))
def test_tool_signal_rejects_invalid_message(message: object) -> None:
    for signal_type in (ToolCallRetry, ToolCallFailed):
        with pytest.raises((TypeError, ValueError)):
            signal_type(message)  # type: ignore[arg-type]


def test_pydantic_tool_control_is_outermost() -> None:
    capability = PydanticToolControlCapability()

    assert capability.get_ordering().position == "outermost"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("signal", "expected_type"),
    (
        (ToolCallRetry("correct it"), ModelRetry),
        (ToolCallFailed("failed"), ToolFailed),
    ),
)
async def test_pydantic_tool_control_converts_only_linktools_signals(
    signal: ToolCallRetry | ToolCallFailed,
    expected_type: type[Exception],
) -> None:
    capability = PydanticToolControlCapability()

    with pytest.raises(expected_type) as raised:
        await capability.on_tool_execute_error(
            None,  # type: ignore[arg-type]
            call=ToolCallPart("tool", tool_call_id="call"),
            tool_def=ToolDefinition(name="tool"),
            args={},
            error=signal,
        )

    assert str(raised.value) == signal.message
    assert raised.value.__cause__ is signal


@pytest.mark.asyncio
async def test_pydantic_tool_control_propagates_other_errors() -> None:
    capability = PydanticToolControlCapability()
    error = RuntimeError("unexpected")

    with pytest.raises(RuntimeError) as raised:
        await capability.on_tool_execute_error(
            None,  # type: ignore[arg-type]
            call=ToolCallPart("tool", tool_call_id="call"),
            tool_def=ToolDefinition(name="tool"),
            args={},
            error=error,
        )

    assert raised.value is error


def test_deferred_control_constructors_remain_in_the_owner() -> None:
    assert isinstance(build_model_retry("retry"), ModelRetry)
    assert isinstance(build_tool_failed("failed"), ToolFailed)


@pytest.mark.asyncio
@pytest.mark.parametrize("signal_type", (ToolCallRetry, ToolCallFailed))
@pytest.mark.parametrize("streamed", (False, True))
async def test_argument_validator_signal_reaches_model(
    signal_type: type[ToolCallRetry] | type[ToolCallFailed],
    streamed: bool,
) -> None:
    message = "value must be non-negative; correct value and retry"
    requests = 0
    executed: list[int] = []

    async def checked(value: int) -> str:
        executed.append(value)
        return "accepted"

    async def validate(_ctx: RunContext[None], value: int) -> None:
        if value < 0:
            raise signal_type(message)

    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal requests
        requests += 1
        if requests == 1:
            return ModelResponse(parts=[ToolCallPart("checked", {"value": -1}, tool_call_id="invalid")])
        if requests == 2:
            assert isinstance(messages[-1], ModelRequest)
            feedback = messages[-1].parts[0]
            if signal_type is ToolCallRetry:
                assert isinstance(feedback, RetryPromptPart)
                assert feedback.content == message
                assert feedback.tool_call_id == "invalid"
                return ModelResponse(parts=[ToolCallPart("checked", {"value": 1}, tool_call_id="corrected")])
            assert isinstance(feedback, ToolReturnPart)
            assert feedback.outcome == "failed"
            assert feedback.content == message
            assert feedback.tool_call_id == "invalid"
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(
        _UsageFunctionModel(respond),
        tools=[Tool(checked, args_validator=validate)],
        capabilities=[PydanticToolControlCapability()],
        retries=1 if signal_type is ToolCallRetry else 0,
    )

    if streamed:
        async with agent.run_stream("Use checked") as result:
            assert await result.get_output() == "done"
    else:
        assert (await agent.run("Use checked")).output == "done"
    assert executed == ([1] if signal_type is ToolCallRetry else [])
    assert requests == (3 if signal_type is ToolCallRetry else 2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    (
        AIError(ErrorCode.STORAGE_UNAVAILABLE),
        AIError(ErrorCode.AUTHORIZATION_DENIED),
        ValueError("backend invariant failed"),
    ),
)
async def test_argument_validator_unrelated_error_remains_terminal(error: Exception) -> None:
    requests = 0

    async def checked(value: int) -> str:
        pytest.fail("invalid arguments must not reach the tool")

    async def validate(_ctx: RunContext[None], value: int) -> None:
        raise error

    def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal requests
        requests += 1
        return ModelResponse(parts=[ToolCallPart("checked", {"value": 1}, tool_call_id="call")])

    agent = Agent(
        FunctionModel(respond),
        tools=[Tool(checked, args_validator=validate)],
        capabilities=[PydanticToolControlCapability()],
    )

    with pytest.raises(type(error)) as raised:
        await agent.run("Use checked")

    assert raised.value is error
    assert requests == 1


@pytest.mark.asyncio
async def test_argument_schema_validation_preserves_sdk_feedback() -> None:
    requests = 0

    async def checked(value: int) -> str:
        return str(value)

    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal requests
        requests += 1
        if requests == 1:
            return ModelResponse(parts=[ToolCallPart("checked", {"value": []}, tool_call_id="call")])
        assert isinstance(messages[-1], ModelRequest)
        feedback = messages[-1].parts[0]
        assert isinstance(feedback, RetryPromptPart)
        assert isinstance(feedback.content, list)
        assert feedback.content[0]["loc"] == ("value",)
        assert feedback.content[0]["type"] == "int_type"
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(
        FunctionModel(respond),
        tools=[checked],
        capabilities=[PydanticToolControlCapability()],
    )

    assert (await agent.run("Use checked")).output == "done"
    assert requests == 2


@pytest.mark.asyncio
async def test_argument_validator_retry_respects_existing_limit() -> None:
    requests = 0

    async def checked(value: int) -> str:
        pytest.fail("invalid arguments must not reach the tool")

    async def validate(_ctx: RunContext[None], value: int) -> None:
        raise ToolCallRetry("Correct value")

    def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal requests
        requests += 1
        return ModelResponse(parts=[ToolCallPart("checked", {"value": 1}, tool_call_id="call")])

    agent = Agent(
        FunctionModel(respond),
        tools=[Tool(checked, args_validator=validate)],
        capabilities=[PydanticToolControlCapability()],
        retries=0,
    )

    with pytest.raises(UnexpectedModelBehavior):
        await agent.run("Use checked")

    assert requests == 1
