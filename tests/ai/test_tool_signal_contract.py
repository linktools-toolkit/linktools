#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LinkTools tool signal and final Pydantic boundary contracts."""

import pytest
from linktools.ai.capability import ToolCallFailed, ToolCallRetry
from linktools.ai.errors import AIError
from linktools.ai.runtime._pydantic_tool_control import (
    PydanticToolControlCapability,
    build_model_retry,
    build_tool_failed,
)
from pydantic_ai.exceptions import ModelRetry, ToolFailed
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition


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
