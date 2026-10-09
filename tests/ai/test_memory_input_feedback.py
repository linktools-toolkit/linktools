#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model-authored memory text rejections preserve backend failure semantics."""

import traceback
from unittest.mock import AsyncMock

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.memory import InMemoryStore, MemoryFile

from linktools.ai.capability import ToolCallRetry
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._harness_memory import build_harness_memory
from linktools.ai.runtime._pydantic_tool_control import PydanticToolControlCapability

from ._runtime_test_helpers import tool_run_context


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "hint"),
    (("private\x00content", "NUL"), ("private\ud800content", "UTF-8")),
)
async def test_memory_input_rejection_precedes_storage(content: str, hint: str) -> None:
    store = InMemoryStore()
    store.get_operation = AsyncMock(side_effect=AssertionError("storage must not be accessed"))
    store.read = AsyncMock(side_effect=AssertionError("storage must not be accessed"))
    store.write = AsyncMock(side_effect=AssertionError("storage must not be accessed"))
    capability = build_harness_memory(store, allow_tools=("write_memory",), capability_id="memory")
    toolset = capability.get_toolset()
    assert toolset is not None
    context = tool_run_context()
    tools = await toolset.get_tools(context)
    validator = tools["write_memory"].args_validator_func
    assert validator is not None

    with pytest.raises(ToolCallRetry, match=hint) as raised:
        validator(context, content=content, file="MEMORY.md", old_text=None)

    assert "private" not in str(raised.value)
    diagnostic = "".join(traceback.format_exception(raised.value))
    assert "private" not in diagnostic
    assert "UnicodeEncodeError" not in diagnostic
    store.get_operation.assert_not_called()
    store.read.assert_not_called()
    store.write.assert_not_called()


@pytest.mark.asyncio
async def test_model_can_correct_memory_content_after_feedback() -> None:
    store = InMemoryStore()
    capability = build_harness_memory(store, allow_tools=("write_memory",), capability_id="memory")
    requests = 0

    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal requests
        requests += 1
        if requests == 1:
            return ModelResponse(parts=[ToolCallPart("write_memory", {"content": "private\x00content"}, tool_call_id="invalid")])
        if requests == 2:
            assert isinstance(messages[-1], ModelRequest)
            feedback = messages[-1].parts[0]
            assert isinstance(feedback, RetryPromptPart)
            assert isinstance(feedback.content, str)
            assert "NUL" in feedback.content
            assert "private" not in feedback.content
            return ModelResponse(parts=[ToolCallPart("write_memory", {"content": "valid memory"}, tool_call_id="corrected")])
        return ModelResponse(parts=[TextPart("done")])

    result = await Agent(
        FunctionModel(respond),
        capabilities=[capability, PydanticToolControlCapability()],
        retries=1,
    ).run("Remember this")

    assert result.output == "done"
    saved = await store.read("memory/MEMORY.md", max_chars=65536)
    assert saved is not None and saved.content.strip() == "valid memory"
    assert requests == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    (
        ValueError("backend invariant failed"),
        AIError(ErrorCode.STORAGE_UNAVAILABLE),
        AIError(ErrorCode.AUTHORIZATION_DENIED),
    ),
)
async def test_memory_backend_errors_are_not_model_retries(error: Exception) -> None:
    store = InMemoryStore()
    store.write = AsyncMock(side_effect=error)
    capability = build_harness_memory(store, allow_tools=("write_memory",), capability_id="memory")
    requests = 0

    def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal requests
        requests += 1
        return ModelResponse(parts=[ToolCallPart("write_memory", {"content": "valid"}, tool_call_id="call")])

    agent = Agent(
        FunctionModel(respond),
        capabilities=[capability, PydanticToolControlCapability()],
    )

    with pytest.raises(type(error)) as raised:
        await agent.run("Remember this")

    assert raised.value is error
    assert requests == 1


@pytest.mark.asyncio
async def test_clean_memory_input_does_not_reclassify_corrupt_stored_content() -> None:
    store = InMemoryStore()
    store.read = AsyncMock(
        return_value=MemoryFile("existing\x00corruption", "1", None, False),
    )
    error = ValueError("stored content is invalid")
    store.write = AsyncMock(side_effect=error)
    capability = build_harness_memory(store, allow_tools=("write_memory",), capability_id="memory")

    def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart("write_memory", {"content": "clean new content"}, tool_call_id="call")])

    agent = Agent(
        FunctionModel(respond),
        capabilities=[capability, PydanticToolControlCapability()],
    )

    with pytest.raises(ValueError) as raised:
        await agent.run("Remember this")

    assert raised.value is error
    assert "existing\x00corruption" in store.write.await_args.args[1]
