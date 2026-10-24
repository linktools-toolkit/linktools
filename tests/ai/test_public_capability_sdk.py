#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public capability composition without a Runtime host."""

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability
from pydantic_ai.exceptions import ModelRetry, ToolFailed
from pydantic_ai.messages import ModelRequest, ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from linktools.ai.capability import (
    AgentContext, PydanticToolControlCapability, SkillCapability, SkillSourceRegistry,
    SubagentCapability, ToolCallFailed, ToolCallRetry, workspace_capabilities,
)
from linktools.ai.core import Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.workspace import Workspace


def test_public_capability_import_does_not_load_runtime() -> None:
    subprocess.run(
        [sys.executable, "-c", "import sys; from linktools.ai.capability import PydanticToolControlCapability; "
         "assert not any(name == 'linktools.ai.runtime' or name.startswith('linktools.ai.runtime.') "
         "for name in sys.modules)"],
        check=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("skill", "subagent", "workspace"))
async def test_public_capabilities_allow_model_correction_without_runtime(kind: str, tmp_path: Path) -> None:
    class BorrowedSession:
        closed = 0

        async def list_directory(self, path: str = ".") -> list[object]:
            if path != ".":
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            return []

        async def close(self) -> None:
            self.closed += 1

    async def delegate(*args: object, **kwargs: object) -> dict[str, object]:
        pytest.fail("an unavailable subagent must not be delegated")

    session = BorrowedSession()
    if kind == "skill":
        capability = SkillCapability([], SkillSourceRegistry())
        failed_call = ToolCallPart("load_skill", {"skill_id": "missing"}, tool_call_id="missing")
        corrected_call = ToolCallPart("list_skills", {}, tool_call_id="corrected")
    elif kind == "subagent":
        capability = SubagentCapability([], delegate)  # type: ignore[arg-type]
        failed_call = ToolCallPart("delegate_task", {"subagent_id": "missing", "task": "inspect"},
                                   tool_call_id="missing")
        corrected_call = ToolCallPart("list_subagents", {}, tool_call_id="corrected")
    else:
        capability = workspace_capabilities(
            Workspace.load(tmp_path), ("list_directory",), session=session,  # type: ignore[arg-type]
        )[0]
        failed_call = ToolCallPart("list_directory", {"path": "missing"}, tool_call_id="missing")
        corrected_call = ToolCallPart("list_directory", {"path": "."}, tool_call_id="corrected")
    requests = 0

    def respond(messages, _info):
        nonlocal requests
        requests += 1
        if requests == 1:
            return ModelResponse(parts=[failed_call])
        assert isinstance(messages[-1], ModelRequest)
        feedback = messages[-1].parts[0]
        if requests == 2:
            assert isinstance(feedback, RetryPromptPart) or (
                isinstance(feedback, ToolReturnPart) and feedback.outcome == "retried"
            )
            assert feedback.tool_call_id == "missing"
            return ModelResponse(parts=[corrected_call])
        assert isinstance(feedback, ToolReturnPart)
        assert feedback.tool_call_id == "corrected"
        assert feedback.content == []
        return ModelResponse(parts=[TextPart("done")])

    context = AgentContext(None, Principal("reader", "tenant"), "sample", None, "execution", {})
    agent = Agent(FunctionModel(respond), capabilities=[PydanticToolControlCapability(), capability], retries=1)
    async with agent:
        assert (await agent.run("Inspect the available resource", deps=context)).output == "done"
    assert requests == 3
    assert session.closed == 0
    await session.close()
    assert session.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("signal_type", (ToolCallRetry, ToolCallFailed))
async def test_sdk_observer_sees_original_signal_before_public_conversion(signal_type) -> None:
    observed = []
    signal = signal_type("safe model feedback")

    class Observer(AbstractCapability):
        async def on_tool_execute_error(self, ctx, *, call, tool_def, args, error):
            observed.append(error)
            raise error

    async def fail() -> str:
        raise signal

    requests = 0

    def respond(messages, _info):
        nonlocal requests
        requests += 1
        if requests == 1:
            return ModelResponse(parts=[ToolCallPart("fail", {}, tool_call_id="call")])
        feedback = messages[-1].parts[0]
        assert feedback.content == "safe model feedback"
        if signal_type is ToolCallFailed:
            assert isinstance(feedback, ToolReturnPart) and feedback.outcome == "failed"
        else:
            assert isinstance(feedback, RetryPromptPart) or feedback.outcome == "retried"
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(FunctionModel(respond), tools=[fail], capabilities=[CombinedCapability([
        Observer(), PydanticToolControlCapability(),
    ])])
    assert (await agent.run("Call fail")).output == "done"
    assert observed == [signal]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ("validate", "execute"))
async def test_public_control_preserves_cancellation_and_tool_cleanup(stage: str) -> None:
    cancelled = asyncio.CancelledError()
    cleaned = []

    async def validate(_ctx: RunContext[None]) -> None:
        if stage == "validate":
            raise cancelled

    async def tool() -> str:
        try:
            raise cancelled
        finally:
            cleaned.append(True)

    def respond(_messages, _info):
        return ModelResponse(parts=[ToolCallPart("tool", {}, tool_call_id="call")])

    agent = Agent(FunctionModel(respond), tools=[Tool(tool, args_validator=validate)],
                  capabilities=[PydanticToolControlCapability()])
    with pytest.raises(asyncio.CancelledError) as raised:
        await agent.run("Call tool")
    assert raised.value is cancelled
    assert cleaned == ([True] if stage == "execute" else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("error", (ModelRetry("native retry"), ToolFailed("native failure"), ValueError("invariant")))
async def test_public_control_does_not_wrap_other_errors(error: Exception) -> None:
    with pytest.raises(type(error)) as raised:
        await PydanticToolControlCapability().on_tool_execute_error(
            None, call=ToolCallPart("tool", tool_call_id="call"),  # type: ignore[arg-type]
            tool_def=Tool(lambda: None).tool_def, args={}, error=error,
        )
    assert raised.value is error
