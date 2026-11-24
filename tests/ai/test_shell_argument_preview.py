#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded Shell display text on live and durable tool process projections."""

from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from linktools.ai.capability import ExecutionCapability, TraceCapability
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime
from linktools.ai.workspace import Workspace

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features, workspace_features


@pytest.mark.asyncio
async def test_native_shell_preview_keeps_short_arguments_and_bounds_long_commands(tmp_path: Path) -> None:
    short = "printf '%s' 'hello  world'"
    long = "printf hi\n# " + "x" * 90
    expected = {"short": short, "long": ("printf hi # " + "x" * 90)[:79] + "…"}

    def respond(messages: list[ModelMessage], _info: object) -> ModelResponse:
        if any(isinstance(part, ToolReturnPart) for part in messages[-1].parts):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[
            ToolCallPart("run_command", {"command": short}, tool_call_id="short"),
            ToolCallPart("run_command", {"command": long}, tool_call_id="long"),
        ])

    features = workspace_features("shell-preview", workspace=Workspace.load(tmp_path))
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.capability(TraceCapability(), scope="runtime")
    features.agent("default", allow_tools=("run_command",))
    model_features(_Models(_UsageFunctionModel(respond)), features)
    async with Persistence.open("shell-preview", storage=PersistenceStorage.in_memory()) as state, Runtime.open(
        persistence=state, features=features,
    ) as runtime:
        session = await runtime.agents.get("default").create_session("preview")
        execution = await session.start("Use the shell tools")
        observed = {}

        async def on_event(item) -> None:
            if item.event.event_type == "TOOL_CALL_STARTED":
                payload = item.event.payload
                observed[payload["call_id"]] = payload.get("argument_preview")

        assert (await execution.wait(
            timeout_seconds=10, on_event=on_event, include_event_content=True,
        )).result.status.value == "SUCCEEDED"
        assert observed == expected
        metadata_boundaries = []
        async for item in execution.watch():
            if item.event.event_type in ("TOOL_CALL_STARTED", "TOOL_CALL_FINISHED"):
                metadata_boundaries.append(item.event.event_type)
                assert "argument_preview" not in item.event.payload
        assert metadata_boundaries.count("TOOL_CALL_STARTED") == 2
        assert metadata_boundaries.count("TOOL_CALL_FINISHED") == 2
        turn = await runtime.history.session_turn("preview", execution.execution_id, principal=runtime.default_principal)
        assert {item.key.removeprefix("tool:"): item.argument_preview for item in turn.process} == expected
        trace = await execution.model_interactions(include_content=True)
        tool, = trace.items[0].request["parameters"]["function_tools"]
        assert "linktools.ai.argument_preview_field" not in tool["metadata"]
