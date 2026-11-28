#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step-off canonical turns preserve structured tool results for the model."""

import asyncio
import json

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.usage import RequestUsage

from linktools.ai.capability import FileSystemCapability
from linktools.ai.core import TenantAuthorizationPolicy, UsageMetrics
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.turns import SessionTurnKey
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.workspace import ToolPermissionPolicy, Workspace, WorkspacePolicy

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


@pytest.mark.asyncio
async def test_read_only_structured_tool_result_survives_cold_context_not_public_display(tmp_path):
    class WorkspaceAuthorization(TenantAuthorizationPolicy):
        pass

    marker = "example model-only file content"
    project = tmp_path / "project"
    project.mkdir()
    (project / "document.txt").write_text(marker, encoding="utf-8")
    workspace = Workspace.load(project, policy=WorkspacePolicy(
        tool_permissions=ToolPermissionPolicy(default="allow"),
    ))
    root = tmp_path / "runtime"
    seen = []

    async def first(messages, info):
        results = [part.content for message in messages for part in message.parts
                   if isinstance(part, ToolReturnPart) and part.tool_name == "read_file"]
        if not results:
            assert {tool.name for tool in info.function_tools} == {"read_file"}
            return ModelResponse(
                parts=[ToolCallPart("read_file", {"path": "document.txt"}, tool_call_id="read-1")],
                usage=RequestUsage(input_tokens=101, output_tokens=202),
            )
        [content] = results
        assert isinstance(content, dict) and content["files"] == []
        assert content["texts"][0]["content"] == marker
        seen.append(content)
        return ModelResponse(
            parts=[TextPart("Document inspected.")],
            usage=RequestUsage(input_tokens=303, output_tokens=404),
        )

    async def continued(messages, _info):
        results = [part.content for message in messages for part in message.parts
                   if isinstance(part, ToolReturnPart) and part.tool_call_id == "read-1"]
        assert results == seen
        seen.append(results[0])
        return ModelResponse(parts=[TextPart("Context restored.")])

    def features(model):
        value = RuntimeFeatures("structured-tool-context", defaults={"execution": {"retain_history": True}})
        value.capability(FileSystemCapability(workspace), scope="runtime")
        value.agent("default", allow_tools=("read_file",), allow_capabilities=())
        return model_features(_Models(_UsageFunctionModel(model)), value)

    async def send(runtime, prompt, request_id):
        frames = asyncio.Queue()
        detach = runtime.sessions.observe_turn_session("chat", frames.put_nowait)
        try:
            turn_id = await runtime.sessions.start_turn(
                "chat", prompt, agent="default", principal=runtime.default_principal, idempotency_key=request_id,
            )
            async with asyncio.timeout(10):
                while True:
                    frame = await frames.get()
                    assert frame["turn_id"] == turn_id
                    assert marker not in json.dumps(frame)
                    if frame["type"] == "terminal":
                        assert frame["state"] == "succeeded", frame
                        return turn_id
        finally:
            detach()

    async with (
        Persistence.open("structured-tool-context", storage=PersistenceStorage.from_root(root)) as state,
        Runtime.open(persistence=state, features=features(first), authorization=WorkspaceAuthorization()) as runtime,
    ):
        principal = runtime.default_principal
        await runtime.sessions.create_turn_session("chat", principal=principal)
        turn_id = await send(runtime, "Read the document.", "first")
        display = await runtime.history.turn_result("chat", turn_id, principal=principal)
        assert display["conversation_committed"]
        assert "usage" not in display
        assert any(item["item_kind"] == "tool_call" and item["tool_call_id"] == "read-1" for item in display["items"])
        assert not any(item["item_kind"] == "tool_result" for item in display["items"])
        assert marker not in json.dumps(display)

    # New Persistence and Runtime: only durable canonical content can supply this.
    # A changed workspace file makes any accidental replay distinguishable.
    (project / "document.txt").write_text("different contents after reopen", encoding="utf-8")
    async with (
        Persistence.open("structured-tool-context", storage=PersistenceStorage.from_root(root)) as state,
        Runtime.open(persistence=state, features=features(continued), authorization=WorkspaceAuthorization()) as runtime,
    ):
        retained = await state.session_turns.get_history_turn(SessionTurnKey("chat", turn_id), lineage=())
        assert retained["usage"] == UsageMetrics(
            model_requests=2, tool_calls=1, input_tokens=404, output_tokens=606,
        )
        await send(runtime, "Use the existing context.", "second")
    assert len(seen) == 2 and seen[0] == seen[1]
