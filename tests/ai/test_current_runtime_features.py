#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime uses live declarations without changing admitted calls."""

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from linktools.ai.capability import AgentContext, ExecutionCapability, MemoryCapability
from linktools.ai.core import ExecutionStatus
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from ._runtime_test_helpers import _Models, _UsageFunctionModel, RuntimeUsageModels, model_features


@pytest.mark.asyncio
async def test_current_runtime_accepts_later_agent_and_tools(tmp_path):
    stage = 0
    features = RuntimeFeatures(workspace_factory=None)

    async def added(ctx: AgentContext[object]) -> str:
        return "added"

    async def seed(ctx: AgentContext[object]) -> str:
        features.tool(added)
        return "seeded"

    async def respond(messages, info):
        nonlocal stage
        names = {tool.name for tool in info.function_tools}
        results = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        stage += 1
        if stage == 1:
            assert names == {"seed"}
            return ModelResponse(parts=[ToolCallPart("seed", {}, "seed-call")])
        if stage == 2:
            assert results[-1].content == "seeded"
            assert names == {"seed", "added"}
            return ModelResponse(parts=[ToolCallPart("added", {}, "added-call")])
        assert results[-1].content == "added"
        return ModelResponse(parts=[TextPart("done")])

    features.model("default", _Models(_UsageFunctionModel(respond)))
    features.tool(seed)
    async with Runtime.open("current", storage=PersistenceStorage.sqlite(tmp_path / "state.db"), features=features) as runtime:
        future = runtime.agents.get("later")
        assert future.revision is None
        features.agent("later", allow_skills=(), allow_subagents=())
        observed = await future.resolve()
        assert observed.revision == 1
        result = await future.run("go")
        assert result.result.status is ExecutionStatus.SUCCEEDED, result.result
    assert stage == 3


@pytest.mark.asyncio
async def test_current_runtime_uses_new_runtime_hooks_for_next_run(tmp_path):
    features = model_features(RuntimeUsageModels(), RuntimeFeatures(workspace_factory=None))
    async with Runtime.open("current", storage=PersistenceStorage.sqlite(tmp_path / "state.db"), features=features) as runtime:
        first = await runtime.agents.get().start("first", idempotency_key="first")
        assert (await first.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        features.capability(ExecutionCapability(retain_history=True))
        features.capability(MemoryCapability(scope="current"))
        second = await runtime.agents.get().start("second", idempotency_key="second")
        assert (await second.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        record = await runtime._state.execution.executions.get(second.execution_id, tenant_id="default")
        assert record.conversation and record.memory_scope == "current"


@pytest.mark.asyncio
async def test_current_runtime_lazily_opens_workspace_for_new_stdio_mcp(tmp_path, monkeypatch):
    from linktools.ai.spec import MCPServerSpec
    from linktools.ai.workspace import LocalWorkspaceBackend
    from linktools.ai.runtime._mcp import _model_tool_name
    from .test_current_toolsets import _fake_mcp

    events = []
    instances = _fake_mcp(monkeypatch, events)
    opened = []

    class WorkspaceBackend(LocalWorkspaceBackend):
        async def working_dir(self):
            opened.append(self)
            return await super().working_dir()

    features = RuntimeFeatures(workspace_factory=WorkspaceBackend)
    features.agent("default", allow_skills=(), allow_subagents=())
    calls = 0

    async def install(ctx: AgentContext[object]) -> str:
        features.mcp(MCPServerSpec("new-server", "python"))
        return "installed"

    async def respond(messages, info):
        nonlocal calls
        calls += 1
        names = {tool.name for tool in info.function_tools}
        if calls == 1:
            assert names == {"install"} and not instances
            return ModelResponse(parts=[ToolCallPart("install", {}, "install")])
        if calls == 2:
            assert _model_tool_name("new-server", "probe") in names
            return ModelResponse(parts=[ToolCallPart(_model_tool_name("new-server", "probe"), {}, "probe")])
        assert any(isinstance(part, ToolReturnPart) and part.content == "new-server"
                   for message in messages for part in message.parts)
        return ModelResponse(parts=[TextPart("done")])

    features.tool(install)
    features.model("default", _Models(_UsageFunctionModel(respond)))
    async with Runtime.open("late-mcp", storage=PersistenceStorage.sqlite(tmp_path / "state.db"),
                            features=features) as runtime:
        result = await runtime.agents.get().run("go")
        assert result.result.status is ExecutionStatus.SUCCEEDED, result.result
    assert opened and len(instances) == 1
    assert events == ["enter:new-server", "exit:new-server", "close:new-server"]


@pytest.mark.asyncio
async def test_current_runtime_rejects_replacing_an_open_workspace_root(tmp_path):
    from linktools.ai.capability import FileSystem
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.workspace import Workspace

    original = Workspace.initialize(tmp_path / "original")
    replacement = Workspace.initialize(tmp_path / "replacement")
    features = model_features(RuntimeUsageModels())
    features.capability(FileSystem(original))
    async with Runtime.open("root-owner", storage=PersistenceStorage.in_memory(), features=features) as runtime:
        features.capability(FileSystem(replacement))
        with pytest.raises(AIError) as error:
            await runtime.agents.get().start("do not use the old root")
        assert error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
        assert error.value.safe_details["reason"] == "runtime_resource_replacement"
        features.capability(FileSystem(original))
        assert (await runtime.agents.get().run("unchanged root")).result.status is ExecutionStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_current_runtime_rejects_agent_scoped_persistence_added_after_open():
    from linktools.ai.errors import AIError, ErrorCode
    features = model_features(RuntimeUsageModels(), RuntimeFeatures(workspace_factory=None))
    async with Runtime.open("late-invalid-scope", storage=PersistenceStorage.in_memory(), features=features) as runtime:
        features.capability(ExecutionCapability(), scope="agent")
        with pytest.raises(AIError) as error:
            await runtime.agents.get().start("invalid scope")
        assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
