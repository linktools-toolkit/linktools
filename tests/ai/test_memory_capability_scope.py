#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One host declaration supplies new execution memory without rewriting captures."""

import asyncio
import inspect
from pathlib import Path

import pytest
from pydantic_ai import Tool
from pydantic_ai.capabilities import CombinedCapability, WrapperCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo

from linktools.ai.capability import AgentContext, ExecutionCapability, MemoryCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionInputContext
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime import Agent, Runtime, RuntimeFeatures, Session
from linktools.ai.runtime.agent_execution import AgentExecutionPort
from linktools.commands.ai._common import _open_local_runtime
from linktools.ai.storage import InMemoryCacheBackend
from linktools.ai.workspace import Workspace

from ._runtime_test_helpers import model_features
from .test_explicit_runtime_features import _FeatureFunctionModel, _FeatureModels


def _features(scope: str | None) -> RuntimeFeatures:
    features = model_features(_FeatureModels())
    features.agent("default", model="default", allow_tools=())
    features.capability(ExecutionCapability(retain_history=True))
    if scope is not None:
        features.capability(MemoryCapability(scope=scope))
    return features


@pytest.mark.parametrize("value", [None, "", " leading", "trailing ", "a\x00b", 1])
def test_memory_capability_validates_scope(value: object) -> None:
    with pytest.raises(AIError) as raised:
        MemoryCapability(scope=value)  # type: ignore[arg-type]
    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID


def test_memory_declaration_requires_runtime_and_invocations_have_no_scope() -> None:
    capability = MemoryCapability()
    assert capability.scope == "default"
    with pytest.raises(AIError) as raised:
        capability.for_agent(object())
    assert raised.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
    for owner, methods in ((Agent, ("start", "run", "plan")),
                           (Session, ("start", "run", "plan")),
                           (AgentExecutionPort, ("start",))):
        for method in methods:
            assert "memory_scope" not in inspect.signature(getattr(owner, method)).parameters


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [None, "default", "workspace-memory"])
@pytest.mark.parametrize("owner", ["agent", "session", "lazy-session"])
@pytest.mark.parametrize("method", ["start", "run", "plan"])
async def test_memory_scope_is_injected_across_agent_and_session_entries(scope: str | None, owner: str, method: str) -> None:
    async with Runtime.open("memory-entries", storage=PersistenceStorage.in_memory(),
                            features=_features(scope)) as runtime:
        agent = runtime.agents.get()
        if owner == "agent":
            handle = agent
        else:
            session = agent.session("session")
            if owner == "session":
                handle = session
            else:
                await agent.create_session("session")
                handle = await runtime.sessions.get("session")
        result = await getattr(handle, method)("hello", idempotency_key="memory-entry")
        if method == "start":
            result = await result.wait()
        assert result.result.status is ExecutionStatus.SUCCEEDED
        record = await runtime._state.execution.executions.get(
            result.result.execution_id, tenant_id=runtime.tenant_id,
        )
        assert record.memory_scope == scope


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["direct", "combined", "wrapper"])
@pytest.mark.parametrize("duplicate", [False, True])
async def test_memory_declaration_rejects_agent_scope_and_duplicates(shape: str, duplicate: bool) -> None:
    def wrap(capability):
        if shape == "combined":
            return CombinedCapability([capability])
        return WrapperCapability(capability) if shape == "wrapper" else capability

    with pytest.raises(AIError) as raised:
        features = model_features(_FeatureModels())
        first = wrap(MemoryCapability())
        features.capability(first, id=first.id or "memory-one", scope="runtime" if duplicate else "agent")
        if duplicate:
            second = wrap(MemoryCapability(scope="other"))
            features.capability(second, id=second.id or "memory-two")
        async with Runtime.open("memory-declarations", storage=PersistenceStorage.in_memory(), features=features):
            pytest.fail("invalid declaration was accepted")
    assert raised.value.code is (ErrorCode.CAPABILITY_CONFLICT if duplicate else ErrorCode.CAPABILITY_RESOLUTION_INVALID)


@pytest.mark.asyncio
async def test_retry_retains_admitted_memory_after_host_configuration_changes(tmp_path: Path) -> None:
    storage = PersistenceStorage.filesystem(tmp_path)
    async with Runtime.open("memory-retry", storage=storage, features=_features("original")) as runtime:
        source = await runtime.agents.get().start("first", idempotency_key="first")
        assert (await source.wait()).result.status is ExecutionStatus.SUCCEEDED
        execution_id = source.execution_id
    async with Runtime.open("memory-retry", storage=storage, features=_features("changed")) as runtime:
        source = await runtime.executions.get(execution_id)
        retry = await source.retry("again", idempotency_key="again")
        assert (await retry.wait()).result.status is ExecutionStatus.SUCCEEDED
        record = await runtime._state.execution.executions.get(retry.execution_id, tenant_id=runtime.tenant_id)
        assert record.memory_scope == "original"
        fresh = await runtime.agents.get().start("new", idempotency_key="new")
        assert (await fresh.wait()).result.status is ExecutionStatus.SUCCEEDED
        record = await runtime._state.execution.executions.get(fresh.execution_id, tenant_id=runtime.tenant_id)
        assert record.memory_scope == "changed"


@pytest.mark.asyncio
async def test_local_cli_composition_registers_default_memory(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    async with _open_local_runtime(workspace, model=_FeatureModels()) as runtime:
        assert runtime.agents.get().execution_port.memory_scope == "default"


@pytest.mark.asyncio
async def test_memory_configuration_is_captured_once_by_runtime() -> None:
    capability = MemoryCapability(scope="admitted-default")
    features = _features(None)
    features.capability(capability)
    async with Runtime.open("memory-capture", storage=PersistenceStorage.in_memory(), features=features) as runtime:
        capability._scope = "mutated-authoring-object"
        execution = await runtime.agents.get().start("hello")
        assert (await execution.wait()).result.status is ExecutionStatus.SUCCEEDED
        record = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert record.memory_scope == "admitted-default"


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_tools", [None, ("*",), ("read_memory",), ()])
async def test_memory_tools_follow_capability_and_agent_selection(allow_tools: tuple[str, ...] | None) -> None:
    expected = {"read_memory", "search_memory", "write_memory", "delete_memory"}
    if allow_tools is not None and "*" not in allow_tools:
        expected.intersection_update(allow_tools)
    observed: list[set[str]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages
        observed.append({tool.name for tool in info.function_tools})
        return ModelResponse(parts=[TextPart("done")])

    features = model_features(_FeatureModels(_FeatureFunctionModel(respond)))
    features.capability(MemoryCapability())
    if allow_tools is not None:
        features.agent("default", allow_tools=allow_tools)
    async with Runtime.open("memory-tool-selection", storage=PersistenceStorage.in_memory(), features=features) as runtime:
        result = await runtime.agents.get().run("hello")
        assert result.result.status is ExecutionStatus.SUCCEEDED
    assert observed == [expected]


@pytest.mark.asyncio
@pytest.mark.parametrize("next_scope", [None, "changed", "business"])
async def test_memory_tools_retry_uses_admitted_store_after_host_change(tmp_path: Path, next_scope: str | None) -> None:
    observations: list[tuple[str, set[str]]] = []
    reads: list[object] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        returns = [part for part in request.parts if isinstance(part, ToolReturnPart)]
        if returns:
            reads.extend(part.content for part in returns if part.tool_name == "read_memory")
            return ModelResponse(parts=[TextPart("done")])
        prompt = next(part.content for part in reversed(request.parts) if isinstance(part, UserPromptPart))
        assert isinstance(prompt, str)
        observations.append((prompt, {tool.name for tool in info.function_tools}))
        if prompt == "remember":
            return ModelResponse(parts=[ToolCallPart("write_memory", {"content": "admitted memory"}, tool_call_id="write")])
        if prompt == "recall":
            return ModelResponse(parts=[ToolCallPart("read_memory", {"file": "MEMORY.md"}, tool_call_id="read")])
        return ModelResponse(parts=[TextPart("fresh")])

    def features(scope: str | None, *, selected: bool = False) -> RuntimeFeatures:
        result = model_features(_FeatureModels(_FeatureFunctionModel(respond)))
        result.capability(ExecutionCapability(retain_history=True))
        if scope == "business":
            def read_memory(context: AgentContext[object]) -> str:
                del context
                return "unrelated business tool"
            result.tool(read_memory, effect_policy="none")
        elif scope is not None:
            result.capability(MemoryCapability(scope=scope))
        if selected:
            result.agent("default", allow_tools=("read_memory", "write_memory"))
        return result

    storage = PersistenceStorage.filesystem(tmp_path)
    async with Runtime.open("admitted-memory-tools", storage=storage, features=features("original", selected=True)) as runtime:
        original = await runtime.agents.get().run("remember")
        assert original.result.status is ExecutionStatus.SUCCEEDED
        execution_id = original.result.execution_id
    async with Runtime.open("admitted-memory-tools", storage=storage, features=features(next_scope)) as runtime:
        fresh = await runtime.agents.get().run("fresh")
        assert fresh.result.status is ExecutionStatus.SUCCEEDED
        source = await runtime.executions.get(execution_id)
        retry = await source.retry("recall")
        assert (await retry.wait()).result.status is ExecutionStatus.SUCCEEDED
    assert observations[0] == ("remember", {"read_memory", "write_memory"})
    fresh_tools = {"read_memory", "search_memory", "write_memory", "delete_memory"}
    if next_scope is None:
        fresh_tools = set()
    elif next_scope == "business":
        fresh_tools = {"read_memory"}
    assert observations[1] == ("fresh", fresh_tools)
    assert observations[2] == ("recall", {"read_memory", "write_memory"})
    assert len(reads) == 1 and "admitted memory" in str(reads[0])


@pytest.mark.asyncio
async def test_memory_tool_selection_requires_current_capability() -> None:
    features = model_features(_FeatureModels())
    features.agent("default", allow_tools=("read_memory",))
    with pytest.raises(AIError) as raised:
        async with Runtime.open("missing-memory", storage=PersistenceStorage.in_memory(), features=features):
            pytest.fail("Memory tools must not enable their own capability")
    assert raised.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["combined", "wrapper"])
async def test_wrapped_memory_tools_respect_candidate_filter(shape: str) -> None:
    observed: list[set[str]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages
        observed.append({tool.name for tool in info.function_tools})
        return ModelResponse(parts=[TextPart("done")])

    features = RuntimeFeatures(filter=lambda _spec, candidate: candidate.id != "write_memory")
    model_features(_FeatureModels(_FeatureFunctionModel(respond)), features)
    memory = MemoryCapability()
    capability = CombinedCapability([memory]) if shape == "combined" else WrapperCapability(memory)
    features.capability(capability, id=capability.id or "memory-tools")
    async with Runtime.open("filtered-memory", storage=PersistenceStorage.in_memory(), features=features) as runtime:
        result = await runtime.agents.get().run("hello")
        assert result.result.status is ExecutionStatus.SUCCEEDED
    assert observed == [{"read_memory", "search_memory", "delete_memory"}]


@pytest.mark.asyncio
async def test_imported_memory_retains_tools_without_host_memory(tmp_path: Path) -> None:
    observed: list[set[str]] = []
    reads: list[object] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        returns = [part for part in request.parts if isinstance(part, ToolReturnPart)]
        if returns:
            reads.extend(part.content for part in returns)
        else:
            names = {tool.name for tool in info.function_tools}
            observed.append(names)
            if "read_memory" in names:
                return ModelResponse(parts=[ToolCallPart("read_memory", {"file": "MEMORY.md"}, tool_call_id="read")])
        return ModelResponse(parts=[TextPart("done")])

    features = model_features(_FeatureModels(_FeatureFunctionModel(respond)))
    features.capability(ExecutionCapability(retain_history=True))
    context = ExecutionInputContext.from_messages((), memory={
        "memory/MEMORY.md": {"content": "captured memory", "version": "m1:" + "1" * 64},
    })
    storage = PersistenceStorage.filesystem(tmp_path)
    async with Runtime.open("imported-memory-tools", storage=storage, features=features) as runtime:
        agent = runtime.agents.get()
        assert (await agent.run("ordinary")).result.status is ExecutionStatus.SUCCEEDED
        imported = await agent.run("imported", input_context=context)
        assert imported.result.status is ExecutionStatus.SUCCEEDED
    async with Runtime.open("imported-memory-tools", storage=storage, features=features) as runtime:
        source = await runtime.executions.get(imported.result.execution_id)
        retry = await source.retry("recall")
        assert (await retry.wait()).result.status is ExecutionStatus.SUCCEEDED
    memory_tools = {"read_memory", "search_memory", "write_memory", "delete_memory"}
    assert observed == [set(), memory_tools, memory_tools]
    assert len(reads) == 2 and all("captured memory" in str(content) for content in reads)


@pytest.mark.asyncio
async def test_imported_memory_rejects_collision_with_business_tool() -> None:
    calls: list[str] = []

    def read_memory(context: AgentContext[object]) -> str:
        del context
        calls.append("business")
        return "business result"

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert {tool.name for tool in info.function_tools} == {"read_memory"}
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        if any(isinstance(part, ToolReturnPart) for part in request.parts):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("read_memory", {}, tool_call_id="read")])

    features = model_features(_FeatureModels(_FeatureFunctionModel(respond)))
    features.tool(read_memory, effect_policy="none")
    async with Runtime.open("business-memory-name", storage=PersistenceStorage.in_memory(), features=features) as runtime:
        agent = runtime.agents.get()
        assert (await agent.run("ordinary")).result.status is ExecutionStatus.SUCCEEDED
        context = ExecutionInputContext.from_messages((), memory={
            "memory/MEMORY.md": {"content": "captured memory", "version": "m1:" + "1" * 64},
        })
        imported = await agent.run("imported", input_context=context)
        assert imported.result.status is ExecutionStatus.FAILED
        assert imported.result.error_code == ErrorCode.CAPABILITY_CONFLICT.value
    assert calls == ["business"]


@pytest.mark.asyncio
async def test_memory_restore_keeps_other_same_named_business_pins(tmp_path: Path) -> None:
    class ReadMemoryOnly(MemoryCapability):
        def tool_definitions(self) -> tuple[Tool[object], ...]:
            return tuple(tool for tool in super().tool_definitions() if tool.name == "read_memory")

    calls: list[str] = []

    def write_memory(context: AgentContext[object]) -> str:
        del context
        calls.append("business")
        return "business result"

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert {tool.name for tool in info.function_tools} == {"read_memory", "write_memory"}
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        if any(isinstance(part, ToolReturnPart) for part in request.parts):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("write_memory", {}, tool_call_id="write")])

    def features(memory: bool) -> RuntimeFeatures:
        result = model_features(_FeatureModels(_FeatureFunctionModel(respond)))
        result.capability(ExecutionCapability(retain_history=True))
        if memory:
            result.capability(ReadMemoryOnly())
        result.tool(write_memory, effect_policy="none")
        return result

    storage = PersistenceStorage.filesystem(tmp_path)
    async with Runtime.open("mixed-memory-pins", storage=storage, features=features(True)) as runtime:
        original = await runtime.agents.get().run("original")
        assert original.result.status is ExecutionStatus.SUCCEEDED
    async with Runtime.open("mixed-memory-pins", storage=storage, features=features(False)) as runtime:
        source = await runtime.executions.get(original.result.execution_id)
        retry = await source.retry("retry")
        assert (await retry.wait()).result.status is ExecutionStatus.SUCCEEDED
    assert calls == ["business", "business"]


@pytest.mark.asyncio
async def test_imported_memory_keeps_durable_source_with_terminal_commit(tmp_path: Path) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert {tool.name for tool in info.function_tools} == {
            "read_memory", "search_memory", "write_memory", "delete_memory",
        }
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        if any(isinstance(part, ToolReturnPart) for part in request.parts):
            return ModelResponse(parts=[TextPart("done")])
        entered.set()
        await release.wait()
        return ModelResponse(parts=[ToolCallPart("write_memory", {"file": "NEW.md", "content": "new captured note"}, tool_call_id="write")])

    features = model_features(_FeatureModels(_FeatureFunctionModel(respond)))
    features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal()))
    storage = PersistenceStorage.sqlite(tmp_path / "memory.db", coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))
    context = ExecutionInputContext.from_messages((), memory={
        "memory/MEMORY.md": {"content": "captured memory", "version": "m1:" + "1" * 64},
    })
    async with Runtime.open("imported-terminal-memory", storage=storage, features=features) as runtime:
        execution = await runtime.agents.get().start("remember", input_context=context)
        try:
            await asyncio.wait_for(entered.wait(), 5)
            source = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
            assert source is not None and source.context_imported and source.input_context is not None
        finally:
            release.set()
        assert (await execution.wait()).result.status is ExecutionStatus.SUCCEEDED
