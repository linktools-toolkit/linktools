#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Current contribution resolution and one-step tool/resource ownership."""

import asyncio
from dataclasses import replace
from typing import Any

import pytest
from pydantic_ai import Agent, Tool
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

from linktools.ai.agent import AgentCompiler, CompiledAgent
from linktools.ai.capability import CapabilityContribution, tool_metadata
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import create_model
from linktools.ai.runtime._current_toolsets import create_current_toolset
from linktools.ai.runtime._mcp import _MCPModelToolset, _model_tool_name
from linktools.ai.runtime._tool_boundary import BoundaryToolset, ManagedToolDescriptor
from linktools.ai.spec import AgentSpec, MCPServerSpec

from ._runtime_test_helpers import tool_run_context
from .test_tool_effect_semantics import _Bridge


def _compiled(candidates: list[CapabilityContribution], selectors: tuple[str, ...] = ("*",)) -> CompiledAgent:
    spec = AgentSpec("agent", allow_tools=selectors)
    return AgentCompiler(
        models={"default": create_model(api="openai", model="test")},
        candidates=candidates, agents={"agent": spec},
    ).compile(spec)


def _tool(name: str, value: str, revision: int = 1) -> CapabilityContribution:
    def read() -> str:
        return value
    return CapabilityContribution.from_opaque(
        "tool", name,
        Tool(read, name=name, metadata=tool_metadata(effect_policy="none", tool_class="business")),
        revision=revision,
    )


@pytest.mark.asyncio
async def test_current_toolsets_update_next_model_step_and_keep_issued_tool() -> None:
    current = [_tool("value", "old")]
    results = []
    requests = 0

    async def provider() -> CompiledAgent:
        return _compiled(current)

    def model(messages: list[Any], info: Any) -> ModelResponse:
        nonlocal requests
        requests += 1
        names = {tool.name for tool in info.function_tools}
        if requests == 1:
            assert names == {"value"}
            current[:] = [_tool("value", "updated", 2), _tool("added", "new")]
            return ModelResponse(parts=[ToolCallPart("value", {}, tool_call_id="old")])
        results.append(messages[-1].parts[0].content)
        if requests == 2:
            assert names == {"value", "added"}
            current.clear()
            return ModelResponse(parts=[ToolCallPart("value", {}, tool_call_id="updated")])
        assert names == set()
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(FunctionModel(model), toolsets=[create_current_toolset(
        provider, asset_readers={}, workspace_backend=None, tool_operations=None,
    )])
    assert (await agent.run("go")).output == "done"
    assert results == ["old", "updated"]


@pytest.mark.asyncio
async def test_boundary_dispatch_retains_prepared_leaf_and_descriptor() -> None:
    first, second = FunctionToolset([_tool("value", "first").materialize_tool()]), FunctionToolset([
        _tool("value", "second").materialize_tool(),
    ])

    class Changing(AbstractToolset):
        @property
        def id(self) -> str:
            return "changing"

        async def get_tools(self, ctx: Any) -> dict[str, Any]:
            return await current.get_tools(ctx)

        async def call_tool(self, name: str, args: dict[str, Any], ctx: Any, tool: Any) -> Any:
            return await tool.toolset.call_tool(name, args, ctx, tool)

    current = first
    boundary = BoundaryToolset((Changing(),), {"value": ManagedToolDescriptor("none", "none", "business")}, id="boundary")
    ctx = tool_run_context()
    original = (await boundary.get_tools(ctx))["value"]
    current = second
    changed = (await boundary.get_tools(ctx))["value"]
    assert await boundary.call_tool("value", {}, ctx, original) == "first"
    assert await boundary.call_tool("value", {}, ctx, changed) == "second"
    with pytest.raises(AIError):
        await boundary.call_tool("other", {}, ctx, original)


def _fake_mcp(monkeypatch: pytest.MonkeyPatch, events: list[str], *, entering: asyncio.Event | None = None,
              release: asyncio.Event | None = None, closing: asyncio.Event | None = None) -> list[FunctionToolset]:
    import linktools.ai.runtime._mcp as module
    instances = []

    class FakeMCP(FunctionToolset):
        def __init__(self, transport: Any, *, server: MCPServerSpec) -> None:
            self.server = server
            def probe() -> str:
                return server.url or server.id
            super().__init__([probe], id=f"mcp:{server.id}")
            instances.append(self)

        async def __aenter__(self) -> Any:
            events.append(f"enter:{self.server.id}")
            if entering is not None:
                entering.set()
                assert release is not None
                await release.wait()
            return self

        async def __aexit__(self, *args: Any) -> None:
            events.append(f"exit:{self.server.id}")

        async def close_resources(self) -> None:
            if closing is not None:
                closing.set()
                assert release is not None
                await release.wait()
            events.append(f"close:{self.server.id}")

    monkeypatch.setattr(module, "MCPToolset", FakeMCP)
    return instances


@pytest.mark.asyncio
async def test_current_mcp_server_set_updates_same_revision_connection_and_removal(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    _fake_mcp(monkeypatch, events)
    server = MCPServerSpec("one", transport="streamable-http", url="https://first.invalid/mcp")
    current = [CapabilityContribution.from_declaration(server)]

    async def provider() -> CompiledAgent:
        return _compiled(current)

    bridge = _Bridge(True)
    dynamic = create_current_toolset(provider, asset_readers={}, workspace_backend=None, tool_operations=bridge)
    ctx = tool_run_context()
    dynamic = await dynamic.for_run(ctx)
    async with dynamic:
        await dynamic.for_run_step(ctx)
        name = _model_tool_name("one", "probe")
        first = (await dynamic.get_tools(ctx))[name]
        current[:] = [CapabilityContribution.from_declaration(replace(server, url="https://updated.invalid/mcp"))]
        assert await dynamic.call_tool(name, {}, ctx, first) == "https://first.invalid/mcp"
        await dynamic.for_run_step(replace(ctx, run_step=2))
        second = (await dynamic.get_tools(ctx))[name]
        assert await dynamic.call_tool(name, {}, ctx, second) == "https://updated.invalid/mcp"
        current[:] = [CapabilityContribution.from_declaration(replace(server, id="two"))]
        await dynamic.for_run_step(replace(ctx, run_step=3))
        assert set(await dynamic.get_tools(ctx)) == {_model_tool_name("two", "probe")}
        current.clear()
        await dynamic.for_run_step(replace(ctx, run_step=4))
        assert await dynamic.get_tools(ctx) == {}
    assert events == ["enter:one", "exit:one", "close:one", "enter:one", "exit:one", "close:one",
                      "enter:two", "exit:two", "close:two"]


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("enter", "exit"))
async def test_current_mcp_cancellation_finishes_native_owned_cleanup(monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    events: list[str] = []
    started, release = asyncio.Event(), asyncio.Event()
    _fake_mcp(monkeypatch, events, entering=started if phase == "enter" else None,
              closing=started if phase == "exit" else None, release=release)
    current = [CapabilityContribution.from_declaration(MCPServerSpec(
        "one", transport="streamable-http", url="https://example.invalid/mcp",
    ))]

    async def provider() -> CompiledAgent:
        return _compiled(current)

    dynamic = await create_current_toolset(provider, asset_readers={}, workspace_backend=None,
                                           tool_operations=_Bridge(True)).for_run(tool_run_context())
    await dynamic.__aenter__()
    if phase == "exit":
        await dynamic.for_run_step(tool_run_context())
        current.clear()
    task = asyncio.create_task(dynamic.for_run_step(replace(tool_run_context(), run_step=2)))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    if phase == "exit":
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await dynamic.__aexit__(None, None, None)
    assert events.count("close:one") == 1
    assert events.count("exit:one") == (1 if phase == "exit" else 0)


@pytest.mark.asyncio
async def test_mcp_direct_dispatch_requires_prepared_allowed_projection() -> None:
    raw = FunctionToolset([_tool("allowed", "yes").materialize_tool(), _tool("denied", "no").materialize_tool()])
    mapped = _MCPModelToolset(raw, "server", frozenset({"allowed"}))
    ctx = tool_run_context()
    tools = await mapped.get_tools(ctx)
    allowed = _model_tool_name("server", "allowed")
    denied = _model_tool_name("server", "denied")
    assert set(tools) == {allowed}
    with pytest.raises(AIError) as error:
        await mapped.call_tool(denied, {}, ctx, tools[allowed])
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
    with pytest.raises(AIError):
        await mapped.call_tool(allowed, {}, ctx, (await raw.get_tools(ctx))["allowed"])


@pytest.mark.asyncio
async def test_current_tool_admission_uses_explicit_capability_reference() -> None:
    from linktools.ai.spec import CapabilityRef

    calls = []

    class Bridge(_Bridge):
        async def begin(self, *args: Any, **kwargs: Any) -> Any:
            calls.append((args, kwargs))
            return await super().begin(*args, **kwargs)

    def effect(value: int = 7) -> int:
        return value

    contribution = CapabilityContribution.from_opaque("tool", "effect", Tool(
        effect, metadata=tool_metadata(effect_policy="replay_safe", tool_class="business"),
    ), revision=3)

    async def provider() -> CompiledAgent:
        return _compiled([contribution])

    def model(messages: list[Any], info: Any) -> ModelResponse:
        if not any(isinstance(message, ModelResponse) for message in messages):
            return ModelResponse(parts=[ToolCallPart("effect", {}, tool_call_id="raw-call")])
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(FunctionModel(model), toolsets=[create_current_toolset(
        provider, asset_readers={}, workspace_backend=None, tool_operations=Bridge(True),
    )])
    await agent.run("go")
    args, kwargs = calls[0]
    assert args[3] == {"value": 7}
    assert kwargs["capability_ref"] == CapabilityRef("tool", "effect", 3)


@pytest.mark.asyncio
async def test_current_toolsets_reuse_unchanged_clients_with_current_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    instances = _fake_mcp(monkeypatch, events)
    current = [_tool("value", "same"), CapabilityContribution.from_declaration(MCPServerSpec(
        "one", transport="streamable-http", url="https://example.invalid/mcp",
    ))]
    resolutions = 0

    async def provider() -> CompiledAgent:
        nonlocal resolutions
        resolutions += 1
        return _compiled(current)

    dynamic = await create_current_toolset(provider, asset_readers={}, workspace_backend=None,
                                           tool_operations=_Bridge(True)).for_run(tool_run_context())
    async with dynamic:
        for step in (1, 2, 3):
            ctx = replace(tool_run_context(), run_step=step)
            await dynamic.for_run_step(ctx)
            expected = {_model_tool_name("one", "probe")}
            if step < 3:
                expected.add("value" if step == 1 else "changed")
            if step > 1:
                expected.add(_model_tool_name("one", "new_tool"))
            assert set(await dynamic.get_tools(ctx)) == expected
            if step == 1:
                def new_tool() -> str:
                    return "new"
                instances[0].add_function(new_tool)
                current[0] = _tool("changed", "new implementation", 2)
            elif step == 2:
                current.pop(0)
        assert events == ["enter:one"]
    assert resolutions == 3
    assert events == ["enter:one", "exit:one", "close:one"]


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("failed_exit", "partial_entry", "switch"))
async def test_failed_step_materializations_stay_owned_until_quiescence(
    monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    import linktools.ai.runtime._current_toolsets as adapter
    import linktools.ai.runtime._mcp as module
    from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend
    from linktools.ai.storage import StorageOverlay

    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    roots, cleanups, events = [], [], []
    client_can_close = False
    failure = AIError(ErrorCode.MCP_CLEANUP_FAILED)
    primary = ValueError("entry failed")

    class FailingMCP(FunctionToolset):
        def __init__(self, transport: Any, *, server: MCPServerSpec) -> None:
            super().__init__([], id=server.id)

        async def __aenter__(self) -> Any:
            events.append("enter")
            if phase == "partial_entry":
                raise primary
            return self

        async def __aexit__(self, *args: Any) -> None:
            events.append("exit")

        async def close_resources(self) -> None:
            events.append("client-close")
            if not client_can_close:
                raise failure

    async def prepare(servers: Any, bindings: Any, *, materializer: Any, **kwargs: Any) -> dict[str, Any]:
        projected = await materializer.materialize(store, {"owned.txt": reference})
        roots.append(projected.root)
        return {}

    monkeypatch.setattr(module, "MCPToolset", FailingMCP)
    monkeypatch.setattr(adapter, "prepare_mcp_projections", prepare)
    try:
        key = AssetKey("mcp", "owned.txt")
        await store.put(key, b"must outlive the process")
        reference, = await store.resolve_versions((key,))
        current = [CapabilityContribution.from_declaration(MCPServerSpec(
            "one", transport="streamable-http", url="https://example.invalid/mcp",
        ))]

        async def provider() -> CompiledAgent:
            return _compiled(current)

        ctx = tool_run_context()
        dynamic = await create_current_toolset(
            provider, asset_readers={}, workspace_backend=None, tool_operations=None,
            register_cleanup=cleanups.append,
        ).for_run(ctx)
        await dynamic.__aenter__()
        assert len(cleanups) == 1
        if phase == "partial_entry":
            with pytest.raises(ValueError) as raised:
                await dynamic.for_run_step(ctx)
            assert raised.value is primary and raised.value.__cause__ is failure
        else:
            await dynamic.for_run_step(ctx)
            with pytest.raises(AIError) as raised:
                if phase == "switch":
                    current[:] = [CapabilityContribution.from_declaration(MCPServerSpec(
                        "one", transport="streamable-http", url="https://changed.invalid/mcp",
                    ))]
                    await dynamic.for_run_step(replace(ctx, run_step=2))
                else:
                    await dynamic.__aexit__(None, None, None)
            assert raised.value is failure
        await dynamic.__aexit__(None, None, None)
        assert len(roots) == 1 and roots[0].exists()
        if phase == "switch":
            with pytest.raises(AIError) as retry:
                await dynamic.for_run_step(replace(ctx, run_step=3))
            assert retry.value.safe_details["reason"] == "previous_resources_unclosed"
            assert len(roots) == 1 and events.count("enter") == 1

        # A failed or absent workspace close is not proof that the files are unused.
        with pytest.raises(AIError) as unproven:
            await cleanups[0](False)
        assert unproven.value is failure and roots[0].exists()
        # The run owner has now proved workspace process-tree quiescence. Client
        # diagnostics remain failures, but no longer prevent materializer cleanup.
        with pytest.raises(AIError) as closed:
            await cleanups[0](True)
        assert closed.value is failure and not roots[0].exists()
        client_can_close = True
        await cleanups[0](False)
        calls = len(events)
        await cleanups[0](False)
        assert len(events) == calls
    finally:
        client_can_close = True
        for cleanup in cleanups:
            await cleanup(True)
        await store.close()


@pytest.mark.asyncio
async def test_new_stdio_server_uses_run_owned_lazy_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    import linktools.ai.runtime._mcp as module

    events: list[str] = []
    _fake_mcp(monkeypatch, events)
    backends = []

    class Backend:
        def stdio_execution_policy(self) -> dict[str, Any]:
            return {"version": 1, "boundary": "host-stdio"}

        async def open_stdio_process(self, *args: Any, **kwargs: Any) -> Any:
            pytest.fail("fake MCP must not start a real process")

    backend = Backend()
    workspace_calls = 0

    async def workspace_provider() -> Any:
        nonlocal workspace_calls
        workspace_calls += 1
        return backend

    def transport(server: Any, **kwargs: Any) -> object:
        backends.append(kwargs["workspace_backend"])
        return object()

    monkeypatch.setattr(module, "_create_mcp_transport", transport)
    current = [_tool("business", "ready")]

    async def provider() -> CompiledAgent:
        return _compiled(current)

    ctx = tool_run_context()
    dynamic = await create_current_toolset(
        provider, asset_readers={}, workspace_backend=None, tool_operations=_Bridge(True),
        workspace_provider=workspace_provider,
    ).for_run(ctx)
    async with dynamic:
        await dynamic.for_run_step(ctx)
        assert workspace_calls == 0
        current.append(CapabilityContribution.from_declaration(MCPServerSpec("later", "python")))
        await dynamic.for_run_step(replace(ctx, run_step=2))
        assert backends == [backend] and workspace_calls == 1
        await dynamic.for_run_step(replace(ctx, run_step=3))
        assert backends == [backend] and workspace_calls == 2
        assert events == ["enter:later"]
    assert events == ["enter:later", "exit:later", "close:later"]
