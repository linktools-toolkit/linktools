#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline native MCP catalogs, version binding, and response semantics."""

import asyncio
import copy
from dataclasses import FrozenInstanceError, replace
from typing import Any, Literal

import pytest
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.mcp import MCPToolset

from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.capability import AgentContext, ToolCallFailed, ToolCallRetry
from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError
from linktools.ai.runtime._mcp import (
    _MCPBinding,
    _model_tool_name,
    close_mcp_resources,
    materialize_mcp_capabilities,
)
from linktools.ai.runtime._tool import ToolOperationDecision
from linktools.ai.runtime._tool_response_fixture import ToolResponseFixture
from linktools.ai.spec import MCPServerSpec, mcp_server_selector, mcp_tool_selector
from linktools.ai.storage import StorageOverlay

from ._runtime_test_helpers import tool_run_context
from .test_tool_effect_semantics import _Bridge


def _manifest() -> dict[str, Any]:
    return {
        "version": 1,
        "kind": "mcp-tool-responses",
        "servers": [{
            "ref": {"kind": "mcp", "id": "server", "revision": 3},
            "tools": [{
                "name": "lookup",
                "definition": {
                    "name": _model_tool_name("server", "lookup"),
                    "description": '[MCP identity: {"server_id":"server","tool_name":"lookup"}]\nLookup',
                    "parameters_json_schema": {
                        "type": "object", "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                    "return_schema": {"type": "object"},
                    "include_return_schema": True,
                    "strict": False,
                    "sequential": True,
                },
                "responses": [{
                    "arguments": {"query": "ready", "options": {"a": 1, "b": 2}},
                    "outcome": {"kind": "success", "value": {"url": "https://example.test/info"}},
                }],
            }],
        }],
    }


async def _store(payload: dict[str, Any]) -> tuple[AssetStore, ToolResponseFixture]:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    key = AssetKey("fixture", "one-case.json")
    await store.put(key, canonical_json_bytes(payload))
    ref, = await store.resolve_versions((key,))
    return store, ToolResponseFixture(ref, store)


async def _capabilities(
    fixture: ToolResponseFixture, bridge: _Bridge, selectors: tuple[str, ...] = (),
    *, transport: Literal["stdio", "streamable-http", "sse"] = "stdio",
) -> tuple[AbstractCapability[AgentContext[object]], ...]:
    server = MCPServerSpec(
        "server", "missing-command", revision=3,
    ) if transport == "stdio" else MCPServerSpec(
        "server", transport=transport, url="https://example.invalid/mcp", revision=3,
    )
    manifest = await fixture.load(fixture.ref)
    return await materialize_mcp_capabilities(
        (server,),
        selectors or (mcp_server_selector("server"),),
        sandbox=object(), sandbox_session=None, host_cwd=None,
        bindings={"server": _MCPBinding(None, None, {"boundary": "saved-stdio"})},
        projections={}, tool_operations=bridge, tool_metrics=None,
        response_fixture=manifest,
    )


@pytest.mark.asyncio
async def test_fixture_reads_saved_version_with_borrowed_reader_after_default_changes() -> None:
    store, original = await _store(_manifest())
    try:
        updated = _manifest()
        updated["servers"][0]["tools"][0]["responses"][0]["outcome"]["value"] = "new"
        await store.put(original.ref.key, canonical_json_bytes(updated))
        current_ref, = await store.resolve_versions((original.ref.key,))
        fixture = ToolResponseFixture(current_ref, store)
        loaded = await fixture.load(original.ref)
        tool = loaded.server("server").tools[0]
        assert tool.result({"options": {"b": 2, "a": 1}, "query": "ready"}) == {
            "url": "https://example.test/info",
        }
        definition = tool.tool_definition()
        definition.parameters_json_schema["properties"]["mutated"] = {}
        assert "mutated" not in tool.tool_definition().parameters_json_schema["properties"]
        result = tool.result({"query": "ready", "options": {"a": 1, "b": 2}})
        result["url"] = "mutated"
        assert tool.result({"query": "ready", "options": {"a": 1, "b": 2}})["url"].startswith("https://")
        with pytest.raises(FrozenInstanceError):
            fixture.ref = original.ref
        assert await store.get(original.ref.key) == canonical_json_bytes(updated)
        with pytest.raises(AIError):
            await fixture.load(replace(original.ref, etag="0" * 64))
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("stdio", "streamable-http", "sse"))
async def test_fixture_catalog_and_repeated_calls_stay_offline_and_preserve_effect_policy(
    monkeypatch: pytest.MonkeyPatch, transport: Literal["stdio", "streamable-http", "sse"],
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("fixture must not construct or admit a live MCP transport")

    async def forbidden_enter(self: MCPToolset) -> None:
        raise AssertionError("fixture must not enter a live MCP connection")

    monkeypatch.setattr(MCPToolset, "__aenter__", forbidden_enter)
    for name in ("_create_mcp_transport", "_MCPDiscoveryToolset", "_mcp_execution_policy"):
        monkeypatch.setattr(f"linktools.ai.runtime._mcp.{name}", forbidden)
    payload = _manifest()
    extra = copy.deepcopy(payload["servers"][0]["tools"][0])
    extra["name"] = "other"
    extra["definition"]["name"] = _model_tool_name("server", "other")
    payload["servers"][0]["tools"].append(extra)
    store, fixture = await _store(payload)

    class Bridge(_Bridge):
        async def begin(self, *args: object, **kwargs: object) -> ToolOperationDecision:
            assert args[-1] is False
            return await super().begin(*args, **kwargs)

    bridge = Bridge(False)
    capabilities = ()
    try:
        capabilities = await _capabilities(
            fixture, bridge, (mcp_tool_selector("server", "lookup"),), transport=transport,
        )
        boundary = capabilities[0].get_toolset()
        context = tool_run_context()
        async with boundary:
            tools = await boundary.get_tools(context)
            name = _model_tool_name("server", "lookup")
            assert set(tools) == {name}
            definition = tools[name].tool_def
            saved = payload["servers"][0]["tools"][0]["definition"]
            assert definition.name == saved["name"]
            assert definition.description == saved["description"]
            assert definition.parameters_json_schema == saved["parameters_json_schema"]
            assert definition.return_schema == saved["return_schema"]
            assert definition.include_return_schema is True
            assert definition.strict is False and definition.sequential is True
            for call_id in ("first", "second"):
                result = await boundary.call_tool(
                    name, {"options": {"b": 2, "a": 1}, "query": "ready"},
                    replace(context, tool_call_id=call_id), tools[name],
                )
                assert result == {"url": "https://example.test/info"}
            concurrent = await asyncio.gather(*(
                boundary.call_tool(
                    name, {"query": "ready", "options": {"a": 1, "b": 2}},
                    replace(context, tool_call_id=call_id), tools[name],
                )
                for call_id in ("concurrent-first", "concurrent-second")
            ))
            assert concurrent == [{"url": "https://example.test/info"}] * 2
            with pytest.raises(ToolCallFailed, match="No tool response fixture"):
                await boundary.call_tool(name, {"query": "missing"}, context, tools[name])
            assert "complete" in bridge.calls and "fail" in bridge.calls
            assert "unknown" not in bridge.calls
    finally:
        await close_mcp_resources(capabilities)
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(("kind", "signal"), (("failed", ToolCallFailed), ("retry", ToolCallRetry)))
async def test_fixture_native_failure_outcomes_settle_without_unknown(kind: str, signal: type[Exception]) -> None:
    payload = _manifest()
    payload["servers"][0]["tools"][0]["responses"][0]["outcome"] = {"kind": kind, "message": "Try another query"}
    store, fixture = await _store(payload)
    bridge = _Bridge(False)
    capabilities = ()
    try:
        capabilities = await _capabilities(fixture, bridge)
        boundary = capabilities[0].get_toolset()
        context = tool_run_context()
        tools = await boundary.get_tools(context)
        name, tool = next(iter(tools.items()))
        with pytest.raises(signal, match="Try another query"):
            await boundary.call_tool(name, {"query": "ready", "options": {"a": 1, "b": 2}}, context, tool)
        assert "fail" in bridge.calls and "unknown" not in bridge.calls
    finally:
        await close_mcp_resources(capabilities)
        await store.close()


@pytest.mark.asyncio
async def test_fixture_coverage_and_named_revision_must_match_selected_servers() -> None:
    store, fixture = await _store(_manifest())
    try:
        manifest = await fixture.load(fixture.ref)
        for server, selectors in (
            (MCPServerSpec("missing", "unused"), (mcp_server_selector("missing"),)),
            (MCPServerSpec("server", "unused", revision=4), (mcp_server_selector("server"),)),
            (MCPServerSpec("server", "unused", revision=3), (mcp_tool_selector("server", "missing"),)),
            (MCPServerSpec("server", "unused", revision=3), (mcp_server_selector("other"),)),
        ):
            with pytest.raises(AIError):
                manifest.validate_servers((server,), selectors)
        manifest.validate_servers((), ())
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", (
    "version", "unknown_field", "missing_schema", "unknown_outcome", "raw_mcp",
    "multimodal", "duplicate_tool", "conflicting_response", "wrong_model_name",
))
async def test_fixture_rejects_unsupported_or_conflicting_inputs(invalid: str) -> None:
    payload = _manifest()
    tool = payload["servers"][0]["tools"][0]
    response = tool["responses"][0]
    if invalid == "version":
        payload["version"] = 2
    elif invalid == "unknown_field":
        tool["definition"]["unknown"] = "ignored"
    elif invalid == "missing_schema":
        del tool["definition"]["parameters_json_schema"]
    elif invalid == "unknown_outcome":
        response["outcome"] = {"kind": "unknown", "message": "uncertain"}
    elif invalid == "raw_mcp":
        response["outcome"] = {"isError": True, "content": [{"type": "text", "text": "failed"}]}
    elif invalid == "multimodal":
        response["outcome"] = {"kind": "resource_link", "uri": "https://example.test/file"}
    elif invalid == "duplicate_tool":
        conflicting = copy.deepcopy(tool)
        conflicting["definition"]["parameters_json_schema"] = {"type": "object", "properties": {}}
        payload["servers"][0]["tools"].append(conflicting)
    elif invalid == "conflicting_response":
        conflicting = copy.deepcopy(response)
        conflicting["outcome"]["value"] = "different"
        tool["responses"].append(conflicting)
    else:
        tool["definition"]["name"] = "wrong"
    store, fixture = await _store(payload)
    try:
        with pytest.raises(AIError):
            await _capabilities(fixture, _Bridge(False))
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_identical_argument_rows_do_not_consume_results() -> None:
    payload = _manifest()
    tool = payload["servers"][0]["tools"][0]
    tool["responses"].append(copy.deepcopy(tool["responses"][0]))
    tool["responses"][0]["outcome"]["value"] = {"kind": "binary", "isError": True, "content": [], "url": "https://example.test"}
    tool["responses"][1] = copy.deepcopy(tool["responses"][0])
    store, fixture = await _store(payload)
    try:
        manifest = await fixture.load(fixture.ref)
        recorded = manifest.server("server").tools[0]
        arguments = tool["responses"][0]["arguments"]
        assert recorded.result(arguments) == recorded.result(arguments)
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", (
    b'{"version":1,"version":1,"kind":"mcp-tool-responses","servers":[]}',
    b'{"version":1,"kind":"mcp-tool-responses","servers":[],"number":NaN}',
))
async def test_fixture_rejects_ambiguous_or_nonfinite_json(raw: bytes) -> None:
    store, fixture = await _store(_manifest())
    try:
        await store.put(fixture.ref.key, raw)
        invalid_ref, = await store.resolve_versions((fixture.ref.key,))
        with pytest.raises(AIError):
            await fixture.load(invalid_ref)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_native_runtime_consumes_fixture_through_durable_mcp_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections.abc import Sequence

    from pydantic import TypeAdapter
    from pydantic_ai.messages import (
        ModelMessage, ModelMessagesTypeAdapter, ModelRequest, ModelResponse,
        TextPart, ToolCallPart, ToolReturnPart,
    )
    from pydantic_ai.models.function import AgentInfo
    from pydantic_ai.tools import ToolDefinition
    from pydantic_ai.toolsets import FunctionToolset

    from linktools.ai.asset import AssetMaterializer
    from linktools.ai.capability import CapabilityGroup
    from linktools.ai.core import ExecutionStatus, JsonValue, ToolOperationStatus
    from linktools.ai.runtime import Runtime, RuntimeStorage
    from linktools.ai.runtime._mcp import _MCPModelToolset, prepare_mcp_projections
    from linktools.ai.workspace import SandboxResource

    from ._runtime_test_helpers import _UsageFunctionModel
    from .test_live_history_readback_integration import _Models

    def forbidden_transport(*args: object, **kwargs: object) -> None:
        raise AssertionError("fixture execution must not construct an MCP transport")

    async def forbidden_live_path(*args: object, **kwargs: object) -> None:
        raise AssertionError("fixture execution must not enter MCP or project its resources")

    async def no_live_projection(
        servers: Sequence[MCPServerSpec], *args: Any, **kwargs: Any,
    ) -> dict:
        assert not servers
        return await prepare_mcp_projections(servers, *args, **kwargs)

    for name in ("_create_mcp_transport", "_MCPDiscoveryToolset"):
        monkeypatch.setattr(f"linktools.ai.runtime._mcp.{name}", forbidden_transport)
    monkeypatch.setattr(MCPToolset, "__aenter__", forbidden_live_path)
    monkeypatch.setattr(SandboxResource, "from_asset_versions", forbidden_live_path)
    monkeypatch.setattr(AssetMaterializer, "materialize", forbidden_live_path)
    monkeypatch.setattr(
        "linktools.ai.runtime._agent_executor.prepare_mcp_projections", no_live_projection,
    )

    async def lookup(query: str, options: dict[str, int]) -> dict[str, str]:
        """Look up a query using the requested options."""
        raise AssertionError("the captured upstream leaf must never execute")

    mapped = _MCPModelToolset(FunctionToolset([lookup]), "server", None)
    captured = await mapped.get_tools(tool_run_context())
    captured_definition = next(iter(captured.values())).tool_def
    payload = _manifest()
    recorded_tool = payload["servers"][0]["tools"][0]
    recorded_tool["definition"] = TypeAdapter(ToolDefinition).dump_python(
        captured_definition, mode="json",
    )
    sentinel = "UNUSED_FIXTURE_RESPONSE_94B"
    recorded_tool["responses"].append({
        "arguments": {"query": "unmatched", "options": {}},
        "outcome": {"kind": "success", "value": {"expected_answer": sentinel}},
    })
    name = recorded_tool["definition"]["name"]
    expected = recorded_tool["responses"][0]["outcome"]["value"]
    consumed: list[JsonValue] = []

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        visible = "\n".join((
            ModelMessagesTypeAdapter.dump_json(messages).decode("utf-8"),
            TypeAdapter(list[ToolDefinition]).dump_json(
                info.model_request_parameters.function_tools,
            ).decode("utf-8"),
            info.instructions or "",
        ))
        for private_value in (sentinel, "expected_answer", "mcp-tool-responses", "one-case.json"):
            assert private_value not in visible
        returned = [
            part for message in messages if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_call_id == "fixture-call"
        ]
        if returned:
            assert returned[-1].content == expected
            consumed.append(returned[-1].content)
            return ModelResponse(parts=[TextPart("Fixture consumed")])
        definition = next(tool for tool in info.function_tools if tool.name == name)
        assert definition.parameters_json_schema == recorded_tool["definition"]["parameters_json_schema"]
        assert definition.description == recorded_tool["definition"]["description"]
        for key, value in recorded_tool["definition"]["metadata"].items():
            assert definition.metadata[key] == value
        return ModelResponse(parts=[ToolCallPart(
            name, {"query": "ready", "options": {"b": 2, "a": 1}},
            tool_call_id="fixture-call",
        )])

    assets, fixture = await _store(payload)
    try:
        await assets.put(AssetKey("mcp", "server/server.py"), b"raise AssertionError('must not run')")
        group = CapabilityGroup("fixture-native-runtime", assets=assets)
        group.mcp(MCPServerSpec(
            "server", "fixture-server-must-not-start", ("resource:server.py",),
            AssetKey("mcp", "server"), revision=3,
        ))
        group.agent("default", model="default", allow_tools=(mcp_tool_selector("server", "lookup"),))
        storage = RuntimeStorage.in_memory()
        async with Runtime.open(
            "fixture-native-runtime", models=_Models(_UsageFunctionModel(model)),
            storage=storage, capabilities=(group,), tool_responses=fixture,
        ) as runtime:
            execution = await runtime.agents.get("default").start("Look up the fixture")
            result = (await execution.wait(timeout_seconds=10)).result
            assert result.status is ExecutionStatus.SUCCEEDED
            assert result.output == {"text": "Fixture consumed"}
            assert consumed == [expected]
            operations = await storage.recovery.tools.list_by_execution(
                execution.execution_id, tenant_id=runtime.default_principal.tenant_id,
            )
            assert len(operations) == 1
            operation = operations[0]
            assert operation.tool_name == name and operation.tool_call_id == "fixture-call"
            assert operation.status is ToolOperationStatus.COMPLETED
            assert operation.replay_safe is False
            assert operation.result_payload is not None
        assert await assets.read_versions((fixture.ref,)) == (canonical_json_bytes(payload),)
    finally:
        await assets.close()
