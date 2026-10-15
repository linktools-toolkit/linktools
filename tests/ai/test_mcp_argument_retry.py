#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCP failures without a known execution outcome retain effect uncertainty."""

from typing import Any

import pytest
from fastmcp.client.transports import StdioTransport
from fastmcp.exceptions import McpError, ToolError
from mcp.types import (
    ErrorData, Implementation, InitializeResult, INTERNAL_ERROR, INVALID_PARAMS, ServerCapabilities,
)
from pydantic_ai.tools import ToolDefinition

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._mcp import _MCPDiscoveryToolset
from linktools.ai.runtime._tool_boundary import ManagedToolDescriptor
from linktools.ai.spec import MCPServerSpec

from ._runtime_test_helpers import tool_run_context
from .test_tool_effect_semantics import _Bridge, _call


def _protocol_error(code: int, message: str) -> McpError:
    try:
        return McpError(ErrorData(code=code, message=message))
    except TypeError:
        return McpError(code, message)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("invalid_params", "internal", "tool", "transport", "task_invalid_params"))
async def test_mcp_client_failures_do_not_claim_known_no_effect(
    monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    errors = {
        "invalid_params": _protocol_error(INVALID_PARAMS, "rejected arguments"),
        "task_invalid_params": _protocol_error(INVALID_PARAMS, "task result request rejected"),
        "internal": _protocol_error(INTERNAL_ERROR, "server failed"),
        "tool": ToolError("tool execution failed"),
        "transport": ConnectionError("response lost"),
    }
    error = errors[failure]
    calls = 0
    toolset = _MCPDiscoveryToolset(
        StdioTransport(command="unused-in-memory-client", args=[]),
        server=MCPServerSpec("effect-boundary", "in-memory"),
    )

    async def reject(**kwargs: Any) -> None:
        nonlocal calls
        assert kwargs["name"] == "mutate"
        calls += 1
        raise error

    class Client:
        initialize_result = InitializeResult(
            protocolVersion="2025-03-26", capabilities=ServerCapabilities(),
            serverInfo=Implementation(name="effect-boundary", version="1"),
        )

        async def __aenter__(self) -> "Client":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def call_tool(self, **kwargs: Any) -> None:
            return await reject(**kwargs)

    monkeypatch.setattr(toolset, "client", Client())
    context = tool_run_context()
    task = failure == "task_invalid_params"
    tool = toolset.tool_for_tool_def(
        ToolDefinition(name="mutate", metadata={"task": task}), ctx=context,
    )
    if task:
        async def reject_task(name: str, args: dict[str, Any], metadata: object) -> None:
            await reject(name=name, arguments=args, meta=metadata)

        monkeypatch.setattr(toolset, "_call_tool_as_task", reject_task)

    async def mutate() -> object:
        return await toolset.call_tool("mutate", {}, context, tool)

    bridge = _Bridge(False)
    with pytest.raises(AIError) as raised:
        await _call(
            mutate,
            ManagedToolDescriptor(
                effect_owner="tool_operation", effect_policy="non_replay_safe", tool_class="mcp",
            ),
            bridge=bridge,
        )
    assert raised.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
    assert bridge.calls == ["begin", "unknown"]
    assert calls == 1


def _schema_tool(schema: dict[str, Any]) -> Any:
    toolset = _MCPDiscoveryToolset(
        StdioTransport(command="unused-in-memory-client", args=[]),
        server=MCPServerSpec("schema-boundary", "in-memory"),
    )
    return toolset.tool_for_tool_def(
        ToolDefinition(name="mutate", parameters_json_schema=schema), ctx=tool_run_context(),
    )


@pytest.mark.parametrize(
    ("schema", "arguments", "keyword"),
    (
        ({"type": "object", "required": ["value"]}, {}, "required"),
        ({"type": "object", "properties": {"value": {"type": "integer"}}},
         {"value": "private-instance-value"}, "type"),
        ({"type": "object", "properties": {"value": {"enum": ["allowed"]}}},
         {"value": "private-instance-value"}, "enum"),
        ({"$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
          "properties": {"value": {"type": "array", "items": [{"type": "integer"}]}}},
         {"value": ["private-instance-value"]}, "type"),
        ({"type": "object", "$defs": {"node": {"type": "object", "properties": {
            "next": {"$ref": "#/$defs/node"}, "value": {"type": "integer"}}}},
          "properties": {"root": {"$ref": "#/$defs/node"}}},
         {"root": {"next": {"value": "private-instance-value"}}}, "type"),
    ),
)
def test_mcp_schema_rejections_are_safe_argument_feedback(
    schema: dict[str, Any], arguments: dict[str, Any], keyword: str,
) -> None:
    from linktools.ai.capability import ToolCallRetry

    tool = _schema_tool(schema)
    assert tool.args_validator_func is not None
    with pytest.raises(ToolCallRetry) as caught:
        tool.args_validator_func(tool_run_context(), **arguments)
    assert keyword in str(caught.value)
    assert "private-instance-value" not in str(caught.value)
    import traceback

    assert "private-instance-value" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize(
    ("schema", "arguments"),
    (
        ({"type": "object", "properties": {
            "value": {"type": "integer"}, "context": {"type": "string"}}},
         {"value": 1, "context": "tool context", "extra": "permitted"}),
        ({"$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
          "properties": {"value": {"type": "array", "items": [{"type": "integer"}, {"type": "string"}],
                                    "additionalItems": False}}}, {"value": [1, "item"]}),
        ({"type": "object", "$defs": {"node": {"type": "object", "properties": {
            "next": {"$ref": "#/$defs/node"}, "value": {"type": "integer"}}}},
          "properties": {"root": {"$ref": "#/$defs/node"}}},
         {"root": {"value": 1, "next": {"value": 2}}}),
    ),
)
def test_mcp_schema_accepts_draft_semantics_and_local_references(
    schema: dict[str, Any], arguments: dict[str, Any],
) -> None:
    tool = _schema_tool(schema)
    assert tool.args_validator_func is not None
    tool.args_validator_func(tool_run_context(), **arguments)


def test_mcp_malformed_schema_is_not_model_argument_feedback() -> None:
    from jsonschema.exceptions import SchemaError

    with pytest.raises(SchemaError):
        tool = _schema_tool({"type": "object", "properties": {"value": {"type": "not-a-type"}}})
        assert tool.args_validator_func is not None
        tool.args_validator_func(tool_run_context(), value="value")


@pytest.mark.asyncio
@pytest.mark.parametrize("correct", (True, False))
async def test_mcp_schema_feedback_precedes_dispatch_and_obeys_retry_budget(
    monkeypatch: pytest.MonkeyPatch, correct: bool,
) -> None:
    from types import SimpleNamespace

    from mcp.types import TextContent
    from pydantic_ai import Agent
    from pydantic_ai.exceptions import UnexpectedModelBehavior
    from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, TextPart, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    from linktools.ai.runtime._pydantic_tool_control import PydanticToolControlCapability

    toolset = _MCPDiscoveryToolset(
        StdioTransport(command="unused-in-memory-client", args=[]),
        server=MCPServerSpec("schema-agent", "in-memory"),
    )
    requests = 0
    dispatched: list[dict[str, Any]] = []

    class Client:
        initialize_result = InitializeResult(
            protocolVersion="2025-03-26", capabilities=ServerCapabilities(),
            serverInfo=Implementation(name="schema-agent", version="1"),
        )

        async def __aenter__(self) -> "Client":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def call_tool(self, **kwargs: Any) -> object:
            dispatched.append(kwargs["arguments"])
            return SimpleNamespace(
                content=[TextContent(type="text", text="effect complete")],
                structured_content=None, is_error=False,
            )

    async def get_tools(ctx: Any) -> dict[str, Any]:
        return {"mutate": toolset.tool_for_tool_def(ToolDefinition(
            name="mutate", parameters_json_schema={
                "type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"],
            },
        ), ctx=ctx)}

    monkeypatch.setattr(toolset, "client", Client())
    monkeypatch.setattr(toolset, "get_tools", get_tools)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal requests
        del info
        requests += 1
        if requests == 2:
            assert dispatched == []
            feedback = [part for part in messages[-1].parts if isinstance(part, RetryPromptPart)]
            assert len(feedback) == 1
            assert feedback[0].tool_call_id == "invalid"
            assert "private-instance-value" not in str(feedback[0].content)
            assert "type" in str(feedback[0].content)
        if requests <= 2:
            value = 1 if correct and requests == 2 else "private-instance-value"
            return ModelResponse(parts=[ToolCallPart(
                "mutate", {"value": value}, tool_call_id="invalid" if requests == 1 else "next",
            )])
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(FunctionModel(respond), toolsets=[toolset],
                  capabilities=[PydanticToolControlCapability()], retries=1)
    if correct:
        result = await agent.run("Call the tool")
        assert result.output == "done"
        assert requests == 3
        assert dispatched == [{"value": 1}]
    else:
        with pytest.raises(UnexpectedModelBehavior):
            await agent.run("Call the tool")
        assert requests == 2
        assert dispatched == []


def test_mcp_unknown_schema_dialect_defers_validation_to_server() -> None:
    tool = _schema_tool({
        "$schema": "https://example.invalid/custom-schema", "type": "object",
        "properties": {"value": {"type": "integer"}},
    })
    assert tool.args_validator_func is None


def test_mcp_external_schema_reference_does_not_download_or_reject_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket

    connections: list[object] = []

    def reject_connection(sock: socket.socket, address: object) -> None:
        connections.append(address)
        raise AssertionError("schema validation must not access the network")

    monkeypatch.setattr(socket.socket, "connect", reject_connection)
    tool = _schema_tool({
        "type": "object", "properties": {
            "value": {"$ref": "https://example.invalid/types.json#/$defs/value"},
        },
    })
    assert tool.args_validator_func is not None
    tool.args_validator_func(tool_run_context(), value="server-validated")
    assert connections == []


@pytest.mark.parametrize("reference", ("#/$defs/missing", "#missing-anchor"))
def test_mcp_broken_local_schema_reference_is_not_argument_feedback(reference: str) -> None:
    from referencing.exceptions import Unresolvable

    tool = _schema_tool({"type": "object", "properties": {"value": {"$ref": reference}}})
    assert tool.args_validator_func is not None
    with pytest.raises(Unresolvable):
        tool.args_validator_func(tool_run_context(), value="not-the-problem")
