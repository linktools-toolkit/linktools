#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asset-backed native MCP JSON/text responses for one evaluation case."""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import TypeAdapter, ValidationError
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool

from ..asset import AssetStoreReader, AssetVersionRef
from ..capability import ToolCallFailed, ToolCallRetry
from ..core import JsonValue, canonical_json_bytes, normalize_json_value
from ..errors import AIError, ErrorCode
from ..spec import MCPServerSpec, mcp_tool_selector, parse_mcp_tool_selector

_DEFINITION_FIELDS = frozenset({
    "name", "parameters_json_schema", "description", "outer_typed_dict_key",
    "strict", "sequential", "kind", "metadata", "timeout", "defer_loading",
    "unless_native", "with_native", "tool_kind", "return_schema",
    "include_return_schema", "toolset_id", "capability_id",
})
_DEFINITION_ADAPTER = TypeAdapter(ToolDefinition)
_ARGUMENTS_ADAPTER = TypeAdapter(dict[str, Any])


@dataclass(frozen=True, slots=True)
class ToolResponseFixture:
    """Bind a default immutable response Asset to a borrowed version reader.

    The Runtime pins ``ref`` at admission. Recovery reads its saved reference
    through the same reader, even when the Runtime's default has changed.
    This object never initializes or closes the reader.
    """

    ref: AssetVersionRef
    reader: AssetStoreReader = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.ref, AssetVersionRef):
            raise TypeError("tool response fixture ref must be AssetVersionRef")
        if not isinstance(self.reader, AssetStoreReader):
            raise TypeError("tool response fixture reader must be AssetStoreReader")

    async def load(self, saved_ref: AssetVersionRef) -> "_ToolResponseManifest":
        """Read exactly the execution's pinned Asset version, without fallback."""
        if not isinstance(saved_ref, AssetVersionRef):
            raise TypeError("saved fixture ref must be AssetVersionRef")
        values = await self.reader.read_versions((saved_ref,))
        if len(values) != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            payload = json.loads(values[0], object_pairs_hook=_unique_object)
            return _decode_manifest(normalize_json_value(payload))
        except (ValueError, TypeError, UnicodeError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


@dataclass(frozen=True, slots=True)
class _ToolResponse:
    arguments: bytes
    outcome: bytes

    def result(self) -> JsonValue:
        value = json.loads(self.outcome)
        if value["kind"] == "failed":
            raise ToolCallFailed(value["message"])
        if value["kind"] == "retry":
            raise ToolCallRetry(value["message"])
        return value["value"]


@dataclass(frozen=True, slots=True)
class _FixtureTool:
    name: str
    definition: bytes
    responses: tuple[_ToolResponse, ...]

    def tool_definition(self) -> ToolDefinition:
        return _DEFINITION_ADAPTER.validate_json(self.definition, strict=True)

    def result(self, arguments: dict[str, Any]) -> JsonValue:
        try:
            canonical = canonical_json_bytes(normalize_json_value(arguments))
        except (TypeError, ValueError) as error:
            raise ToolCallFailed("Tool response fixture arguments are not JSON") from error
        for response in self.responses:
            if response.arguments == canonical:
                return response.result()
        raise ToolCallFailed("No tool response fixture matches these arguments")


@dataclass(frozen=True, slots=True)
class _FixtureServer:
    id: str
    revision: int
    tools: tuple[_FixtureTool, ...]


@dataclass(frozen=True, slots=True)
class _ToolResponseManifest:
    servers: tuple[_FixtureServer, ...]

    def server(self, server_id: str) -> _FixtureServer:
        for server in self.servers:
            if server.id == server_id:
                return server
        raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING)

    def validate_servers(
        self,
        servers: Sequence[MCPServerSpec],
        selectors: Sequence[str],
    ) -> None:
        """Require every selected MCP server and explicitly named tool offline."""
        selected: dict[str, set[str]] = {}
        for selector in selectors:
            parsed = parse_mcp_tool_selector(selector)
            if parsed is not None:
                server_id, tool_name = parsed
                names = selected.setdefault(server_id, set())
                if tool_name is not None:
                    names.add(tool_name)
        if set(selected) != {server.id for server in servers}:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        for declaration in servers:
            server = self.server(declaration.id)
            if server.revision != declaration.revision:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            if not selected[server.id].issubset(tool.name for tool in server.tools):
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


class _FixtureToolset(AbstractToolset[object]):
    def __init__(
        self,
        server: _FixtureServer,
        allowed_tools: frozenset[str] | None,
    ) -> None:
        self._server_id = server.id
        self._tools = {
            tool.tool_definition().name: tool
            for tool in server.tools
            if allowed_tools is None or tool.name in allowed_tools
        }

    @property
    def id(self) -> str:
        return f"mcp:{self._server_id}"

    async def get_tools(self, ctx: RunContext[object]) -> dict[str, ToolsetTool[object]]:
        return {
            name: ToolsetTool(
                toolset=self,
                tool_def=tool.tool_definition(),
                max_retries=ctx.max_retries,
                args_validator=_ARGUMENTS_ADAPTER.validator,
            )
            for name, tool in self._tools.items()
        }

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[object],
        tool: ToolsetTool[object],
    ) -> JsonValue:
        fixture = self._tools.get(name)
        if fixture is None:
            raise ToolCallFailed("Tool is not present in the response fixture")
        return fixture.result(tool_args)


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    value: dict[str, JsonValue] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("fixture JSON has duplicate object keys")
        value[key] = item
    return value


def _object(value: JsonValue, keys: set[str]) -> dict[str, JsonValue]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("fixture object fields are invalid")
    return value


def _decode_manifest(value: JsonValue) -> _ToolResponseManifest:
    payload = _object(value, {"version", "kind", "servers"})
    if type(payload["version"]) is not int or payload["version"] != 1:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    if payload["kind"] != "mcp-tool-responses" or not isinstance(payload["servers"], list):
        raise ValueError("tool response fixture format is invalid")
    servers: list[_FixtureServer] = []
    server_ids: set[str] = set()
    model_names: set[str] = set()
    for item in payload["servers"]:
        row = _object(item, {"ref", "tools"})
        ref = _object(row["ref"], {"kind", "id", "revision"})
        identity, revision = ref["id"], ref["revision"]
        if (
            ref["kind"] != "mcp" or not isinstance(identity, str) or not identity.strip()
            or type(revision) is not int or revision < 1 or not isinstance(row["tools"], list)
        ):
            raise ValueError("fixture MCP reference is invalid")
        if identity in server_ids:
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        server_ids.add(identity)
        tools: list[_FixtureTool] = []
        names: set[str] = set()
        for tool_row in row["tools"]:
            tool = _decode_tool(identity, tool_row)
            model_name = tool.tool_definition().name
            if tool.name in names or model_name in model_names:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            names.add(tool.name)
            model_names.add(model_name)
            tools.append(tool)
        servers.append(_FixtureServer(identity, revision, tuple(tools)))
    return _ToolResponseManifest(tuple(servers))


def _decode_tool(server_id: str, value: JsonValue) -> _FixtureTool:
    row = _object(value, {"name", "definition", "responses"})
    name, definition = row["name"], row["definition"]
    if not isinstance(name, str) or not name:
        raise ValueError("fixture original tool name is invalid")
    mcp_tool_selector(server_id, name)
    if (
        not isinstance(definition, dict)
        or not {"name", "parameters_json_schema"}.issubset(definition)
        or not set(definition).issubset(_DEFINITION_FIELDS)
        or not isinstance(definition["parameters_json_schema"], dict)
    ):
        raise ValueError("fixture tool definition is incomplete or unsupported")
    encoded = canonical_json_bytes(definition)
    try:
        decoded = _DEFINITION_ADAPTER.validate_json(encoded, strict=True)
    except ValidationError as error:
        raise ValueError("fixture tool definition is invalid") from error
    if not decoded.name or decoded.kind != "function" or decoded.tool_kind is not None:
        raise ValueError("fixture requires ordinary MCP function tools")
    rows = row["responses"]
    if not isinstance(rows, list):
        raise ValueError("fixture responses must be a list")
    responses: dict[bytes, bytes] = {}
    for item in rows:
        response = _object(item, {"arguments", "outcome"})
        if not isinstance(response["arguments"], dict):
            raise ValueError("fixture arguments must be an object")
        arguments = canonical_json_bytes(response["arguments"])
        outcome = _decode_outcome(response["outcome"])
        previous = responses.get(arguments)
        if previous is not None and previous != outcome:
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        responses[arguments] = outcome
    return _FixtureTool(name, encoded, tuple(
        _ToolResponse(arguments, outcome) for arguments, outcome in responses.items()
    ))


def _decode_outcome(value: JsonValue) -> bytes:
    if not isinstance(value, dict):
        raise ValueError("fixture outcome must be an object")
    kind = value.get("kind")
    if kind == "success":
        _object(value, {"kind", "value"})
    elif kind in {"failed", "retry"}:
        _object(value, {"kind", "message"})
        signal = ToolCallFailed if kind == "failed" else ToolCallRetry
        signal(value["message"])
    else:
        raise ValueError("fixture outcome must be native success, failed, or retry")
    return canonical_json_bytes(value)


__all__ = ["ToolResponseFixture"]
