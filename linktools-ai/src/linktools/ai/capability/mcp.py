#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCP protocol tools and their owned client lifecycle."""

import asyncio
import json
from collections.abc import Mapping
from dataclasses import replace
from types import TracebackType
from typing import Any, NoReturn

import anyio
import httpx
from fastmcp.client.transports import ClientTransport
from fastmcp.exceptions import McpError
from httpx2 import HTTPError as HTTPErrorV2, HTTPStatusError as HTTPStatusErrorV2
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from jsonschema.validators import validator_for
from mcp.types import Tool as MCPTool
from pydantic import ValidationError
from pydantic_ai.mcp import MCPToolset as SDKMCPToolset
from pydantic_ai.tools import RunContext as PydanticRunContext, ToolDefinition
from pydantic_ai.toolsets import ToolsetTool
from referencing import Registry
from referencing.exceptions import InvalidAnchor, NoSuchAnchor, PointerToNowhere, Unresolvable

from ..core import JsonValue
from ..errors import AIError, ErrorCode
from ..spec import MCPServerSpec
from ._tool_signal import ToolCallRetry


class MCPToolset(SDKMCPToolset[object]):
    """Own MCP discovery, argument feedback and final client cleanup.

    The caller supplies a prepared transport and retains its workspace/assets.
    SDK entry and reference counting are retained; call close_resources during final
    owned cleanup, including after failed entry or exit. This does not provide
    Runtime authorization, stable model names or execution accounting.
    """
    def __init__(self, transport: ClientTransport, *, server: MCPServerSpec) -> None:
        super().__init__(
            transport,
            id=f"mcp:{server.id}",
            cache_tools=True,
            init_timeout=server.init_timeout,
            read_timeout=server.read_timeout,
        )
        self._init_timeout = server.init_timeout
        self._server_id = server.id
        self._transport_kind = server.transport
        self._cleanup_error: BaseException | None = None

    def tool_for_tool_def(
        self,
        tool_def: ToolDefinition,
        *,
        ctx: PydanticRunContext[object],
    ) -> ToolsetTool[object]:
        tool = super().tool_for_tool_def(tool_def, ctx=ctx)
        schema = tool_def.parameters_json_schema
        validator_type = validator_for(
            schema, default=None if "$schema" in schema else Draft202012Validator,
        )
        if validator_type is None:
            # The server remains authoritative for a dialect this client cannot evaluate.
            return tool
        validator_type.check_schema(schema)
        validator = validator_type(schema, registry=Registry())

        def validate_arguments(context: PydanticRunContext[object], /, **arguments: Any) -> None:
            del context
            try:
                validator.validate(arguments)
            except Unresolvable as error:
                if isinstance(error.__cause__ or error, (InvalidAnchor, NoSuchAnchor, PointerToNowhere)):
                    raise
                # The registry never fetches external documents. Their semantics
                # remain with the server, unlike broken references within this schema.
                return
            except JsonSchemaValidationError as error:
                schema_path = tuple(error.absolute_schema_path)
                field = next((schema_path[index + 1] for index, item in enumerate(schema_path[:-1])
                              if item == "properties"), None)
                location = f" field {json.dumps(str(field)[:80])}" if field is not None else ""
                raise ToolCallRetry(
                    f"MCP tool arguments{location} violate the {error.validator} schema constraint. "
                    "Check the tool schema, correct the argument fields and types, and retry."
                ) from None

        return replace(tool, args_validator_func=validate_arguments)

    def _details(self, phase: str) -> dict[str, JsonValue]:
        return {
            "server_id": self._server_id,
            "transport": self._transport_kind,
            "phase": phase,
        }

    async def __aenter__(self) -> "MCPToolset":
        try:
            with anyio.fail_after(self._init_timeout):
                await super().__aenter__()
        except BaseException as error:
            _raise_connection_failure(error, self._details("connect"))
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        try:
            return await super().__aexit__(exc_type, exc_value, traceback)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            try:
                _raise_cleanup_failure(error, self._details("close"))
            except BaseException as cleanup_error:
                # Upstream wrapper exits may discard the primary exception.
                # Final owned cleanup still has it and can report both safely.
                if self._cleanup_error is None:
                    self._cleanup_error = cleanup_error
            return None

    async def list_tools(self) -> list[MCPTool]:
        try:
            tools = await super().list_tools()
        except BaseException as error:
            _raise_connection_failure(error, self._details("tools/list"))
        names = tuple(tool.name for tool in tools)
        if len(names) != len(set(names)):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        return tools

    async def close_resources(self) -> None:
        """Close the owned client and report any earlier SDK exit failure."""
        try:
            await self.client.close()
        except asyncio.CancelledError as cancelled:
            if self._cleanup_error is not None:
                raise cancelled from self._cleanup_error
            raise
        except BaseException as error:
            try:
                _raise_cleanup_failure(error, self._details("close"))
            except BaseException as cleanup_error:
                if self._cleanup_error is None:
                    self._cleanup_error = cleanup_error
        if self._cleanup_error is not None:
            raise self._cleanup_error


def _raise_cleanup_failure(
    error: BaseException,
    details: Mapping[str, JsonValue] | None = None,
) -> NoReturn:
    if isinstance(error, (AIError, asyncio.CancelledError)):
        raise error
    raise AIError(ErrorCode.MCP_CLEANUP_FAILED, safe_details=details) from None


def _connection_errors(error: BaseException) -> tuple[BaseException, ...] | None:
    if isinstance(error, (AIError, asyncio.CancelledError)):
        raise error
    nested = getattr(error, "exceptions", None)
    if isinstance(nested, tuple):
        values: list[BaseException] = []
        for child in nested:
            result = _connection_errors(child)
            if result is None:
                return None
            values.extend(result)
        return tuple(values) if values else None
    if isinstance(error, (
        OSError, TimeoutError, httpx.HTTPError, HTTPErrorV2, McpError, ValidationError,
        anyio.EndOfStream, anyio.ClosedResourceError, anyio.BrokenResourceError,
    )):
        return (error,)
    if type(error) is RuntimeError:
        message = str(error)
        cause = error.__cause__
        # FastMCP exposes its connection wrapper as a plain RuntimeError.
        # Other RuntimeError causes can be secondary cleanup failures.
        if cause is not None and message == f"Client failed to connect: {cause}":
            return _connection_errors(cause)
        # SDK negotiation failures have no public error type, and transport
        # exception-group unwrapping can remove their original timeout cause.
        if message in {
            "Failed to initialize server session",
            "Server session was closed unexpectedly",
        } or message.startswith("Unsupported protocol version from the server:"):
            return (error,)
    return None


def _raise_connection_failure(
    error: BaseException,
    details: Mapping[str, JsonValue],
) -> NoReturn:
    failures = _connection_errors(error)
    if failures is None:
        raise error
    safe_details = dict(details)
    for failure in failures:
        if isinstance(failure, (httpx.HTTPStatusError, HTTPStatusErrorV2)):
            safe_details["status_code"] = failure.response.status_code
            break
    raise AIError(ErrorCode.MCP_CONNECTION_FAILED, safe_details=safe_details) from None



__all__ = ["MCPToolset"]
