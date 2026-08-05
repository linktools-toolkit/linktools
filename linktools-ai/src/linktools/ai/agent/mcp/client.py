#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Construct MCP clients and own session-scoped MCP resources."""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Mapping

from linktools.core import environ

from ...errors import (
    McpCleanupRequiredError,
    MCPAuthenticationError,
    MCPConnectionError,
    MCPConnectionUnavailableError,
    MCPDiscoveryError,
    MCPDiscoveryUnsupportedError,
    MCPToolDefinitionError,
)
from ...execution.cancellation import TaskTermination, cancel_task
from ...json import canonical_json_bytes
from .models import MCPConnectionRef, MCPDiscoveryResult, MCPToolInfo

logger = environ.get_logger("ai.agent.mcp.client")

if TYPE_CHECKING:
    from .spec import MCPServerSpec


def _resolved_tool_prefix(spec: "MCPServerSpec") -> "str | None":
    value = spec.tool_prefix
    if value is False:
        return None
    if value is None or value is True:
        return spec.id
    return value


@dataclass(frozen=True, slots=True)
class MCPToolsetHandle:
    connection_ref: MCPConnectionRef
    toolset: Any


def _digest_mapping(values: "Mapping[str, str]") -> str:
    canonical = json.dumps(
        sorted(values.items()), ensure_ascii=False, separators=(",", ":")
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _config_fingerprint(spec: "MCPServerSpec") -> str:
    payload = {
        "transport": spec.transport,
        "command": list(spec.command) if spec.command is not None else None,
        "url": spec.url,
        "cwd": spec.cwd,
        "timeout_seconds": spec.timeout_seconds,
        "tool_prefix": spec.tool_prefix,
        "enabled_tools": list(spec.enabled_tools) if spec.enabled_tools is not None else None,
        "disabled_tools": list(spec.disabled_tools),
        "env_digest": _digest_mapping(spec.env),
        "headers_digest": _digest_mapping(spec.headers),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()[:16]


def build_mcp_server(spec: "MCPServerSpec") -> Any:
    from pydantic_ai.mcp import MCPServerSSE, MCPServerStdio, MCPServerStreamableHTTP

    prefix = _resolved_tool_prefix(spec)
    if spec.transport == "stdio":
        if not spec.command:
            raise MCPConnectionError(f"mcp {spec.id}: stdio requires a command")
        return MCPServerStdio(
            command=spec.command[0],
            args=list(spec.command[1:]),
            cwd=spec.cwd,
            env=dict(spec.env),
            timeout=spec.timeout_seconds,
            tool_prefix=prefix,
        )
    if spec.transport == "sse":
        if not spec.url:
            raise MCPConnectionError(f"mcp {spec.id}: sse requires a url")
        return MCPServerSSE(
            url=spec.url,
            headers=dict(spec.headers),
            timeout=spec.timeout_seconds,
            tool_prefix=prefix,
        )
    if spec.transport == "http":
        if not spec.url:
            raise MCPConnectionError(f"mcp {spec.id}: http requires a url")
        return MCPServerStreamableHTTP(
            url=spec.url,
            headers=dict(spec.headers),
            timeout=spec.timeout_seconds,
            tool_prefix=prefix,
        )
    raise MCPConnectionError(
        f"mcp {spec.id}: unknown transport {spec.transport!r}"
    )


@dataclass(slots=True)
class MCPClient:
    """Low-level calls against one already constructed SDK client."""

    toolset: Any

    async def discover(
        self,
        *,
        server_id: str,
        connection_ref: "MCPConnectionRef",
    ) -> MCPDiscoveryResult:
        try:
            lister = getattr(self.toolset, "list_tools", None)
            if lister is None:
                return MCPDiscoveryResult(
                    (),
                    False,
                    MCPDiscoveryUnsupportedError(
                        f"MCP server {server_id!r} cannot enumerate tools"
                    ),
                    connection_ref,
                )
            raw_tools = await lister()
            return MCPDiscoveryResult(
                tuple(self.convert_tool_info(tool) for tool in raw_tools or ()),
                True,
                None,
                connection_ref,
            )
        except Exception as error:
            normalized = self.normalize_discovery_error(error)
            logger.warning(
                "MCP discovery failed (server=%s tools may be unavailable): %s: %s",
                server_id, type(normalized).__name__, normalized,
            )
            return MCPDiscoveryResult(
                (),
                False,
                normalized,
                connection_ref,
            )

    async def call(
        self,
        *,
        server_id: str,
        tool_name: str,
        arguments: "Mapping[str, Any]",
    ) -> Any:
        caller = getattr(self.toolset, "direct_call_tool", None)
        if caller is None:
            raise MCPConnectionError(
                f"MCP server {server_id!r} has no direct tool caller"
            )
        return await caller(tool_name, dict(arguments))

    async def close(self) -> None:
        closer = getattr(self.toolset, "close", None)
        if closer is None:
            return
        result = closer()
        if hasattr(result, "__await__"):
            await result

    @staticmethod
    def convert_tool_info(tool: Any) -> MCPToolInfo:
        name = getattr(tool, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise MCPToolDefinitionError("MCP tool name must be non-empty")
        schema = (
            getattr(tool, "inputSchema", None)
            or getattr(tool, "input_schema", None)
            or getattr(tool, "parameters_json_schema", None)
            or {"type": "object", "properties": {}}
        )
        if not isinstance(schema, Mapping):
            raise MCPToolDefinitionError(
                f"invalid schema for MCP tool {name!r}"
            )
        annotations = getattr(tool, "annotations", None)
        hint = (
            getattr(annotations, "readOnlyHint", None)
            if annotations is not None
            else None
        )
        return MCPToolInfo(
            name=name,
            description=getattr(tool, "description", None),
            parameters_json_schema=schema,
            read_only=True if hint is True else False if hint is False else None,
            metadata=getattr(tool, "metadata", {}) or {},
        )

    @staticmethod
    def normalize_discovery_error(error: BaseException) -> BaseException:
        if isinstance(error, MCPDiscoveryError):
            return error
        name = type(error).__name__.lower()
        text = str(error).lower()
        if "auth" in name or "unauthorized" in text or "forbidden" in text:
            return MCPAuthenticationError("MCP authentication failed")
        if "unsupported" in name or "notimplemented" in name:
            return MCPDiscoveryUnsupportedError(
                "MCP discovery is unsupported"
            )
        if "connect" in name or "timeout" in name or "transport" in name:
            return MCPConnectionError("MCP connection failed")
        return MCPDiscoveryError("MCP discovery failed")


class MCPConnectionPool:
    """Own all live MCP toolsets and deduplicate connection creation."""

    def __init__(self) -> None:
        self._toolsets: "dict[tuple[str, str], Any]" = {}
        self._lock = asyncio.Lock()

    async def get_toolset(self, server: "MCPServerSpec") -> MCPToolsetHandle:
        key = (server.id, _config_fingerprint(server))
        cached = self._toolsets.get(key)
        if cached is not None:
            return MCPToolsetHandle(MCPConnectionRef(*key), cached)
        async with self._lock:
            cached = self._toolsets.get(key)
            if cached is None:
                cached = build_mcp_server(server)
                self._toolsets[key] = cached
            return MCPToolsetHandle(MCPConnectionRef(*key), cached)

    async def list_tools(self, server: "MCPServerSpec") -> "tuple[str, ...]":
        result = await self.list_tools_result(server)
        return tuple(item.name for item in result.tools)

    async def list_tools_result(self, server: "MCPServerSpec") -> MCPDiscoveryResult:
        handle = await self.get_toolset(server)
        return await MCPClient(handle.toolset).discover(
            server_id=server.id, connection_ref=handle.connection_ref
        )

    async def call_tool(
        self,
        *,
        connection_ref: MCPConnectionRef,
        tool_name: str,
        arguments: "Mapping[str, Any]",
    ) -> Any:
        key = (connection_ref.server_id, connection_ref.fingerprint)
        toolset = self._toolsets.get(key)
        if toolset is None:
            raise MCPConnectionUnavailableError(f"MCP connection {key!r} is not available")
        return await MCPClient(toolset).call(
            server_id=connection_ref.server_id,
            tool_name=tool_name,
            arguments=arguments,
        )

    async def close_server(self, server_id: str) -> "McpCloseResult":
        failures: "list[McpCloseFailure]" = []
        for key in tuple(key for key in self._toolsets if key[0] == server_id):
            toolset = self._toolsets.get(key)
            if toolset is not None:
                try:
                    await MCPClient(toolset).close()
                except Exception as exc:
                    failures.append(McpCloseFailure(key[0], key[1], type(exc).__name__))
                    logger.error(
                        "event=agent.mcp.close_failed server_id=%s fingerprint=%s error_id=%s",
                        key[0],
                        key[1],
                        type(exc).__name__,
                    )
                else:
                    self._toolsets.pop(key, None)
        return McpCloseResult(not failures, tuple(failures))

    async def close(self) -> "McpCloseResult":
        failures: "list[McpCloseFailure]" = []
        for key, toolset in tuple(self._toolsets.items()):
            try:
                await MCPClient(toolset).close()
            except Exception as exc:
                failure = McpCloseFailure(key[0], key[1], type(exc).__name__)
                failures.append(failure)
                logger.error(
                    "event=agent.mcp.close_failed server_id=%s fingerprint=%s error_id=%s",
                    key[0],
                    key[1],
                    failure.error_id,
                )
            else:
                self._toolsets.pop(key, None)
        return McpCloseResult(not failures, tuple(failures))


class McpResourceState(StrEnum):
    NEW = "new"
    CONNECTING = "connecting"
    OPEN = "open"
    CLOSING = "closing"
    CLEANUP_REQUIRED = "cleanup_required"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class McpResourceFailure:
    owner: str
    resource_id: "str | None"
    error_id: str


@dataclass(frozen=True, slots=True)
class McpCloseFailure:
    server_id: str
    fingerprint: str
    error_id: str


@dataclass(frozen=True, slots=True)
class McpCloseResult:
    closed: bool
    failures: "tuple[McpCloseFailure, ...]" = ()


@dataclass(frozen=True, slots=True)
class McpCleanupFailure:
    server_id: str
    fingerprint: str
    error_id: str


@dataclass(slots=True)
class McpSessionResources:
    """Session owner for domain MCP specs; ACP schemas never enter here."""

    specs: "tuple[MCPServerSpec, ...]" = ()
    pool: Any = None
    state: McpResourceState = McpResourceState.NEW
    connect_task: "asyncio.Task[tuple[Any, ...]] | None" = None
    close_task: "asyncio.Task[tuple[McpResourceFailure, ...]] | None" = None
    _toolsets: "tuple[Any, ...] | None" = None
    last_close_failures: "tuple[McpCloseFailure, ...]" = ()
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def toolsets(self) -> "tuple[Any, ...]":
        async with self._lock:
            if self.state is McpResourceState.CLOSED:
                raise RuntimeError("MCP session resources are closed")
            if self.state in {
                McpResourceState.CLOSING,
                McpResourceState.CLEANUP_REQUIRED,
            }:
                raise RuntimeError("MCP session resources require cleanup")
            if self.state is McpResourceState.OPEN:
                return self._toolsets or ()
            if self.connect_task is None:
                self.state = McpResourceState.CONNECTING
                # task-owner: mcp.connect
                self.connect_task = asyncio.create_task(self._connect())
            task = self.connect_task
        return await asyncio.shield(task)

    async def _connect(self) -> "tuple[Any, ...]":
        pool = MCPConnectionPool()
        try:
            handles = tuple(
                [await pool.get_toolset(spec) for spec in self.specs]
            )
        except BaseException as connect_error:
            try:
                close_result = await asyncio.shield(pool.close())
            except BaseException as close_error:
                close_result = McpCloseResult(
                    False,
                    (McpCloseFailure("unknown", "unknown", type(close_error).__name__),),
                )
            async with self._lock:
                self.connect_task = None
                if close_result.closed:
                    if self.state is McpResourceState.CONNECTING:
                        self.state = McpResourceState.NEW
                else:
                    self.pool = pool
                    self.last_close_failures = close_result.failures
                    self.state = McpResourceState.CLEANUP_REQUIRED
            if close_result.closed:
                raise connect_error
            failures = tuple(
                McpCleanupFailure(item.server_id, item.fingerprint, item.error_id)
                for item in close_result.failures
            )
            logger.error(
                "event=agent.mcp.connect_rollback_failed failure_count=%s",
                len(failures),
            )
            raise McpCleanupRequiredError(
                failures=failures,
                connect_error=connect_error,
            ) from connect_error
        async with self._lock:
            stale = self.state is not McpResourceState.CONNECTING
            if not stale:
                self.pool = pool
                self._toolsets = tuple(handle.toolset for handle in handles)
                self.connect_task = None
                self.state = McpResourceState.OPEN
                toolsets = self._toolsets
            else:
                self.connect_task = None
                toolsets = ()
        if stale:
            close_result = await asyncio.shield(pool.close())
            if not close_result.closed:
                async with self._lock:
                    self.pool = pool
                    self.last_close_failures = close_result.failures
                    self.connect_task = None
                    self.state = McpResourceState.CLEANUP_REQUIRED
                failures = tuple(
                    McpCleanupFailure(item.server_id, item.fingerprint, item.error_id)
                    for item in close_result.failures
                )
                raise McpCleanupRequiredError(
                    "stale MCP connect requires cleanup", failures
                )
            return ()
        logger.info("event=agent.mcp.connected resource_count=%s", len(handles))
        return toolsets

    async def close(self, session_id: str) -> "tuple[McpResourceFailure, ...]":
        async with self._lock:
            if self.state is McpResourceState.CLOSED:
                return ()
            retry = self.state is McpResourceState.CLEANUP_REQUIRED
            if self.close_task is None:
                self.state = McpResourceState.CLOSING
                # task-owner: mcp.close
                self.close_task = asyncio.create_task(self._close_once())
            task = self.close_task
        if retry:
            logger.info(
                "event=agent.mcp.cleanup_retry session_id=%s resource_count=%s",
                session_id,
                len(self.specs),
            )
        return await asyncio.shield(task)

    async def _close_once(self) -> "tuple[McpResourceFailure, ...]":
        failures: "list[McpResourceFailure]" = []
        close_failures: "tuple[McpCloseFailure, ...]" = ()
        async with self._lock:
            connect_task = self.connect_task
            pool = self.pool
        if connect_task is not None and not connect_task.done():
            termination = await cancel_task(connect_task, 10.0)
            if termination is TaskTermination.TIMED_OUT:
                failures.append(McpResourceFailure("mcp", None, "connect_timeout"))
            async with self._lock:
                if self.state is McpResourceState.CLEANUP_REQUIRED:
                    failures.extend(
                        McpResourceFailure("mcp", item.server_id, item.error_id)
                        for item in self.last_close_failures
                    )
                    pool = None
                else:
                    pool = self.pool
        if pool is not None:
            try:
                result = await asyncio.wait_for(pool.close(), 10.0)
            except Exception as exc:
                failures.append(McpResourceFailure("mcp", None, type(exc).__name__))
            else:
                close_failures = result.failures
                failures.extend(
                    McpResourceFailure("mcp", item.server_id, item.error_id)
                    for item in result.failures
                )
        async with self._lock:
            if not failures:
                self.pool = None
                self._toolsets = None
                self.specs = ()
                self.connect_task = None
                self.last_close_failures = ()
                self.state = McpResourceState.CLOSED
                self.close_task = None
                logger.info("event=agent.mcp.closed resource_count=0")
            else:
                self.state = McpResourceState.CLEANUP_REQUIRED
                self.last_close_failures = close_failures
                self.close_task = None
        return tuple(failures)

    def is_empty(self, session_id: str) -> bool:
        return self.state is McpResourceState.CLOSED or (
            self.state is McpResourceState.NEW and self.pool is None and self.connect_task is None
        )


__all__ = [
    "MCPClient",
    "McpCloseFailure",
    "McpCloseResult",
    "McpCleanupFailure",
    "McpResourceFailure",
    "McpResourceState",
    "McpSessionResources",
    "build_mcp_server",
]
