#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""MCP tool client, gateway, and tool discovery."""

import asyncio
import datetime
import hashlib
import json
import os
import sys
import time
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client

import logging

from ..support.utils import resolve_ref as _resolve_ref, truthy as _truthy
from ..support.workspace import TraceStore
from .registry import MCPRegistry, MCPServerSpec

logger = logging.getLogger("linktools.ai.core.mcp_client")


@dataclass(frozen=True, slots=True)
class MCPCacheScope:
    trace_id: str
    server_id: str
    cache_key: str
    spec: MCPServerSpec
    cache_control: dict[str, Any]
    mcp_tool_name: str | None = None


@asynccontextmanager
async def _ephemeral_session(spec: MCPServerSpec):
    """Open and close a MCP session in the same task.

    The stdio transport uses anyio cancel scopes internally; keeping it in a
    cross-task persistent pool can raise "exit cancel scope in a different task".
    """
    stack = AsyncExitStack()
    try:
        transport = MCPConnection(spec)._build_transport()
        read, write = await stack.enter_async_context(transport)
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        yield session
    finally:
        await stack.aclose()


class MCPConnection:
    """Persistent connection to a single MCP server."""

    def __init__(self, spec: MCPServerSpec) -> None:
        self.spec = spec
        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError(f"MCP connection not open: {self.spec.name}")
        return self._session

    @property
    def is_open(self) -> bool:
        return self._session is not None

    async def open(self) -> None:
        stack = AsyncExitStack()
        try:
            transport = self._build_transport()
            read, write = await stack.enter_async_context(transport)
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._exit_stack = stack
            self._session = session
        except Exception:
            await stack.aclose()
            raise

    async def close(self) -> None:
        if self._exit_stack is not None:
            stack, self._exit_stack = self._exit_stack, None
            self._session = None
            await stack.aclose()

    def _build_transport(self):
        if self.spec.mcp_type == "stdio":
            return self._stdio_transport()
        if self.spec.mcp_type == "sse":
            return self._sse_transport()
        if self.spec.mcp_type == "http":
            return self._streamable_http_transport()
        raise ValueError(f"Unsupported MCP type: {self.spec.mcp_type}")

    def _stdio_transport(self):
        command = str(_resolve_ref(self.spec.command) or sys.executable)
        base_dir = self.spec.base_dir or Path()
        args = [_resolve_arg(base_dir, item) for item in self.spec.args]
        if not args:
            default_script = base_dir / "script.py"
            if default_script.exists():
                args = [str(default_script.resolve())]
        if not args:
            raise ValueError(f"Server '{self.spec.name}': stdio requires args or script.py")
        env = os.environ.copy()
        env.update({str(k): str(_resolve_ref(v) or "") for k, v in self.spec.env.items()})
        return stdio_client(StdioServerParameters(command=command, args=args, env=env))

    def _sse_transport(self):
        endpoint = str(_resolve_ref(self.spec.url) or "")
        if not endpoint:
            raise ValueError(f"SSE MCP server '{self.spec.name}' requires url")
        headers = {str(k): str(_resolve_ref(v) or "") for k, v in self.spec.headers.items()}
        timeout = float(self.spec.circuit_breaker.get("timeout_seconds") or 30)
        return sse_client(endpoint, headers=headers, timeout=timeout, sse_read_timeout=timeout)

    def _streamable_http_transport(self):
        endpoint = str(_resolve_ref(self.spec.url) or "")
        if not endpoint:
            raise ValueError(f"HTTP MCP server '{self.spec.name}' requires url")
        return streamable_http_client(endpoint)


class MCPConnectionPool:
    """One persistent MCPConnection per server, lazily connected."""

    def __init__(self) -> None:
        self._connections: dict[str, MCPConnection] = {}
        self._global_lock = asyncio.Lock()
        self._server_locks: dict[str, asyncio.Lock] = {}

    async def get_session(self, spec: MCPServerSpec) -> ClientSession:
        lock = await self._server_lock(spec.name)
        async with lock:
            conn = self._connections.get(spec.name)
            if conn is None or not conn.is_open:
                conn = MCPConnection(spec)
                await conn.open()
                self._connections[spec.name] = conn
            return conn.session

    async def invalidate(self, name: str) -> None:
        lock = await self._server_lock(name)
        async with lock:
            conn = self._connections.pop(name, None)
        if conn is not None:
            await conn.close()

    async def close_all(self) -> None:
        async with self._global_lock:
            names = list(self._connections.keys())
        for name in names:
            await self.invalidate(name)

    async def _server_lock(self, name: str) -> asyncio.Lock:
        async with self._global_lock:
            if name not in self._server_locks:
                self._server_locks[name] = asyncio.Lock()
            return self._server_locks[name]


class MCPClient:
    """MCP protocol client. Uses a shared persistent connection pool per instance."""

    def __init__(self, registry: MCPRegistry, memory: TraceStore) -> None:
        self.registry = registry
        self.memory = memory
        self.pool = MCPConnectionPool()

    async def query(
        self,
        server_or_capability: str,
        request: dict[str, Any],
        trace_id: str,
        mcp_tool_name: str | None = None,
    ) -> dict[str, Any]:
        spec = self._resolve_spec(server_or_capability)
        request = dict(request)
        cache_control = self._extract_cache_control(request)
        cache = MCPCacheScope(
            trace_id=trace_id,
            server_id=spec.name,
            cache_key=self._cache_key(spec.name, mcp_tool_name, request),
            spec=spec,
            cache_control=cache_control,
            mcp_tool_name=mcp_tool_name,
        )
        if self._cache_enabled(spec, cache_control):
            cached = self._read_cache(cache)
            if cached is not None:
                return cached

        retry_count = int(spec.circuit_breaker.get("retry_count", 0))
        retry_interval = float(spec.circuit_breaker.get("retry_interval_seconds", 0))
        last_error: Exception | None = None

        for attempt in range(retry_count + 1):
            try:
                result = await self._query_once(spec, request, mcp_tool_name=mcp_tool_name)
                if self._cache_enabled(spec, cache_control) and result.get("status", "ok") != "failed":
                    self._write_cache(cache, request, result)
                return result
            except Exception as exc:
                last_error = exc
                if attempt < retry_count and retry_interval > 0:
                    await asyncio.sleep(retry_interval)

        return {
            "server": spec.name,
            "status": "failed",
            "data": [],
            "data_gaps": [f"{spec.name}: {last_error}"],
        }

    def _resolve_spec(self, server_or_capability: str) -> MCPServerSpec:
        if server_or_capability in self.registry:
            return self.registry.get(server_or_capability)
        spec = self.registry.resolve_by_capability(server_or_capability)
        if spec is not None:
            return spec
        raise KeyError(f"MCP server not found: {server_or_capability}")

    def _extract_cache_control(self, request: dict[str, Any]) -> dict[str, Any]:
        control: dict[str, Any] = {}
        cache_control = request.pop("cache_control", None)
        if isinstance(cache_control, dict):
            control.update(cache_control)
        for key in ("force_refresh", "_force_refresh", "refresh_cache"):
            if key in request:
                control["force_refresh"] = request.pop(key)
        for key in ("bypass_cache", "no_cache", "_bypass_cache"):
            if key in request:
                control["bypass_cache"] = request.pop(key)
        if "ttl_seconds" in request:
            control["ttl_seconds"] = request.pop("ttl_seconds")
        return control

    def _cache_enabled(self, spec: MCPServerSpec, cache_control: dict[str, Any]) -> bool:
        if _truthy(cache_control.get("bypass_cache")):
            return False
        cache_config = spec.cache
        if cache_config.get("enabled") is False:
            return False
        return spec.kind == "read" or bool(cache_config.get("enabled", False))

    def _read_cache(self, cache: MCPCacheScope) -> dict[str, Any] | None:
        if _truthy(cache.cache_control.get("force_refresh")):
            return None
        path = self._cache_path(cache.trace_id, cache.server_id, cache.cache_key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        cached_at = float(payload.get("cached_at", 0) or 0)
        ttl = self._ttl_seconds(cache.spec, cache.cache_control)
        if ttl > 0 and time.time() - cached_at > ttl:
            return None
        result = payload.get("result")
        return result if isinstance(result, dict) else None

    def _write_cache(self, cache: MCPCacheScope, request: dict[str, Any], result: dict[str, Any]) -> None:
        path = self._cache_path(cache.trace_id, cache.server_id, cache.cache_key)
        self.memory.write_json(path, {
            "server": cache.server_id,
            "mcp_tool_name": cache.mcp_tool_name,
            "request": request,
            "cached_at": time.time(),
            "ttl_seconds": self._ttl_seconds(cache.spec, cache.cache_control),
            "result": result,
        })

    def _cache_path(self, trace_id: str, server_id: str, cache_key: str) -> Path:
        return self.memory.for_trace(trace_id).slot_path("cache") / "mcp_results" / server_id / f"{cache_key}.json"

    def _cache_key(self, server_id: str, mcp_tool_name: str | None, request: dict[str, Any]) -> str:
        payload = {"server": server_id, "mcp_tool_name": mcp_tool_name, "request": request}
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:32]

    def _ttl_seconds(self, spec: MCPServerSpec, cache_control: dict[str, Any]) -> int:
        override = cache_control.get("ttl_seconds")
        if override is not None:
            try:
                return max(0, int(override))
            except (TypeError, ValueError):
                pass
        c = spec.cache
        return int(c.get("ttl_seconds", c.get("default_ttl_seconds", 300)))

    async def _query_once(
        self,
        spec: MCPServerSpec,
        request: dict[str, Any],
        mcp_tool_name: str | None = None,
    ) -> dict[str, Any]:
        if not mcp_tool_name:
            raise ValueError(f"mcp_tool_name required for server '{spec.name}'")
        timeout_secs = int(spec.circuit_breaker.get("timeout_seconds") or 30)
        try:
            if spec.mcp_type == "stdio":
                async with _ephemeral_session(spec) as session:
                    call_result = await session.call_tool(
                        mcp_tool_name,
                        dict(request),
                        read_timeout_seconds=datetime.timedelta(seconds=timeout_secs),
                    )
            else:
                session = await self.pool.get_session(spec)
                call_result = await session.call_tool(
                    mcp_tool_name,
                    dict(request),
                    read_timeout_seconds=datetime.timedelta(seconds=timeout_secs),
                )
        except Exception:
            if spec.mcp_type != "stdio":
                await self.pool.invalidate(spec.name)
            raise
        return _coerce_mcp_call_result(spec.name, call_result)


class MCPToolDiscovery:
    """Discover and resolve tools from MCP servers using a shared connection pool."""

    def __init__(self, registry: MCPRegistry, pool: MCPConnectionPool) -> None:
        self.registry = registry
        self.pool = pool

    async def get_tool_specs(self, required_data_sources: list[str]) -> list[dict[str, Any]]:
        """Return tool spec entries for the specified data sources."""
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for source in required_data_sources:
            spec = self.registry.get(source) if source in self.registry else self.registry.resolve_by_capability(source)
            if spec is None or spec.name in seen:
                continue
            seen.add(spec.name)
            tools = await self._discover_tools(spec)
            result.extend(self._tools_to_entries(spec, tools))
        return result

    def _tools_to_entries(self, spec: MCPServerSpec, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        server = spec.server_name or spec.name
        return [
            {
                "name": f"mcp__{server}__{tool['name']}",
                "protocol_tool_name": tool["name"],
                "server": server,
                "description": tool.get("description"),
                "input_schema": tool.get("input_schema"),
                "kind": spec.kind,
                "provides": spec.provides,
                "mcp_type": spec.mcp_type,
            }
            for tool in tools
        ]

    async def _discover_tools(self, spec: MCPServerSpec) -> list[dict[str, Any]]:
        try:
            if spec.mcp_type == "stdio":
                async with _ephemeral_session(spec) as session:
                    result = await session.list_tools()
            else:
                session = await self.pool.get_session(spec)
                result = await session.list_tools()
        except Exception:
            if spec.mcp_type != "stdio":
                await self.pool.invalidate(spec.name)
            logger.warning("failed to discover tools for %s", spec.name, exc_info=True)
            return []
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema or {"type": "object", "properties": {}},
            }
            for tool in result.tools
        ]


def _coerce_mcp_call_result(server_id: str, call_result: Any) -> dict[str, Any]:
    if call_result.isError:
        raise RuntimeError(f"MCP tool error from {server_id}: {_extract_text(call_result)}")

    structured = call_result.structuredContent
    if isinstance(structured, dict):
        extracted = structured.get("result", structured)
        if isinstance(extracted, dict):
            extracted.setdefault("server", server_id)
            extracted.setdefault("status", "ok")
            extracted.setdefault("data_gaps", [])
            return extracted

    text = _extract_text(call_result)
    if text:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(parsed, dict):
                parsed.setdefault("server", server_id)
                parsed.setdefault("status", "ok")
                parsed.setdefault("data_gaps", [])
                inner = parsed.get("result")
                if isinstance(inner, dict):
                    inner.setdefault("server", server_id)
                    inner.setdefault("status", "ok")
                    inner.setdefault("data_gaps", [])
                    return inner
                return parsed
            return {"server": server_id, "status": "ok", "data_gaps": [], "data": parsed}

    return {"server": server_id, "status": "ok", "data_gaps": [], "data": text or ""}


def _extract_text(call_result: Any) -> str:
    for item in call_result.content or []:
        if hasattr(item, "text"):
            return item.text
    return ""


def _resolve_arg(base_dir: Path, value: Any) -> str:
    text = str(_resolve_ref(value))
    if text.startswith("-") or "://" in text:
        return text
    path = Path(text)
    if path.is_absolute():
        return str(path)
    candidate = base_dir / path
    return str(candidate.resolve()) if candidate.exists() else text
