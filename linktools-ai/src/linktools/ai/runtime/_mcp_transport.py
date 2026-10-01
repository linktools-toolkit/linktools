#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construct explicit MCP transports and bridge sandbox-owned stdio bytes."""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import anyio
from fastmcp.client.transports import (
    ClientTransport,
    SSETransport,
    StdioTransport,
    StreamableHttpTransport,
)
from mcp import ClientSession
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage
from pydantic import TypeAdapter

from ..errors import AIError, ErrorCode
from ..spec import MCPServerSpec
from ..workspace import (
    SandboxResource,
    SandboxResourcePath,
    SandboxStdioProcess,
    StdioSandboxSession,
)

_JSON_RPC_MESSAGE_ADAPTER = TypeAdapter(JSONRPCMessage)


def _create_mcp_transport(
    server: MCPServerSpec,
    *,
    sandboxed: bool,
    sandbox_session: object | None,
    host_cwd: str | None,
    args: tuple[str | SandboxResourcePath, ...] = (),
    resources: tuple[SandboxResource, ...] = (),
) -> ClientTransport:
    if server.transport == "streamable-http":
        return StreamableHttpTransport(
            server.url,
            headers=dict(server.headers) or None,
        )
    if server.transport == "sse":
        options = (
            {} if server.read_timeout is None
            else {"sse_read_timeout": server.read_timeout}
        )
        return SSETransport(
            server.url,
            headers=dict(server.headers) or None,
            **options,
        )
    if server.transport != "stdio" or server.command is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if sandboxed:
        if not isinstance(sandbox_session, StdioSandboxSession):
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
        return _SandboxMCPTransport(
            sandbox_session,
            server.command,
            args,
            resources,
            server.env,
        )
    if host_cwd is None:
        raise AIError(
            ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
            safe_details={"reason": "mcp_cwd_unavailable"},
        )
    return StdioTransport(
        server.command,
        list(args),
        cwd=host_cwd,
        env=dict(server.env) or None,
    )


class _SandboxMCPTransport(ClientTransport):
    """Use a Sandbox session's supervised stdio process as an MCP transport."""

    def __init__(
        self,
        session: StdioSandboxSession,
        command: str,
        args: tuple[str | SandboxResourcePath, ...],
        resources: tuple[SandboxResource, ...],
        environment: "Mapping[str, str] | None" = None,
    ) -> None:
        self._session = session
        self._command = command
        self._args = args
        self._resources = resources
        self._environment = dict(environment or {})
        self._process: SandboxStdioProcess | None = None

    @asynccontextmanager
    async def connect_session(self, **session_kwargs: Any) -> AsyncIterator[ClientSession]:
        kwargs: dict[str, object] = {"resources": self._resources}
        if self._environment:
            kwargs["environment"] = self._environment
        process = await self._session.open_stdio_process(
            self._command,
            self._args,
            **kwargs,
        )
        self._process = process
        read_send, read_receive = anyio.create_memory_object_stream(1)
        write_send, write_receive = anyio.create_memory_object_stream(1)
        client_session = ClientSession(
            read_receive,
            write_send,
            **session_kwargs,
        )
        reader_scope = anyio.CancelScope()
        writer_scope = anyio.CancelScope()
        reader_done = anyio.Event()
        writer_done = anyio.Event()
        try:
            async with client_session:
                async with anyio.create_task_group() as tasks:
                    tasks.start_soon(
                        self._read_worker,
                        process,
                        read_send,
                        reader_scope,
                        reader_done,
                    )
                    tasks.start_soon(
                        self._write_worker,
                        process,
                        write_receive,
                        writer_scope,
                        writer_done,
                    )
                    try:
                        yield client_session
                    finally:
                        with anyio.CancelScope(shield=True):
                            await _close_process(
                                process,
                                reader_scope,
                                writer_scope,
                                reader_done,
                                writer_done,
                                read_send,
                                read_receive,
                                write_send,
                                write_receive,
                            )
        except BaseException as error:
            nested = _find_ai_error(error)
            if nested is None:
                raise
            raise nested from error
        finally:
            if self._process is process:
                self._process = None

    async def close(self) -> None:
        process = self._process
        if process is None:
            return
        await process.close_stdin()
        await process.close()

    @staticmethod
    async def _read_messages(
        process: SandboxStdioProcess,
        output: anyio.abc.ObjectSendStream[SessionMessage | Exception],
    ) -> None:
        pending = bytearray()
        async with output:
            try:
                while True:
                    chunk = await process.read_stdout(65536)
                    if not chunk:
                        if pending:
                            raise _invalid_protocol()
                        raise AIError(ErrorCode.SANDBOX_SESSION_LOST)
                    pending.extend(chunk)
                    while True:
                        separator = pending.find(b"\n")
                        if separator < 0:
                            break
                        frame = bytes(pending[:separator])
                        del pending[: separator + 1]
                        if not frame:
                            raise _invalid_protocol()
                        try:
                            message = _JSON_RPC_MESSAGE_ADAPTER.validate_json(frame)
                        except (TypeError, ValueError) as error:
                            raise _invalid_protocol() from error
                        await output.send(SessionMessage(message))
            except anyio.ClosedResourceError:
                return

    @classmethod
    async def _read_worker(
        cls,
        process: SandboxStdioProcess,
        output: anyio.abc.ObjectSendStream[SessionMessage | Exception],
        cancel_scope: anyio.CancelScope,
        done: anyio.Event,
    ) -> None:
        try:
            with cancel_scope:
                await cls._read_messages(process, output)
        finally:
            done.set()

    @staticmethod
    async def _write_messages(
        process: SandboxStdioProcess,
        incoming: anyio.abc.ObjectReceiveStream[SessionMessage],
    ) -> None:
        async with incoming:
            async for session_message in incoming:
                payload = session_message.message.model_dump_json(
                    by_alias=True,
                    exclude_none=True,
                )
                await process.write_stdin(payload.encode("utf-8") + b"\n")

    @classmethod
    async def _write_worker(
        cls,
        process: SandboxStdioProcess,
        incoming: anyio.abc.ObjectReceiveStream[SessionMessage],
        cancel_scope: anyio.CancelScope,
        done: anyio.Event,
    ) -> None:
        try:
            with cancel_scope:
                await cls._write_messages(process, incoming)
        finally:
            done.set()


def _invalid_protocol() -> AIError:
    return AIError(
        ErrorCode.CAPABILITY_RESOLUTION_INVALID,
        safe_details={"reason": "mcp_protocol_invalid"},
    )


async def _close_process(
    process: SandboxStdioProcess,
    reader_scope: anyio.CancelScope,
    writer_scope: anyio.CancelScope,
    reader_done: anyio.Event,
    writer_done: anyio.Event,
    read_send: anyio.abc.ObjectSendStream[SessionMessage | Exception],
    read_receive: anyio.abc.ObjectReceiveStream[SessionMessage | Exception],
    write_send: anyio.abc.ObjectSendStream[SessionMessage],
    write_receive: anyio.abc.ObjectReceiveStream[SessionMessage],
) -> None:
    await read_receive.aclose()
    await write_send.aclose()
    reader_scope.cancel()
    writer_scope.cancel()
    await reader_done.wait()
    await writer_done.wait()
    try:
        await process.close_stdin()
    finally:
        try:
            await process.close()
        finally:
            await read_send.aclose()
            await write_receive.aclose()


def _find_ai_error(error: BaseException) -> AIError | None:
    if isinstance(error, AIError):
        return error
    nested = getattr(error, "exceptions", None)
    if isinstance(nested, tuple):
        for value in nested:
            if isinstance(value, BaseException):
                result = _find_ai_error(value)
                if result is not None:
                    return result
    return None


__all__ = []
