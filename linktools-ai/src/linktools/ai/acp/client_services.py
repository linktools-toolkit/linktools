#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Session-scoped calls from the Agent to the ACP Client."""

import asyncio
import os
from pathlib import Path
from typing import Any

from .errors import request_error
from .sessions import ActiveAcpSession


MAX_FILE_BYTES = 2 * 1024 * 1024
DEFAULT_TERMINAL_OUTPUT_LIMIT = 256 * 1024
MAX_TERMINAL_OUTPUT_LIMIT = 1024 * 1024


class AcpClientServices:
    def __init__(self, *, project_root: "str | Path") -> None:
        self.project_root = Path(project_root).resolve()
        self._connection: Any = None
        self._client_capabilities: Any = None

    def set_connection(self, connection: Any, client_capabilities: Any) -> None:
        self._connection = connection
        self._client_capabilities = client_capabilities

    def _connection_or_error(self) -> Any:
        if self._connection is None:
            raise request_error("client_not_connected")
        return self._connection

    def _allowed_path(self, session: ActiveAcpSession, value: str, *, parent: bool = False) -> Path:
        roots = tuple(Path(os.path.normcase(item)) for item in (session.record.cwd,) + session.record.additional_directories)
        target = Path(value)
        if not target.is_absolute():
            target = Path(session.record.cwd) / target
        try:
            resolved = Path(os.path.normcase(str(target.resolve(strict=not parent))))
            compare = resolved.parent if parent else resolved
        except OSError as exc:
            raise request_error("invalid_path", session_id=session.record.session_id) from exc
        if not any(_contained(compare, root) for root in roots):
            raise request_error("path_outside_allowed_roots", session_id=session.record.session_id)
        return resolved

    def _has_fs(self, name: str) -> bool:
        fs = getattr(self._client_capabilities, "fs", None)
        return bool(getattr(fs, name, False))

    async def read_text_file(self, session: ActiveAcpSession, path: str, line: "int | None" = None, limit: "int | None" = None) -> Any:
        if not self._has_fs("read_text_file"):
            raise request_error("client_capability_not_declared", session_id=session.record.session_id)
        target = self._allowed_path(session, path)
        try:
            content = await asyncio.to_thread(target.read_text, encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise request_error("client_file_read_failed", session_id=session.record.session_id) from exc
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise request_error("file_too_large", session_id=session.record.session_id)
        lines = content.splitlines()
        if line is not None:
            start = max(0, line - 1)
            lines = lines[start:]
        if limit is not None:
            lines = lines[:limit]
        import acp.schema as schema

        return schema.ReadTextFileResponse(content="\n".join(lines))

    async def write_text_file(self, session: ActiveAcpSession, path: str, content: str) -> Any:
        if not self._has_fs("write_text_file"):
            raise request_error("client_capability_not_declared", session_id=session.record.session_id)
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise request_error("file_too_large", session_id=session.record.session_id)
        target = self._allowed_path(session, path, parent=True)
        if target.exists() and target.is_symlink():
            raise request_error("symlink_path_rejected", session_id=session.record.session_id)
        try:
            await asyncio.to_thread(target.write_text, content, encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise request_error("client_file_write_failed", session_id=session.record.session_id) from exc
        return None

    async def create_terminal(self, session: ActiveAcpSession, **kwargs: Any) -> Any:
        connection = self._connection_or_error()
        if not bool(getattr(self._client_capabilities, "terminal", False)):
            raise request_error("client_capability_not_declared", session_id=session.record.session_id)
        cwd = kwargs.get("cwd")
        if cwd is not None:
            kwargs["cwd"] = str(self._allowed_path(session, cwd))
        output_limit = kwargs.get("output_byte_limit") or DEFAULT_TERMINAL_OUTPUT_LIMIT
        if not 0 < output_limit <= MAX_TERMINAL_OUTPUT_LIMIT:
            raise request_error("invalid_terminal_output_limit", session_id=session.record.session_id)
        env = kwargs.get("env") or []
        for item in env:
            if not item.name or "\x00" in item.name or "\x00" in item.value:
                raise request_error("invalid_terminal_environment", session_id=session.record.session_id)
        response = await connection.create_terminal(session.record.session_id, **kwargs)
        terminal_id = response.terminal_id
        session.terminal_handles.add(terminal_id)
        return response

    async def terminal_output(self, session: ActiveAcpSession, terminal_id: str) -> Any:
        self._check_terminal(session, terminal_id)
        return await self._connection_or_error().terminal_output(session.record.session_id, terminal_id)

    async def wait_for_terminal_exit(self, session: ActiveAcpSession, terminal_id: str) -> Any:
        self._check_terminal(session, terminal_id)
        return await self._connection_or_error().wait_for_terminal_exit(session.record.session_id, terminal_id)

    async def kill_terminal(self, session: ActiveAcpSession, terminal_id: str) -> Any:
        self._check_terminal(session, terminal_id)
        return await self._connection_or_error().kill_terminal(session.record.session_id, terminal_id)

    async def release_terminal(self, session: ActiveAcpSession, terminal_id: str) -> Any:
        if terminal_id not in session.terminal_handles:
            return None
        session.terminal_handles.discard(terminal_id)
        return await self._connection_or_error().release_terminal(session.record.session_id, terminal_id)

    async def create_elicitation(self, session: ActiveAcpSession, message: str, mode: Any) -> Any:
        if not getattr(self._client_capabilities, "elicitation", None):
            raise request_error("client_capability_not_declared", session_id=session.record.session_id)
        if session.active_execution_id is None:
            raise request_error("no_active_execution", session_id=session.record.session_id)
        response = await self._connection_or_error().create_elicitation(message, mode)
        return response

    def _check_terminal(self, session: ActiveAcpSession, terminal_id: str) -> None:
        if terminal_id not in session.terminal_handles:
            raise request_error("unknown_terminal", session_id=session.record.session_id)


def _contained(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


__all__ = ["AcpClientServices", "MAX_FILE_BYTES", "MAX_TERMINAL_OUTPUT_LIMIT"]
