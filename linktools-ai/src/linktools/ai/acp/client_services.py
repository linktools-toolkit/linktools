#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Session-scoped calls from the Agent to the ACP Client."""

import logging
import os
import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

from .errors import request_error
from .sessions import ActiveAcpSession


MAX_FILE_BYTES = 2 * 1024 * 1024
DEFAULT_TERMINAL_OUTPUT_LIMIT = 256 * 1024
MAX_TERMINAL_OUTPUT_LIMIT = 1024 * 1024

logger = logging.getLogger("linktools.ai.acp.client_services")


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
        self._ensure_session_open(session)
        roots = tuple(Path(os.path.normcase(item)) for item in (session.record.cwd,) + session.record.additional_directories)
        target = Path(value)
        if not target.is_absolute():
            target = Path(session.record.cwd) / target
        resolved = Path(os.path.normcase(os.path.abspath(os.path.normpath(str(target)))))
        compare = resolved.parent if parent else resolved
        if not any(_contained(compare, root) for root in roots):
            raise request_error("path_outside_allowed_roots", session_id=session.record.session_id)
        return resolved

    @staticmethod
    def _ensure_session_open(session: ActiveAcpSession) -> None:
        if session.record.closed or session.closing_requested or session.cleanup_required:
            raise request_error("session_closed", session_id=session.record.session_id)

    def _has_fs(self, name: str) -> bool:
        fs = getattr(self._client_capabilities, "fs", None)
        return bool(getattr(fs, name, False))

    async def read_text_file(self, session: ActiveAcpSession, path: str, line: "int | None" = None, limit: "int | None" = None) -> Any:
        if not self._has_fs("read_text_file"):
            raise request_error("client_capability_not_declared", session_id=session.record.session_id)
        target = self._allowed_path(session, path)
        if (line is not None and line < 0) or (limit is not None and limit < 0):
            raise request_error("invalid_file_range", session_id=session.record.session_id)
        connection = self._connection_or_error()
        try:
            kwargs = {}
            if line is not None:
                kwargs["line"] = line
            if limit is not None:
                kwargs["limit"] = limit
            response = await connection.read_text_file(
                session.record.session_id,
                str(target),
                **kwargs,
            )
        except Exception as exc:
            logger.warning(
                "event=acp.client.fs.read_failed session_id=%s error_type=%s",
                session.record.session_id,
                type(exc).__name__,
            )
            raise request_error("client_file_read_failed", session_id=session.record.session_id) from exc
        content = getattr(response, "content", None)
        if not isinstance(content, str):
            raise request_error("client_file_read_failed", session_id=session.record.session_id)
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise request_error("file_too_large", session_id=session.record.session_id)
        return response

    async def write_text_file(self, session: ActiveAcpSession, path: str, content: str) -> Any:
        if not self._has_fs("write_text_file"):
            raise request_error("client_capability_not_declared", session_id=session.record.session_id)
        self._ensure_session_open(session)
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise request_error("file_too_large", session_id=session.record.session_id)
        target = self._allowed_path(session, path, parent=True)
        try:
            return await self._connection_or_error().write_text_file(
                session.record.session_id,
                str(target),
                content,
            )
        except Exception as exc:
            logger.warning(
                "event=acp.client.fs.write_failed session_id=%s error_type=%s",
                session.record.session_id,
                type(exc).__name__,
            )
            raise request_error("client_file_write_failed", session_id=session.record.session_id) from exc

    async def create_terminal(self, session: ActiveAcpSession, **kwargs: Any) -> Any:
        self._ensure_session_open(session)
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
        task = asyncio.create_task(
            connection.create_terminal(session.record.session_id, **kwargs)
        )
        async with session.lock:
            session.terminal_create_tasks.add(task)
        try:
            response = await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.gather(task, return_exceptions=True)
            raise
        finally:
            async with session.lock:
                session.terminal_create_tasks.discard(task)
        terminal_id = response.terminal_id
        async with session.lock:
            operation = session.operation
            valid = (
                not session.record.closed
                and not session.closing_requested
                and operation is not None
                and operation.kind.value == "prompt"
            )
            if valid:
                session.terminal_handles.add(terminal_id)
        if not valid:
            try:
                await connection.kill_terminal(session.record.session_id, terminal_id)
            finally:
                await connection.release_terminal(session.record.session_id, terminal_id)
            logger.info(
                "event=acp.terminal.stale_create_compensated session_id=%s terminal_count=0",
                session.record.session_id,
            )
            raise request_error("session_closing", session_id=session.record.session_id)
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
        async with session.lock:
            if terminal_id not in session.terminal_handles:
                return None
            task = session.terminal_release_tasks.get(terminal_id)
            if task is None:
                task = asyncio.create_task(
                    self._connection_or_error().release_terminal(
                        session.record.session_id,
                        terminal_id,
                    )
                )
                session.terminal_release_tasks[terminal_id] = task
        try:
            response = await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.gather(task, return_exceptions=True)
            raise
        except Exception:
            async with session.lock:
                if session.terminal_release_tasks.get(terminal_id) is task:
                    session.terminal_release_tasks.pop(terminal_id, None)
            raise
        else:
            async with session.lock:
                if session.terminal_release_tasks.get(terminal_id) is task:
                    session.terminal_release_tasks.pop(terminal_id, None)
                    session.terminal_handles.discard(terminal_id)
            return response

    async def create_elicitation(self, session: ActiveAcpSession, message: str, mode: Any) -> Any:
        self._ensure_session_open(session)
        if not getattr(self._client_capabilities, "elicitation", None):
            raise request_error("client_capability_not_declared", session_id=session.record.session_id)
        if session.active_execution_id is None:
            raise request_error("no_active_execution", session_id=session.record.session_id)
        root = getattr(mode, "root", mode)
        elicitation_id = getattr(root, "elicitation_id", None)
        mode_session_id = getattr(root, "session_id", None)
        mode_name = type(root).__name__
        is_url_session = mode_name == "ElicitationUrlSessionMode"
        is_known = mode_name in {
            "ElicitationUrlSessionMode",
            "ElicitationUrlRequestMode",
            "ElicitationFormSessionMode",
            "ElicitationFormRequestMode",
        }
        if is_url_session:
            if not isinstance(elicitation_id, str) or not elicitation_id:
                raise request_error("invalid_elicitation_id", session_id=session.record.session_id)
            if mode_session_id is not None and mode_session_id != session.record.session_id:
                raise request_error("elicitation_session_mismatch", session_id=session.record.session_id)
        elif not is_known:
            raise request_error("unsupported_elicitation_mode", session_id=session.record.session_id)
        task_id = elicitation_id if is_url_session else uuid4().hex
        task = asyncio.create_task(
            self._connection_or_error().create_elicitation(message, mode)
        )
        async with session.lock:
            session.pending_elicitation_tasks[task_id] = task
            if is_url_session:
                session.pending_elicitation_ids.add(elicitation_id)
        try:
            response = await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.gather(task, return_exceptions=True)
            raise
        finally:
            async with session.lock:
                if session.pending_elicitation_tasks.get(task_id) is task:
                    session.pending_elicitation_tasks.pop(task_id, None)
        if is_url_session:
            try:
                await self._connection_or_error().complete_elicitation(elicitation_id)
            except Exception:
                raise
            async with session.lock:
                session.pending_elicitation_ids.discard(elicitation_id)
        return response

    async def close_session_resources(self, session: ActiveAcpSession) -> "tuple[tuple[str, str | None, BaseException], ...]":
        failures = []
        async with session.lock:
            terminal_ids = tuple(session.terminal_handles)
            elicitation_ids = tuple(session.pending_elicitation_ids)
        connection = self._connection
        if connection is None:
            failures.extend(
                ("terminal", terminal_id, RuntimeError("client connection is unavailable"))
                for terminal_id in terminal_ids
            )
            failures.extend(
                ("elicitation", elicitation_id, RuntimeError("client connection is unavailable"))
                for elicitation_id in elicitation_ids
            )
            return tuple(failures)
        for terminal_id in terminal_ids:
            try:
                await asyncio.wait_for(
                    connection.kill_terminal(session.record.session_id, terminal_id),
                    timeout=5,
                )
            except Exception as exc:
                failures.append(("terminal", terminal_id, exc))
            try:
                await asyncio.wait_for(
                    connection.release_terminal(session.record.session_id, terminal_id),
                    timeout=5,
                )
            except Exception as exc:
                failures.append(("terminal", terminal_id, exc))
                continue
            if not any(
                resource_type == "terminal" and resource_id == terminal_id
                for resource_type, resource_id, _ in failures
            ):
                async with session.lock:
                    session.terminal_handles.discard(terminal_id)
        for elicitation_id in elicitation_ids:
            try:
                await asyncio.wait_for(
                    connection.complete_elicitation(elicitation_id),
                    timeout=10,
                )
            except Exception as exc:
                failures.append(("elicitation", elicitation_id, exc))
                continue
            async with session.lock:
                session.pending_elicitation_ids.discard(elicitation_id)
            logger.info(
                "event=acp.elicitation.completed_on_close session_id=%s elicitation_id=%s",
                session.record.session_id,
                elicitation_id,
            )
        async with session.lock:
            remaining_tasks = tuple(session.pending_elicitation_tasks.values())
        for task in remaining_tasks:
            task.cancel()
        if remaining_tasks:
            await asyncio.gather(*remaining_tasks, return_exceptions=True)
        return tuple(failures)

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
