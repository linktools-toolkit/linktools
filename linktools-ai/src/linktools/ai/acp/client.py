#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ACP Client callbacks and resources owned by one ACP connection."""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from linktools.core import environ

from ..execution.cancellation import TaskTermination, cancel_task
from ..execution.domain import ApprovalDecision
from ..runtime.session import ResourceFailure
from .protocol import request_error


MAX_FILE_BYTES = 2 * 1024 * 1024
DEFAULT_TERMINAL_OUTPUT_LIMIT = 256 * 1024
MAX_TERMINAL_OUTPUT_LIMIT = 1024 * 1024
logger = environ.get_logger("ai.acp.client")


@dataclass(slots=True)
class AcpClientSessionResources:
    client: "AcpClient"
    session_id: str
    terminal_handles: "set[str]" = field(default_factory=set)
    terminal_create_tasks: "set[asyncio.Task[Any]]" = field(default_factory=set)
    terminal_release_tasks: "dict[str, asyncio.Task[Any]]" = field(default_factory=dict)
    elicitation_tasks: "dict[str, asyncio.Task[Any]]" = field(default_factory=dict)
    accepted_elicitations: "set[str]" = field(default_factory=set)

    async def close(self, session_id: str) -> "tuple[ResourceFailure, ...]":
        failures: "list[ResourceFailure]" = []
        connection = self.client._connection
        for task in tuple(self.terminal_create_tasks):
            termination = await cancel_task(task, 1.0)
            if termination is TaskTermination.TIMED_OUT:
                failures.append(ResourceFailure("acp.client.terminal", None, "task_timeout"))
        for task in tuple(self.elicitation_tasks.values()):
            termination = await cancel_task(task, 1.0)
            if termination is TaskTermination.TIMED_OUT:
                failures.append(ResourceFailure("acp.client.elicitation", None, "task_timeout"))
        self.accepted_elicitations.clear()
        if connection is None:
            failures.extend(
                ResourceFailure("acp.client", terminal_id, "client_connection_missing")
                for terminal_id in self.terminal_handles
            )
            return tuple(failures)
        for terminal_id in tuple(self.terminal_handles):
            try:
                await connection.kill_terminal(session_id, terminal_id)
            except Exception as exc:
                failures.append(ResourceFailure("acp.client.terminal", terminal_id, type(exc).__name__))
                continue
            try:
                await self._release(session_id, terminal_id)
            except Exception as exc:
                failures.append(ResourceFailure("acp.client.terminal", terminal_id, type(exc).__name__))
        return tuple(failures)

    async def _release(self, session_id: str, terminal_id: str) -> Any:
        connection = self.client._connection
        if connection is None:
            raise RuntimeError("client connection is unavailable")
        request = connection.release_terminal(session_id, terminal_id)
        task = self.terminal_release_tasks.get(terminal_id)
        if task is None:
            # task-owner: acp.client.terminal_release
            task = asyncio.create_task(request)
            self.terminal_release_tasks[terminal_id] = task
        elif hasattr(request, "close"):
            request.close()
        try:
            response = await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                response = await task
            except BaseException:
                self.terminal_release_tasks.pop(terminal_id, None)
                raise
            self.terminal_release_tasks.pop(terminal_id, None)
            self.terminal_handles.discard(terminal_id)
            raise
        except BaseException:
            self.terminal_release_tasks.pop(terminal_id, None)
            raise
        self.terminal_release_tasks.pop(terminal_id, None)
        self.terminal_handles.discard(terminal_id)
        return response

    def is_empty(self, session_id: str) -> bool:
        return not (
            self.terminal_handles
            or self.terminal_create_tasks
            or self.terminal_release_tasks
            or self.elicitation_tasks
            or self.accepted_elicitations
        )


class AcpClientSessionOwner:
    def __init__(
        self,
        parent: "AcpClient",
        session_id: str,
        resources: AcpClientSessionResources,
    ) -> None:
        self._parent = parent
        self._session_id = session_id
        self._resources = resources

    async def close(self, session_id: str) -> "tuple[ResourceFailure, ...]":
        failures = await self._resources.close(session_id)
        if not failures and self._resources.is_empty(session_id):
            self._parent._remove_resources(session_id, expected=self._resources)
            logger.info(
                "event=acp.client.resources_removed session_id=%s resource_count=%s",
                session_id,
                len(self._parent._resources),
            )
        return failures

    def is_empty(self, session_id: str) -> bool:
        return self._resources.is_empty(session_id)


class AcpClient:
    def __init__(self, *, project_root: "str | Path") -> None:
        self.project_root = Path(project_root).resolve()
        self._connection: Any = None
        self._client_capabilities: Any = None
        self._resources: "dict[str, AcpClientSessionResources]" = {}

    def set_connection(self, connection: Any, client_capabilities: Any = None) -> None:
        self._connection = connection
        self._client_capabilities = client_capabilities

    def resources(self, session_id: str) -> AcpClientSessionResources:
        return self._resources.setdefault(
            session_id, AcpClientSessionResources(self, session_id)
        )

    def resource_owner(self, session_id: str) -> AcpClientSessionOwner:
        resources = self.resources(session_id)
        return AcpClientSessionOwner(self, session_id, resources)

    def _remove_resources(
        self,
        session_id: str,
        *,
        expected: AcpClientSessionResources,
    ) -> None:
        if self._resources.get(session_id) is expected:
            self._resources.pop(session_id, None)

    async def close_resources(self, session_id: str) -> "tuple[ResourceFailure, ...]":
        owner = self._owner_for_existing(session_id)
        if owner is None:
            return ()
        return await owner.close(session_id)

    def _owner_for_existing(self, session_id: str) -> "AcpClientSessionOwner | None":
        resources = self._resources.get(session_id)
        if resources is None:
            return None
        return AcpClientSessionOwner(self, session_id, resources)

    async def session_update(self, session_id: str, update: Any) -> None:
        if self._connection is None:
            raise request_error("client_not_connected", session_id=session_id)
        await self._connection.session_update(session_id, update)

    async def replay(self, session_id: str, updates: "tuple[Any, ...]") -> None:
        for update in updates:
            await self.session_update(session_id, update)

    async def read_text_file(self, session: Any, path: str, line: "int | None" = None, limit: "int | None" = None) -> Any:
        self._ensure_capability("fs", "read_text_file", session.record.id)
        target = self._allowed_path(session, path)
        if (line is not None and line < 0) or (limit is not None and limit < 0):
            raise request_error("invalid_file_range", session_id=session.record.id)
        kwargs = {}
        if line is not None:
            kwargs["line"] = line
        if limit is not None:
            kwargs["limit"] = limit
        response = await self._connection_or_error().read_text_file(session.record.id, str(target), **kwargs)
        if not isinstance(getattr(response, "content", None), str) or len(response.content.encode()) > MAX_FILE_BYTES:
            raise request_error("client_file_read_failed", session_id=session.record.id)
        return response

    async def write_text_file(self, session: Any, path: str, content: str) -> Any:
        self._ensure_capability("fs", "write_text_file", session.record.id)
        if len(content.encode()) > MAX_FILE_BYTES:
            raise request_error("file_too_large", session_id=session.record.id)
        target = self._allowed_path(session, path, parent=True)
        return await self._connection_or_error().write_text_file(session.record.id, str(target), content)

    async def create_terminal(self, session: Any, **kwargs: Any) -> Any:
        self._ensure_capability("terminal", None, session.record.id)
        resources = self.resources(session.record.id)
        if session.closing_requested or session.record.state.value != "open":
            raise request_error("session_closed", session_id=session.record.id)
        if session.active_execution_id is None:
            raise request_error("no_active_execution", session_id=session.record.id)
        cwd = kwargs.get("cwd")
        if cwd is not None:
            kwargs["cwd"] = str(self._allowed_path(session, cwd))
        output_limit = kwargs.get("output_byte_limit") or DEFAULT_TERMINAL_OUTPUT_LIMIT
        if not 0 < output_limit <= MAX_TERMINAL_OUTPUT_LIMIT:
            raise request_error("invalid_terminal_output_limit", session_id=session.record.id)
        for item in kwargs.get("env") or ():
            if not getattr(item, "name", None) or "\x00" in item.name or "\x00" in item.value:
                raise request_error("invalid_terminal_environment", session_id=session.record.id)
        request = self._connection_or_error().create_terminal(session.record.id, **kwargs)
        # task-owner: acp.client.terminal_create
        task = asyncio.create_task(request)
        resources.terminal_create_tasks.add(task)
        cancelled = False
        try:
            try:
                response = await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
                response = await task
            terminal_id = response.terminal_id
            resources.terminal_handles.add(terminal_id)
            stale = (
                cancelled
                or session.closing_requested
                or session.record.state.value != "open"
                or session.active_execution_id is None
            )
            if stale:
                try:
                    await self._compensate_terminal(session.record.id, resources, terminal_id)
                except Exception as exc:
                    logger.error("event=acp.terminal.compensation_failed session_id=%s resource_count=%s", session.record.id, len(resources.terminal_handles))
                    raise request_error("terminal_compensation_failed", session_id=session.record.id) from exc
                if cancelled:
                    raise asyncio.CancelledError
                raise request_error("session_closing", session_id=session.record.id)
            return response
        finally:
            resources.terminal_create_tasks.discard(task)

    async def _compensate_terminal(self, session_id: str, resources: AcpClientSessionResources, terminal_id: str) -> None:
        connection = self._connection_or_error()
        await connection.kill_terminal(session_id, terminal_id)
        await resources._release(session_id, terminal_id)

    async def terminal_output(self, session: Any, terminal_id: str) -> Any:
        self._check_terminal(session, terminal_id)
        return await self._connection_or_error().terminal_output(session.record.id, terminal_id)

    async def wait_for_terminal_exit(self, session: Any, terminal_id: str) -> Any:
        self._check_terminal(session, terminal_id)
        return await self._connection_or_error().wait_for_terminal_exit(session.record.id, terminal_id)

    async def kill_terminal(self, session: Any, terminal_id: str) -> Any:
        self._check_terminal(session, terminal_id)
        return await self._connection_or_error().kill_terminal(session.record.id, terminal_id)

    async def release_terminal(self, session: Any, terminal_id: str) -> Any:
        self._check_terminal(session, terminal_id)
        return await self.resources(session.record.id)._release(session.record.id, terminal_id)

    async def create_elicitation(self, session: Any, message: str, mode: Any) -> Any:
        if session.closing_requested or session.active_execution_id is None:
            raise request_error("no_active_execution", session_id=session.record.id)
        root = getattr(mode, "root", mode)
        is_url = type(root).__name__ in {"ElicitationUrlSessionMode", "ElicitationUrlRequestMode"}
        capability = "url" if is_url else "form"
        self._ensure_elicitation_capability(capability, session.record.id)
        elicitation_id = getattr(root, "elicitation_id", None)
        mode_session_id = getattr(root, "session_id", None)
        if type(root).__name__ == "ElicitationUrlSessionMode":
            if not isinstance(elicitation_id, str) or not elicitation_id:
                raise request_error("invalid_elicitation_id", session_id=session.record.id)
            if mode_session_id is not None and mode_session_id != session.record.id:
                raise request_error("elicitation_session_mismatch", session_id=session.record.id)
        elicitation_id = elicitation_id or uuid4().hex
        resources = self.resources(session.record.id)
        # task-owner: acp.client.elicitation
        task = asyncio.create_task(self._connection_or_error().create_elicitation(message, mode))
        resources.elicitation_tasks[elicitation_id] = task
        cancelled = False
        try:
            try:
                response = await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
                response = await task
            outcome = getattr(response, "outcome", None)
            accepted = getattr(response, "action", None) == "accept" or type(outcome).__name__ == "ElicitationAcceptAction"
            if is_url and accepted:
                resources.accepted_elicitations.add(elicitation_id)
            if cancelled:
                raise asyncio.CancelledError
            return response
        finally:
            resources.elicitation_tasks.pop(elicitation_id, None)

    async def request_approval(
        self, request: Any, cancellation: asyncio.Event
    ) -> "ApprovalDecision | None":
        if self._connection is None:
            raise request_error("client_not_connected", session_id=request.session_id)
        if cancellation.is_set():
            return None
        import acp.schema as schema

        tool_call = schema.ToolCallUpdate(toolCallId=request.tool_call_id, title=request.tool_name, status="pending", rawInput=request.arguments)
        options = [
            schema.PermissionOption(optionId="allow_once", name="Allow once", kind="allow_once"),
            schema.PermissionOption(optionId="reject_once", name="Reject once", kind="reject_once"),
        ]
        response = await self._connection.request_permission(
            request.session_id, tool_call, options
        )
        outcome = response.outcome
        option_id = getattr(outcome, "option_id", getattr(outcome, "optionId", None))
        return (
            ApprovalDecision.ALLOW
            if option_id == "allow_once"
            else ApprovalDecision.DENY
            if option_id == "reject_once"
            else None
        )

    def _ensure_capability(self, name: str, operation: "str | None", session_id: str) -> None:
        value = getattr(self._client_capabilities, name, None)
        if operation is None:
            allowed = bool(value)
        else:
            allowed = bool(getattr(value, operation, False))
        if not allowed:
            raise request_error("client_capability_not_declared", session_id=session_id)

    def _ensure_elicitation_capability(self, mode: str, session_id: str) -> None:
        value = getattr(self._client_capabilities, "elicitation", None)
        if not getattr(value, mode, False):
            raise request_error("client_capability_not_declared", session_id=session_id)

    def _allowed_path(self, session: Any, value: str, *, parent: bool = False) -> Path:
        roots = tuple(Path(item).resolve() for item in (session.record.workspace.cwd,) + session.record.workspace.additional_directories)
        target = Path(value)
        if not target.is_absolute():
            target = Path(session.record.workspace.cwd) / target
        resolved = target.resolve()
        compare = resolved.parent if parent else resolved
        if not any(self._contained(compare, root) for root in roots):
            raise request_error("path_outside_allowed_roots", session_id=session.record.id)
        return resolved

    @staticmethod
    def _contained(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
        except ValueError:
            return False
        return True

    def _check_terminal(self, session: Any, terminal_id: str) -> None:
        if terminal_id not in self.resources(session.record.id).terminal_handles:
            raise request_error("unknown_terminal", session_id=session.record.id)

    def _connection_or_error(self) -> Any:
        if self._connection is None:
            raise request_error("client_not_connected")
        return self._connection


__all__ = [
    "AcpClient",
    "AcpClientSessionOwner",
    "AcpClientSessionResources",
    "MAX_FILE_BYTES",
    "MAX_TERMINAL_OUTPUT_LIMIT",
]
