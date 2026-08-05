#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ACP Client callbacks and resources owned by one ACP connection."""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from linktools.core import environ

from ..execution.cancellation import TaskTermination, observe_task
from ..execution.domain import ApprovalDecision
from ..runtime.session import ResourceFailure
from .protocol import request_error


MAX_FILE_BYTES = 2 * 1024 * 1024
DEFAULT_TERMINAL_OUTPUT_LIMIT = 256 * 1024
MAX_TERMINAL_OUTPUT_LIMIT = 1024 * 1024
logger = environ.get_logger("ai.acp.client")


@dataclass(slots=True, eq=False)
class _ClientOperation:
    task: "asyncio.Task[Any]"
    cancel_requested: bool = False
    detached: bool = False
    terminal_id: "str | None" = None


@dataclass(slots=True)
class AcpClientSessionResources:
    client: "AcpClient"
    session_id: str
    terminal_handles: "set[str]" = field(default_factory=set)
    terminal_create_operations: "set[_ClientOperation]" = field(default_factory=set)
    terminal_release_operations: "dict[str, _ClientOperation]" = field(default_factory=dict)
    terminal_kill_operations: "dict[str, _ClientOperation]" = field(default_factory=dict)
    elicitation_operations: "dict[str, _ClientOperation]" = field(default_factory=dict)
    accepted_elicitations: "set[str]" = field(default_factory=set)
    closing: bool = False

    async def close(self, session_id: str) -> "tuple[ResourceFailure, ...]":
        failures: "list[ResourceFailure]" = []
        self.closing = True
        connection = self.client._connection
        for operation in tuple(self.terminal_create_operations):
            self._detach(operation)
            termination = await observe_task(operation.task, 1.0)
            if termination is TaskTermination.TIMED_OUT:
                failures.append(ResourceFailure("acp.client.terminal", None, "task_timeout"))
        for operation in tuple(self.elicitation_operations.values()):
            self._detach(operation)
            termination = await observe_task(operation.task, 1.0)
            if termination is TaskTermination.TIMED_OUT:
                failures.append(ResourceFailure("acp.client.elicitation", None, "task_timeout"))
        for operation in tuple(self.terminal_release_operations.values()):
            termination = await observe_task(operation.task, 1.0)
            if termination is TaskTermination.TIMED_OUT:
                failures.append(ResourceFailure("acp.client.terminal", None, "task_timeout"))
        for operation in tuple(self.terminal_kill_operations.values()):
            termination = await observe_task(operation.task, 1.0)
            if termination is TaskTermination.TIMED_OUT:
                failures.append(ResourceFailure("acp.client.terminal", None, "task_timeout"))
        self.accepted_elicitations.clear()
        if connection is None:
            failures.extend(
                ResourceFailure("acp.client", terminal_id, "client_connection_missing")
                for terminal_id in self.terminal_handles
            )
            return tuple(failures)
        for terminal_id in tuple(self.terminal_handles):
            try:
                await self._kill_with_timeout(session_id, terminal_id, 1.0)
            except asyncio.TimeoutError:
                failures.append(ResourceFailure("acp.client.terminal", terminal_id, "task_timeout"))
                continue
            except Exception as exc:
                failures.append(ResourceFailure("acp.client.terminal", terminal_id, type(exc).__name__))
                continue
            try:
                await self._release_with_timeout(session_id, terminal_id, 1.0)
            except asyncio.TimeoutError:
                failures.append(ResourceFailure("acp.client.terminal", terminal_id, "task_timeout"))
            except Exception as exc:
                failures.append(ResourceFailure("acp.client.terminal", terminal_id, type(exc).__name__))
        return tuple(failures)

    async def _release(self, session_id: str, terminal_id: str) -> Any:
        connection = self.client._connection
        if connection is None:
            raise RuntimeError("client connection is unavailable")
        operation = self.terminal_release_operations.get(terminal_id)
        if operation is None:
            request = connection.release_terminal(session_id, terminal_id)
            # task-owner: acp.client.terminal_release
            task = asyncio.create_task(request)
            operation = _ClientOperation(task)
            self.terminal_release_operations[terminal_id] = operation
            task.add_done_callback(
                lambda completed: self._finish_release(terminal_id, operation, completed)
            )
        try:
            response = await asyncio.shield(operation.task)
        except asyncio.CancelledError:
            self._detach(operation)
            logger.info(
                "event=acp.client.operation_detached session_id=%s operation=terminal_release",
                session_id,
            )
            raise
        except BaseException:
            raise
        self.terminal_handles.discard(terminal_id)
        return response

    async def _kill(self, session_id: str, terminal_id: str) -> Any:
        operation = self._start_kill(session_id, terminal_id)
        try:
            return await asyncio.shield(operation.task)
        except asyncio.CancelledError:
            self._detach(operation)
            logger.info(
                "event=acp.client.operation_detached session_id=%s operation=terminal_kill",
                session_id,
            )
            raise

    async def _kill_with_timeout(
        self, session_id: str, terminal_id: str, timeout: float
    ) -> Any:
        operation = self._start_kill(session_id, terminal_id)
        return await asyncio.wait_for(asyncio.shield(operation.task), timeout)

    def _start_kill(self, session_id: str, terminal_id: str) -> _ClientOperation:
        operation = self.terminal_kill_operations.get(terminal_id)
        if operation is not None:
            return operation
        connection = self.client._connection
        if connection is None:
            raise RuntimeError("client connection is unavailable")
        request = connection.kill_terminal(session_id, terminal_id)
        # task-owner: acp.client.terminal_kill
        task = asyncio.create_task(request)
        operation = _ClientOperation(task)
        self.terminal_kill_operations[terminal_id] = operation
        task.add_done_callback(
            lambda completed: self._finish_kill(terminal_id, operation, completed)
        )
        return operation

    def _finish_kill(
        self,
        terminal_id: str,
        operation: _ClientOperation,
        task: "asyncio.Task[Any]",
    ) -> None:
        try:
            task.exception()
        except asyncio.CancelledError:
            pass
        except BaseException:
            pass
        if self.terminal_kill_operations.get(terminal_id) is operation:
            self.terminal_kill_operations.pop(terminal_id, None)
        logger.info(
            "event=acp.client.operation_completed session_id=%s operation=terminal_kill",
            self.session_id,
        )

    async def _release_with_timeout(
        self, session_id: str, terminal_id: str, timeout: float
    ) -> Any:
        operation = self.terminal_release_operations.get(terminal_id)
        if operation is None:
            connection = self.client._connection
            if connection is None:
                raise RuntimeError("client connection is unavailable")
            request = connection.release_terminal(session_id, terminal_id)
            # task-owner: acp.client.terminal_release
            task = asyncio.create_task(request)
            operation = _ClientOperation(task)
            self.terminal_release_operations[terminal_id] = operation
            task.add_done_callback(
                lambda completed: self._finish_release(terminal_id, operation, completed)
            )
        response = await asyncio.wait_for(asyncio.shield(operation.task), timeout)
        self.terminal_handles.discard(terminal_id)
        return response

    def _finish_release(
        self,
        terminal_id: str,
        operation: _ClientOperation,
        task: "asyncio.Task[Any]",
    ) -> None:
        succeeded = False
        try:
            succeeded = task.exception() is None
        except asyncio.CancelledError:
            pass
        except BaseException:
            pass
        if self.terminal_release_operations.get(terminal_id) is operation:
            self.terminal_release_operations.pop(terminal_id, None)
        if succeeded:
            self.terminal_handles.discard(terminal_id)
        logger.info(
            "event=acp.client.operation_completed session_id=%s operation=terminal_release",
            self.session_id,
        )

    def _detach(self, operation: _ClientOperation) -> None:
        operation.cancel_requested = True
        operation.detached = True
        if not operation.task.done():
            operation.task.cancel()

    def is_empty(self, session_id: str) -> bool:
        return not (
            self.terminal_handles
            or self.terminal_create_operations
            or self.terminal_release_operations
            or self.terminal_kill_operations
            or self.elicitation_operations
            or self.accepted_elicitations
        )

    @property
    def operation_count(self) -> int:
        return (
            len(self.terminal_create_operations)
            + len(self.terminal_release_operations)
            + len(self.terminal_kill_operations)
            + len(self.elicitation_operations)
        )

    @property
    def detached_operation_count(self) -> int:
        operations = (
            tuple(self.terminal_create_operations)
            + tuple(self.terminal_release_operations.values())
            + tuple(self.terminal_kill_operations.values())
            + tuple(self.elicitation_operations.values())
        )
        return sum(1 for operation in operations if operation.detached)


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

    def discard_if_empty(self) -> None:
        if self._resources.is_empty(self._session_id):
            self._parent._remove_resources(
                self._session_id,
                expected=self._resources,
            )


class AcpClient:
    def __init__(self, *, project_root: "str | Path") -> None:
        self.project_root = Path(project_root).resolve()
        self._connection: Any = None
        self._client_capabilities: Any = None
        self._resources: "dict[str, AcpClientSessionResources]" = {}
        self._owners: "dict[str, AcpClientSessionOwner]" = {}

    def set_connection(self, connection: Any, client_capabilities: Any = None) -> None:
        self._connection = connection
        self._client_capabilities = client_capabilities

    @property
    def client_operation_count(self) -> int:
        return sum(resource.operation_count for resource in self._resources.values())

    @property
    def client_detached_operation_count(self) -> int:
        return sum(
            resource.detached_operation_count for resource in self._resources.values()
        )

    def resources(self, session_id: str) -> AcpClientSessionResources:
        return self._resources.setdefault(
            session_id, AcpClientSessionResources(self, session_id)
        )

    def resource_owner(self, session_id: str) -> AcpClientSessionOwner:
        resources = self.resources(session_id)
        owner = self._owners.get(session_id)
        if owner is not None and owner._resources is resources:
            return owner
        owner = AcpClientSessionOwner(self, session_id, resources)
        self._owners[session_id] = owner
        return owner

    def _remove_resources(
        self,
        session_id: str,
        *,
        expected: AcpClientSessionResources,
    ) -> None:
        if self._resources.get(session_id) is expected:
            self._resources.pop(session_id, None)
            owner = self._owners.get(session_id)
            if owner is not None and owner._resources is expected:
                self._owners.pop(session_id, None)

    async def close_resources(self, session_id: str) -> "tuple[ResourceFailure, ...]":
        owner = self._owner_for_existing(session_id)
        if owner is None:
            return ()
        return await owner.close(session_id)

    def _owner_for_existing(self, session_id: str) -> "AcpClientSessionOwner | None":
        resources = self._resources.get(session_id)
        if resources is None:
            return None
        return self.resource_owner(session_id)

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
        operation: _ClientOperation

        async def run_request() -> Any:
            response = await request
            terminal_id = response.terminal_id
            operation.terminal_id = terminal_id
            resources.terminal_handles.add(terminal_id)
            stale = (
                operation.cancel_requested
                or operation.detached
                or resources.closing
                or session.closing_requested
                or session.record.state.value != "open"
                or session.active_execution_id is None
            )
            if not stale:
                return response
            try:
                await self._compensate_terminal(session.record.id, resources, terminal_id)
            except Exception as exc:
                logger.error(
                    "event=acp.client.late_terminal_compensated session_id=%s terminal_id=%s error_id=%s",
                    session.record.id,
                    terminal_id,
                    type(exc).__name__,
                )
                raise
            if operation.cancel_requested or operation.detached:
                raise asyncio.CancelledError
            raise request_error("session_closing", session_id=session.record.id)

        # task-owner: acp.client.terminal_create
        task = asyncio.create_task(run_request())
        operation = _ClientOperation(task)
        resources.terminal_create_operations.add(operation)
        task.add_done_callback(
            lambda completed: self._finish_terminal_create(resources, operation, completed)
        )
        try:
            try:
                response = await asyncio.shield(operation.task)
            except asyncio.CancelledError:
                operation.cancel_requested = True
                operation.detached = True
                operation.task.cancel()
                logger.info(
                    "event=acp.client.operation_detached session_id=%s operation=terminal_create",
                    session.record.id,
                )
                raise
            return response
        except asyncio.CancelledError:
            raise

    @staticmethod
    def _finish_terminal_create(
        resources: AcpClientSessionResources,
        operation: _ClientOperation,
        task: "asyncio.Task[Any]",
    ) -> None:
        try:
            task.exception()
        except asyncio.CancelledError:
            pass
        except BaseException:
            pass
        resources.terminal_create_operations.discard(operation)
        logger.info(
            "event=acp.client.operation_completed session_id=%s operation=terminal_create",
            resources.session_id,
        )

    async def _compensate_terminal(self, session_id: str, resources: AcpClientSessionResources, terminal_id: str) -> None:
        await resources._kill(session_id, terminal_id)
        await resources._release(session_id, terminal_id)

    async def terminal_output(self, session: Any, terminal_id: str) -> Any:
        self._check_terminal(session, terminal_id)
        return await self._connection_or_error().terminal_output(session.record.id, terminal_id)

    async def wait_for_terminal_exit(self, session: Any, terminal_id: str) -> Any:
        self._check_terminal(session, terminal_id)
        return await self._connection_or_error().wait_for_terminal_exit(session.record.id, terminal_id)

    async def kill_terminal(self, session: Any, terminal_id: str) -> Any:
        self._check_terminal(session, terminal_id)
        return await self.resources(session.record.id)._kill(session.record.id, terminal_id)

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
        request = self._connection_or_error().create_elicitation(message, mode)
        operation: _ClientOperation

        async def run_request() -> Any:
            response = await request
            outcome = getattr(response, "outcome", None)
            accepted = getattr(response, "action", None) == "accept" or type(outcome).__name__ == "ElicitationAcceptAction"
            if (
                is_url
                and accepted
                and not resources.closing
                and not operation.cancel_requested
                and not operation.detached
                and not session.closing_requested
            ):
                resources.accepted_elicitations.add(elicitation_id)
            return response

        # task-owner: acp.client.elicitation
        task = asyncio.create_task(run_request())
        operation = _ClientOperation(task)
        resources.elicitation_operations[elicitation_id] = operation
        task.add_done_callback(
            lambda completed: self._finish_elicitation(resources, elicitation_id, operation, completed)
        )
        try:
            try:
                response = await asyncio.shield(operation.task)
            except asyncio.CancelledError:
                operation.cancel_requested = True
                operation.detached = True
                operation.task.cancel()
                logger.info(
                    "event=acp.client.operation_detached session_id=%s operation=elicitation",
                    session.record.id,
                )
                raise
            return response
        except asyncio.CancelledError:
            raise

    @staticmethod
    def _finish_elicitation(
        resources: AcpClientSessionResources,
        elicitation_id: str,
        operation: _ClientOperation,
        task: "asyncio.Task[Any]",
    ) -> None:
        try:
            task.exception()
        except asyncio.CancelledError:
            pass
        except BaseException:
            pass
        if resources.elicitation_operations.get(elicitation_id) is operation:
            resources.elicitation_operations.pop(elicitation_id, None)
        logger.info(
            "event=acp.client.operation_completed session_id=%s operation=elicitation",
            resources.session_id,
        )

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
