#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ACP Client callbacks and resources owned by one ACP connection."""

import asyncio
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Awaitable, Callable
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


class _OutboundRequestKind(StrEnum):
    PERMISSION = "permission"
    FS_READ = "fs_read"
    FS_WRITE = "fs_write"
    TERMINAL_CREATE = "terminal_create"
    TERMINAL_OUTPUT = "terminal_output"
    TERMINAL_WAIT = "terminal_wait"
    TERMINAL_KILL = "terminal_kill"
    TERMINAL_RELEASE = "terminal_release"
    ELICITATION_CREATE = "elicitation_create"


@dataclass(slots=True, eq=False)
class _OutboundRequest:
    id: str
    session_id: str
    kind: _OutboundRequestKind
    task: "asyncio.Task[Any]"
    detached: bool = False
    completed: bool = False
    terminal_id: "str | None" = None
    started_at: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class AcpClientSessionResources:
    client: "AcpClient"
    session_id: str
    terminal_handles: "set[str]" = field(default_factory=set)
    _outbound: "dict[str, _OutboundRequest]" = field(default_factory=dict)
    accepted_elicitations: "set[str]" = field(default_factory=set)
    closing: bool = False

    async def close(self, session_id: str) -> "tuple[ResourceFailure, ...]":
        failures: "list[ResourceFailure]" = []
        self.closing = True
        deadline = asyncio.get_running_loop().time() + 1.0
        timed_out: "dict[str, _OutboundRequest]" = {}
        while self._outbound:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                timed_out.update(self._outbound)
                break
            operations = tuple(self._outbound.values())
            for operation in operations:
                if operation.task.done():
                    continue
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    timed_out[operation.id] = operation
                    break
                termination = await observe_task(operation.task, remaining)
                if termination is TaskTermination.TIMED_OUT:
                    timed_out[operation.id] = operation
                    break
            await asyncio.sleep(0)
        failures.extend(
            ResourceFailure(
                f"acp.client.{operation.kind.value}",
                operation.id,
                "task_timeout",
            )
            for operation in timed_out.values()
        )
        self.accepted_elicitations.clear()
        if self.client._connection is None:
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

    def _start_request(
        self,
        kind: _OutboundRequestKind,
        factory: "Callable[[], Awaitable[Any]]",
        completion: "Callable[[Any, bool], Awaitable[Any]] | None" = None,
        terminal_id: "str | None" = None,
    ) -> _OutboundRequest:
        if self.client._closing and not self.closing:
            raise request_error("client_closing", session_id=self.session_id)
        operation: _OutboundRequest

        async def run_request() -> Any:
            response = await factory()
            if completion is None:
                return response
            return await completion(response, operation.detached or self.closing)

        # task-owner: acp.client.outbound
        task = asyncio.create_task(run_request())
        operation = _OutboundRequest(
            uuid4().hex,
            self.session_id,
            kind,
            task,
            terminal_id=terminal_id,
        )
        self._outbound[operation.id] = operation
        task.add_done_callback(
            lambda completed: self._finish_request(operation, completed)
        )
        logger.info(
            "event=acp.client.operation_started session_id=%s operation=%s operation_id=%s",
            self.session_id,
            kind.value,
            operation.id,
        )
        return operation

    async def _request(
        self,
        kind: _OutboundRequestKind,
        factory: "Callable[[], Awaitable[Any]]",
        completion: "Callable[[Any, bool], Awaitable[Any]] | None" = None,
    ) -> Any:
        operation = self._start_request(kind, factory, completion)
        try:
            return await asyncio.shield(operation.task)
        except asyncio.CancelledError:
            operation.detached = True
            logger.info(
                "event=acp.client.operation_detached session_id=%s operation=%s operation_id=%s",
                self.session_id,
                kind.value,
                operation.id,
            )
            raise

    def _finish_request(
        self,
        operation: _OutboundRequest,
        task: "asyncio.Task[Any]",
    ) -> None:
        try:
            task.exception()
        except BaseException:
            pass
        operation.completed = True
        if self._outbound.get(operation.id) is operation:
            self._outbound.pop(operation.id, None)
        logger.info(
            "event=acp.client.operation_completed session_id=%s operation=%s operation_id=%s",
            self.session_id,
            operation.kind.value,
            operation.id,
        )

    def _find_request(
        self, kind: _OutboundRequestKind, terminal_id: "str | None" = None
    ) -> "_OutboundRequest | None":
        for operation in self._outbound.values():
            if operation.kind is not kind:
                continue
            if terminal_id is None or getattr(operation, "terminal_id", None) == terminal_id:
                return operation
        return None

    async def _release(self, session_id: str, terminal_id: str) -> Any:
        operation = self._find_request(_OutboundRequestKind.TERMINAL_RELEASE, terminal_id)
        if operation is None:
            operation = self._start_request(
                _OutboundRequestKind.TERMINAL_RELEASE,
                lambda: self._connection_request().release_terminal(session_id, terminal_id),
                self._release_completion(terminal_id),
                terminal_id,
            )
        try:
            return await asyncio.shield(operation.task)
        except asyncio.CancelledError:
            operation.detached = True
            raise

    async def _kill(self, session_id: str, terminal_id: str) -> Any:
        operation = self._find_request(_OutboundRequestKind.TERMINAL_KILL, terminal_id)
        if operation is None:
            operation = self._start_request(
                _OutboundRequestKind.TERMINAL_KILL,
                lambda: self._connection_request().kill_terminal(session_id, terminal_id),
                terminal_id=terminal_id,
            )
        try:
            return await asyncio.shield(operation.task)
        except asyncio.CancelledError:
            operation.detached = True
            raise

    async def _kill_with_timeout(
        self, session_id: str, terminal_id: str, timeout: float
    ) -> Any:
        operation = self._find_request(_OutboundRequestKind.TERMINAL_KILL, terminal_id)
        if operation is None:
            operation = self._start_request(
                _OutboundRequestKind.TERMINAL_KILL,
                lambda: self._connection_request().kill_terminal(session_id, terminal_id),
                terminal_id=terminal_id,
            )
        return await asyncio.wait_for(asyncio.shield(operation.task), timeout)

    async def _release_with_timeout(
        self, session_id: str, terminal_id: str, timeout: float
    ) -> Any:
        operation = self._find_request(_OutboundRequestKind.TERMINAL_RELEASE, terminal_id)
        if operation is None:
            operation = self._start_request(
                _OutboundRequestKind.TERMINAL_RELEASE,
                lambda: self._connection_request().release_terminal(session_id, terminal_id),
                self._release_completion(terminal_id),
                terminal_id,
            )
        return await asyncio.wait_for(asyncio.shield(operation.task), timeout)

    def _release_completion(
        self, terminal_id: str
    ) -> "Callable[[Any, bool], Awaitable[Any]]":
        async def complete(response: Any, detached: bool) -> Any:
            self.terminal_handles.discard(terminal_id)
            return response

        return complete

    def _connection_request(self) -> Any:
        if self.client._connection is None:
            raise RuntimeError("client connection is unavailable")
        return self.client._connection

    def is_empty(self, session_id: str) -> bool:
        return not (
            self.terminal_handles
            or self._outbound
            or self.accepted_elicitations
        )

    @property
    def operation_count(self) -> int:
        return len(self._outbound)

    @property
    def outbound_request_count(self) -> int:
        return self.operation_count

    @property
    def detached_operation_count(self) -> int:
        return sum(1 for operation in self._outbound.values() if operation.detached)

    @property
    def detached_outbound_request_count(self) -> int:
        return self.detached_operation_count

    @property
    def outbound_request_age_ms(self) -> float:
        if not self._outbound:
            return 0.0
        return max(
            0.0,
            (time.monotonic() - min(item.started_at for item in self._outbound.values()))
            * 1000,
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
        self._closing = False
        self._close_task: "asyncio.Task[tuple[ResourceFailure, ...]] | None" = None

    def set_connection(self, connection: Any, client_capabilities: Any = None) -> None:
        if self._closing:
            raise request_error("client_closing")
        self._connection = connection
        self._client_capabilities = client_capabilities

    @property
    def client_operation_count(self) -> int:
        return sum(resource.operation_count for resource in self._resources.values())

    @property
    def outbound_request_count(self) -> int:
        return sum(
            resource.outbound_request_count for resource in self._resources.values()
        )

    @property
    def client_detached_operation_count(self) -> int:
        return sum(
            resource.detached_operation_count for resource in self._resources.values()
        )

    @property
    def detached_outbound_request_count(self) -> int:
        return sum(
            resource.detached_outbound_request_count
            for resource in self._resources.values()
        )

    @property
    def client_outbound_request_age_ms(self) -> float:
        return max(
            (resource.outbound_request_age_ms for resource in self._resources.values()),
            default=0.0,
        )

    def resources(self, session_id: str) -> AcpClientSessionResources:
        if self._closing and session_id not in self._resources:
            raise request_error("client_closing", session_id=session_id)
        return self._resources.setdefault(
            session_id, AcpClientSessionResources(self, session_id)
        )

    def resource_owner(self, session_id: str) -> AcpClientSessionOwner:
        if self._closing and session_id not in self._resources:
            raise request_error("client_closing", session_id=session_id)
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

    async def close(self) -> "tuple[ResourceFailure, ...]":
        task = self._close_task
        if task is None:
            task = asyncio.create_task(self._close_once())
            self._close_task = task
        else:
            logger.info("event=acp.client.close_joined")
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    async def _close_once(self) -> "tuple[ResourceFailure, ...]":
        self._closing = True
        connection = self._connection
        failures: "list[ResourceFailure]" = []
        if connection is not None:
            close = getattr(connection, "close", None)
            if close is not None:
                try:
                    result = close()
                    if hasattr(result, "__await__"):
                        await result
                except Exception as exc:
                    failures.append(
                        ResourceFailure("acp.client.connection", None, type(exc).__name__)
                    )
        self._connection = None
        for session_id in tuple(self._resources):
            failures.extend(await self.close_resources(session_id))
        logger.info(
            "event=acp.client.closed client_failure_count=%s resource_count=%s owner_count=%s",
            len(failures),
            len(self._resources),
            len(self._owners),
        )
        return tuple(failures)

    def _owner_for_existing(self, session_id: str) -> "AcpClientSessionOwner | None":
        resources = self._resources.get(session_id)
        if resources is None:
            return None
        return self.resource_owner(session_id)

    async def session_update(self, session_id: str, update: Any) -> None:
        if self._closing:
            raise request_error("client_closing", session_id=session_id)
        if self._connection is None:
            raise request_error("client_not_connected", session_id=session_id)
        await self._connection.session_update(session_id, update)

    async def replay(self, session_id: str, updates: "tuple[Any, ...]") -> None:
        for update in updates:
            await self.session_update(session_id, update)

    async def read_text_file(
        self,
        session: Any,
        path: str,
        line: "int | None" = None,
        limit: "int | None" = None,
    ) -> Any:
        self._ensure_capability("fs", "read_text_file", session.record.id)
        target = self._allowed_path(session, path)
        if (line is not None and line < 0) or (limit is not None and limit < 0):
            raise request_error("invalid_file_range", session_id=session.record.id)
        kwargs = {}
        if line is not None:
            kwargs["line"] = line
        if limit is not None:
            kwargs["limit"] = limit

        async def complete(response: Any, detached: bool) -> Any:
            if detached:
                return None
            if not isinstance(getattr(response, "content", None), str) or len(response.content.encode()) > MAX_FILE_BYTES:
                raise request_error("client_file_read_failed", session_id=session.record.id)
            return response

        return await self.resources(session.record.id)._request(
            _OutboundRequestKind.FS_READ,
            lambda: self._connection_or_error().read_text_file(
                session.record.id, str(target), **kwargs
            ),
            complete,
        )

    async def write_text_file(self, session: Any, path: str, content: str) -> Any:
        self._ensure_capability("fs", "write_text_file", session.record.id)
        if len(content.encode()) > MAX_FILE_BYTES:
            raise request_error("file_too_large", session_id=session.record.id)
        target = self._allowed_path(session, path, parent=True)

        async def complete(response: Any, detached: bool) -> Any:
            return None if detached else response

        return await self.resources(session.record.id)._request(
            _OutboundRequestKind.FS_WRITE,
            lambda: self._connection_or_error().write_text_file(
                session.record.id, str(target), content
            ),
            complete,
        )

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
        resources = self.resources(session.record.id)
        operation: _OutboundRequest

        async def complete(response: Any, detached: bool) -> Any:
            terminal_id = response.terminal_id
            operation.terminal_id = terminal_id
            resources.terminal_handles.add(terminal_id)
            stale = (
                detached
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
            if detached:
                raise asyncio.CancelledError
            raise request_error("session_closing", session_id=session.record.id)

        operation = resources._start_request(
            _OutboundRequestKind.TERMINAL_CREATE,
            lambda: self._connection_or_error().create_terminal(
                session.record.id, **kwargs
            ),
            complete,
        )
        try:
            return await asyncio.shield(operation.task)
        except asyncio.CancelledError:
            operation.detached = True
            raise

    async def _compensate_terminal(self, session_id: str, resources: AcpClientSessionResources, terminal_id: str) -> None:
        await resources._kill(session_id, terminal_id)
        await resources._release(session_id, terminal_id)

    async def terminal_output(self, session: Any, terminal_id: str) -> Any:
        self._check_terminal(session, terminal_id)
        return await self.resources(session.record.id)._request(
            _OutboundRequestKind.TERMINAL_OUTPUT,
            lambda: self._connection_or_error().terminal_output(
                session.record.id, terminal_id
            ),
        )

    async def wait_for_terminal_exit(self, session: Any, terminal_id: str) -> Any:
        self._check_terminal(session, terminal_id)
        return await self.resources(session.record.id)._request(
            _OutboundRequestKind.TERMINAL_WAIT,
            lambda: self._connection_or_error().wait_for_terminal_exit(
                session.record.id, terminal_id
            ),
        )

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

        async def complete(response: Any, detached: bool) -> Any:
            outcome = getattr(response, "outcome", None)
            accepted = getattr(response, "action", None) == "accept" or type(outcome).__name__ == "ElicitationAcceptAction"
            if (
                is_url
                and accepted
                and not detached
                and not resources.closing
                and not session.closing_requested
            ):
                resources.accepted_elicitations.add(elicitation_id)
            return None if detached else response

        return await resources._request(
            _OutboundRequestKind.ELICITATION_CREATE,
            lambda: self._connection_or_error().create_elicitation(message, mode),
            complete,
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
        resources = self.resources(request.session_id)

        async def complete(response: Any, detached: bool) -> Any:
            if detached:
                return None
            outcome = response.outcome
            option_id = getattr(outcome, "option_id", getattr(outcome, "optionId", None))
            return (
                ApprovalDecision.ALLOW
                if option_id == "allow_once"
                else ApprovalDecision.DENY
                if option_id == "reject_once"
                else None
            )

        operation = resources._start_request(
            _OutboundRequestKind.PERMISSION,
            lambda: self._connection_or_error().request_permission(
                request.session_id, tool_call, options
            ),
            complete,
        )
        cancel_wait = asyncio.create_task(cancellation.wait())
        try:
            done, _ = await asyncio.wait(
                {operation.task, cancel_wait},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_wait in done and not operation.task.done():
                operation.detached = True
                return None
            return await asyncio.shield(operation.task)
        except asyncio.CancelledError:
            operation.detached = True
            raise
        finally:
            if not cancel_wait.done():
                cancel_wait.cancel()
            try:
                await cancel_wait
            except asyncio.CancelledError:
                pass

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
        resources = self._resources.get(session.record.id)
        if resources is None or terminal_id not in resources.terminal_handles:
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
