#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ACP session ownership and lifecycle state machine."""

import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping
from uuid import uuid4

from ..execution.domain import RunStatus
from ..governance.identity import PrincipalContext
from ..runtime.facade import Runtime
from .errors import request_error
from .persistence import AcpSessionRecord, AcpSessionRepository, mcp_descriptor_fingerprint
from .session_state import (
    SessionOperationCoordinator,
    SessionOperationKind,
    SessionOperationToken,
    assert_session_invariants,
)

logger = logging.getLogger("linktools.ai.acp.sessions")


@dataclass(slots=True)
class SessionMcpResources:
    descriptors: "tuple[Any, ...]" = ()
    state: "McpResourceState" = field(default_factory=lambda: McpResourceState.NEW)
    lock: "asyncio.Lock" = field(default_factory=asyncio.Lock)
    connect_task: "asyncio.Task[tuple[Any, ...]] | None" = None
    close_task: "asyncio.Task[None] | None" = None
    pool: Any = None
    _toolsets: "tuple[Any, ...] | None" = None

    async def toolsets(self) -> "tuple[Any, ...]":
        async with self.lock:
            if self.state is McpResourceState.CLOSED:
                raise request_error("session_closed")
            if self.state is McpResourceState.CLOSING:
                raise request_error("session_closing")
            if self.state is McpResourceState.OPEN:
                return self._toolsets or ()
            if self.connect_task is None:
                self.state = McpResourceState.CONNECTING
                self.connect_task = asyncio.create_task(self._connect_once())
            task = self.connect_task
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.gather(task, return_exceptions=True)
            raise

    async def _connect_once(self) -> "tuple[Any, ...]":
        from ..agent.mcp.connection import MCPConnectionPool

        pool = MCPConnectionPool()
        try:
            values = [
                await pool.get_toolset(mcp_spec(descriptor))
                for descriptor in self.descriptors
            ]
            toolsets = tuple(handle.toolset for handle in values)
        except asyncio.CancelledError:
            await pool.close()
            async with self.lock:
                if self.connect_task is asyncio.current_task():
                    self.connect_task = None
                    self.state = McpResourceState.NEW
            raise
        except Exception as exc:
            await pool.close()
            async with self.lock:
                if self.connect_task is asyncio.current_task():
                    self.connect_task = None
                    self.state = McpResourceState.NEW
            raise request_error("mcp_connection_failed") from exc
        async with self.lock:
            stale = self.state is not McpResourceState.CONNECTING
            if not stale:
                self.pool = pool
                self._toolsets = toolsets
                self.connect_task = None
                self.state = McpResourceState.OPEN
        if stale:
            await pool.close()
            async with self.lock:
                self.connect_task = None
            return ()
        return toolsets

    async def close(self) -> None:
        async with self.lock:
            if self.state is McpResourceState.CLOSED:
                return
            if self.close_task is not None:
                task = self.close_task
            else:
                self.state = McpResourceState.CLOSING
                task = asyncio.create_task(self._close_once())
                self.close_task = task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.gather(task, return_exceptions=True)
            raise

    async def _close_once(self) -> None:
        async with self.lock:
            connect_task = self.connect_task
            pool = self.pool
        if connect_task is not None and not connect_task.done():
            connect_task.cancel()
            await asyncio.gather(connect_task, return_exceptions=True)
            logger.info(
                "event=acp.mcp.connect_cancelled pending_task_count=0 state=closing"
            )
        if pool is not None:
            try:
                await asyncio.wait_for(pool.close(), timeout=10)
            except Exception:
                async with self.lock:
                    self.state = McpResourceState.OPEN
                    self.close_task = None
                raise
        async with self.lock:
            self.pool = None
            self._toolsets = None
            self.connect_task = None
            self.state = McpResourceState.CLOSED
            self.descriptors = ()
            self.close_task = None


class McpResourceState(str, Enum):
    NEW = "new"
    CONNECTING = "connecting"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass(slots=True)
class ActiveAcpSession:
    record: AcpSessionRecord
    lock: asyncio.Lock
    active_execution_id: "str | None"
    mcp_resources: SessionMcpResources
    terminal_handles: "set[str]"
    pending_elicitation_ids: "set[str]"
    operation_epoch: int = 0
    operation: "SessionOperationToken | None" = None
    closing_requested: bool = False
    cleanup_required: bool = False
    close_task: "asyncio.Task[SessionCloseResult] | None" = None
    pending_permission: "PendingPermissionToken | None" = None
    pending_permission_task: "asyncio.Task[Any] | None" = None
    pending_elicitation_tasks: "dict[str, asyncio.Task[Any]]" = field(default_factory=dict)
    terminal_create_tasks: "set[asyncio.Task[Any]]" = field(default_factory=set)
    terminal_release_tasks: "dict[str, asyncio.Task[Any]]" = field(default_factory=dict)

    @property
    def closing(self) -> bool:
        """Deprecated read-only view retained for local callers during cleanup."""
        return self.closing_requested or self.cleanup_required


@dataclass(frozen=True, slots=True)
class PendingPermissionToken:
    session_id: str
    execution_id: str
    approval_id: str
    tool_call_id: str
    epoch: int


@dataclass(frozen=True, slots=True)
class SessionCloseFailure:
    resource_type: "Literal['operation', 'execution', 'permission', 'elicitation', 'terminal', 'mcp', 'persistence']"
    resource_id: "str | None"
    error_id: str


@dataclass(frozen=True, slots=True)
class SessionCloseResult:
    closed: bool
    failures: "tuple[SessionCloseFailure, ...]"


CloseReason = Literal["client", "eof", "signal", "error"]


def validate_session_paths(
    *, project_root: "str | Path", cwd: str, additional_directories: "list[str] | None"
) -> "tuple[str, tuple[str, ...]]":
    project = Path(os.path.normcase(str(Path(project_root).resolve(strict=True))))
    target = Path(os.path.normcase(str(Path(cwd).resolve(strict=True))))
    if not target.is_dir() or not _contained(target, project):
        raise request_error("invalid_cwd")
    additional = []
    for value in additional_directories or ():
        path = Path(os.path.normcase(str(Path(value).resolve(strict=True))))
        if not path.is_dir():
            raise request_error("invalid_additional_directory")
        additional.append(str(path))
    return str(target), tuple(sorted(set(additional)))


def _contained(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


class AcpSessionService:
    def __init__(
        self,
        *,
        runtime: Runtime,
        repository: AcpSessionRepository,
        project_root: "str | Path",
        principal: PrincipalContext,
        default_mode_id: str,
        mode_ids: "tuple[str, ...]" = (),
        config_defaults: "Mapping[str, Any] | None" = None,
        client_services: Any = None,
    ) -> None:
        self.runtime = runtime
        self.repository = repository
        self.project_root = Path(os.path.normcase(str(Path(project_root).resolve())))
        self.principal = principal
        self.default_mode_id = default_mode_id
        self.mode_ids = tuple(mode_ids) or (default_mode_id,)
        self.config_defaults = dict(config_defaults or {})
        self.client_services = client_services
        self._active: "dict[str, ActiveAcpSession]" = {}
        self.coordinator = SessionOperationCoordinator()

    @property
    def active_sessions(self) -> Mapping[str, ActiveAcpSession]:
        return self._active

    async def create(
        self,
        *,
        cwd: str,
        additional_directories: "list[str] | None" = None,
        mcp_servers: "list[Any] | None" = None,
    ) -> ActiveAcpSession:
        _validate_mcp_descriptors(mcp_servers or ())
        normalized_cwd, directories = validate_session_paths(
            project_root=self.project_root,
            cwd=cwd,
            additional_directories=additional_directories,
        )
        session_id = uuid4().hex
        await self.runtime.create_session(session_id, principal=self.principal)
        now = datetime.now(timezone.utc)
        record = AcpSessionRecord(
            schema_version=1,
            session_id=session_id,
            cwd=normalized_cwd,
            additional_directories=directories,
            mode_id=self.default_mode_id,
            config_values=dict(self.config_defaults),
            mcp_server_fingerprints=tuple(
                sorted(mcp_descriptor_fingerprint(server) for server in (mcp_servers or ()))
            ),
            title=None,
            closed=False,
            created_at=now,
            updated_at=now,
        )
        self.repository.save(record)
        active = ActiveAcpSession(record, asyncio.Lock(), None, SessionMcpResources(tuple(mcp_servers or ())), set(), set())
        self._active[session_id] = active
        return active

    async def get(self, session_id: str) -> ActiveAcpSession:
        active = self._active.get(session_id)
        if active is not None:
            return active
        record = self.repository.load(session_id)
        if record is None:
            raise request_error("unknown_session", session_id=session_id)
        active = ActiveAcpSession(record, asyncio.Lock(), None, SessionMcpResources(), set(), set())
        self._active[session_id] = active
        return active

    async def load_or_resume(
        self,
        *,
        session_id: str,
        cwd: str,
        additional_directories: "list[str] | None",
        mcp_servers: "list[Any] | None",
        replay: bool,
    ) -> ActiveAcpSession:
        active = await self.get(session_id)
        _validate_mcp_descriptors(mcp_servers or ())
        normalized_cwd, directories = validate_session_paths(
            project_root=self.project_root,
            cwd=cwd,
            additional_directories=additional_directories,
        )
        operation = await self.coordinator.reserve(
            active,
            SessionOperationKind.LOAD if replay else SessionOperationKind.RESUME,
        )
        new_resources: SessionMcpResources | None = None
        old_descriptors: tuple[Any, ...] = ()
        try:
            async with active.lock:
                if active.record.cwd != normalized_cwd:
                    raise request_error("cwd_mismatch", session_id=session_id)
                fingerprints = tuple(
                    sorted(
                        mcp_descriptor_fingerprint(server)
                        for server in (mcp_servers or ())
                    )
                )
                if fingerprints != active.record.mcp_server_fingerprints:
                    raise request_error("mcp_descriptor_mismatch", session_id=session_id)
                base_record = active.record
                old_mcp_resources = active.mcp_resources
                old_descriptors = old_mcp_resources.descriptors
            if replay:
                await self._assert_complete_history(session_id)
            new_resources = SessionMcpResources(tuple(mcp_servers or ()))
            await old_mcp_resources.close()
            record = replace(
                base_record,
                additional_directories=directories,
                closed=False,
                updated_at=datetime.now(timezone.utc),
            )
            try:
                self.repository.save(record)
            except Exception:
                async with active.lock:
                    active.mcp_resources = SessionMcpResources(old_descriptors)
                raise
            if not await self.coordinator.validate(active, operation):
                raise request_error("session_state_changed", session_id=session_id)
            async with active.lock:
                active.record = record
                active.mcp_resources = new_resources
                new_resources = None
            return active
        except Exception:
            if new_resources is not None:
                await new_resources.close()
            raise
        finally:
            await self.coordinator.release(active, operation)

    async def fork(self, source_session_id: str, *, cwd: str, additional_directories: "list[str] | None", mcp_servers: "list[Any] | None") -> ActiveAcpSession:
        source = await self.get(source_session_id)
        normalized_cwd, directories = validate_session_paths(
            project_root=self.project_root,
            cwd=cwd,
            additional_directories=additional_directories,
        )
        _validate_mcp_descriptors(mcp_servers or ())
        fingerprints = tuple(sorted(mcp_descriptor_fingerprint(server) for server in (mcp_servers or ())))
        operation = await self.coordinator.reserve(source, SessionOperationKind.FORK)
        try:
            await self._assert_complete_history(source_session_id)
            async with source.lock:
                source_mode_id = source.record.mode_id
                source_config_values = dict(source.record.config_values)
                source_title = source.record.title
                source_fingerprints = source.record.mcp_server_fingerprints
            if fingerprints != source_fingerprints:
                raise request_error("mcp_descriptor_mismatch", session_id=source_session_id)
            target_id = uuid4().hex
            now = datetime.now(timezone.utc)
            record = AcpSessionRecord(
                schema_version=1,
                session_id=target_id,
                cwd=normalized_cwd,
                additional_directories=directories,
                mode_id=source_mode_id,
                config_values=source_config_values,
                mcp_server_fingerprints=fingerprints,
                title=source_title,
                closed=False,
                created_at=now,
                updated_at=now,
            )
            staged = self.repository.stage(record)
            try:
                await self.runtime.fork_session(source_session_id, target_id, self.principal)
                self.repository.publish(staged, target_id)
            except Exception:
                self.repository.discard(staged)
                raise
            active = ActiveAcpSession(
                record,
                asyncio.Lock(),
                None,
                SessionMcpResources(tuple(mcp_servers or ())),
                set(),
                set(),
            )
            self._active[target_id] = active
            return active
        finally:
            await self.coordinator.release(source, operation)

    async def close(self, session_id: str) -> SessionCloseResult:
        return await self.close_session_resources(session_id, reason="client")

    async def close_session_resources(
        self,
        session_id: str,
        *,
        reason: CloseReason,
    ) -> SessionCloseResult:
        active = await self.get(session_id)
        async with active.lock:
            if (
                active.record.closed
                and active.active_execution_id is None
                and active.operation is None
                and not active.terminal_handles
                and not active.pending_elicitation_ids
                and not active.pending_elicitation_tasks
                and not active.mcp_resources.descriptors
            ):
                return SessionCloseResult(True, ())
        task = await self.coordinator.request_close(
            active,
            lambda: self._run_close_once(active, reason),
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.gather(task, return_exceptions=True)
            raise

    async def _run_close_once(
        self,
        active: ActiveAcpSession,
        reason: CloseReason,
    ) -> SessionCloseResult:
        task = asyncio.current_task()
        try:
            return await self._close_once(active, reason)
        finally:
            if task is not None:
                await self.coordinator.clear_close_task(active, task)

    async def cancel_prompt(self, session_id: str) -> None:
        active = await self.get(session_id)
        await self._cancel_prompt(active)

    async def _cancel_prompt(self, active: ActiveAcpSession) -> None:
        async with active.lock:
            operation = active.operation
            if operation is not None and operation.kind is not SessionOperationKind.PROMPT:
                return
            permission_task = active.pending_permission_task
            active.pending_permission_task = None
            active.pending_permission = None
            active.operation_epoch += 1
            execution_id = active.active_execution_id
            if operation is None:
                active.active_execution_id = None
        if permission_task is not None:
            permission_task.cancel()
            await asyncio.gather(permission_task, return_exceptions=True)
        if execution_id is not None:
            await asyncio.wait_for(
                self.runtime.cancel(execution_id, principal=self.principal),
                timeout=5,
            )

    async def _close_once(
        self,
        active: ActiveAcpSession,
        reason: CloseReason,
    ) -> SessionCloseResult:
        failures: "list[SessionCloseFailure]" = []
        async with active.lock:
            operation = active.operation
            execution_id = active.active_execution_id
        if operation is not None:
            if operation.kind is SessionOperationKind.PROMPT:
                try:
                    await self._cancel_prompt(active)
                    await asyncio.wait_for(operation.done.wait(), timeout=10)
                except Exception as exc:
                    failures.append(
                        self._close_failure(
                            "execution",
                            execution_id,
                            exc,
                        )
                    )
            else:
                try:
                    await asyncio.wait_for(operation.done.wait(), timeout=20)
                except Exception as exc:
                    failures.append(self._close_failure("operation", None, exc))
        try:
            close_operation = await self.coordinator.reserve_close(active)
        except Exception as exc:
            failures.append(self._close_failure("operation", None, exc))
            return await self._finish_close_failure(active, failures, reason)
        try:
            await self._cancel_pending_tasks(active)
            if self.client_services is not None:
                try:
                    resource_failures = await self.client_services.close_session_resources(active)
                except Exception as exc:
                    resource_failures = (("terminal", None, exc),)
                seen_resources: set[tuple[str, str | None]] = set()
                for resource_type, resource_id, error in resource_failures:
                    key = (resource_type, resource_id)
                    if key in seen_resources:
                        continue
                    seen_resources.add(key)
                    failures.append(self._close_failure(resource_type, resource_id, error))
            async with active.lock:
                mcp_resources = active.mcp_resources
            try:
                await mcp_resources.close()
            except Exception as exc:
                failures.append(self._close_failure("mcp", None, exc))
            close_state_error: BaseException | None = None
            async with active.lock:
                if active.active_execution_id is not None:
                    failures.append(
                        self._close_failure(
                            "execution",
                            active.active_execution_id,
                            RuntimeError("execution is still active"),
                        )
                    )
                if active.pending_permission_task is not None:
                    failures.append(self._close_failure("permission", None, RuntimeError("permission task remains")))
                if active.pending_elicitation_tasks:
                    failures.append(self._close_failure("elicitation", None, RuntimeError("elicitation task remains")))
                if active.terminal_create_tasks:
                    failures.append(self._close_failure("terminal", None, RuntimeError("terminal create task remains")))
                if active.terminal_handles:
                    failures.append(self._close_failure("terminal", None, RuntimeError("terminal resources remain")))
                if active.pending_elicitation_ids:
                    failures.append(self._close_failure("elicitation", None, RuntimeError("elicitation resources remain")))
                if failures:
                    close_state_error = RuntimeError("session resources remain")
                else:
                    record = replace(active.record, closed=True, updated_at=datetime.now(timezone.utc))
            if close_state_error is not None:
                return await self._finish_close_failure(active, failures, reason, close_operation)
            self.repository.save(record)
            if not await self.coordinator.validate(active, close_operation):
                failures.append(self._close_failure("persistence", None, RuntimeError("session state changed during close")))
                return await self._finish_close_failure(active, failures, reason, close_operation)
            async with active.lock:
                active.record = record
                active.cleanup_required = False
                active.closing_requested = False
            await self.coordinator.release(active, close_operation)
            if logger.isEnabledFor(logging.DEBUG):
                assert_session_invariants(active)
            async with active.lock:
                self._active.pop(active.record.session_id, None)
            logger.info(
                "event=acp.session.closed session_id=%s close_reason=%s pending_task_count=0 terminal_count=0 elicitation_count=0 mcp_state=closed cleanup_required=false",
                active.record.session_id,
                reason,
            )
            return SessionCloseResult(True, ())
        except Exception as exc:
            failures.append(self._close_failure("persistence", None, exc))
            return await self._finish_close_failure(active, failures, reason, close_operation)

    async def _finish_close_failure(
        self,
        active: ActiveAcpSession,
        failures: "list[SessionCloseFailure]",
        reason: CloseReason,
        close_operation: "SessionOperationToken | None" = None,
    ) -> SessionCloseResult:
        async with active.lock:
            active.cleanup_required = True
            active.closing_requested = False
        if close_operation is not None:
            await self.coordinator.release(active, close_operation)
        logger.warning(
            "event=acp.session.cleanup_failed session_id=%s close_reason=%s pending_task_count=%s terminal_count=%s elicitation_count=%s cleanup_required=true error_ids=%s",
            active.record.session_id,
            reason,
            len(active.pending_elicitation_tasks),
            len(active.terminal_handles),
            len(active.pending_elicitation_ids),
            ",".join(item.error_id for item in failures),
        )
        return SessionCloseResult(False, tuple(failures))

    async def _cancel_pending_tasks(self, active: ActiveAcpSession) -> None:
        async with active.lock:
            permission_task = active.pending_permission_task
            active.pending_permission_task = None
            active.pending_permission = None
            tasks = tuple(active.pending_elicitation_tasks.values())
            create_tasks = tuple(active.terminal_create_tasks)
            release_tasks = tuple(active.terminal_release_tasks.values())
        owned = tuple(
            task
            for task in (permission_task, *tasks, *create_tasks, *release_tasks)
            if task is not None
        )
        for task in owned:
            if not task.done():
                task.cancel()
        if owned:
            await asyncio.gather(*owned, return_exceptions=True)

    @staticmethod
    def _close_failure(resource_type: Any, resource_id: "str | None", error: BaseException) -> SessionCloseFailure:
        return SessionCloseFailure(resource_type, resource_id, uuid4().hex)

    async def set_mode(self, session_id: str, mode_id: str) -> ActiveAcpSession:
        active = await self.get(session_id)
        operation = await self.coordinator.reserve(active, SessionOperationKind.SET_MODE)
        try:
            if mode_id not in self.mode_ids:
                raise request_error("unknown_mode", session_id=session_id)
            async with active.lock:
                record = replace(active.record, mode_id=mode_id, updated_at=datetime.now(timezone.utc))
            self.repository.save(record)
            if not await self.coordinator.validate(active, operation):
                raise request_error("session_state_changed", session_id=session_id)
            async with active.lock:
                active.record = record
                return active
        finally:
            await self.coordinator.release(active, operation)

    async def set_config(self, session_id: str, config_id: str, value: Any) -> ActiveAcpSession:
        active = await self.get(session_id)
        operation = await self.coordinator.reserve(active, SessionOperationKind.SET_CONFIG)
        try:
            if config_id not in self.config_defaults:
                raise request_error("unknown_config_option", session_id=session_id)
            if not isinstance(value, type(self.config_defaults[config_id])):
                raise request_error("invalid_config_value", session_id=session_id)
            async with active.lock:
                values = dict(active.record.config_values)
                values[config_id] = value
                record = replace(active.record, config_values=values, updated_at=datetime.now(timezone.utc))
            self.repository.save(record)
            if not await self.coordinator.validate(active, operation):
                raise request_error("session_state_changed", session_id=session_id)
            async with active.lock:
                active.record = record
                return active
        finally:
            await self.coordinator.release(active, operation)

    async def list(self, *, cwd: "str | None" = None, cursor: "str | None" = None) -> "tuple[tuple[AcpSessionRecord, ...], str | None]":
        records = list(self.repository.list())
        if cwd is not None:
            normalized, _ = validate_session_paths(project_root=self.project_root, cwd=cwd, additional_directories=[])
            records = [record for record in records if record.cwd == normalized]
        records.sort(key=lambda record: (-record.updated_at.timestamp(), record.session_id))
        offset = self._decode_cursor(records, cursor)
        page = records[offset: offset + 50]
        next_cursor = self._encode_cursor(page[-1]) if offset + 50 < len(records) else None
        return tuple(page), next_cursor

    async def _assert_complete_history(self, session_id: str) -> None:
        views = await self.runtime.get_session_messages(session_id=session_id, principal=self.principal)
        for view in views:
            capture_state = getattr(view, "capture_state", None)
            if getattr(view, "status", None) is not RunStatus.COMPLETED or getattr(capture_state, "value", capture_state) != "complete":
                raise request_error("incomplete_history", session_id=session_id)

    @staticmethod
    def _encode_cursor(record: AcpSessionRecord) -> str:
        raw = json.dumps(
            {
                "v": 1,
                "updated_at": record.updated_at.isoformat().replace("+00:00", "Z"),
                "session_id": record.session_id,
            },
            separators=(",", ":"),
        ).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    @staticmethod
    def _decode_cursor(records: "list[AcpSessionRecord]", cursor: "str | None") -> int:
        if cursor is None:
            return 0
        try:
            raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
            value = json.loads(raw)
            if value.get("v") != 1 or not isinstance(value.get("updated_at"), str) or not isinstance(value.get("session_id"), str):
                raise ValueError
            updated_at = datetime.fromisoformat(value["updated_at"].replace("Z", "+00:00"))
            for index, record in enumerate(records):
                if record.updated_at == updated_at and record.session_id == value["session_id"]:
                    return index + 1
            raise ValueError("cursor record is no longer available")
        except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise request_error("invalid_cursor") from exc


__all__ = [
    "ActiveAcpSession",
    "AcpSessionService",
    "CloseReason",
    "PendingPermissionToken",
    "SessionCloseFailure",
    "SessionCloseResult",
    "SessionMcpResources",
    "McpResourceState",
    "mcp_spec",
    "validate_session_paths",
]


def mcp_spec(descriptor: Any) -> Any:
    """Convert one official ACP descriptor to the existing MCP domain type."""
    from ..agent.mcp.spec import MCPServerSpec

    kind = _mcp_transport(descriptor)
    name = getattr(descriptor, "name", "")
    if not name:
        raise request_error("invalid_mcp_descriptor")
    if kind == "stdio":
        env = {
            item.name: item.value
            for item in getattr(descriptor, "env", ())
        }
        command = (descriptor.command, *tuple(getattr(descriptor, "args", ())))
        return MCPServerSpec(
            id=name,
            name=name,
            transport="stdio",
            command=command,
            env=env,
        )
    if kind in {"http", "sse"}:
        headers = {
            item.name: item.value
            for item in getattr(descriptor, "headers", ())
        }
        return MCPServerSpec(
            id=name,
            name=name,
            transport=kind,
            url=descriptor.url,
            headers=headers,
        )
    raise request_error(
        "unsupported_mcp_transport",
        details={"transport": kind},
    )


def _mcp_transport(descriptor: Any) -> str:
    kind = getattr(descriptor, "type", None)
    if kind is not None:
        return str(kind)
    return {
        "McpServerStdio": "stdio",
        "McpServerHttp": "http",
        "McpServerSse": "sse",
        "McpServerAcp": "acp",
    }.get(type(descriptor).__name__, "unknown")


def _validate_mcp_descriptors(descriptors: Any) -> None:
    try:
        for descriptor in descriptors:
            mcp_spec(descriptor)
    except Exception as exc:
        if getattr(exc, "code", None) == -32602:
            raise
        raise request_error("invalid_mcp_descriptor") from exc
