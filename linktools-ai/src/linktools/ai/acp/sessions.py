#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ACP session ownership and lifecycle state machine."""

import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Mapping
from uuid import uuid4

from ..execution.domain import RunStatus
from ..governance.identity import PrincipalContext
from ..runtime.facade import Runtime
from .errors import request_error
from .persistence import AcpSessionRecord, AcpSessionRepository, mcp_descriptor_fingerprint

logger = logging.getLogger("linktools.ai.acp.sessions")


@dataclass(slots=True)
class SessionMcpResources:
    descriptors: "tuple[Any, ...]" = ()
    close_callback: "Callable[[], Awaitable[None]] | None" = None
    _toolsets: "tuple[Any, ...] | None" = None

    async def toolsets(self) -> "tuple[Any, ...]":
        if self._toolsets is not None:
            return self._toolsets
        if not self.descriptors:
            self._toolsets = ()
            return self._toolsets
        from ..agent.mcp.connection import MCPConnectionPool

        pool = MCPConnectionPool()
        try:
            values = []
            for descriptor in self.descriptors:
                values.append(await pool.get_toolset(mcp_spec(descriptor)))
        except Exception:
            await pool.close()
            raise request_error("mcp_connection_failed")
        self.close_callback = pool.close
        self._toolsets = tuple(handle.toolset for handle in values)
        return self._toolsets

    async def close(self) -> None:
        if self.close_callback is not None:
            await self.close_callback()
        self.descriptors = ()
        self.close_callback = None
        self._toolsets = None


@dataclass(slots=True)
class ActiveAcpSession:
    record: AcpSessionRecord
    lock: asyncio.Lock
    active_execution_id: "str | None"
    mcp_resources: SessionMcpResources
    terminal_handles: "set[str]"
    pending_elicitation_ids: "set[str]"
    operation_epoch: int = 0
    closing: bool = False
    pending_permission: "PendingPermissionToken | None" = None


@dataclass(frozen=True, slots=True)
class PendingPermissionToken:
    session_id: str
    execution_id: str
    approval_id: str
    tool_call_id: str
    epoch: int


@dataclass(frozen=True, slots=True)
class SessionCloseFailure:
    resource_type: "Literal['execution', 'permission', 'elicitation', 'terminal', 'mcp', 'persistence']"
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
        async with active.lock:
            if active.record.cwd != normalized_cwd:
                raise request_error("cwd_mismatch", session_id=session_id)
            if active.closing or active.active_execution_id is not None:
                raise request_error("session_busy", session_id=session_id)
            fingerprints = tuple(sorted(mcp_descriptor_fingerprint(server) for server in (mcp_servers or ())))
            if fingerprints != active.record.mcp_server_fingerprints:
                raise request_error("mcp_descriptor_mismatch", session_id=session_id)
            epoch = active.operation_epoch
            base_record = active.record
            old_mcp_resources = active.mcp_resources
        if replay:
            await self._assert_complete_history(session_id)
        await old_mcp_resources.close()
        record = replace(
            base_record,
            additional_directories=directories,
            closed=False,
            updated_at=datetime.now(timezone.utc),
        )
        self.repository.save(record)
        async with active.lock:
            if active.operation_epoch != epoch or active.closing:
                raise request_error("session_state_changed", session_id=session_id)
            active.record = record
            active.mcp_resources = SessionMcpResources(tuple(mcp_servers or ()))
            return active

    async def fork(self, source_session_id: str, *, cwd: str, additional_directories: "list[str] | None", mcp_servers: "list[Any] | None") -> ActiveAcpSession:
        source = await self.get(source_session_id)
        async with source.lock:
            if source.active_execution_id is not None or source.closing:
                raise request_error("session_busy", session_id=source_session_id)
            source_mode_id = source.record.mode_id
            source_config_values = dict(source.record.config_values)
            source_title = source.record.title
            source_fingerprints = source.record.mcp_server_fingerprints
        await self._assert_complete_history(source_session_id)
        normalized_cwd, directories = validate_session_paths(
            project_root=self.project_root,
            cwd=cwd,
            additional_directories=additional_directories,
        )
        _validate_mcp_descriptors(mcp_servers or ())
        fingerprints = tuple(sorted(mcp_descriptor_fingerprint(server) for server in (mcp_servers or ())))
        if fingerprints != source_fingerprints:
            raise request_error("mcp_descriptor_mismatch", session_id=source_session_id)
        target_id = uuid4().hex
        await self.runtime.fork_session(source_session_id, target_id, self.principal)
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
        self.repository.save(record)
        active = ActiveAcpSession(record, asyncio.Lock(), None, SessionMcpResources(tuple(mcp_servers or ())), set(), set())
        self._active[target_id] = active
        return active

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
                and not active.terminal_handles
                and not active.pending_elicitation_ids
                and not active.mcp_resources.descriptors
            ):
                return SessionCloseResult(True, ())
            active.closing = True
            active.operation_epoch += 1
            active.pending_permission = None
            execution_id = active.active_execution_id
            mcp_resources = active.mcp_resources
        failures: "list[SessionCloseFailure]" = []
        if execution_id is not None:
            try:
                await self.runtime.cancel(execution_id, principal=self.principal)
                await self._wait_execution_terminal(execution_id)
            except Exception as exc:
                failures.append(self._close_failure("execution", execution_id, exc))
            else:
                async with active.lock:
                    if active.active_execution_id == execution_id:
                        active.active_execution_id = None
        if self.client_services is not None:
            try:
                resource_failures = await self.client_services.close_session_resources(active)
            except Exception as exc:
                resource_failures = (("terminal", None, exc),)
            for resource_type, resource_id, error in resource_failures:
                failures.append(self._close_failure(resource_type, resource_id, error))
        try:
            await mcp_resources.close()
        except Exception as exc:
            failures.append(self._close_failure("mcp", None, exc))
        async with active.lock:
            if active.active_execution_id is not None:
                failures.append(self._close_failure("execution", active.active_execution_id, RuntimeError("execution is still active")))
            if active.terminal_handles:
                failures.append(self._close_failure("terminal", None, RuntimeError("terminal resources remain")))
            if active.pending_elicitation_ids:
                failures.append(self._close_failure("elicitation", None, RuntimeError("elicitation resources remain")))
            if failures:
                logger.warning(
                    "event=acp.session.cleanup_failed session_id=%s close_reason=%s remaining_resource_count=%s",
                    session_id,
                    reason,
                    len(active.terminal_handles) + len(active.pending_elicitation_ids),
                )
                return SessionCloseResult(False, tuple(failures))
            record = replace(active.record, closed=True, updated_at=datetime.now(timezone.utc))
            epoch = active.operation_epoch
        try:
            self.repository.save(record)
        except Exception as exc:
            failure = self._close_failure("persistence", None, exc)
            logger.warning(
                "event=acp.session.close_persistence_failed session_id=%s error_id=%s",
                session_id,
                failure.error_id,
            )
            return SessionCloseResult(False, (failure,))
        async with active.lock:
            if active.operation_epoch != epoch or active.active_execution_id is not None:
                failure = self._close_failure("persistence", None, RuntimeError("session state changed during close"))
                return SessionCloseResult(False, (failure,))
            active.record = record
            active.closing = False
            self._active.pop(session_id, None)
            logger.info(
                "event=acp.session.closed session_id=%s close_reason=%s remaining_resource_count=0",
                session_id,
                reason,
            )
            return SessionCloseResult(True, ())

    async def _wait_execution_terminal(self, execution_id: str) -> None:
        deadline = asyncio.get_running_loop().time() + 10
        while True:
            record = await self.runtime.get_execution_record(execution_id, principal=self.principal)
            if record is not None and record.status in {
                RunStatus.COMPLETED,
                RunStatus.CANCELLED,
                RunStatus.FAILED,
            }:
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("execution did not reach a terminal state")
            await asyncio.sleep(0.01)

    @staticmethod
    def _close_failure(resource_type: Any, resource_id: "str | None", error: BaseException) -> SessionCloseFailure:
        return SessionCloseFailure(resource_type, resource_id, uuid4().hex)

    async def set_mode(self, session_id: str, mode_id: str) -> ActiveAcpSession:
        active = await self.get(session_id)
        async with active.lock:
            if active.active_execution_id is not None or active.closing:
                raise request_error("session_busy", session_id=session_id)
            if mode_id not in self.mode_ids:
                raise request_error("unknown_mode", session_id=session_id)
            epoch = active.operation_epoch
            record = replace(active.record, mode_id=mode_id, updated_at=datetime.now(timezone.utc))
        self.repository.save(record)
        async with active.lock:
            if active.operation_epoch != epoch or active.closing:
                raise request_error("session_state_changed", session_id=session_id)
            active.record = record
            return active

    async def set_config(self, session_id: str, config_id: str, value: Any) -> ActiveAcpSession:
        active = await self.get(session_id)
        async with active.lock:
            if active.active_execution_id is not None or active.closing:
                raise request_error("session_busy", session_id=session_id)
            if config_id not in self.config_defaults:
                raise request_error("unknown_config_option", session_id=session_id)
            if not isinstance(value, type(self.config_defaults[config_id])):
                raise request_error("invalid_config_value", session_id=session_id)
            values = dict(active.record.config_values)
            values[config_id] = value
            epoch = active.operation_epoch
            record = replace(active.record, config_values=values, updated_at=datetime.now(timezone.utc))
        self.repository.save(record)
        async with active.lock:
            if active.operation_epoch != epoch or active.closing:
                raise request_error("session_state_changed", session_id=session_id)
            active.record = record
            return active

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
