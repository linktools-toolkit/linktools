#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Protocol-neutral session lifecycle and operation ownership."""

import asyncio
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol
from uuid import uuid4

from linktools.core import environ

from ..errors import (
    SessionBusyError,
    SessionCleanupRequiredError,
    StorageConflictError,
    UnknownSessionError,
)
from ..execution.session import (
    CreateSession,
    ForkSession,
    SessionRecord,
    SessionSettings,
    SessionState,
    SessionWorkspace,
    UpdateSession,
)
from ..execution.store import ExecutionStore

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent.mcp.spec import MCPServerSpec
    from ..governance.identity import PrincipalContext


logger = environ.get_logger("ai.runtime.session")


class SessionOperationKind(StrEnum):
    PROMPT = "prompt"
    LOAD = "load"
    RESUME = "resume"
    FORK = "fork"
    UPDATE = "update"
    CLOSE = "close"


@dataclass(frozen=True, slots=True)
class ResourceFailure:
    owner: str
    resource_id: "str | None"
    error_id: str


@dataclass(frozen=True, slots=True)
class SessionCloseResult:
    closed: bool
    failures: "tuple[ResourceFailure, ...]" = ()


class SessionResourceOwner(Protocol):
    async def close(self, session_id: str) -> "tuple[ResourceFailure, ...]": ...

    def is_empty(self, session_id: str) -> bool: ...


@dataclass(slots=True)
class ActiveRuntimeSession:
    record: SessionRecord
    tool_sources: "tuple[MCPServerSpec, ...]" = ()
    operation: "SessionOperationLease | None" = None
    active_execution_id: "str | None" = None
    closing_requested: bool = False
    cleanup_required: bool = False
    owners: "dict[str, SessionResourceOwner]" = None  # type: ignore[assignment]
    lock: asyncio.Lock = None  # type: ignore[assignment]
    close_task: "asyncio.Task[SessionCloseResult] | None" = None
    mcp_resources: Any = None

    def __post_init__(self) -> None:
        if self.owners is None:
            self.owners = {}
        if self.lock is None:
            self.lock = asyncio.Lock()


class SessionOperationLease:
    def __init__(
        self,
        coordinator: "SessionOperationCoordinator",
        active: ActiveRuntimeSession,
        kind: SessionOperationKind,
        operation_id: str,
        execution_id: "str | None",
    ) -> None:
        self._coordinator = coordinator
        self.active = active
        self.kind = kind
        self.operation_id = operation_id
        self.execution_id = execution_id
        self._released = False
        self.done = asyncio.Event()

    async def __aenter__(self) -> "SessionOperationLease":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.release()

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._coordinator.release(self)
        self.done.set()


class SessionOperationCoordinator:
    async def reserve(
        self,
        active: ActiveRuntimeSession,
        kind: SessionOperationKind,
        *,
        execution_id: "str | None" = None,
    ) -> SessionOperationLease:
        async with active.lock:
            if active.cleanup_required:
                raise SessionCleanupRequiredError(active.record.id)
            if active.record.state is SessionState.CLOSED and kind is not SessionOperationKind.CLOSE:
                raise SessionBusyError(active.record.id)
            if active.closing_requested and kind is not SessionOperationKind.CLOSE:
                raise SessionBusyError(active.record.id)
            if active.operation is not None:
                raise SessionBusyError(active.record.id)
            lease = SessionOperationLease(
                self,
                active,
                kind,
                uuid4().hex,
                execution_id,
            )
            active.operation = lease
            if execution_id is not None:
                active.active_execution_id = execution_id
            logger.info(
                "event=runtime.session.operation_reserved session_id=%s operation_kind=%s operation_id=%s",
                active.record.id,
                kind.value,
                lease.operation_id,
            )
            return lease

    async def release(self, lease: SessionOperationLease) -> None:
        active = lease.active
        async with active.lock:
            if active.operation is lease:
                active.operation = None
                if lease.kind is SessionOperationKind.PROMPT:
                    active.active_execution_id = None
        logger.info(
            "event=runtime.session.operation_released session_id=%s operation_kind=%s operation_id=%s",
            active.record.id,
            lease.kind.value,
            lease.operation_id,
        )


class SessionLoadTransaction:
    def __init__(
        self,
        service: "RuntimeSessionService",
        active: ActiveRuntimeSession,
        lease: SessionOperationLease,
        workspace: SessionWorkspace,
        settings: SessionSettings,
        history: tuple[Any, ...],
    ) -> None:
        self._service = service
        self._active = active
        self._lease = lease
        self._workspace = workspace
        self._settings = settings
        self.session = active.record
        self.history = history
        self._committed = False
        self._rolled_back = False

    async def __aenter__(self) -> "SessionLoadTransaction":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if not self._committed:
            await self.rollback()
        else:
            await self._lease.release()

    async def commit(self) -> SessionRecord:
        if self._committed:
            return self._active.record
        if self._rolled_back:
            raise RuntimeError("session load transaction is rolled back")
        record = await self._service._update_active(
            self._active,
            workspace=self._workspace,
            settings=self._settings,
        )
        self._committed = True
        self.session = record
        return record

    async def rollback(self) -> None:
        if not self._committed:
            self._rolled_back = True
            await self._lease.release()

    async def release(self) -> None:
        await self._lease.release()


class RuntimeSessionService:
    def __init__(
        self, store: ExecutionStore, authorization: Any = None, mcp_resource_factory: Any = None
    ) -> None:
        self._store = store
        self._authorization = authorization
        self._active: dict[str, ActiveRuntimeSession] = {}
        self._coordinator = SessionOperationCoordinator()
        self._interaction_cancel: Any = None
        self._interaction_owner: "SessionResourceOwner | None" = None
        self._mcp_resource_factory = mcp_resource_factory

    def set_interaction_canceller(self, callback: Any) -> None:
        self._interaction_cancel = callback

    def set_interaction_owner(self, owner: SessionResourceOwner) -> None:
        self._interaction_owner = owner

    async def create(
        self,
        *,
        session_id: str,
        workspace: SessionWorkspace,
        settings: SessionSettings,
        principal: "PrincipalContext",
        tool_sources: "tuple[MCPServerSpec, ...]" = (),
    ) -> ActiveRuntimeSession:
        command = CreateSession(
            session_id=session_id,
            user_id=principal.user_id,
            tenant_id=principal.tenant_id,
            workspace=workspace,
            settings=settings,
        )
        record = await self._store.create_session(command)
        self._authorize(principal, record)
        active = self._active.setdefault(
            record.id, ActiveRuntimeSession(record, tool_sources)
        )
        active.tool_sources = tool_sources
        self._ensure_mcp_resources(active)
        return active

    async def get(
        self, session_id: str, *, principal: "PrincipalContext"
    ) -> ActiveRuntimeSession:
        active = self._active.get(session_id)
        if active is not None:
            self._authorize(principal, active.record)
            return active
        record = await self._store.get_session(session_id)
        if record is None:
            raise UnknownSessionError(session_id)
        self._authorize(principal, record)
        active = ActiveRuntimeSession(record)
        self._active[session_id] = active
        return active

    async def list(
        self, *, principal: "PrincipalContext"
    ) -> "tuple[ActiveRuntimeSession, ...]":
        result = []
        for record in await self._store.list_all_sessions():
            try:
                self._authorize(principal, record)
            except Exception:
                continue
            result.append(self._active.setdefault(record.id, ActiveRuntimeSession(record)))
        return tuple(result)

    async def prepare_load(
        self,
        *,
        session_id: str,
        workspace: SessionWorkspace,
        settings: SessionSettings,
        principal: "PrincipalContext",
        tool_sources: "tuple[MCPServerSpec, ...]" = (),
    ) -> SessionLoadTransaction:
        active = await self.get(session_id, principal=principal)
        lease = await self._coordinator.reserve(active, SessionOperationKind.LOAD)
        try:
            history = await self._store.get_session_messages(session_id)
            active.tool_sources = tool_sources
            self._ensure_mcp_resources(active)
            return SessionLoadTransaction(
                self, active, lease, workspace, settings, history
            )
        except BaseException:
            await lease.release()
            raise

    async def resume(
        self,
        session_id: str,
        *,
        workspace: SessionWorkspace,
        settings: SessionSettings,
        principal: "PrincipalContext",
        tool_sources: "tuple[MCPServerSpec, ...]" = (),
    ) -> ActiveRuntimeSession:
        active = await self.get(session_id, principal=principal)
        async with await self._coordinator.reserve(active, SessionOperationKind.RESUME):
            active.tool_sources = tool_sources
            self._ensure_mcp_resources(active)
            await self._update_active(active, workspace=workspace, settings=settings)
        return active

    async def fork(
        self,
        source_session_id: str,
        target_session_id: str,
        *,
        workspace: SessionWorkspace,
        settings: SessionSettings,
        principal: "PrincipalContext",
        tool_sources: "tuple[MCPServerSpec, ...]" = (),
    ) -> ActiveRuntimeSession:
        source = await self.get(source_session_id, principal=principal)
        async with await self._coordinator.reserve(source, SessionOperationKind.FORK):
            if any(
                run.session_id == source_session_id
                and run.status.value in {"pending", "running", "paused", "cancelling"}
                for run in await self._store.list_all_runs()
            ):
                raise SessionBusyError(source_session_id)
            record = await self._store.fork_session(
                ForkSession(
                    source_session_id=source_session_id,
                    target_session_id=target_session_id,
                    user_id=principal.user_id,
                    tenant_id=principal.tenant_id,
                    workspace=workspace,
                    settings=settings,
                )
            )
        active = self._active.setdefault(record.id, ActiveRuntimeSession(record, tool_sources))
        self._ensure_mcp_resources(active)
        return active

    async def update(
        self,
        session_id: str,
        *,
        workspace: "SessionWorkspace | None" = None,
        settings: "SessionSettings | None" = None,
        principal: "PrincipalContext",
    ) -> SessionRecord:
        active = await self.get(session_id, principal=principal)
        async with await self._coordinator.reserve(active, SessionOperationKind.UPDATE):
            return await self._update_active(active, workspace=workspace, settings=settings)

    async def reserve(
        self,
        session_id: str,
        kind: SessionOperationKind,
        *,
        principal: "PrincipalContext",
        execution_id: "str | None" = None,
    ) -> SessionOperationLease:
        active = await self.get(session_id, principal=principal)
        return await self._coordinator.reserve(active, kind, execution_id=execution_id)

    async def register_owner(
        self, session_id: str, name: str, owner: SessionResourceOwner
    ) -> None:
        active = self._active.get(session_id)
        if active is None:
            raise UnknownSessionError(session_id)
        async with active.lock:
            active.owners[name] = owner

    async def toolsets(
        self, session_id: str, *, principal: "PrincipalContext"
    ) -> "tuple[Any, ...]":
        active = await self.get(session_id, principal=principal)
        self._ensure_mcp_resources(active)
        if active.mcp_resources is None:
            return ()
        return await active.mcp_resources.toolsets()

    async def close(
        self, session_id: str, *, principal: "PrincipalContext", reason: str = "client"
    ) -> SessionCloseResult:
        active = await self.get(session_id, principal=principal)
        async with active.lock:
            if active.close_task is None:
                active.close_task = asyncio.create_task(self._close(active, reason))
            task = active.close_task
        return await asyncio.shield(task)

    async def shutdown(self) -> "tuple[SessionCloseResult, ...]":
        return tuple(
            await asyncio.gather(
                *(self._close(active, "shutdown") for active in tuple(self._active.values())),
                return_exceptions=False,
            )
        )

    async def _close(
        self, active: ActiveRuntimeSession, reason: str
    ) -> SessionCloseResult:
        async with active.lock:
            active.closing_requested = True
            execution_id = active.active_execution_id
            operation = active.operation
            owners = list(active.owners.items())
        ordered_owners: list[tuple[str, SessionResourceOwner]] = []
        if self._interaction_owner is not None:
            ordered_owners.append(("interaction", self._interaction_owner))
        ordered_owners.extend((name, owner) for name, owner in owners if name != "mcp")
        ordered_owners.extend((name, owner) for name, owner in owners if name == "mcp")
        owners = ordered_owners
        if execution_id is not None and self._interaction_cancel is not None:
            await self._interaction_cancel(active.record.id)
        if operation is not None and operation.kind is not SessionOperationKind.CLOSE:
            await operation.done.wait()
        failures: "list[ResourceFailure]" = []
        for name, owner in owners:
            try:
                failures.extend(await owner.close(active.record.id))
            except Exception as exc:
                failures.append(ResourceFailure(name, None, type(exc).__name__))
        failures.extend(
            ResourceFailure(name, None, "owner_not_empty")
            for name, owner in owners
            if not owner.is_empty(active.record.id)
        )
        if failures:
            async with active.lock:
                active.cleanup_required = True
                active.closing_requested = False
                active.close_task = None
            logger.error(
                "event=runtime.session.cleanup_failed session_id=%s resource_count=%s",
                active.record.id,
                len(failures),
            )
            return SessionCloseResult(False, tuple(failures))
        record = await self._update_active(active, state=SessionState.CLOSED)
        self._active.pop(active.record.id, None)
        logger.info("event=runtime.session.closed session_id=%s", record.id)
        return SessionCloseResult(True)

    async def _update_active(
        self,
        active: ActiveRuntimeSession,
        *,
        workspace: "SessionWorkspace | None" = None,
        settings: "SessionSettings | None" = None,
        state: "SessionState | None" = None,
    ) -> SessionRecord:
        command = UpdateSession(
            session_id=active.record.id,
            expected_revision=active.record.revision,
            workspace=workspace,
            settings=settings,
            state=state,
        )
        task = asyncio.create_task(self._store.update_session(command))
        try:
            record = await asyncio.shield(task)
        except asyncio.CancelledError:
            record = await task
            active.record = record
            raise
        active.record = record
        return record

    def _authorize(self, principal: "PrincipalContext", record: SessionRecord) -> None:
        if self._authorization is not None:
            self._authorization.assert_session_access(principal=principal, session=record)

    def _ensure_mcp_resources(self, active: ActiveRuntimeSession) -> None:
        if self._mcp_resource_factory is None or active.mcp_resources is not None:
            return
        active.mcp_resources = self._mcp_resource_factory(active.tool_sources)
        active.owners.setdefault("mcp", active.mcp_resources)


__all__ = [
    "ActiveRuntimeSession",
    "ResourceFailure",
    "RuntimeSessionService",
    "SessionCloseResult",
    "SessionLoadTransaction",
    "SessionOperationCoordinator",
    "SessionOperationKind",
    "SessionOperationLease",
    "SessionResourceOwner",
]
