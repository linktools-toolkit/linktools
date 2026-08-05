#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Protocol-neutral session lifecycle and operation ownership."""

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable, Protocol
from uuid import uuid4

from linktools.core import environ

from ..errors import (
    McpCleanupRequiredError,
    McpReplacementError,
    SessionBusyError,
    SessionClosedError,
    SessionCleanupRequiredError,
    SessionInvariantError,
    SessionOperationError,
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

if TYPE_CHECKING:
    from ..agent.mcp.spec import MCPServerSpec
    from ..governance.identity import PrincipalContext


logger = environ.get_logger("ai.runtime.session")

ActiveApply = Callable[[SessionRecord], None]


class SessionOperationKind(StrEnum):
    CREATE = "create"
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


@dataclass(slots=True)
class McpReplacement:
    old_specs: "tuple[MCPServerSpec, ...]"
    old_resources: Any
    new_specs: "tuple[MCPServerSpec, ...]"
    candidate: Any
    committed: bool = False


class SessionCommitter:
    """Serialize persisted session changes with the active lease."""

    def __init__(self, store: ExecutionStore) -> None:
        self._store = store

    async def commit_update(
        self,
        active: "ActiveRuntimeSession",
        lease: "SessionOperationLease",
        command: "UpdateSession | None" = None,
        *,
        workspace: "SessionWorkspace | None" = None,
        settings: "SessionSettings | None" = None,
        state: "SessionState | None" = None,
        apply: "ActiveApply | None" = None,
    ) -> SessionRecord:
        async with active.lock:
            if active.operation is not lease:
                raise SessionInvariantError("session operation lease changed")
            if command is None:
                command = UpdateSession(
                    session_id=active.record.id,
                    expected_revision=active.record.revision,
                    workspace=workspace,
                    settings=settings,
                    state=state,
                )
            elif (
                command.session_id != active.record.id
                or command.expected_revision != active.record.revision
            ):
                raise SessionInvariantError("session commit command does not match active record")
        session_id = command.session_id
        logger.info(
            "event=runtime.session.commit_started session_id=%s operation_id=%s",
            session_id,
            lease.operation_id,
        )

        async def commit_once() -> SessionRecord:
            record = await self._store.update_session(command)
            async with active.lock:
                if active.operation is not lease:
                    active.cleanup_required = True
                    raise SessionInvariantError(
                        "session operation lease changed after store commit"
                    )
                try:
                    active.record = record
                    if apply is not None:
                        apply(record)
                except asyncio.CancelledError:
                    active.cleanup_required = True
                    raise
                except Exception as exc:
                    active.cleanup_required = True
                    raise SessionInvariantError(
                        "active session apply failed"
                    ) from exc
                except BaseException:
                    active.cleanup_required = True
                    raise
            logger.info(
                "event=runtime.session.commit_completed session_id=%s operation_id=%s session_revision=%s session_state=%s",
                record.id,
                lease.operation_id,
                record.revision,
                record.state.value,
            )
            return record

        # task-owner: runtime.session.commit
        task = asyncio.create_task(commit_once())
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            logger.info(
                "event=runtime.session.commit_cancel_joined session_id=%s operation_id=%s",
                session_id,
                lease.operation_id,
            )
            await task
            raise


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
        self._release_task: "asyncio.Task[None] | None" = None
        self._released = False
        self.done = asyncio.Event()

    async def __aenter__(self) -> "SessionOperationLease":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.release()

    async def release(self) -> None:
        if self._released:
            return
        if self._release_task is None:
            # task-owner: runtime.session.lease_release
            self._release_task = asyncio.create_task(self._coordinator.release(self))
        try:
            await asyncio.shield(self._release_task)
        except asyncio.CancelledError:
            await self._release_task
            self._released = True
            logger.info(
                "event=runtime.session.lease_release_joined session_id=%s operation_id=%s",
                self.active.record.id,
                self.operation_id,
            )
            raise
        self._released = True


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
            if active.closing_requested and kind is not SessionOperationKind.CLOSE:
                raise SessionBusyError(active.record.id)
            if active.record.state is SessionState.CLOSED and kind not in {
                SessionOperationKind.LOAD,
                SessionOperationKind.RESUME,
                SessionOperationKind.CLOSE,
            }:
                raise SessionClosedError(active.record.id)
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
        try:
            async with active.lock:
                if active.operation is lease:
                    active.operation = None
                if (
                    lease.execution_id is not None
                    and active.active_execution_id == lease.execution_id
                ):
                    active.active_execution_id = None
        finally:
            lease.done.set()
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
        replacement: McpReplacement,
    ) -> None:
        self._service = service
        self._active = active
        self._lease = lease
        self.lease = lease
        self._workspace = workspace
        self._settings = settings
        self.session = active.record
        self.history = history
        self.replacement = replacement
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
        record = await self._service._commit_replacement(
            self._active,
            self._lease,
            self.replacement,
            workspace=self._workspace,
            settings=self._settings,
        )
        self._committed = True
        self.session = record
        return record

    async def rollback(self) -> None:
        if self._committed or self._rolled_back:
            return
        self._rolled_back = True
        failures = await self._service._close_candidate(
            self._active.record.id, self.replacement.candidate
        )
        if failures:
            async with self._active.lock:
                self._active.cleanup_required = True
                self._service._set_candidate_owner(
                    self._active, self.replacement.candidate
                )
            logger.error(
                "event=runtime.session.mcp_replacement_failed session_id=%s mcp_close_failure_count=%s",
                self._active.record.id,
                len(failures),
            )
        await self._lease.release()

    async def release(self) -> None:
        await self._lease.release()


class RuntimeSessionService:
    def __init__(
        self, store: ExecutionStore, authorization: Any = None, mcp_resource_factory: Any = None
    ) -> None:
        self._store = store
        self._authorization = authorization
        self._active_lock = asyncio.Lock()
        self._active: dict[str, ActiveRuntimeSession] = {}
        self._active_identity_conflict_count = 0
        self._coordinator = SessionOperationCoordinator()
        self._committer = SessionCommitter(store)
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
        owner: "SessionResourceOwner | None" = None,
        owner_name: str = "",
    ) -> ActiveRuntimeSession:
        self._validate_owner(owner, owner_name)
        command = CreateSession(
            session_id=session_id,
            user_id=principal.user_id,
            tenant_id=principal.tenant_id,
            workspace=workspace,
            settings=settings,
        )
        record = await self._store.create_session(command)
        self._authorize(principal, record)
        active = await self._activate_record(record, tool_sources=tool_sources)
        if owner is not None:
            lease = await self._coordinator.reserve(active, SessionOperationKind.CREATE)
            try:
                await self.register_owner(record.id, owner_name, owner, lease=lease)
            except BaseException:
                await lease.release()
                await self._join_or_start_close(active, "owner_registration")
                self._discard_owner_if_empty(owner, record.id)
                raise
            await lease.release()
        return active

    async def get(
        self, session_id: str, *, principal: "PrincipalContext"
    ) -> ActiveRuntimeSession:
        active = self._active.get(session_id)
        if active is not None:
            self._authorize(principal, active.record)
            logger.debug(
                "event=runtime.session.active_cache_hit session_id=%s", session_id
            )
            return active
        record = await self._store.get_session(session_id)
        if record is None:
            raise UnknownSessionError(session_id)
        self._authorize(principal, record)
        return await self._activate_record(record)

    async def list(
        self, *, principal: "PrincipalContext"
    ) -> "tuple[SessionRecord, ...]":
        result = []
        for record in await self._store.list_all_sessions():
            try:
                self._authorize(principal, record)
            except Exception:
                continue
            result.append(record)
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
            replacement = await self._prepare_mcp_replacement(active, tool_sources)
            return SessionLoadTransaction(
                self, active, lease, workspace, settings, history, replacement
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
        owner: "SessionResourceOwner | None" = None,
        owner_name: str = "",
    ) -> ActiveRuntimeSession:
        self._validate_owner(owner, owner_name)
        active = await self.get(session_id, principal=principal)
        lease = await self._coordinator.reserve(active, SessionOperationKind.RESUME)
        try:
            replacement = await self._prepare_mcp_replacement(active, tool_sources)
            await self._commit_replacement(
                active,
                lease,
                replacement,
                workspace=workspace,
                settings=settings,
            )
        except BaseException:
            await lease.release()
            raise
        if owner is not None:
            try:
                await self.register_owner(session_id, owner_name, owner, lease=lease)
            except BaseException:
                await lease.release()
                await self._join_or_start_close(active, "owner_registration")
                self._discard_owner_if_empty(owner, session_id)
                raise
        await lease.release()
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
        owner: "SessionResourceOwner | None" = None,
        owner_name: str = "",
    ) -> ActiveRuntimeSession:
        self._validate_owner(owner, owner_name)
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
        active = await self._activate_record(record, tool_sources=tool_sources)
        if owner is not None:
            lease = await self._coordinator.reserve(active, SessionOperationKind.CREATE)
            try:
                await self.register_owner(record.id, owner_name, owner, lease=lease)
            except BaseException:
                await lease.release()
                await self._join_or_start_close(active, "owner_registration")
                self._discard_owner_if_empty(owner, record.id)
                raise
            await lease.release()
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
        lease = await self._coordinator.reserve(active, SessionOperationKind.UPDATE)
        try:
            return await self._committer.commit_update(
                active, lease, workspace=workspace, settings=settings
            )
        finally:
            await lease.release()

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
        self,
        session_id: str,
        name: str,
        owner: SessionResourceOwner,
        *,
        lease: "SessionOperationLease | None" = None,
    ) -> None:
        if not name:
            raise ValueError("owner name is required")
        active = self._active.get(session_id)
        if active is None:
            raise UnknownSessionError(session_id)
        async with active.lock:
            if active.cleanup_required:
                raise SessionCleanupRequiredError(session_id)
            if active.closing_requested:
                raise SessionBusyError(session_id)
            if active.record.state is SessionState.CLOSED:
                raise SessionClosedError(session_id)
            if lease is not None and active.operation is not lease:
                raise SessionInvariantError("session owner lease changed")
            current = active.owners.get(name)
            if current is owner:
                return
            if current is not None and not current.is_empty(session_id):
                raise StorageConflictError(f"resource owner {name!r} is active")
            active.owners[name] = owner
            logger.info(
                "event=runtime.session.owner_registered session_id=%s owner_name=%s",
                session_id,
                name,
            )

    async def toolsets(
        self,
        session_id: str,
        *,
        principal: "PrincipalContext",
        lease: SessionOperationLease,
    ) -> "tuple[Any, ...]":
        active = await self.get(session_id, principal=principal)
        async with active.lock:
            if active.operation is not lease or lease.kind is not SessionOperationKind.PROMPT:
                raise SessionOperationError("MCP toolsets require the active PROMPT lease")
            if active.record.state is not SessionState.OPEN:
                raise SessionOperationError("MCP toolsets require an open session")
            if active.closing_requested or active.cleanup_required:
                raise SessionOperationError("MCP toolsets are unavailable during session cleanup")
            if active.active_execution_id != lease.execution_id:
                raise SessionOperationError("MCP toolsets require the active execution")
            resources = active.mcp_resources
            tool_sources = active.tool_sources
        if resources is None and tool_sources and self._mcp_resource_factory is not None:
            candidate = self._mcp_resource_factory(tool_sources)
            discarded_candidate = None
            operation_changed = False
            async with active.lock:
                if active.operation is not lease:
                    discarded_candidate = candidate
                    operation_changed = True
                elif active.mcp_resources is None:
                    active.mcp_resources = candidate
                    active.owners["mcp"] = candidate
                    resources = candidate
                else:
                    discarded_candidate = candidate
                    resources = active.mcp_resources
            if discarded_candidate is not None:
                failures = await self._close_candidate(session_id, discarded_candidate)
                if failures:
                    async with active.lock:
                        active.cleanup_required = True
                        self._set_candidate_owner(active, discarded_candidate)
                    raise McpCleanupRequiredError(failures=failures)
                if operation_changed:
                    raise SessionInvariantError("session operation lease changed")
        if resources is None:
            return ()
        try:
            return await resources.toolsets()
        except McpCleanupRequiredError:
            async with active.lock:
                active.cleanup_required = True
            raise

    async def close(
        self, session_id: str, *, principal: "PrincipalContext", reason: str = "client"
    ) -> SessionCloseResult:
        active = await self.get(session_id, principal=principal)
        return await self._join_or_start_close(active, reason)

    async def shutdown(self) -> "tuple[SessionCloseResult, ...]":
        return tuple(
            await asyncio.gather(
                *(self._join_or_start_close(active, "shutdown") for active in tuple(self._active.values())),
                return_exceptions=False,
            )
        )

    async def _join_or_start_close(
        self, active: ActiveRuntimeSession, reason: str
    ) -> SessionCloseResult:
        async with active.lock:
            if active.close_task is None:
                # task-owner: runtime.close
                active.close_task = asyncio.create_task(self._close_once(active, reason))
                joined = False
            else:
                joined = True
            task = active.close_task
        logger.info(
            "event=runtime.session.close_joined session_id=%s close_reason=%s close_joined=%s",
            active.record.id,
            reason,
            joined,
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                await task
            except BaseException:
                pass
            raise

    async def _close_once(
        self, active: ActiveRuntimeSession, reason: str
    ) -> SessionCloseResult:
        async with active.lock:
            session_id = active.record.id
            if active.record.state is SessionState.CLOSED and not active.owners:
                await self._remove_active(active)
                active.close_task = None
                return SessionCloseResult(True)
            active.closing_requested = True
            execution_id = active.active_execution_id
            operation = active.operation
        if execution_id is not None and self._interaction_cancel is not None:
            await self._interaction_cancel(session_id)
        if operation is not None and operation.kind is not SessionOperationKind.CLOSE:
            await operation.done.wait()
        async with active.lock:
            owners_by_name = dict(active.owners)
        if self._interaction_owner is not None:
            owners_by_name.setdefault("interaction", self._interaction_owner)
        ordered_names = sorted(
            owners_by_name,
            key=lambda name: (
                0 if name == "interaction" else 1 if name == "acp.client" else 3 if name == "mcp" else 2,
                name,
            ),
        )
        failures: "list[ResourceFailure]" = []
        for name in ordered_names:
            owner = owners_by_name[name]
            try:
                failures.extend(await owner.close(session_id))
            except Exception as exc:
                failures.append(ResourceFailure(name, None, type(exc).__name__))
        failures.extend(
            ResourceFailure(name, None, "owner_not_empty")
            for name in ordered_names
            if not owners_by_name[name].is_empty(session_id)
        )
        if failures:
            async with active.lock:
                active.cleanup_required = True
                active.closing_requested = False
                active.close_task = None
            logger.error(
                "event=runtime.session.cleanup_failed session_id=%s resource_count=%s",
                session_id,
                len(failures),
            )
            return SessionCloseResult(False, tuple(failures))
        async with active.lock:
            already_closed = active.record.state is SessionState.CLOSED
            closed_record = active.record
        if already_closed:
            record = closed_record
        else:
            lease = SessionOperationLease(
                self._coordinator, active, SessionOperationKind.CLOSE, uuid4().hex, None
            )
            async with active.lock:
                active.operation = lease
            try:
                record = await self._committer.commit_update(
                    active, lease, state=SessionState.CLOSED
                )
            except Exception as exc:
                async with active.lock:
                    active.cleanup_required = True
                    active.closing_requested = False
                    active.close_task = None
                return SessionCloseResult(
                    False, (ResourceFailure("session", None, type(exc).__name__),)
                )
            finally:
                await lease.release()
        async with active.lock:
            active.owners.clear()
            active.close_task = None
        await self._remove_active(active)
        logger.info("event=runtime.session.closed session_id=%s", record.id)
        return SessionCloseResult(True)

    async def _prepare_mcp_replacement(
        self,
        active: ActiveRuntimeSession,
        tool_sources: "tuple[MCPServerSpec, ...]",
    ) -> McpReplacement:
        async with active.lock:
            old_specs = active.tool_sources
            old_resources = active.mcp_resources
        candidate = (
            self._mcp_resource_factory(tool_sources)
            if self._mcp_resource_factory is not None and tool_sources
            else None
        )
        return McpReplacement(
            old_specs=old_specs,
            old_resources=old_resources,
            new_specs=tool_sources,
            candidate=candidate,
        )

    async def _commit_replacement(
        self,
        active: ActiveRuntimeSession,
        lease: SessionOperationLease,
        replacement: McpReplacement,
        *,
        workspace: "SessionWorkspace",
        settings: "SessionSettings",
    ) -> SessionRecord:
        # task-owner: runtime.session.mcp_replacement
        task = asyncio.create_task(
            self._commit_replacement_once(
                active,
                lease,
                replacement,
                workspace=workspace,
                settings=settings,
            )
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            logger.info(
                "event=runtime.session.replacement_commit_joined session_id=%s operation_id=%s",
                active.record.id,
                lease.operation_id,
            )
            raise

    async def _commit_replacement_once(
        self,
        active: ActiveRuntimeSession,
        lease: SessionOperationLease,
        replacement: McpReplacement,
        *,
        workspace: "SessionWorkspace",
        settings: "SessionSettings",
    ) -> SessionRecord:
        async with active.lock:
            session_id = active.record.id
            was_closed = active.record.state is SessionState.CLOSED
        old_failures: tuple[ResourceFailure, ...] = ()
        if replacement.old_resources is not None:
            old_failures = await replacement.old_resources.close(session_id)
        if old_failures:
            candidate_failures = await self._close_candidate(session_id, replacement.candidate)
            failures = old_failures + candidate_failures
            async with active.lock:
                active.cleanup_required = True
                if candidate_failures:
                    self._set_candidate_owner(active, replacement.candidate)
            logger.error(
                "event=runtime.session.mcp_replacement_failed session_id=%s mcp_close_failure_count=%s",
                session_id,
                len(failures),
            )
            raise McpReplacementError(
                "MCP resource replacement failed", tuple(failures)
            )
        def apply(record: SessionRecord) -> None:
            active.tool_sources = replacement.new_specs
            active.mcp_resources = replacement.candidate
            active.owners.pop("mcp", None)
            if replacement.candidate is not None:
                active.owners["mcp"] = replacement.candidate

        try:
            record = await self._committer.commit_update(
                active,
                lease,
                workspace=workspace,
                settings=settings,
                state=SessionState.OPEN,
                apply=apply,
            )
        except BaseException:
            candidate_failures = await self._close_candidate(
                session_id, replacement.candidate
            )
            restored_resources = None
            restore_error: "BaseException | None" = None
            if (
                not candidate_failures
                and replacement.old_specs
                and self._mcp_resource_factory is not None
            ):
                try:
                    restored_resources = self._mcp_resource_factory(
                        replacement.old_specs
                    )
                except BaseException as exc:
                    restore_error = exc
            if candidate_failures:
                logger.error(
                    "event=runtime.session.mcp_replacement_failed session_id=%s mcp_close_failure_count=%s",
                    session_id,
                    len(candidate_failures),
                )
            async with active.lock:
                if candidate_failures:
                    active.cleanup_required = True
                    self._set_candidate_owner(active, replacement.candidate)
                else:
                    active.mcp_resources = restored_resources
                    active.tool_sources = replacement.old_specs
                    active.owners.pop("mcp", None)
                    if active.mcp_resources is not None:
                        active.owners["mcp"] = active.mcp_resources
                    if restore_error is not None:
                        active.cleanup_required = True
            raise
        replacement.committed = True
        logger.info(
            "event=runtime.session.mcp_replacement_started session_id=%s mcp_old_count=%s mcp_new_count=%s",
            session_id,
            len(replacement.old_specs),
            len(replacement.new_specs),
        )
        if was_closed:
            logger.info(
                "event=runtime.session.reopened session_id=%s session_revision=%s session_state=%s",
                session_id,
                record.revision,
                record.state.value,
            )
        return record

    async def _activate_record(
        self,
        record: SessionRecord,
        *,
        tool_sources: "tuple[MCPServerSpec, ...]" = (),
    ) -> ActiveRuntimeSession:
        async with self._active_lock:
            active = self._active.get(record.id)
            if active is not None:
                logger.debug(
                    "event=runtime.session.active_cache_hit session_id=%s", record.id
                )
                logger.info(
                    "event=runtime.session.active_identity_conflict session_id=%s",
                    record.id,
                )
                self._active_identity_conflict_count += 1
                return active
            active = ActiveRuntimeSession(record, tool_sources)
            self._active[record.id] = active
            logger.info(
                "event=runtime.session.active_cache_created session_id=%s active_count=%s",
                record.id,
                len(self._active),
            )
            return active

    async def _remove_active(self, active: ActiveRuntimeSession) -> None:
        async with self._active_lock:
            if self._active.get(active.record.id) is active:
                self._active.pop(active.record.id, None)

    @property
    def active_session_count(self) -> int:
        return len(self._active)

    @property
    def active_identity_conflict_count(self) -> int:
        return self._active_identity_conflict_count

    @property
    def cleanup_required_count(self) -> int:
        return sum(1 for active in self._active.values() if active.cleanup_required)

    @staticmethod
    def _validate_owner(
        owner: "SessionResourceOwner | None", owner_name: str
    ) -> None:
        if owner is not None and not owner_name:
            raise ValueError("owner_name is required when owner is provided")

    @staticmethod
    def _discard_owner_if_empty(owner: SessionResourceOwner, session_id: str) -> None:
        discard = getattr(owner, "discard_if_empty", None)
        if discard is not None:
            discard()

    @staticmethod
    def _set_candidate_owner(
        active: ActiveRuntimeSession, candidate: Any
    ) -> None:
        if candidate is not None:
            active.owners["mcp.candidate"] = candidate

    async def _close_candidate(
        self, session_id: str, candidate: Any
    ) -> tuple[ResourceFailure, ...]:
        if candidate is None:
            return ()
        try:
            return tuple(await candidate.close(session_id))
        except Exception as exc:
            return (ResourceFailure("mcp.candidate", None, type(exc).__name__),)

    def _authorize(self, principal: "PrincipalContext", record: SessionRecord) -> None:
        if self._authorization is not None:
            self._authorization.assert_session_access(principal=principal, session=record)

__all__ = [
    "ActiveRuntimeSession",
    "McpReplacement",
    "ResourceFailure",
    "RuntimeSessionService",
    "SessionCloseResult",
    "SessionCommitter",
    "SessionLoadTransaction",
    "SessionOperationCoordinator",
    "SessionOperationKind",
    "SessionOperationLease",
    "SessionResourceOwner",
]
