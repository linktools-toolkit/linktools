#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Session lifecycle coordination over narrow ports."""

from linktools.core import environ

from ...domain.execution import ExecutionHandle, ExecutionStatus, Page
from ...domain.session import Session, SessionStatus
from ...foundation.clock import Clock, SystemClock
from ...foundation.errors import ErrorCode, LinktoolsAIError
from ...foundation.ids import deterministic_id
from ...ports.session import SessionRepository, SessionResourcePort

logger = environ.get_logger("ai.application.services.session")


class SessionLifecycleService:
    """Implement Session actions without executing an Agent or SDK."""

    def __init__(
        self,
        repository: SessionRepository,
        resources: "SessionResourcePort | None" = None,
        lease: "object | None" = None,
        clock: "Clock | None" = None,
    ) -> None:
        self._repository = repository
        self._resources = resources
        self._lease = lease
        self._clock = clock or SystemClock()
        self._lease_fences: "dict[str, int]" = {}

    async def create(self, request: object) -> Session:
        session_id = request.session_id or deterministic_id(b"session", request.owner_id, request.project_id, request.agent_id, request.agent_revision)
        session = Session(
            session_id=session_id, owner_id=request.owner_id, project_id=request.project_id,
            agent_id=request.agent_id, agent_revision=request.agent_revision, profile=request.profile,
            bundle_digest=request.bundle_digest, settings=request.settings,
        )
        logger.info("session create id=%s project=%s", session_id, request.project_id)
        return await self._repository.create(session)

    async def get(self, session_id: str) -> Session:
        return await self._require(session_id)

    async def list(self, request: object) -> "Page[Session]":
        value = await self._repository.list(request.owner_id, request.limit, request.cursor)
        if isinstance(value, Page):
            return Page(items=value.items, next_cursor=value.next_cursor)
        return Page(items=tuple(value))

    async def load(self, session_id: str, request: object) -> "tuple[Session, tuple[str, ...]]":
        session = await self._require(session_id)
        if session.owner_id != request.owner_id or session.project_id != request.project_id:
            raise LinktoolsAIError(ErrorCode.AUTHORIZATION_DENIED, "session scope does not match")
        if request.bundle_digest is not None and request.bundle_digest != session.bundle_digest:
            raise LinktoolsAIError(ErrorCode.SESSION_CONFLICT, "session bundle digest does not match")
        if self._resources is not None:
            await self._resources.reconcile(session)
        return session, session.mcp_resource_refs

    async def resume(self, session_id: str, request: object) -> ExecutionHandle:
        session = await self._require(session_id)
        if session.status not in {SessionStatus.OPEN, SessionStatus.BUSY}:
            raise LinktoolsAIError(ErrorCode.SESSION_CONFLICT, "closed session cannot resume")
        if self._lease is not None:
            operation_lease = await self._lease.claim(session_id, session.owner_id, f"resume:{session.revision}")
            self._lease_fences[session_id] = operation_lease.fence
        execution_id = deterministic_id(b"session-run", session.session_id, session.revision, request.input)
        next_session = session.model_copy(update={"status": SessionStatus.BUSY, "current_execution_id": execution_id, "revision": session.revision + 1})
        await self._repository.compare_and_set(session_id, session.revision, next_session)
        logger.info("session resume id=%s execution=%s", session_id, execution_id)
        return ExecutionHandle(execution_id=execution_id, status=ExecutionStatus.ACCEPTED)

    async def fork(self, session_id: str, request: object) -> Session:
        source = await self._require(session_id)
        new_id = request.session_id or deterministic_id(b"session-fork", source.session_id, source.revision, request.owner_id)
        forked = source.model_copy(update={"session_id": new_id, "owner_id": request.owner_id, "project_id": request.project_id or source.project_id, "current_execution_id": None, "status": SessionStatus.OPEN, "revision": 1})
        return await self._repository.create(forked)

    async def update(self, session_id: str, request: object) -> Session:
        session = await self._require(session_id)
        if session.revision != request.expected_revision:
            raise LinktoolsAIError(ErrorCode.SESSION_CONFLICT, "session revision is stale")
        settings = session.settings if request.settings is None else request.settings
        resources = session.mcp_resource_refs
        if request.mcp_resource_refs is not None:
            resources = request.mcp_resource_refs
        updated = session.model_copy(
            update={
                "settings": settings,
                "mcp_resource_refs": resources,
                "revision": session.revision + 1,
            }
        )
        if request.mcp_resource_refs is not None and self._resources is not None:
            prepared = await self._resources.prepare(updated)
            await self._resources.swap(session, prepared)
        return await self._repository.compare_and_set(session_id, request.expected_revision, updated)

    async def close(self, session_id: str, request: object) -> Session:
        session = await self._require(session_id)
        if session.status is SessionStatus.CLOSED:
            return session
        if session.revision != request.expected_revision:
            raise LinktoolsAIError(ErrorCode.SESSION_CONFLICT, "session revision is stale")
        closing = session.model_copy(update={"status": SessionStatus.CLOSING, "revision": session.revision + 1})
        await self._repository.compare_and_set(session_id, request.expected_revision, closing)
        try:
            if self._resources is not None:
                await self._resources.release(closing)
            if self._lease is not None and session_id in self._lease_fences:
                await self._lease.release(
                    session_id,
                    session.owner_id,
                    self._lease_fences.pop(session_id),
                )
            closed = closing.model_copy(update={"status": SessionStatus.CLOSED, "revision": closing.revision + 1})
        except Exception as error:
            logger.warning("session cleanup required id=%s error=%s", session_id, type(error).__name__)
            closed = closing.model_copy(update={"status": SessionStatus.CLEANUP_REQUIRED, "revision": closing.revision + 1})
        return await self._repository.compare_and_set(session_id, closing.revision, closed)

    async def _require(self, session_id: str) -> Session:
        value = await self._repository.get(session_id)
        if value is None:
            raise LinktoolsAIError(ErrorCode.SESSION_NOT_FOUND, "session not found")
        return value

__all__ = ["SessionLifecycleService"]
