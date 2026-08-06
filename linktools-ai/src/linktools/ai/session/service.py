#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Narrow public facade for the Session application service."""

from .api import SessionApi
from .model import (
    CloseSessionRequest,
    CreateSessionRequest,
    ForkSessionRequest,
    ListSessionsRequest,
    LoadedSession,
    LoadSessionRequest,
    ResumeSessionRequest,
    SessionView,
    UpdateSessionRequest,
)
from ..domain.execution import ExecutionHandle, Page
from ..domain.session import Session


class SessionService(SessionApi):
    """Forward Session API calls to one injected lifecycle action object."""

    def __init__(self, actions: object) -> None:
        self._actions = actions

    async def create(self, request: CreateSessionRequest) -> SessionView:
        return self._view(await self._actions.create(request))

    async def get(self, session_id: str) -> SessionView:
        return self._view(await self._actions.get(session_id))

    async def list(self, request: ListSessionsRequest) -> "Page[SessionView]":
        page = await self._actions.list(request)
        return Page(items=tuple(self._view(item) for item in page.items), next_cursor=page.next_cursor)

    async def load(self, session_id: str, request: LoadSessionRequest) -> LoadedSession:
        session, resource_refs = await self._actions.load(session_id, request)
        return LoadedSession(session=self._view(session), resource_refs=resource_refs)

    async def resume(self, session_id: str, request: ResumeSessionRequest) -> ExecutionHandle:
        return await self._actions.resume(session_id, request)

    async def fork(self, session_id: str, request: ForkSessionRequest) -> SessionView:
        return self._view(await self._actions.fork(session_id, request))

    async def update(self, session_id: str, request: UpdateSessionRequest) -> SessionView:
        return self._view(await self._actions.update(session_id, request))

    async def close(self, session_id: str, request: CloseSessionRequest) -> SessionView:
        return self._view(await self._actions.close(session_id, request))

    @staticmethod
    def _view(session: object) -> SessionView:
        if isinstance(session, Session):
            return SessionView(
                session_id=session.session_id,
                owner_id=session.owner_id,
                project_id=session.project_id,
                agent_id=session.agent_id,
                agent_revision=session.agent_revision,
                bundle_digest=session.bundle_digest,
                profile=session.profile,
                settings=session.settings,
                status=session.status,
                revision=session.revision,
                current_execution_id=session.current_execution_id,
            )
        return session


__all__ = ["SessionService"]
