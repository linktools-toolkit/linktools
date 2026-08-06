#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session query and mutation API."""

from typing import Protocol

from ..core import Page, Principal
from .services import (
    CloseSessionRequest,
    CreateSessionRequest,
    ExecutionHandle,
    ForkSessionRequest,
    ListSessionRequest,
    LoadedSession,
    ResumeSessionRequest,
    SessionView,
    UpdateSessionRequest,
)


class SessionQueryApi(Protocol):
    async def get(self, session_id: str, *, principal: Principal) -> SessionView: ...
    async def list(self, request: ListSessionRequest) -> 'Page[SessionView]': ...
    async def load(self, session_id: str, *, principal: Principal) -> LoadedSession: ...


class SessionApi(SessionQueryApi, Protocol):
    async def create(self, request: CreateSessionRequest) -> SessionView: ...
    async def resume(self, session_id: str, request: ResumeSessionRequest) -> ExecutionHandle: ...
    async def fork(self, session_id: str, request: ForkSessionRequest) -> SessionView: ...
    async def update(self, session_id: str, request: UpdateSessionRequest) -> SessionView: ...
    async def close(self, session_id: str, request: CloseSessionRequest) -> SessionView: ...


__all__ = ["SessionApi", "SessionQueryApi"]
