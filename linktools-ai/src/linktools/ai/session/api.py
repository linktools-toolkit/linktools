#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Public Session runtime protocol."""

from typing import Protocol

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


class SessionApi(Protocol):
    async def create(self, request: CreateSessionRequest) -> SessionView: ...
    async def get(self, session_id: str) -> SessionView: ...
    async def list(self, request: ListSessionsRequest) -> "Page[SessionView]": ...
    async def load(self, session_id: str, request: LoadSessionRequest) -> LoadedSession: ...
    async def resume(self, session_id: str, request: ResumeSessionRequest) -> ExecutionHandle: ...
    async def fork(self, session_id: str, request: ForkSessionRequest) -> SessionView: ...
    async def update(self, session_id: str, request: UpdateSessionRequest) -> SessionView: ...
    async def close(self, session_id: str, request: CloseSessionRequest) -> SessionView: ...


__all__ = ["SessionApi"]
