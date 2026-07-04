#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local (file / in-memory) session store implementations."""

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pydantic_ai.messages import ModelMessage

from linktools.core import environ

from .history import (
    SessionContextSnapshot,
    load_message_history,
    write_session_context,
)
from .protocols import RunStatus, SessionStatusInfo

if TYPE_CHECKING:
    from .types import FileSession, Session, SessionTurn


def local_session(session_id: str) -> "FileSession":
    """Persistent FileSession rooted under this package's local data directory,
    keyed by `session_id` — the common case for CLI/local-tool callers who want
    conversation history to survive across process invocations."""
    from .types import FileSessionSpec, Session

    root = environ.get_data_path("ai", "sessions", session_id, create_parent=True)
    return Session.create(root, FileSessionSpec(session_id=session_id))


class InMemorySessionStatusStore:
    def __init__(self) -> None:
        self._states: "dict[str, SessionStatusInfo]" = {}

    async def get(self, session_id: str) -> SessionStatusInfo:
        return self._states.get(
            session_id,
            SessionStatusInfo(type="idle", updated_at=datetime.now(timezone.utc).isoformat()),
        )

    async def set(self, session_id: str, status: SessionStatusInfo) -> None:
        self._states[session_id] = status


class FileHistoryStore:
    async def load(self, session: "Session") -> "list[ModelMessage]":
        return await asyncio.to_thread(load_message_history, session.root)

    async def persist(self, session: "Session", turn: "SessionTurn") -> None:
        await asyncio.to_thread(
            write_session_context,
            session.root,
            SessionContextSnapshot(
                session_id=session.session_id,
                messages=turn.all_messages,
                model=turn.model,
                token_usage=turn.token_usage,
                llm_call=turn.llm_call,
            ),
        )


class InMemoryRunStatusStore:
    def __init__(self) -> None:
        self._statuses: "dict[str, RunStatus]" = {}

    async def start(self, run_id: str) -> None:
        self._statuses[run_id] = RunStatus(state="running")

    async def update(self, run_id: str, status: RunStatus) -> None:
        self._statuses[run_id] = status

    async def get(self, run_id: str) -> RunStatus:
        return self._statuses[run_id]


