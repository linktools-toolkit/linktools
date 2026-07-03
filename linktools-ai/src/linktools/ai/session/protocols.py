#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session store Protocols: TranscriptStore / HistoryStore / ArtifactStore /
SessionStatusStore."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic_ai.messages import ModelMessage

if TYPE_CHECKING:
    from .types import Session, SessionTranscript, SessionTranscriptHead, SessionTranscriptWrite, SessionTurn


@runtime_checkable
class TranscriptStore(Protocol):
    """Durable backend for a DB-backed session history store.

    This is the DB analog of `context.json`, declared here so the agent layer stays
    decoupled from secops/DB internals.
    """

    async def head(self, session_id: str) -> "SessionTranscriptHead":
        """Return the stored session head for `session_id` (or an empty head)."""
        ...

    async def load(
        self,
        session_id: str,
        *,
        budget_tokens: int,
        after_seq: "int | None" = None,
        batch_size: int = 64,
    ) -> "SessionTranscript":
        """Return the restore snapshot plus an incrementally-read tail event window."""
        ...

    async def save(self, transcript: "SessionTranscriptWrite") -> None:
        """Upsert the session head and append any new ordered events for `session_id`."""
        ...


class HistoryStore(Protocol):
    """Persistence backend for multi-turn session history."""

    async def load(self, session: "Session") -> "list[ModelMessage]":
        ...

    async def persist(self, session: "Session", turn: "SessionTurn") -> None: ...


class ArtifactStore(Protocol):
    """Persistence backend for non-history session artifacts."""

    async def persist_call_sidecar(self, session: "Session", turn: "SessionTurn") -> None: ...


SessionStatus = Literal["idle", "busy", "retry", "error"]


@dataclass(frozen=True, slots=True)
class SessionStatusInfo:
    type: SessionStatus
    updated_at: str
    message: "str | None" = None


class SessionStatusStore(Protocol):
    """Runtime-only session activity state, separate from durable session metadata."""

    async def get(self, session_id: str) -> SessionStatusInfo:
        ...

    async def set(self, session_id: str, status: SessionStatusInfo) -> None:
        ...


@dataclass(frozen=True, slots=True)
class RunStatus:
    state: 'Literal["running", "done", "failed"]'
    result: Any = None
    error: "str | None" = None


@runtime_checkable
class RunStatusStore(Protocol):
    """Background-run status tracking for AgentKernel.start_background/check_background."""

    async def start(self, run_id: str) -> None:
        ...

    async def update(self, run_id: str, status: RunStatus) -> None:
        ...

    async def get(self, run_id: str) -> RunStatus:
        ...
