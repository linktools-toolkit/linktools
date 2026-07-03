#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Execution session context for agent runs.

A `Session` is the canonical execution context for one agent instance. It coordinates:

- multi-turn history through an injected `HistoryStore`
- sidecars (FileSession only) through an injected `ArtifactStore`

`FileSession` and `RemoteSession` remain as compatibility construction helpers for the two
current assembly modes:

- local file-backed runtime history for pipeline / worker executions
- DB-backed transcript history for long-lived conversational lines

Neither carries a "working directory" concept -- that belongs to `RuntimeAgent`
(see agent.py), not to the session. `Session` only manages the session/history
lifecycle; it does no file/bash tool execution and knows nothing about traces.
"""

import abc
import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelMessage

import logging

from ..core.model_runtime import RuntimeModelConfig
from .window import (
    NoopSummaryPolicy,
    RecentWindowPolicy,
    SessionSummary,
    SessionSummaryPolicy,
    SessionWindowPolicy,
)
from .coordination import InMemorySessionCoordinator, SessionCoordinator, coordinator_for_store
from .protocols import (
    ArtifactStore,
    HistoryStore,
    SessionStatus,
    SessionStatusInfo,
    SessionStatusStore,
    TranscriptStore,
)
from .local import FileHistoryStore, LocalArtifactStore
from .remote import RemoteHistoryStore

logger = logging.getLogger("linktools.ai.session.types")
_AUTO_COORDINATION = object()


@dataclass(frozen=True, slots=True)
class SessionTurn:
    history: "list[ModelMessage]"
    all_messages: "list[ModelMessage]"
    model: RuntimeModelConfig
    token_usage: "dict[str, Any]"
    llm_call: "dict[str, Any]"
    display_entries: "list[dict[str, Any]] | None" = None
    system_prompt: str = ""


@dataclass(frozen=True, slots=True)
class FileSessionSpec:
    session_id: str
    history_store: "HistoryStore | None" = None
    artifact_store: "ArtifactStore | None" = None
    status_store: "SessionStatusStore | None" = None
    coordination: Any = _AUTO_COORDINATION
    window_policy: SessionWindowPolicy = field(default_factory=RecentWindowPolicy)


@dataclass(frozen=True, slots=True)
class RemoteSessionSpec:
    session_id: str
    store: TranscriptStore
    status_store: "SessionStatusStore | None" = None
    coordination: Any = _AUTO_COORDINATION
    window_policy: SessionWindowPolicy = field(default_factory=RecentWindowPolicy)
    summary_policy: SessionSummaryPolicy = field(default_factory=NoopSummaryPolicy)


@dataclass(frozen=True, slots=True)
class SessionEvent:
    seq: int
    event_id: str
    event_type: str
    role: str
    token_estimate: int
    payload: "dict[str, Any]"
    turn_id: "str | None" = None
    call_id: "str | None" = None
    parent_call_id: "str | None" = None
    visible_in_ui: bool = False

    def as_dict(self) -> "dict[str, Any]":
        return {
            "seq": self.seq,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "role": self.role,
            "turn_id": self.turn_id,
            "call_id": self.call_id,
            "parent_call_id": self.parent_call_id,
            "visible_in_ui": 1 if self.visible_in_ui else 0,
            "token_estimate": self.token_estimate,
            "payload": self.payload,
        }


def normalize_display_entry(entry: "dict[str, Any]") -> "dict[str, Any]":
    normalized = dict(entry)
    text = normalized.get("text")
    if isinstance(text, str):
        text = text.strip()
        if text:
            normalized["text"] = text
            normalized.setdefault("content", text)
    content = normalized.get("content")
    if isinstance(content, str):
        content = content.strip()
        if content:
            normalized["content"] = content
            normalized.setdefault("text", content)
    return normalized


def normalize_display_entries(entries: "list[dict[str, Any]] | None") -> "list[dict[str, Any]]":
    if not entries:
        return []
    return [normalize_display_entry(item) for item in entries if isinstance(item, dict)]


def build_input_display_entries(inputs: Any) -> "list[dict[str, Any]] | None":
    if isinstance(inputs, dict):
        explicit_entries = normalize_display_entries(inputs.get("display_entries"))
        if explicit_entries:
            return explicit_entries
        for key in ("display_text", "text", "content", "question"):
            value = inputs.get(key)
            if isinstance(value, str) and value.strip():
                return [normalize_display_entry({"kind": "user_text", "role": "user", "text": value})]
        return None
    if isinstance(inputs, str) and inputs.strip():
        return [normalize_display_entry({"kind": "user_text", "role": "user", "text": inputs})]
    return None


@dataclass(frozen=True, slots=True)
class SessionTranscriptHead:
    session_kind: "str | None" = None
    parent_session_id: "str | None" = None
    head_seq: int = 0
    snapshot_seq: int = 0
    snapshot_messages: "list[dict[str, Any]]" = field(default_factory=list)
    snapshot_token_estimate: int = 0
    history_token_budget: int = 24000
    status: "str | None" = None
    meta_json: "dict[str, Any] | None" = None

    def as_dict(self) -> "dict[str, Any]":
        return {
            "session_kind": self.session_kind,
            "parent_session_id": self.parent_session_id,
            "head_seq": self.head_seq,
            "snapshot_seq": self.snapshot_seq,
            "snapshot_messages": list(self.snapshot_messages),
            "snapshot_token_estimate": self.snapshot_token_estimate,
            "history_token_budget": self.history_token_budget,
            "status": self.status,
            "meta_json": self.meta_json,
        }

    @property
    def is_empty(self) -> bool:
        return (
            self.session_kind is None
            and self.parent_session_id is None
            and self.head_seq == 0
            and self.snapshot_seq == 0
            and not self.snapshot_messages
            and self.snapshot_token_estimate == 0
            and self.history_token_budget == 24000
            and self.status is None
            and self.meta_json is None
        )


@dataclass(frozen=True, slots=True)
class SessionTranscript:
    head: SessionTranscriptHead
    events: "list[SessionEvent]" = field(default_factory=list)
    loaded_until_seq: int = 0


@dataclass(frozen=True, slots=True)
class SessionTranscriptWrite:
    session_id: str
    head: SessionTranscriptHead
    events: "list[SessionEvent]" = field(default_factory=list)


class Session(abc.ABC):
    """Execution context for an agent run + its message-history persistence backend.

    `Session` is the stable agent-facing API. `FileSession` and `RemoteSession` remain as
    compatibility wrappers around the two current backend assembly modes. `Session` itself
    does no file I/O and knows nothing about "traces" or working directories -- only
    `FileSession` (via its own `root`/`artifact_store`) and `RuntimeAgent` (via its own
    `workdir`) touch the filesystem.
    """

    session_id: str
    parent_session_id: "str | None"
    history_store: HistoryStore
    status_store: "SessionStatusStore | None"
    coordination: "SessionCoordinator | None"

    @classmethod
    def create(cls, root: Path, spec: FileSessionSpec) -> "FileSession":
        """Create a filesystem-backed session (pipeline / runtime default).

        `root` is used exactly as given -- the session's own directory, not a
        parent directory the session computes a subpath under. Callers who
        want per-session subdirectories build that path themselves before
        calling this."""
        return FileSession.create(root, spec)

    @classmethod
    def create_db(cls, spec: RemoteSessionSpec) -> "Session":
        """Create a DB-backed session through the unified Session factory."""
        return RemoteSession.create(spec)

    async def get_status(self) -> SessionStatusInfo:
        if self.status_store is None:
            return SessionStatusInfo(type="idle", updated_at=datetime.now(timezone.utc).isoformat())
        return await self.status_store.get(self.session_id)

    async def set_status(self, status: SessionStatus, *, message: "str | None" = None) -> None:
        if self.status_store is None:
            return
        await self.status_store.set(
            self.session_id,
            SessionStatusInfo(type=status, message=message, updated_at=datetime.now(timezone.utc).isoformat()),
        )

    # -- persistence contract (driven by base.py) -----------------------------

    async def load_history(self) -> "list[ModelMessage]":
        """Restore the prior multi-turn history for this session (or [])."""
        return await self.history_store.load(self)

    async def persist(self, turn: SessionTurn) -> None:
        """Persist the completed turn's history for the next call to restore.

        Base implementation only touches `history_store` -- `FileSession`
        overrides this to additionally write its per-call sidecar inside the
        same coordination-gated block (see `FileSession.persist`)."""
        coordination = getattr(self, "coordination", None)
        if coordination is None:
            await self.history_store.persist(self, turn)
            return
        idempotency_key = _persist_idempotency_key(turn)
        decision = await coordination.begin_persist(self.session_id, idempotency_key)
        if not decision.should_persist:
            return
        history_committed = bool(decision.history_already_committed)
        completed = False
        try:
            if not history_committed:
                await self.history_store.persist(self, turn)
                history_committed = True
            completed = True
        finally:
            await coordination.complete_persist(
                self.session_id,
                idempotency_key,
                history_committed=history_committed,
                completed=completed,
            )

    @abc.abstractmethod
    def copy(self, *, child_session_id: str) -> "Session":
        """Derive a child session for a stage/subagent run."""
        ...

@dataclass(slots=True)
class FileSession(Session):
    """Filesystem-backed session: history in `root/context.json`, per-call
    prompt sidecars under `root/calls/`."""

    session_id: str
    root: Path
    parent_session_id: "str | None" = None
    history_store: HistoryStore = field(default_factory=FileHistoryStore, repr=False)
    artifact_store: ArtifactStore = field(default_factory=LocalArtifactStore, repr=False)
    status_store: "SessionStatusStore | None" = field(default=None, repr=False)
    coordination: "SessionCoordinator | None" = field(default_factory=InMemorySessionCoordinator, repr=False)
    window_policy: SessionWindowPolicy = field(default_factory=RecentWindowPolicy, repr=False)

    @classmethod
    def create(cls, root: Path, spec: FileSessionSpec) -> "FileSession":
        session_id = spec.session_id or str(uuid.uuid1())
        return cls(
            session_id=session_id,
            root=root,
            history_store=spec.history_store or FileHistoryStore(),
            artifact_store=spec.artifact_store or LocalArtifactStore(),
            status_store=spec.status_store,
            coordination=(
                InMemorySessionCoordinator()
                if spec.coordination is _AUTO_COORDINATION
                else spec.coordination
            ),
            window_policy=spec.window_policy,
        )

    def copy(self, *, child_session_id: str, root: "Path | None" = None) -> "FileSession":
        session_id = child_session_id or str(uuid.uuid1())
        return FileSession(
            session_id=session_id,
            root=root if root is not None else self.root / session_id,
            parent_session_id=self.session_id,
            history_store=self.history_store,
            artifact_store=self.artifact_store,
            status_store=self.status_store,
            coordination=self.coordination,
            window_policy=self.window_policy,
        )

    async def persist(self, turn: SessionTurn) -> None:
        """Same coordination/idempotency gating as `Session.persist`, but also
        writes the per-call prompt sidecar (read by the console's prompt
        inspector) inside the same gated block. Does NOT call
        `super().persist()` -- duplicates the gating structure so the sidecar
        write stays gated on the same `should_persist` decision as history,
        matching prior behavior exactly (a skipped/duplicate turn skips both)."""
        coordination = self.coordination
        if coordination is None:
            await asyncio.gather(
                self.history_store.persist(self, turn),
                self.artifact_store.persist_call_sidecar(self, turn),
            )
            return
        idempotency_key = _persist_idempotency_key(turn)
        decision = await coordination.begin_persist(self.session_id, idempotency_key)
        if not decision.should_persist:
            return
        history_committed = bool(decision.history_already_committed)
        completed = False
        try:
            if not history_committed:
                await self.history_store.persist(self, turn)
                history_committed = True
            await self.artifact_store.persist_call_sidecar(self, turn)
            completed = True
        finally:
            await coordination.complete_persist(
                self.session_id,
                idempotency_key,
                history_committed=history_committed,
                completed=completed,
            )

@dataclass(slots=True)
class RemoteSession(Session):
    """DB-backed session for conversational lines: the durable multi-turn history persists
    to a `TranscriptStore` keyed by `session_id`, so chat survives pod loss and works on
    any pod. No file I/O of any kind -- no `root`, no `artifact_store`."""

    session_id: str
    store: TranscriptStore = field(repr=False)
    parent_session_id: "str | None" = None
    history_store: HistoryStore = field(init=False, repr=False)
    status_store: "SessionStatusStore | None" = field(default=None, repr=False)
    coordination: Any = field(default=_AUTO_COORDINATION, repr=False)
    window_policy: SessionWindowPolicy = field(default_factory=RecentWindowPolicy, repr=False)
    summary_policy: SessionSummaryPolicy = field(default_factory=NoopSummaryPolicy, repr=False)
    history_token_budget: int = 24000
    loaded_head_seq: int = field(default=0, repr=False)
    loaded_summary: "SessionSummary | None" = field(default=None, repr=False)
    loaded_messages: "list[ModelMessage]" = field(default_factory=list, repr=False)
    pending_request_id: "str | None" = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.history_store = RemoteHistoryStore(self.store)
        if self.coordination is _AUTO_COORDINATION:
            self.coordination = coordinator_for_store(self.store)

    @classmethod
    def create(cls, spec: RemoteSessionSpec) -> "RemoteSession":
        return cls(
            session_id=spec.session_id,
            store=spec.store,
            status_store=spec.status_store,
            coordination=spec.coordination,
            window_policy=spec.window_policy,
            summary_policy=spec.summary_policy,
        )

    def copy(self, *, child_session_id: str) -> "RemoteSession":
        # Child sessions exist for stage/subagent fan-out, which conversational (chat) lines
        # don't use — a RemoteSession never spawns subagents. Fail loudly rather than invent a
        # transcript-key scheme no path exercises.
        raise NotImplementedError("RemoteSession does not support child sessions (no subagents in chat lines)")

    async def load_history(self) -> "list[ModelMessage]":
        try:
            return await Session.load_history(self)
        except Exception:
            logger.warning("failed to load db session history [%s]", self.session_id, exc_info=True)
            return []

    async def persist(self, turn: SessionTurn) -> None:
        if self.coordination is None:
            await Session.persist(self, turn)
            return
        lease = await self.coordination.acquire_lease(self.session_id)
        try:
            await Session.persist(self, turn)
        finally:
            await lease.release()


def _persist_idempotency_key(turn: SessionTurn) -> "str | None":
    llm_call = turn.llm_call if isinstance(turn.llm_call, dict) else {}
    for key in ("idempotency_key", "call_id", "turn_id"):
        value = llm_call.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None
