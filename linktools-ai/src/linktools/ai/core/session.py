#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Execution session context for agent runs.

A `Session` is the canonical execution context for one agent instance. It coordinates:

- runtime filesystem layout through `RunContext`
- multi-turn history through an injected `HistoryStore`
- sidecars / finalize / restore through an injected `ArtifactStore`

`FileSession` and `DbSession` remain as compatibility construction helpers for the two
current assembly modes:

- local file-backed runtime history for pipeline / worker executions
- DB-backed transcript history for long-lived conversational lines
"""

from __future__ import annotations

import abc
import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelMessage

import logging

from .environment import AgentEnvironment
from .model_runtime import RuntimeModelConfig
from .session_window import (
    NoopSummaryPolicy,
    RecentWindowPolicy,
    SessionSummary,
    SessionSummaryPolicy,
    SessionWindowPolicy,
)
from .session_coordination import InMemorySessionCoordinator, SessionCoordinator, coordinator_for_store
from .stores import (
    ArtifactStore,
    DbHistoryStore,
    FileHistoryStore,
    HistoryStore,
    LocalArtifactStore,
    SessionStatus,
    SessionStatusInfo,
    SessionStatusStore,
    TranscriptStore,
)

logger = logging.getLogger("linktools.ai.core.session")
_AUTO_COORDINATION = object()


@dataclass(frozen=True, slots=True)
class RunContext:
    """Runtime-local filesystem layout for one session execution context."""

    workspace_root: Path
    trace_root: Path
    runtime_dir: Path
    session_dir: Path


@dataclass(frozen=True, slots=True)
class SessionTurn:
    history: list[ModelMessage]
    all_messages: list[ModelMessage]
    model: RuntimeModelConfig
    token_usage: dict[str, Any]
    llm_call: dict[str, Any]
    display_entries: list[dict[str, Any]] | None = None
    system_prompt: str = ""


@dataclass(frozen=True, slots=True)
class FileSessionSpec:
    trace_id: str
    session_id: str
    history_store: HistoryStore | None = None
    artifact_store: ArtifactStore | None = None
    status_store: SessionStatusStore | None = None
    coordination: Any = _AUTO_COORDINATION
    window_policy: SessionWindowPolicy = field(default_factory=RecentWindowPolicy)


@dataclass(frozen=True, slots=True)
class DbSessionSpec:
    session_id: str
    store: TranscriptStore
    trace_id: str | None = None
    artifact_store: ArtifactStore | None = None
    status_store: SessionStatusStore | None = None
    coordination: Any = _AUTO_COORDINATION
    trace_root: Path | None = None
    window_policy: SessionWindowPolicy = field(default_factory=RecentWindowPolicy)
    summary_policy: SessionSummaryPolicy = field(default_factory=NoopSummaryPolicy)


@dataclass(frozen=True, slots=True)
class SessionEvent:
    seq: int
    event_id: str
    event_type: str
    role: str
    token_estimate: int
    payload: dict[str, Any]
    turn_id: str | None = None
    call_id: str | None = None
    parent_call_id: str | None = None
    visible_in_ui: bool = False

    def as_dict(self) -> dict[str, Any]:
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


def normalize_display_entry(entry: dict[str, Any]) -> dict[str, Any]:
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


def normalize_display_entries(entries: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not entries:
        return []
    return [normalize_display_entry(item) for item in entries if isinstance(item, dict)]


def build_input_display_entries(inputs: Any) -> list[dict[str, Any]] | None:
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
    trace_id: str | None = None
    session_kind: str | None = None
    parent_session_id: str | None = None
    head_seq: int = 0
    snapshot_seq: int = 0
    snapshot_messages: list[dict[str, Any]] = field(default_factory=list)
    snapshot_token_estimate: int = 0
    history_token_budget: int = 24000
    status: str | None = None
    meta_json: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
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
            self.trace_id is None
            and self.session_kind is None
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
    events: list[SessionEvent] = field(default_factory=list)
    loaded_until_seq: int = 0


@dataclass(frozen=True, slots=True)
class SessionTranscriptWrite:
    session_id: str
    trace_id: str
    head: SessionTranscriptHead
    events: list[SessionEvent] = field(default_factory=list)


class Session(abc.ABC):
    """Execution context for an agent run + its message-history persistence backend.

    `Session` is the stable agent-facing API. `FileSession` and `DbSession` remain as
    compatibility wrappers around the two current backend assembly modes.
    """

    session_id: str
    trace_id: str
    run: RunContext
    parent_session_id: str | None
    history_store: HistoryStore
    artifact_store: ArtifactStore
    status_store: SessionStatusStore | None
    coordination: SessionCoordinator | None

    @classmethod
    def create(cls, environ: AgentEnvironment, spec: FileSessionSpec) -> "FileSession":
        """Create a filesystem-backed session (pipeline / runtime default)."""
        return FileSession.create(environ, spec)

    @classmethod
    def create_db(cls, environ: AgentEnvironment, spec: DbSessionSpec) -> "Session":
        """Create a DB-backed session through the unified Session factory."""
        return DbSession.create(environ, spec)

    @property
    def workspace_root(self) -> Path:
        return self.run.workspace_root

    @property
    def trace_root(self) -> Path:
        return self.run.trace_root

    @property
    def runtime_dir(self) -> Path:
        return self.run.runtime_dir

    @property
    def session_dir(self) -> Path:
        return self.run.session_dir

    async def get_status(self) -> SessionStatusInfo:
        if self.status_store is None:
            return SessionStatusInfo(type="idle", updated_at=datetime.now(timezone.utc).isoformat())
        return await self.status_store.get(self.session_id)

    async def set_status(self, status: SessionStatus, *, message: str | None = None) -> None:
        if self.status_store is None:
            return
        await self.status_store.set(
            self.session_id,
            SessionStatusInfo(type=status, message=message, updated_at=datetime.now(timezone.utc).isoformat()),
        )

    # -- persistence contract (driven by base.py) -----------------------------

    async def load_history(self) -> list[ModelMessage]:
        """Restore the prior multi-turn history for this session (or [])."""
        return await self.history_store.load(self)

    async def persist(self, turn: SessionTurn) -> None:
        """Persist the completed turn (full history) for the next call to restore."""
        coordination = getattr(self, "coordination", None)
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

    async def finalize(self) -> dict[str, Any] | None:
        return await self.artifact_store.finalize(self)

    async def restore(self) -> Path | None:
        return await self.artifact_store.restore(self)

    @abc.abstractmethod
    def copy(self, *, child_session_id: str) -> "Session":
        """Derive a child session for a stage/subagent run."""
        ...

@dataclass(slots=True)
class FileSession(Session):
    """Filesystem-backed session: history in `session_dir/context.json`, per-call
    prompt sidecars under `session_dir/calls/`."""

    session_id: str
    trace_id: str
    run: RunContext
    parent_session_id: str | None = None
    history_store: HistoryStore = field(default_factory=FileHistoryStore, repr=False)
    artifact_store: ArtifactStore = field(default_factory=LocalArtifactStore, repr=False)
    status_store: SessionStatusStore | None = field(default=None, repr=False)
    coordination: SessionCoordinator | None = field(default_factory=InMemorySessionCoordinator, repr=False)
    window_policy: SessionWindowPolicy = field(default_factory=RecentWindowPolicy, repr=False)

    @classmethod
    def create(cls, environ: AgentEnvironment, spec: FileSessionSpec) -> "FileSession":
        session_id = spec.session_id or str(uuid.uuid1())
        trace_root = environ.trace_root(spec.trace_id)
        return cls(
            session_id=session_id,
            trace_id=spec.trace_id,
            run=RunContext(
                workspace_root=environ.workspace_root,
                trace_root=trace_root,
                runtime_dir=trace_root / "runtime",
                session_dir=trace_root / "session" / session_id,
            ),
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

    def copy(self, *, child_session_id: str) -> "FileSession":
        session_id = child_session_id or str(uuid.uuid1())
        return FileSession(
            session_id=session_id,
            trace_id=self.trace_id,
            run=RunContext(
                workspace_root=self.workspace_root,
                trace_root=self.trace_root,
                runtime_dir=self.runtime_dir,
                session_dir=self.trace_root / "session" / session_id,
            ),
            parent_session_id=self.session_id,
            history_store=self.history_store,
            artifact_store=self.artifact_store,
            status_store=self.status_store,
            coordination=self.coordination,
            window_policy=self.window_policy,
        )

@dataclass(slots=True)
class DbSession(Session):
    """DB-backed session for conversational lines: the durable multi-turn history persists
    to a `TranscriptStore` keyed by `session_id`, so chat survives pod loss and works on
    any pod. Only the conversation *state* moves to the DB; the per-call prompt sidecar
    (best-effort pod-local debug 留痕) is still written, same as `FileSession`, so the
    trace inspector keeps working uniformly across backends."""

    session_id: str
    trace_id: str
    run: RunContext
    store: TranscriptStore = field(repr=False)
    parent_session_id: str | None = None
    artifact_store: ArtifactStore = field(default_factory=LocalArtifactStore, repr=False)
    history_store: HistoryStore = field(init=False, repr=False)
    status_store: SessionStatusStore | None = field(default=None, repr=False)
    coordination: Any = field(default=_AUTO_COORDINATION, repr=False)
    window_policy: SessionWindowPolicy = field(default_factory=RecentWindowPolicy, repr=False)
    summary_policy: SessionSummaryPolicy = field(default_factory=NoopSummaryPolicy, repr=False)
    history_token_budget: int = 24000
    loaded_head_seq: int = field(default=0, repr=False)
    loaded_summary: SessionSummary | None = field(default=None, repr=False)
    loaded_messages: list[ModelMessage] = field(default_factory=list, repr=False)
    pending_request_id: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.history_store = DbHistoryStore(self.store)
        if self.coordination is _AUTO_COORDINATION:
            self.coordination = coordinator_for_store(self.store)

    @classmethod
    def create(cls, environ: AgentEnvironment, spec: DbSessionSpec) -> "DbSession":
        trace_id = spec.trace_id or spec.session_id
        trace_root = spec.trace_root if spec.trace_root is not None else environ.trace_root(trace_id)
        return cls(
            session_id=spec.session_id,
            trace_id=trace_id,
            run=RunContext(
                workspace_root=environ.workspace_root,
                trace_root=trace_root,
                runtime_dir=trace_root / "runtime",
                session_dir=trace_root / "session" / spec.session_id,
            ),
            store=spec.store,
            artifact_store=spec.artifact_store or LocalArtifactStore(),
            status_store=spec.status_store,
            coordination=spec.coordination,
            window_policy=spec.window_policy,
            summary_policy=spec.summary_policy,
        )

    def copy(self, *, child_session_id: str) -> "DbSession":
        # Child sessions exist for stage/subagent fan-out, which conversational (chat) lines
        # don't use — a DbSession never spawns subagents. Fail loudly rather than invent a
        # transcript-key scheme no path exercises.
        raise NotImplementedError("DbSession does not support child sessions (no subagents in chat lines)")

    async def load_history(self) -> list[ModelMessage]:
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


def _persist_idempotency_key(turn: SessionTurn) -> str | None:
    llm_call = turn.llm_call if isinstance(turn.llm_call, dict) else {}
    for key in ("idempotency_key", "call_id", "turn_id"):
        value = llm_call.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None
