#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SessionStore: the Protocol every Session persistence backend implements.
Session is a pure record + append-only message log -- no physical path, no
copy(), matching the store contract."""

from typing import Any, Mapping, Protocol, runtime_checkable

from .models import NewSessionMessage, SessionMessage, SessionRecord, SessionStatus


@runtime_checkable
class SessionStore(Protocol):
    async def create(self, session: SessionRecord) -> SessionRecord: ...

    async def get(self, session_id: str) -> "SessionRecord | None": ...

    async def append_messages(
        self,
        session_id: str,
        messages: "tuple[NewSessionMessage, ...]",
    ) -> "tuple[SessionMessage, ...]":
        """Persist ``messages``, assigning ``id``/``sequence``/``created_at``
        for each (the store is the SOLE sequence authority,
        not the caller). Returns the persisted messages in the same order,
        with sequence numbers assigned contiguously starting after the
        session's current max sequence."""
        ...

    async def append_messages_once(
        self,
        *,
        commit_id: str,
        session_id: str,
        messages: "tuple[NewSessionMessage, ...]",
    ) -> "tuple[SessionMessage, ...]":
        """Commit-scoped idempotent batch append. The store atomically
        reserves (session_id, commit_id) for the whole batch -- a retried
        call with the SAME commit_id returns the originally-persisted batch
        instead of re-appending. Used by Run lifecycle commits (pause/
        complete/...) so a retry does not duplicate the session messages a
        commit recorded. The store does NOT dedupe via a full list-then-
        filter; the unique constraint on (session_id, commit_id, batch_index)
        is the atomic reserve."""
        ...

    async def list_messages(
        self, session_id: str, *, after_sequence: int = 0, limit: int = 1000
    ) -> "tuple[SessionMessage, ...]": ...

    async def update(
        self,
        session_id: str,
        *,
        status: "SessionStatus | None" = None,
        metadata: "Mapping[str, Any] | None" = None,
    ) -> SessionRecord: ...
