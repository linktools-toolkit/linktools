#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SessionStore: the Protocol every Session persistence backend implements.
Session is a pure record + append-only message log -- no physical path, no
copy(), matching spec docs/linktools-ai.md section 19."""

from typing import Any, Mapping, Protocol, runtime_checkable

from .models import NewSessionMessage, SessionMessage, SessionRecord, SessionStatus


@runtime_checkable
class SessionStore(Protocol):
    async def create(self, session: SessionRecord) -> SessionRecord:
        ...

    async def get(self, session_id: str) -> "SessionRecord | None":
        ...

    async def append_messages(
        self, session_id: str, messages: "tuple[NewSessionMessage, ...]",
    ) -> "tuple[SessionMessage, ...]":
        """Persist ``messages``, assigning ``id``/``sequence``/``created_at``
        for each (G6/review3 §6.3: the store is the SOLE sequence authority,
        not the caller). Returns the persisted messages in the same order,
        with sequence numbers assigned contiguously starting after the
        session's current max sequence."""
        ...

    async def list_messages(self, session_id: str, *, after_sequence: int = 0, limit: int = 1000) -> "tuple[SessionMessage, ...]":
        ...

    async def update(
        self,
        session_id: str,
        *,
        status: "SessionStatus | None" = None,
        metadata: "Mapping[str, Any] | None" = None,
    ) -> SessionRecord:
        ...
