#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local (file / in-memory) session store implementations."""

import asyncio
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai.messages import ModelMessage

from .artifact import ArtifactMeta, ArtifactRef
from .history import (
    CallPromptSnapshot,
    SessionContextSnapshot,
    load_message_history,
    new_call_messages,
    write_call_prompt,
    write_session_context,
)
from .protocols import RunStatus, SessionStatusInfo

if TYPE_CHECKING:
    from .types import Session, SessionTurn


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


class LocalArtifactStore:
    async def persist_call_sidecar(self, session: "Session", turn: "SessionTurn") -> None:
        await asyncio.to_thread(
            write_call_prompt,
            session.root,
            CallPromptSnapshot(
                call_id=str(turn.llm_call.get("call_id") or ""),
                messages=new_call_messages(turn.history, turn.all_messages, system_prompt=turn.system_prompt),
            ),
        )


class ReadOnlyArtifactStore:
    """ArtifactStore that makes no writes — use when a session reads an existing trace."""

    async def persist_call_sidecar(self, session: "Session", turn: "SessionTurn") -> None:
        pass


class InMemoryRunStatusStore:
    def __init__(self) -> None:
        self._statuses: "dict[str, RunStatus]" = {}

    async def start(self, run_id: str) -> None:
        self._statuses[run_id] = RunStatus(state="running")

    async def update(self, run_id: str, status: RunStatus) -> None:
        self._statuses[run_id] = status

    async def get(self, run_id: str) -> RunStatus:
        return self._statuses[run_id]


class LocalAgentArtifactStore:
    """Local filesystem implementation of `AgentArtifactStore` (session/artifact.py) --
    that Protocol had zero implementations before this. Files live at
    `root / ref.key`, which `ArtifactRef.key` already produces as a safe relative path."""

    def __init__(self, root: Path) -> None:
        self.root = root

    async def get(self, ref: ArtifactRef) -> "bytes | None":
        path = self.root / ref.key
        if not await asyncio.to_thread(path.exists):
            return None
        return await asyncio.to_thread(path.read_bytes)

    async def put(self, ref: ArtifactRef, content: bytes, *, idempotency_key: str) -> ArtifactMeta:
        path = self.root / ref.key
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_bytes, content)
        checksum = hashlib.sha256(content).hexdigest()
        return ArtifactMeta(
            ref=ref,
            checksum=checksum,
            size_bytes=len(content),
            backend="local",
            location=str(path),
            status="stored",
            metadata={"idempotency_key": idempotency_key},
        )
